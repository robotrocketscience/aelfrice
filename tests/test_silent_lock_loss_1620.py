"""A lock must never be requested and silently lost (#1620).

A user ran `/aelf:lock <statement>`. It was never locked, nothing
reported a failure, and the same instruction was restated twelve times
over six weeks without binding. On the live store `lock_level='user'`
was set on 1 belief out of 20,176.

Two routes lost it and both are covered here.

**These arms drive the real surfaces.** An earlier attempt filtered
`_should_skip_bm25`, which is the BM25 *retrieval* gate (#674) and has
no belief writer behind it — so the capture path was untouched and the
tests still passed. Capture runs through `ingest.py` ->
`noise_filter.is_transcript_noise`, and the end-to-end arm below drives
`ingest_jsonl` so that mistake cannot repeat silently.

Hypotheses registered before the experiments ran:

H1  lock works on novel text                                  worked
H2  lock works over an agent_inferred collision               worked
H3  lock left a matching phantom unlocked while saying so     DEFECT
H4  capture stored `/aelf:lock ...` as a belief               DEFECT
H5  the L0 read keys on lock_level                            worked
H6  origin does not gate the L0 read                          worked
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
)
from aelfrice.noise_filter import is_transcript_noise
from aelfrice.store import MemoryStore

# Split so this module does not itself contain a matchable command.
_LOCK_CMD = "/aelf:" + "lock "


def _seed(store: MemoryStore, text: str, origin: str, btype: str = "factual") -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bid = content_hash[:16]
    store._conn.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, type,"
        " lock_level, created_at, origin, retention_class, lock_tier, scope)"
        " VALUES (?,?,?,0.6,1,?,?,'2026-01-01T00:00:00Z',?,"
        "'fact','frozen','project')",
        (bid, text, content_hash, btype, LOCK_NONE, origin),
    )
    store._conn.commit()
    return bid


def _row(store: MemoryStore, text: str):
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return store._conn.execute(
        "SELECT id, lock_level, origin FROM beliefs WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()


# --- H3: a lock must end with the statement actually locked -------------


@pytest.mark.timeout(60)
def test_locking_a_matching_phantom_promotes_it_AND_locks_it(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator ruling: promotion is not a lock.

    `_cmd_lock` used to skip the lock write on ORIGIN_SPECULATIVE, so a
    statement matching a phantom was promoted, reported as
    `locked: <id>`, and exited 0 while `aelf locked` did not list it.

    Promotion now runs BEFORE the write. `find_phantom_lock_matches`
    only sees rows still carrying ORIGIN_SPECULATIVE, so promoting first
    means the write can no longer disqualify a phantom from its own
    path — which is the only reason the skip existed.
    """
    from aelfrice import cli

    text = "Cold beliefs should decay toward hibernation after ninety days."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed(store, text, "speculative", btype="speculative")
    finally:
        store.close()

    rc = cli.main(["lock", text])
    out = capsys.readouterr()

    store = MemoryStore(str(db))
    try:
        row = _row(store, text)
        listed = [b.content for b in store.list_locked_beliefs()]
    finally:
        store.close()

    assert rc == 0, f"lock failed: {out.out!r} {out.err!r}"
    assert row is not None and row[1] == LOCK_USER, (
        f"the statement was not locked: {tuple(row) if row else None!r}"
    )
    assert text in listed, "locked but absent from `aelf locked`"
    assert "promoted phantom" in out.out, "the phantom was not promoted"


@pytest.mark.timeout(60)
def test_a_promotion_keeps_its_stronger_origin(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locking a promoted phantom must not demote how it got there.

    ORIGIN_USER_VALIDATED is the promotion target and a stronger
    provenance than ORIGIN_USER_STATED: the user both asserted the
    statement and validated a phantom already carrying it. Overwriting
    it breaks `tests/test_phantom_promotion_trigger.py`.
    """
    from aelfrice import cli

    text = "Retrieval should prefer recent beliefs over corroborated ones."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed(store, text, "speculative", btype="speculative")
    finally:
        store.close()

    cli.main(["lock", text])

    store = MemoryStore(str(db))
    try:
        row = _row(store, text)
    finally:
        store.close()

    assert row is not None
    assert row[1] == LOCK_USER, "the statement must still be locked"
    assert row[2] == ORIGIN_USER_VALIDATED, (
        f"a promotion was demoted to {row[2]!r}"
    )


@pytest.mark.timeout(60)
def test_a_lock_that_did_not_take_reports_failure(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command must never print success over an unlocked statement.

    Simulated by making the write a no-op, which is the shape every
    silent-loss route ends in. What matters is that the verification
    fires, not how the write failed.
    """
    from aelfrice import cli
    from aelfrice.store import MemoryStore as _MS

    text = "A statement whose lock write is made to do nothing."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    # Seed an unlocked collision FIRST. On novel text `derive()` already
    # stamps LOCK_USER at insert, so `update_belief` is never called and
    # neutering it would prove nothing — the arm has to run the path
    # where that write is the one doing the locking.
    store = MemoryStore(str(db))
    try:
        _seed(store, text, ORIGIN_AGENT_INFERRED)
    finally:
        store.close()

    monkeypatch.setattr(_MS, "update_belief", lambda self, b: None)
    rc = cli.main(["lock", text])
    out = capsys.readouterr()

    assert rc != 0, f"reported success over an unlocked statement: {out.out!r}"
    assert "locked:" not in out.out, f"printed a success line: {out.out!r}"
    assert "FAILED" in out.err or "not locked" in out.err, out.err


# --- H1, H2, H5, H6: regression guards ----------------------------------


@pytest.mark.timeout(60)
def test_locking_novel_text_locks_and_confirms(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice import cli

    text = "A statement that exists nowhere else in this store."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    rc = cli.main(["lock", text])
    out = capsys.readouterr()
    assert rc == 0
    assert "locked:" in out.out, f"no confirmation printed: {out.out!r}"

    store = MemoryStore(str(db))
    try:
        assert _row(store, text)[1] == LOCK_USER
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_locking_over_an_agent_inferred_belief_still_locks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common collision must keep working (f6f1d030, 2026-07-29)."""
    from aelfrice import cli

    text = "The upload retry budget is three attempts before surfacing an error."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed(store, text, ORIGIN_AGENT_INFERRED)
    finally:
        store.close()

    assert cli.main(["lock", text]) == 0
    store = MemoryStore(str(db))
    try:
        assert _row(store, text)[1] == LOCK_USER
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_the_locked_tier_keys_on_lock_level_not_origin(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice import cli

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    locked_text = "This statement was locked deliberately."
    cli.main(["lock", locked_text])

    store = MemoryStore(str(db))
    try:
        unlocked = "This statement was never locked."
        _seed(store, unlocked, ORIGIN_AGENT_INFERRED)
        odd = "A locked belief whose origin stayed agent_inferred."
        _seed(store, odd, ORIGIN_AGENT_INFERRED)
        store._conn.execute(
            "UPDATE beliefs SET lock_level = ?, locked_at = '2026-01-02T00:00:00Z'"
            " WHERE content = ?",
            (LOCK_USER, odd),
        )
        store._conn.commit()
        listed = {b.content for b in store.list_locked_beliefs()}
    finally:
        store.close()

    assert locked_text in listed
    assert odd in listed, "origin must not gate the locked tier"
    assert unlocked not in listed


# --- H4: the CAPTURE path, driven end to end ----------------------------


@pytest.mark.timeout(120)
def test_ingest_does_not_write_an_aelfrice_command_as_a_belief(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end arm, and the reason this file exists twice.

    An earlier attempt filtered `_should_skip_bm25`, which gates BM25
    retrieval and has no belief writer behind it. Every unit test
    passed and `ingest_jsonl` still wrote the command as a belief.
    Driving the real ingest path is the only arm that can catch that.
    """
    from aelfrice.ingest import ingest_jsonl

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    keep = "Route all questions, dispositions and decisions through the tool."
    rows = [
        _LOCK_CMD + "Route all questions and decisions through the tool.",
        "aelf lock Route all questions through the tool",
        '<belief id="987f092b61d5ad20" lock="none">prioritization</belief>',
        keep,
    ]
    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "ts": f"2026-01-01T00:0{i}:00Z",
                    "role": "user",
                    "text": t,
                    "session_id": "s1",
                    "turn_id": f"t{i}",
                }
            )
            for i, t in enumerate(rows)
        )
        + "\n",
        encoding="utf-8",
    )

    store = MemoryStore(str(db))
    try:
        ingest_jsonl(store, str(path))
        written = [r[0] for r in store._conn.execute("SELECT content FROM beliefs")]
    finally:
        store.close()

    commands = [
        w
        for w in written
        if w.startswith("/aelf:") or w.startswith("aelf ") or w.startswith("<belief")
    ]
    assert not commands, f"ingest wrote commands as beliefs: {commands}"
    assert keep in written, "the ordinary statement was dropped with them"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        _LOCK_CMD + "Route all questions and decisions through the tool.",
        _LOCK_CMD + "Use the house style guide for all documentation.",
        "/aelf:retire 78b07c65a1a61851 because it is stale now",
        "/aelf:upgrade",
        "aelf lock Route all questions through the tool",
        '<belief id="987f092b61d5ad20" lock="none">prioritization</belief>',
        "<aelfrice-memory>",
        "<core>",
        "<locked>",
        # Removing these four together used to fail nothing.
        "<belief>bare open tag with no attributes</belief>",
        "<session-start>",
        "<recent-work>",
        "<cadence-checkpoint>",
    ],
)
def test_commands_and_rendered_output_are_transcript_noise(text: str) -> None:
    assert is_transcript_noise(text), f"admitted as a belief: {text[:60]!r}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        "Route all questions, dispositions and decisions through the tool.",
        "The scanner should skip files larger than one megabyte.",
        # A sentence that MENTIONS a command is a claim about the world.
        "The command /aelf:lock did not appear to take effect yesterday.",
        # These OPEN with the token and are still prose. An earlier
        # attempt skipped all of them, which is silent memory loss —
        # the same class of defect as the one being fixed.
        "aelf locks should always be injected verbatim in every session.",
        "aelf should never write to the live store during a benchmark run.",
        "uv run aelf is the invocation we document in the README.",
    ],
)
def test_ordinary_prose_is_still_captured(text: str) -> None:
    assert not is_transcript_noise(text), f"dropped a real statement: {text[:60]!r}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        "aelf locks are injected verbatim every session",
        "aelf is the tool we use for durable memory",
    ],
)
def test_unpunctuated_prose_opening_with_the_cli_name_is_dropped(text: str) -> None:
    """A known limitation, pinned so it is visible rather than implied.

    The category-1 escape hatch is punctuation-only
    (`_looks_like_written_prose`), so only a *punctuated* sentence
    opening with `aelf ` survives. The unpunctuated form is dropped.

    That is the pre-existing #1371 design, shared with the `git ` and
    `pytest ` prefixes, not something this change introduced — and the
    module docstring notes the transcript logger writes prompts
    verbatim, so unpunctuated prose is common in this corpus. Recorded
    here because an earlier draft of the changelog claimed such
    sentences "still land", which holds only with a full stop.

    If this arm starts failing, the escape hatch got smarter and the
    claim can be widened.
    """
    assert is_transcript_noise(text), (
        "the limitation changed — update the claim in the changelog too"
    )
