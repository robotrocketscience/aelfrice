"""A lock must never be requested and silently lost (#1620).

A user ran `/aelf:lock <statement>`. It was never locked, nothing
reported a failure, and the same instruction was restated twelve times
over six weeks without binding. On the live store `lock_level='user'`
was set on 1 belief out of 20,176.

These arms come from hypotheses registered before the experiments ran.
Two of them, H3 and H4, fail on the pre-fix code and are the reason this
file exists; the rest are regression guards for behaviour that already
works and must keep working.

H1  lock works on novel text                                  worked
H2  lock works over an agent_inferred collision               worked
H3  lock silently no-ops over a SPECULATIVE collision         DEFECT
H4  the capture path ingests `/aelf:lock …` as a belief       DEFECT
H5  the L0 read keys on lock_level                            worked
H6  origin does not gate the L0 read                          worked
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_SPECULATIVE,
)
from aelfrice.store import MemoryStore


def _seed_unlocked(store: MemoryStore, text: str, origin: str) -> str:
    """Insert an unlocked belief the way passive capture would.

    Written through the raw connection rather than `insert_belief` so the
    row's origin is exactly what the experiment needs; the lock path
    resolves it by `content_hash`, which is a plain sha256 of the text
    for every source.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bid = content_hash[:16]
    store._conn.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, type,"
        " lock_level, created_at, origin, retention_class, lock_tier, scope)"
        " VALUES (?,?,?,0.6,1,'factual',?,'2026-01-01T00:00:00Z',?,"
        "'fact','frozen','project')",
        (bid, text, content_hash, LOCK_NONE, origin),
    )
    store._conn.commit()
    return bid


def _lock_level(store: MemoryStore, text: str) -> str | None:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    row = store._conn.execute(
        "SELECT lock_level FROM beliefs WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return None if row is None else row[0]


# --- H3: the silent-failure hole ----------------------------------------


@pytest.mark.timeout(60)
def test_locking_over_a_speculative_belief_does_not_report_success(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aelf lock` must not print success while leaving the belief unlocked.

    `_cmd_lock` skips the upgrade when the resolved row carries
    ORIGIN_SPECULATIVE — so #550's phantom-promotion path keeps its own
    rows — but printed a success line regardless and exited 0. The user
    is told the statement is locked when it is not, which is the single
    worst outcome for a durability primitive.
    """
    from aelfrice import cli

    text = "Cold beliefs should decay toward hibernation after ninety days."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed_unlocked(store, text, ORIGIN_SPECULATIVE)
    finally:
        store.close()

    rc = cli.main(["lock", text])
    out = capsys.readouterr()
    combined = out.out + out.err

    store = MemoryStore(str(db))
    try:
        level = _lock_level(store, text)
    finally:
        store.close()

    assert level != LOCK_USER, (
        "locking a speculative phantom here would strip its ORIGIN_SPECULATIVE "
        "and disqualify it from the #550 promotion path; it must be refused"
    )
    assert rc != 0, (
        "lock left the belief unlocked but exited 0 — a silent failure; "
        f"output was {combined!r}"
    )
    assert "phantom" in combined.lower() or "speculative" in combined.lower(), (
        "a refusal must name why, so the user can act on it; "
        f"output was {combined!r}"
    )


@pytest.mark.timeout(60)
def test_a_speculative_collision_never_prints_a_bare_success_line(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The word 'locked:' must not appear when nothing was locked.

    Separate from the exit code because the exit code is invisible to a
    human reading a terminal, and the reported failure was a human
    believing the wrong thing.
    """
    from aelfrice import cli

    text = "Retrieval should prefer recent beliefs over corroborated ones."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed_unlocked(store, text, ORIGIN_SPECULATIVE)
    finally:
        store.close()

    cli.main(["lock", text])
    out = capsys.readouterr()

    store = MemoryStore(str(db))
    try:
        level = _lock_level(store, text)
    finally:
        store.close()

    assert level != LOCK_USER, "the phantom must not be locked in place"
    assert "locked:" not in out.out, (
        f"printed a success line while lock_level stayed {level!r}: {out.out!r}"
    )


@pytest.mark.timeout(60)
def test_a_refused_lock_leaves_the_phantom_promotable(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal must not damage the row it declined to lock.

    `_cmd_lock` skips ORIGIN_SPECULATIVE because #550 Surface B owns
    those rows: `find_phantom_lock_matches` only sees beliefs still
    carrying that origin, and `promote_phantom` is what moves them to
    ORIGIN_USER_VALIDATED. Locking one here would rewrite its origin and
    silently disqualify it from its own promotion path.

    So the contract has two halves and this arm pins the second: after a
    refused lock the phantom is still a phantom, unlocked and unrewritten.
    Without it, deleting the speculative guard passes every other arm.
    """
    from aelfrice import cli

    text = "Hibernation should trigger after ninety cold days."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed_unlocked(store, text, ORIGIN_SPECULATIVE)
    finally:
        store.close()

    cli.main(["lock", text])
    capsys.readouterr()

    store = MemoryStore(str(db))
    try:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = store._conn.execute(
            "SELECT lock_level, origin FROM beliefs WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    finally:
        store.close()

    assert row is not None, "the refusal deleted the belief"
    assert row[0] == LOCK_NONE, f"the phantom was locked anyway: {row!r}"
    assert row[1] == ORIGIN_SPECULATIVE, (
        f"the phantom's origin was rewritten to {row[1]!r}, which removes it "
        "from the #550 promotion path"
    )


# --- H2, H1, H5, H6: regression guards for what already works ------------


@pytest.mark.timeout(60)
def test_locking_over_an_agent_inferred_belief_still_locks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common collision must keep working (f6f1d030, 2026-07-29).

    Guards the fix for H3 against being written as "skip every
    collision", which would break the case that motivated f6f1d030.
    """
    from aelfrice import cli

    text = "The upload retry budget is three attempts before surfacing an error."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    try:
        _seed_unlocked(store, text, ORIGIN_AGENT_INFERRED)
    finally:
        store.close()

    rc = cli.main(["lock", text])
    assert rc == 0

    store = MemoryStore(str(db))
    try:
        assert _lock_level(store, text) == LOCK_USER, (
            "a collision with an ordinary captured belief must still lock"
        )
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_locking_novel_text_locks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import cli

    text = "A statement that exists nowhere else in this store."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    rc = cli.main(["lock", text])
    assert rc == 0
    store = MemoryStore(str(db))
    try:
        assert _lock_level(store, text) == LOCK_USER
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_a_successful_lock_still_confirms_itself(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deferring the success line must not delete it.

    The fix for this issue moves the outcome line to AFTER #550 Surface
    B, so it is printed only once the result is known. The obvious way to
    get that wrong is to drop it: the lock then works silently, and a
    user with no confirmation is back to guessing — which is the problem
    this issue is about, in the opposite direction.
    """
    from aelfrice import cli

    text = "A statement locked to prove the confirmation still prints."
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    rc = cli.main(["lock", text])
    out = capsys.readouterr()

    assert rc == 0
    assert "locked:" in out.out, (
        f"a successful lock printed no confirmation: {out.out!r}"
    )


# --- H4: the capture path must not eat aelfrice's own commands ----------


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "prompt",
    [
        "/aelf:lock Route all questions and decisions through the question tool.",
        "/aelf:lock Use the house style guide for all documentation.",
        "/aelf:retire 78b07c65a1a61851 because it is stale now",
        "/aelf:promote 33997aa16979bd3c to user_validated origin",
        "aelf lock Route all questions and decisions through the question tool.",
        "uv run aelf lock Route all questions through the question tool.",
    ],
)
def test_an_aelfrice_command_is_not_captured_as_a_belief(prompt: str) -> None:
    """A command is an instruction to the tool, not a claim about the world.

    Storing `/aelf:lock <statement>` as a belief is worse than noise: it
    is the evidence that a lock was requested, spent on a low-confidence
    row about the request instead of on the request itself. Two such rows
    exist on the live store and neither produced a lock.
    """
    from aelfrice.hook import _should_skip_bm25

    skip, reason = _should_skip_bm25(prompt)
    assert skip, f"an aelfrice command was admitted for capture: {prompt!r}"
    assert reason is not None and "command" in reason, (
        f"expected a command-shaped skip reason, got {reason!r}"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "prompt",
    [
        "Route all questions, dispositions and decisions through the question tool.",
        "The scanner should skip files larger than one megabyte.",
        "We decided to hold the ranking change behind the gold set.",
        "Can you check whether aelf lock is working correctly for me?",
        "The command /aelf:lock did not appear to take effect yesterday.",
    ],
)
def test_ordinary_prose_is_still_captured(prompt: str) -> None:
    """The filter must not swallow a real statement.

    The last two cases matter most: a sentence that MENTIONS a command is
    a claim about the world and must still be captured. Only a prompt
    that IS a command is skipped.
    """
    from aelfrice.hook import _should_skip_bm25

    skip, reason = _should_skip_bm25(prompt)
    assert not skip, f"ordinary prose was skipped as a command: {prompt!r} ({reason})"


# --- H4b: self-ingestion -------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "prompt",
    [
        '<belief id="987f092b61d5ad20" lock="none">prioritization and other '
        "important decisions</belief>",
        "<aelfrice-memory>\nThe memory store contents below are in two trust "
        "tiers.\n</aelfrice-memory>",
    ],
)
def test_rendered_injection_is_not_re_ingested(prompt: str) -> None:
    """aelfrice must not re-admit its own rendered output as a belief.

    One such row exists on the live store, dated 2026-08-19: a belief
    whose content is the XML of another belief.
    """
    from aelfrice.hook import _should_skip_bm25

    skip, _ = _should_skip_bm25(prompt)
    assert skip, f"aelfrice re-ingested its own rendered block: {prompt[:60]!r}"


# --- H5 / H6: the L0 read path ------------------------------------------


@pytest.mark.timeout(60)
def test_the_locked_tier_keys_on_lock_level_not_origin(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked belief is L0 whatever its origin; an unlocked one is not."""
    from aelfrice import cli

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    locked_text = "This statement was locked deliberately."
    cli.main(["lock", locked_text])

    store = MemoryStore(str(db))
    try:
        unlocked_text = "This statement was never locked."
        _seed_unlocked(store, unlocked_text, ORIGIN_AGENT_INFERRED)
        odd_origin = "A locked belief whose origin stayed agent_inferred."
        _seed_unlocked(store, odd_origin, ORIGIN_AGENT_INFERRED)
        store._conn.execute(
            "UPDATE beliefs SET lock_level = ?, locked_at = '2026-01-02T00:00:00Z'"
            " WHERE content = ?",
            (LOCK_USER, odd_origin),
        )
        store._conn.commit()
        ids = {b.content for b in store.list_locked_beliefs()}
    finally:
        store.close()

    assert locked_text in ids
    assert odd_origin in ids, "origin must not gate the locked tier"
    assert unlocked_text not in ids, "an unlocked belief must not reach the locked tier"
