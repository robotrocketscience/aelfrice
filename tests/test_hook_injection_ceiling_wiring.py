"""#1551: the bounds are wired into every block the hooks emit.

`test_hook_injection_ceiling.py` drives the pure functions. Deleting the
call that applies either of them left that whole module green, and the
suite with it, because nothing there went through a hook. These tests
drive the hooks and assert on captured stdout.

Three emit sites carry the same envelope and all three are exercised:

* `user_prompt_submit`'s retrieval branch;
* its `elif gate_skip:` branch, reached when the #674 prompt-shape gate
  refuses BM25 on a session's first prompt — a first prompt under 12
  characters, an acknowledgement. On shipped defaults that is routine,
  and it was unbounded: 16,526 estimated tokens from a 300-lock store
  against a 6,000-token ceiling, with nothing on stderr;
* `session_start`, likewise unbounded.

The drop policy under test: **both bounds stop at `lock="user"`.** Every
lock survives every trim, and a block that cannot fit without dropping
one is emitted over the ceiling with a note saying so.

Fixture sizes are literals, sized by `scripts/measure_block_ceiling.py`
against the shipped ceiling rather than derived from it. Nothing here
reads `AELFRICE_HOOK_BLOCK_CEILING`; it is deleted from the environment
so an exported value cannot change a result.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from aelfrice.hook import (
    AUDIT_HOOK_USER_PROMPT_SUBMIT,
    CLOSE_TAG,
    OPEN_TAG,
    SESSION_START_CLOSE_TAG,
    SESSION_START_OPEN_TAG,
    _audit_path_for_db,
    _audit_tokens_from_block,
    _cap_belief_content,
    _telemetry_path_for_db,
    read_hook_audit,
    read_user_prompt_submit_telemetry,
    session_start,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.session_ring import read_ring_state
from aelfrice.store import MemoryStore

_CEILING_ENV = "AELFRICE_HOOK_BLOCK_CEILING"

# The shipped ceiling, as a literal. A fixture derived from
# `HOOK_BLOCK_TOKEN_CEILING` would follow the constant if someone raised
# it and prove nothing; `test_shipped_constants_are_pinned` guards the
# value itself.
_CEILING = 6000

_WORD = "banana"
_PROMPT = f"tell me everything about the {_WORD} please"
# Under `_MIN_PROMPT_LEN` (12), so `_should_skip_bm25` returns
# ("trivial:short") and the hook takes the `elif gate_skip:` emit path.
_GATED_PROMPT = "ok"


@pytest.fixture(autouse=True)
def _pin_ceiling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_CEILING_ENV, raising=False)


def _mk(
    bid: str,
    content: str,
    *,
    locked: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-26T00:00:00Z" if locked else None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(
    db: Path,
    *,
    n_locks: int = 0,
    lock_chars: int = 150,
    n_core: int = 0,
    core_chars: int = 200,
    core_matches_prompt: bool = False,
    n_hits: int = 0,
    hit_chars: int = 400,
    n_long: int = 0,
    long_chars: int = 5_000,
) -> tuple[list[str], list[str], list[str]]:
    """Seed a store and return `(lock_ids, core_ids, hit_ids)`.

    A `<core>` belief is one that qualifies on posterior: alpha 4, beta 1
    is mu 0.8 over alpha+beta 5, clearing `_CORE_MIN_POSTERIOR` (2/3) and
    `_CORE_MIN_ALPHA_BETA` (4). Core beliefs are not locked, so they are
    what the ceiling is allowed to drop.

    `core_matches_prompt` puts `_WORD` in the core content, so the same
    belief is both a `<core>` entry and a BM25 hit for `_PROMPT`. That is
    the shape #1547's dedupe collapses: the `<core>` entry renders
    verbatim and the hit renders as a `seen` pointer to it, in one
    envelope.

    `n_long` adds unlocked retrieval hits of `long_chars` characters,
    appended to `hit_ids`. They exist so a fixture can put a belief over
    `BELIEF_CONTENT_CHAR_CAP` into the emitted set: below the cap
    `_cap_belief_content` is the identity, and a fixture on which it is
    the identity cannot tell a call site that applies it from one that
    does not.
    """
    lock_ids: list[str] = []
    core_ids: list[str] = []
    hit_ids: list[str] = []
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            bid = f"L{i:031d}"
            store.insert_belief(
                _mk(bid, "lockword " + "q" * lock_chars, locked=True)
            )
            lock_ids.append(bid)
        for i in range(n_core):
            bid = f"C{i:031d}"
            store.insert_belief(
                _mk(
                    bid,
                    (f"{_WORD} core fact " if core_matches_prompt
                     else "coreword ") + "w" * core_chars,
                    alpha=4.0,
                    beta=1.0,
                )
            )
            core_ids.append(bid)
        for i in range(n_hits):
            bid = f"H{i:031d}"
            store.insert_belief(_mk(bid, f"{_WORD} fact " + "z" * hit_chars))
            hit_ids.append(bid)
        for i in range(n_long):
            bid = f"B{i:031d}"
            store.insert_belief(_mk(bid, f"{_WORD} fact " + "y" * long_chars))
            hit_ids.append(bid)
    finally:
        store.close()
    return lock_ids, core_ids, hit_ids


def _fire_ups(
    tmp_path: Path,
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str = _PROMPT,
    session_id: str = "s1",
) -> tuple[str, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )
    rc = user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr
    )
    assert rc == 0
    # The hook fails soft, so an exception anywhere inside it becomes a
    # stderr trace and rc 0 — indistinguishable from a small block.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def _fire_session_start(
    tmp_path: Path,
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": "s-start",
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "SessionStart",
        }
    )
    rc = session_start(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    assert rc == 0
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


# ---------------------------------------------------------------------------
# user_prompt_submit — the retrieval branch
# ---------------------------------------------------------------------------


def test_ups_ceiling_sheds_core_before_the_prompts_own_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling sheds prompt-independent content first.

    `<locked>`, `<core>` and `<recent-work>` are emitted above the
    per-turn hits, so a dropper that pops the body's tail takes the hits
    the prompt selected and keeps the `<core>` pool the prompt had no part
    in choosing — `<core>` is ranked by corroboration and posterior, and
    neither consults the prompt. Measured by
    `scripts/measure_block_ceiling.py --lanes` on this fixture's shape: 50
    locks, 20 `<core>` beliefs without the query term and 20 hits with it
    gave 6/20 hits and 19/20 core untrimmed, 0/20 hits and 17/20 core
    under tail-first, 6/20 hits and 7/20 core under the lane order.

    The control arm is the same store with `AELFRICE_HOOK_BLOCK_CEILING=0`,
    so what a hit lane of this store looks like untrimmed is measured
    rather than transcribed, and the shipped arm is compared against it.
    The three assertions are one expression, so no half can go dead: the
    trim must have run, the hit lane must be untouched by it, and the
    `<core>` lane must be where the bytes came from. A tail-first dropper
    fails the second, a dropper that fires on nothing fails the first.
    """
    _, core_ids, hit_ids = _seed(
        tmp_path / "ceiling.db",
        n_locks=50, lock_chars=150,
        n_core=20, core_chars=200,
        n_hits=20, hit_chars=400,
    )
    # A second store with identical contents: the control fire writes ring
    # and exposure rows, and a shared db would let the first arm move the
    # second arm's ranking.
    _seed(
        tmp_path / "control.db",
        n_locks=50, lock_chars=150,
        n_core=20, core_chars=200,
        n_hits=20, hit_chars=400,
    )

    monkeypatch.setenv(_CEILING_ENV, "0")
    control, control_err = _fire_ups(
        tmp_path, tmp_path / "control.db", monkeypatch, session_id="s-ctl"
    )
    assert control_err == "", control_err
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    out, err = _fire_ups(
        tmp_path, tmp_path / "ceiling.db", monkeypatch, session_id="s-cap"
    )

    dropped = re.search(r"dropped (\d+) belief element", err)
    control_hits = [b for b in hit_ids if b in control]
    control_core = [b for b in core_ids if b in control]
    assert control_hits and control_core, control[:2_000]
    assert {
        "trim_ran": bool(dropped) and int(dropped.group(1)) > 0,
        "hits": [b for b in hit_ids if b in out],
        "core_shed": len([b for b in core_ids if b in out]) < len(control_core),
    } == {
        "trim_ran": True,
        "hits": control_hits,
        "core_shed": True,
    }, (err, len(out), len(control))


def test_ups_retrieval_branch_trims_to_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    lock_ids, _, hit_ids = _seed(
        db, n_locks=60, lock_chars=150, n_hits=20, hit_chars=400
    )
    out, err = _fire_ups(tmp_path, db, monkeypatch)

    assert _audit_tokens_from_block(out) <= _CEILING
    assert "dropped" in err
    # Framing intact and no element half-removed.
    assert OPEN_TAG in out and CLOSE_TAG in out
    assert out.count("<belief ") == out.count("</belief>")
    # Every lock survived; some retrieval hits did not.
    assert [b for b in lock_ids if b not in out] == []
    assert any(b not in out for b in hit_ids)


def test_ups_retrieval_branch_leaves_a_fitting_block_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the mutation: a small store must be untouched."""
    db = tmp_path / "memory.db"
    lock_ids, _, _ = _seed(db, n_locks=3, lock_chars=150, n_hits=2)
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    assert err == ""
    assert [b for b in lock_ids if b not in out] == []


def test_ups_audit_record_omits_the_beliefs_the_ceiling_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped belief must not collect exposure evidence.

    `aelf tail` prints `beliefs[]` as what was injected, and the #779
    Layer-3 sweeper resolves an `injection_events` row against the next
    assistant turn — so a belief credited here but deleted before the
    write is scored `referenced=0` by construction.
    """
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "1")
    _seed(db, n_locks=60, lock_chars=150, n_hits=20, hit_chars=400)
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    assert "dropped" in err

    rows = read_hook_audit(_audit_path_for_db(db))
    ups = [
        r for r in rows
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    assert len(ups) == 1
    audited = {
        b["id"] for b in ups[0]["beliefs"]  # type: ignore[index,union-attr]
    }
    assert audited, ups[0]
    assert all(bid in out for bid in audited)
    assert ups[0]["n_beliefs"] == len(audited)


def test_ups_never_emits_a_seen_pointer_to_a_dropped_element(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trim must not leave a manifest pointer without its referent.

    `retrieval.seen_manifest_line`'s docstring is the contract: the entry
    exists because "the full text is already in this context window,
    above". On a session's first prompt #1547's dedupe puts both halves in
    one envelope — the belief renders verbatim in `<core>` and as a `seen`
    pointer among the per-turn hits — and a `<core>` element carries no
    `lock="user"`, so the ceiling was free to delete it and leave the
    pointer behind. Measured on the shipped 6,000-token ceiling before the
    fix: 57 `seen` ids against 55 elements, 2 of them dangling.

    The assertion is a whole-body invariant rather than a check on the two
    ids that happened to dangle, because the defect is the class and any
    future dropper reintroduces it the same way.
    """
    db = tmp_path / "memory.db"
    _seed(
        db,
        n_locks=55,
        lock_chars=200,
        n_core=40,
        core_chars=2_000,
        core_matches_prompt=True,
    )
    out, err = _fire_ups(tmp_path, db, monkeypatch)

    # The trim must have run, and the dedupe must have emitted pointers:
    # without both, the invariant below holds vacuously.
    assert "dropped" in err, err
    seen_ids = re.findall(r'^  seen (\S+): "', out, re.MULTILINE)
    assert seen_ids, out[:2_000]
    element_ids = set(re.findall(r'<belief id="([^"]+)"', out))
    assert [bid for bid in seen_ids if bid not in element_ids] == []


def test_ups_exposure_writes_omit_the_beliefs_the_ceiling_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three exposure writes the audit test above does not reach.

    The session ring's contract is "already shipped this session", and
    the next `PreToolUse:Grep|Glob|Bash` fire dedups against it — so an
    id entered without being shipped suppresses a belief the model never
    saw, which is a silent drop rather than a deduplication. A
    `belief_touches` row is the same claim in the sidecar table, and it
    outlives the session. An `injection_events` row is the sharpest of
    the three: the #779 Layer-3 sweeper resolves it against the *next*
    assistant turn, so a row for a belief that was deleted before the
    write scores `referenced=0` by construction and feeds that verdict
    into the meta-belief substrate. Measured with
    `hits=emitted_hits` reverted to `hits` on this fixture: 66 rows
    against 63 rendered `<belief>` elements, 3 credited but absent.

    All three are asserted as whole-set invariants against the emitted
    block: every id either side of the ceiling appears in what was
    written, so any future dropper is covered too. The dropped count is
    read off stderr first, because on a fixture the ceiling leaves alone
    every invariant holds vacuously.
    """
    db = tmp_path / "memory.db"
    session_id = "s-ring"
    _seed(db, n_locks=60, lock_chars=150, n_hits=20, hit_chars=400)
    out, err = _fire_ups(tmp_path, db, monkeypatch, session_id=session_id)
    dropped = re.search(r"dropped (\d+) belief element", err)
    assert dropped is not None, err
    assert int(dropped.group(1)) > 0, err

    ring_ids = [
        e["id"] for e in read_ring_state(session_id).get("ring", [])
    ]
    store = MemoryStore(str(db))
    try:
        # `current_fire_idx=0, window_k=1` puts the threshold at -1, so
        # the window covers every row this session has: the assertion is
        # about the whole table, not a recency slice of it.
        touched = store.read_touch_set_in_window(
            session_id, current_fire_idx=0, window_k=1
        )
        # Every row of this session: the sweeper has not run, so all of
        # them are still `referenced IS NULL`.
        injected = [
            row[2] for row in store.list_pending_injection_events(session_id)
        ]
    finally:
        store.close()
    assert ring_ids and touched and injected, (
        len(ring_ids), len(touched), len(injected)
    )

    # One assertion over all three sets, so no one of them can go dead
    # while another reports the failure.
    assert {
        "ring": [bid for bid in ring_ids if bid not in out],
        "belief_touches": [bid for bid in sorted(touched) if bid not in out],
        "injection_events": [bid for bid in injected if bid not in out],
    } == {"ring": [], "belief_touches": [], "injection_events": []}


def test_ups_seen_pointer_on_turn_two_names_a_belief_turn_one_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1382's ledger must not record what the ceiling deleted.

    `test_ups_never_emits_a_seen_pointer_to_a_dropped_element` covers the
    dangling pointer *within* one envelope. This is the cross-turn form,
    and it is worse: the ledger says "this text is in the context
    window", and a dropped belief written there emits a `seen` pointer to
    text the model was never shown on every later turn of the session,
    not just the next one.

    Measured on the 80-lock / 20-core store below with
    `AELFRICE_TURN_DIFFERENTIAL=1`, never-shown `seen` ids per turn are
    0/0/0/0 on the shipped routing and 0/2/2/2 when the ledger is fed
    `hits` instead of `emitted_hits` — the same two `<core>` ids, for the
    rest of the session.
    """
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    db = tmp_path / "memory.db"
    session_id = "s-ledger"
    _seed(
        db,
        n_locks=80,
        lock_chars=200,
        n_core=20,
        core_chars=2_000,
        core_matches_prompt=True,
    )
    first, err = _fire_ups(
        tmp_path, db, monkeypatch, session_id=session_id
    )
    assert "dropped" in err, err
    rendered_on_turn_one = set(re.findall(r'<belief id="([^"]+)"', first))
    assert rendered_on_turn_one, first[:2_000]

    second, _ = _fire_ups(
        tmp_path,
        db,
        monkeypatch,
        prompt=f"turn two: tell me more about the {_WORD} please",
        session_id=session_id,
    )
    # Turn 1 rendered the session-start sub-block; turn 2 does not, so
    # every pointer below comes from the ledger rather than from #1547's
    # in-envelope dedupe.
    assert "<locked>" in first
    assert "<locked>" not in second
    seen_ids = re.findall(r'^  seen (\S+): "', second, re.MULTILINE)
    assert seen_ids, second[:2_000]
    assert [
        bid for bid in seen_ids if bid not in rendered_on_turn_one
    ] == []


def test_ups_total_chars_stays_in_one_unit_across_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B6: the telemetry field must not change units at the boundary.

    `total_chars` is belief-content characters as injected, and `aelf
    doctor` renders it as "injection size p50/p95: N chars". An earlier
    revision of this change overwrote it with `len(body)` — whole
    rendered-block bytes, framing and manifest lines included — but only
    on fires the ceiling trimmed, so the percentiles mixed two units and
    switched between them exactly at the over-ceiling boundary, which is
    where the tail of the distribution is.

    The assertion is distinguishing on purpose: it names the value the
    field must hold AND the value it must not, because the two are within
    an order of magnitude of each other and an `> 0` check passes on both.

    The fixture carries one 5,012-character hit so the sum's per-belief
    cap is not the identity. Without it every seeded belief is under
    `BELIEF_CONTENT_CHAR_CAP` (1,200) — locks of 159 characters, hits of
    412 — `_cap_belief_content` returns its input on every row, and
    `expected` is the same figure whether the shipped sum applies the cap
    or not. The `uncapped` assertion below pins that the fixture keeps
    that property.
    """
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "1")
    _seed(
        db, n_locks=60, lock_chars=150, n_hits=20, hit_chars=400,
        n_long=1, long_chars=5_000,
    )
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    assert "dropped" in err

    tel = read_user_prompt_submit_telemetry(_telemetry_path_for_db(db))
    assert len(tel) == 1
    total_chars = tel[0]["total_chars"]

    # The emitted set, taken from the audit record, costed from the store.
    ups = [
        r for r in read_hook_audit(_audit_path_for_db(db))
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    emitted = ups[0]["beliefs"]  # type: ignore[index]
    store = MemoryStore(str(db))
    try:
        expected = 0
        uncapped = 0
        for row in emitted:  # type: ignore[union-attr]
            b = store.get_belief(row["id"])
            assert b is not None
            expected += len(
                _cap_belief_content(b.content, locked=bool(row["locked"]))
            )
            uncapped += len(b.content)
    finally:
        store.close()

    # The cap bites on the emitted set, so `expected` distinguishes a sum
    # that applies it from one that sums raw content.
    assert expected != uncapped, (expected, uncapped)
    assert total_chars == expected, (total_chars, expected)
    assert total_chars != len(out), "recorded rendered-block bytes, not content"


def test_ups_caps_one_oversized_belief_instead_of_dropping_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-belief cap, on the lane it was added for.

    A 35,000-character belief costs 8,763 tokens under retrieval's own
    element estimate and `DEFAULT_HOOK_TOKEN_BUDGET` is 1,500, so before
    `_ups_belief_line_cost` the packer rejected it and the fire emitted an
    **empty block** — the cap never ran. Charging the line this lane
    actually renders admits it at 321 tokens, capped.
    """
    db = tmp_path / "memory.db"
    _, _, hit_ids = _seed(db, n_hits=1, hit_chars=35_000)
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    assert err == ""
    assert hit_ids[0] in out
    assert "[…truncated]" in out
    assert len(out) < 4_000, len(out)


def test_ups_does_not_cap_a_user_locked_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drop policy: a lock is emitted whole, however long it is."""
    db = tmp_path / "memory.db"
    body = "lockword " + "q" * 35_000
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L" + "0" * 31, body, locked=True))
    finally:
        store.close()
    out, err = _fire_ups(tmp_path, db, monkeypatch)
    assert "[…truncated]" not in out
    assert "q" * 35_000 in out
    assert "still over the" in err


def test_ups_does_not_cap_a_user_locked_belief_on_a_later_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock cap-exemption at the site that renders turns 2 and on.

    Turn 1 emits the lock inside the session-start `<locked>` sub-block,
    which has its own render. Every turn after that emits it through
    `_belief_element_line`, and only a second fire in the same session
    reaches that code with a locked belief — so
    `test_ups_does_not_cap_a_user_locked_belief` above, which fires once,
    leaves it uncovered. Dropping `locked=` from that call site truncated
    a 5,000-character lock on turn 2 while `pytest tests -k hook` stayed
    at 1005 passed.

    The turn-2 assertions are distinguishing on purpose: the block must
    contain the whole content AND no truncation marker. Either alone
    passes on a mutant — the marker is absent from a block that dropped
    the belief entirely, and a prefix check passes on a truncated one.
    """
    db = tmp_path / "memory.db"
    content = "lockword " + "q" * 5_000
    assert len(content) > 1_200, "must exceed BELIEF_CONTENT_CHAR_CAP"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("L" + "0" * 31, content, locked=True))
    finally:
        store.close()

    first, _ = _fire_ups(
        tmp_path, db, monkeypatch, prompt=_PROMPT, session_id="s-two-turn"
    )
    # Turn 1 goes through the `<locked>` sub-block, which is a different
    # render. Asserted so a change that removed the sub-block would not
    # quietly leave this test measuring turn 1 twice.
    assert "<locked>" in first

    second, err = _fire_ups(
        tmp_path,
        db,
        monkeypatch,
        prompt="second turn: tell me about the lockword please",
        session_id="s-two-turn",
    )
    assert "<locked>" not in second
    assert "[…truncated]" not in second
    assert content in second
    assert err == ""


# ---------------------------------------------------------------------------
# user_prompt_submit — the gate-skip branch
# ---------------------------------------------------------------------------


def test_gate_skip_branch_trims_to_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    lock_ids, core_ids, _ = _seed(
        db, n_locks=100, lock_chars=150, n_core=30, core_chars=200
    )
    out, err = _fire_ups(tmp_path, db, monkeypatch, prompt=_GATED_PROMPT)

    assert _audit_tokens_from_block(out) <= _CEILING
    assert "dropped" in err
    assert OPEN_TAG in out and CLOSE_TAG in out
    assert out.count("<belief ") == out.count("</belief>")
    assert [b for b in lock_ids if b not in out] == []
    assert any(b not in out for b in core_ids)


def test_gate_skip_branch_leaves_a_fitting_block_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    lock_ids, _, _ = _seed(db, n_locks=3, lock_chars=150)
    out, err = _fire_ups(tmp_path, db, monkeypatch, prompt=_GATED_PROMPT)
    assert err == ""
    assert [b for b in lock_ids if b not in out] == []


def test_core_section_caps_an_oversized_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1: `<core>` is one of the two lanes the fix names as the cause.

    It is the lane an unbounded belief reaches without being locked — the
    section is selected by corroboration and posterior, neither of which
    is a length — and the cap was not applied at its render site.
    """
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            _mk("C" + "0" * 31, "coreword " + "w" * 35_000, alpha=4.0, beta=1.0)
        )
    finally:
        store.close()
    out, err = _fire_ups(tmp_path, db, monkeypatch, prompt=_GATED_PROMPT)
    assert "<core>" in out
    assert "[…truncated]" in out
    assert len(out) < 4_000, len(out)
    assert err == ""


def test_gate_skip_branch_keeps_every_lock_and_reports_the_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#379 under the ceiling: 130 locks that cannot fit are all emitted.

    Before the exemption the dropper popped from the tail and `<locked>`
    sits at the head, so the locks went last — but they went. On a
    300-lock store every locked element was deleted, leaving an empty
    `<locked>` section under 300 dangling `seen <id>` manifest pointers.
    """
    db = tmp_path / "memory.db"
    lock_ids, _, _ = _seed(db, n_locks=130, lock_chars=200)
    out, err = _fire_ups(tmp_path, db, monkeypatch, prompt=_GATED_PROMPT)

    assert [b for b in lock_ids if b not in out] == []
    assert _audit_tokens_from_block(out) > _CEILING
    assert "still over the 6000-token ceiling" in err
    assert "never dropped" in err


# ---------------------------------------------------------------------------
# session_start
# ---------------------------------------------------------------------------


def test_session_start_keeps_every_lock_and_reports_the_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third emit site, which had no ceiling at all.

    Its block comes from `retrieve(store, "", ...)`, which fires the L0
    lane only, so in practice every element is a lock and the drop arm is
    unreachable here by construction. What this site gains is the overrun
    note — the honest outcome for a baseline #379 forbids trimming — and
    the guarantee that it is on the same write path as its two siblings.
    """
    db = tmp_path / "memory.db"
    lock_ids, _, _ = _seed(db, n_locks=130, lock_chars=200)
    out, err = _fire_session_start(tmp_path, db, monkeypatch)

    assert SESSION_START_OPEN_TAG in out
    assert SESSION_START_CLOSE_TAG in out
    assert [b for b in lock_ids if b not in out] == []
    assert _audit_tokens_from_block(out) > _CEILING
    assert "still over the 6000-token ceiling" in err


def test_session_start_leaves_a_fitting_baseline_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    lock_ids, _, _ = _seed(db, n_locks=3, lock_chars=150)
    out, err = _fire_session_start(tmp_path, db, monkeypatch)
    assert err == ""
    assert [b for b in lock_ids if b not in out] == []
