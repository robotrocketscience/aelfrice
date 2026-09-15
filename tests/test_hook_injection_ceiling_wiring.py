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
    read_hook_audit,
    session_start,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
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
    n_hits: int = 0,
    hit_chars: int = 400,
) -> tuple[list[str], list[str], list[str]]:
    """Seed a store and return `(lock_ids, core_ids, hit_ids)`.

    A `<core>` belief is one that qualifies on posterior: alpha 4, beta 1
    is mu 0.8 over alpha+beta 5, clearing `_CORE_MIN_POSTERIOR` (2/3) and
    `_CORE_MIN_ALPHA_BETA` (4). Core beliefs are not locked, so they are
    what the ceiling is allowed to drop.
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
                _mk(bid, "coreword " + "w" * core_chars, alpha=4.0, beta=1.0)
            )
            core_ids.append(bid)
        for i in range(n_hits):
            bid = f"H{i:031d}"
            store.insert_belief(_mk(bid, f"{_WORD} fact " + "z" * hit_chars))
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
