"""Session-start enrichment tests for the UserPromptSubmit hook (#578).

Verifies:
  - First prompt of a new session injects a <session-start> sub-block
    inside <aelfrice-memory>.
  - Subsequent prompts in the same session omit the sub-block.
  - The sub-block contains <locked> and <core> sections.
  - `is_session_first_prompt` predicate correctly detects new vs continuing
    sessions.
  - `_build_session_start_subblock` content correctness.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    DEFAULT_SESSION_START_CORE_TOKEN_BUDGET,
    SESSION_START_CORE_BUDGET_ENV,
    SESSION_START_SUBBLOCK_CLOSE,
    SESSION_START_SUBBLOCK_OPEN,
    SESSION_STATE_FILENAME,
    _build_session_start_subblock,
    _session_start_core_budget,
    is_session_first_prompt,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    corroboration_count: int = 0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _payload(prompt: str, session_id: str = "sess-abc") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "prompt": prompt,
            "cwd": "/tmp",
        }
    )


# ---------------------------------------------------------------------------
# is_session_first_prompt predicate
# ---------------------------------------------------------------------------


def test_is_session_first_prompt_true_when_state_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No state file → first call is first prompt."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    assert is_session_first_prompt("session-1") is True


def test_is_session_first_prompt_false_on_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same session_id on second call → not first prompt."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    is_session_first_prompt("session-1")  # first call writes state
    assert is_session_first_prompt("session-1") is False


def test_is_session_first_prompt_true_for_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different session_id → first prompt of new session."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    is_session_first_prompt("session-1")
    assert is_session_first_prompt("session-2") is True


def test_is_session_first_prompt_false_for_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None session_id → cannot detect; conservatively returns False."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    assert is_session_first_prompt(None) is False


def test_is_session_first_prompt_false_for_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    assert is_session_first_prompt("") is False


def test_is_session_first_prompt_writes_state_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    state_file = tmp_path / SESSION_STATE_FILENAME
    assert not state_file.exists()
    is_session_first_prompt("session-1")
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["session_id"] == "session-1"


# ---------------------------------------------------------------------------
# _build_session_start_subblock content
# ---------------------------------------------------------------------------


def test_build_subblock_empty_on_empty_store(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        # tmp_path is not a git work-tree, so <recent-work> is also empty.
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert result == ""


def test_build_subblock_contains_locked_beliefs(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "never push to main", LOCK_USER, "2026-01-01T00:00:00Z")])
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store)
    finally:
        store.close()
    assert SESSION_START_SUBBLOCK_OPEN in result
    assert "<locked>" in result
    assert "never push to main" in result
    assert 'lock="user"' in result
    assert SESSION_START_SUBBLOCK_CLOSE in result


def test_build_subblock_contains_core_beliefs(tmp_path: Path) -> None:
    """Unlocked beliefs with corroboration>=2 appear in <core>."""
    db = tmp_path / "memory.db"
    # Belief with high corroboration — use alpha/beta to signal quality
    # Note: corroboration_count is not a direct insert field; we use alpha/beta
    # to satisfy the posterior branch (alpha/(alpha+beta) >= 2/3, alpha+beta >= 4).
    _seed_db(db, [
        _mk("C1", "prefer snake_case", alpha=8.0, beta=3.0),  # mu=0.73 > 2/3, ab=11>=4
    ])
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store)
    finally:
        store.close()
    assert "<core>" in result
    assert "prefer snake_case" in result


def test_build_subblock_locked_only_store(tmp_path: Path) -> None:
    """Locked beliefs appear in <locked>; empty <core> is still present."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "rule one", LOCK_USER, "2026-01-01T00:00:00Z")])
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store)
    finally:
        store.close()
    assert "<locked>" in result
    assert "rule one" in result
    assert "<core>" in result  # section always present


def test_build_subblock_excludes_locked_from_core(tmp_path: Path) -> None:
    """Locked beliefs must not appear in <core> (they're in <locked>)."""
    db = tmp_path / "memory.db"
    _seed_db(db, [
        _mk("L1", "locked rule", LOCK_USER, "2026-01-01T00:00:00Z",
            alpha=9.0, beta=1.0),  # would qualify as core too
    ])
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store)
    finally:
        store.close()
    # locked rule appears once, in <locked>
    assert result.count("locked rule") == 1
    # Only one <belief> element
    assert result.count('<belief id="L1"') == 1


def test_build_subblock_ignores_low_quality_unlocked(tmp_path: Path) -> None:
    """Unlocked beliefs that don't meet core thresholds are absent."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "weak belief", alpha=1.0, beta=2.0)])  # mu=0.33 < 2/3
    store = MemoryStore(str(db))
    try:
        # tmp_path is not a git work-tree, so <recent-work> is also empty.
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert result == ""


# ---------------------------------------------------------------------------
# <core> token-budget cap (#578 follow-up — unbounded-injection fix)
# ---------------------------------------------------------------------------


def _seed_many_core(db_path: Path, n: int, pad: int = 116) -> list[str]:
    """Seed n unlocked beliefs that all qualify as core (posterior arm),
    with a strictly descending posterior so ordering is testable. Returns
    the ids in descending-posterior (= retained-first) order."""
    beliefs = []
    ids = []
    for i in range(n):
        bid = f"K{i:03d}"
        ids.append(bid)
        # alpha descends 30.0 -> ... ; beta=3 -> mu=alpha/(alpha+3) >= 0.667, ab>=4
        content = f"core fact {i:03d} " + ("x" * pad)
        beliefs.append(_mk(bid, content, alpha=30.0 - i * 0.1, beta=3.0))
    _seed_db(db_path, beliefs)
    return ids


def test_core_section_capped_by_token_budget(tmp_path: Path) -> None:
    """A store with far more core-qualifying beliefs than the budget allows
    must emit a bounded <core> section, not all of them."""
    db = tmp_path / "memory.db"
    ids = _seed_many_core(db, n=200)  # ~200 * ~33 tok = ~6600 tok >> 1500
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    emitted = result.count("<belief ")
    assert emitted < 200, "core section was not capped"
    # budget is 1500 tok; per belief ~33 tok -> ~45 kept, never the full 200
    assert emitted <= 70
    # highest-posterior belief retained; lowest dropped
    assert f'id="{ids[0]}"' in result
    assert f'id="{ids[-1]}"' not in result


def test_core_cap_skips_oversized_keeps_smaller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized belief is skipped, not a hard stop: smaller lower-ranked
    beliefs that still fit are kept (no prefix truncation)."""
    monkeypatch.setenv(SESSION_START_CORE_BUDGET_ENV, "100")  # 100 tok = 400 chars
    db = tmp_path / "memory.db"
    # B0 highest posterior but oversized (~250 tok); B1/B2 small and fit.
    _seed_db(db, [
        _mk("B0", "huge " + "z" * 1000, alpha=30.0, beta=1.0),  # ~251 tok > 100
        _mk("B1", "small one " + "a" * 40, alpha=20.0, beta=3.0),
        _mk("B2", "small two " + "b" * 40, alpha=19.0, beta=3.0),
    ])
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert 'id="B0"' not in result  # oversized skipped
    assert 'id="B1"' in result      # smaller still packed despite B0 skip
    assert 'id="B2"' in result


def test_core_cap_disabled_with_zero_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the budget to 0 restores uncapped (pre-fix) behaviour."""
    monkeypatch.setenv(SESSION_START_CORE_BUDGET_ENV, "0")
    db = tmp_path / "memory.db"
    ids = _seed_many_core(db, n=120)
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert result.count("<belief ") == 120
    assert f'id="{ids[-1]}"' in result


def test_core_cap_never_trims_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a tiny core budget must not drop locked beliefs (#379)."""
    monkeypatch.setenv(SESSION_START_CORE_BUDGET_ENV, "1")
    db = tmp_path / "memory.db"
    locks = [_mk(f"L{i}", "rule " + "y" * 80, LOCK_USER,
                 f"2026-01-0{i}T00:00:00Z") for i in range(1, 6)]
    _seed_db(db, locks)
    store = MemoryStore(str(db))
    try:
        result = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    for i in range(1, 6):
        assert f'id="L{i}"' in result


def test_session_start_core_budget_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SESSION_START_CORE_BUDGET_ENV, raising=False)
    assert _session_start_core_budget() == DEFAULT_SESSION_START_CORE_TOKEN_BUDGET
    monkeypatch.setenv(SESSION_START_CORE_BUDGET_ENV, "500")
    assert _session_start_core_budget() == 500
    monkeypatch.setenv(SESSION_START_CORE_BUDGET_ENV, "garbage")
    assert _session_start_core_budget() == DEFAULT_SESSION_START_CORE_TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Integration: user_prompt_submit first-prompt injection
# ---------------------------------------------------------------------------


def test_first_prompt_injects_session_start_subblock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First UserPromptSubmit for a new session includes <session-start>."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "sign all commits", LOCK_USER, "2026-01-01T00:00:00Z")])
    _set_db(monkeypatch, db)
    sin = io.StringIO(_payload("push the release", session_id="fresh-session"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=io.StringIO())
    assert rc == 0
    out = sout.getvalue()
    assert SESSION_START_SUBBLOCK_OPEN in out
    assert "sign all commits" in out
    assert "<locked>" in out


def test_subsequent_prompt_omits_session_start_subblock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second UserPromptSubmit in the same session omits <session-start>."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "sign all commits", LOCK_USER, "2026-01-01T00:00:00Z")])
    _set_db(monkeypatch, db)
    sid = "continuing-session"
    # First call — marks session
    user_prompt_submit(
        stdin=io.StringIO(_payload("prompt one", session_id=sid)),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    # Second call — same session
    sout2 = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("prompt two", session_id=sid)),
        stdout=sout2,
        stderr=io.StringIO(),
    )
    assert rc == 0
    out2 = sout2.getvalue()
    assert SESSION_START_SUBBLOCK_OPEN not in out2


def test_new_session_id_re_triggers_session_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different session_id after a prior session triggers injection again."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "use uv", LOCK_USER, "2026-01-01T00:00:00Z")])
    _set_db(monkeypatch, db)
    # Session 1
    user_prompt_submit(
        stdin=io.StringIO(_payload("hello", session_id="session-A")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    # Session 2 (different session_id)
    sout2 = io.StringIO()
    user_prompt_submit(
        stdin=io.StringIO(_payload("hello", session_id="session-B")),
        stdout=sout2,
        stderr=io.StringIO(),
    )
    out2 = sout2.getvalue()
    assert SESSION_START_SUBBLOCK_OPEN in out2


def test_session_start_subblock_inside_aelfrice_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """<session-start> must appear inside <aelfrice-memory>, not standalone."""
    from aelfrice.hook import CLOSE_TAG, OPEN_TAG

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "locked rule", LOCK_USER, "2026-01-01T00:00:00Z")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    user_prompt_submit(
        stdin=io.StringIO(_payload("anything", session_id="wrap-test")),
        stdout=sout,
        stderr=io.StringIO(),
    )
    out = sout.getvalue()
    open_pos = out.find(OPEN_TAG)
    close_pos = out.find(CLOSE_TAG)
    sub_pos = out.find(SESSION_START_SUBBLOCK_OPEN)
    assert open_pos >= 0
    assert close_pos > open_pos
    assert sub_pos > open_pos
    assert sub_pos < close_pos


def test_no_session_id_in_payload_no_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Payload without session_id: no session-start block (cannot track)."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("L1", "locked rule", LOCK_USER, "2026-01-01T00:00:00Z")])
    _set_db(monkeypatch, db)
    payload = json.dumps({"prompt": "something"})  # no session_id
    sout = io.StringIO()
    user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=sout,
        stderr=io.StringIO(),
    )
    out = sout.getvalue()
    assert SESSION_START_SUBBLOCK_OPEN not in out
