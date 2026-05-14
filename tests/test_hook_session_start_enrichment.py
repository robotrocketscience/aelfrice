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
    SESSION_START_SUBBLOCK_CLOSE,
    SESSION_START_SUBBLOCK_OPEN,
    SESSION_STATE_FILENAME,
    _build_session_start_subblock,
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
        result = _build_session_start_subblock(store)
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
        result = _build_session_start_subblock(store)
    finally:
        store.close()
    assert result == ""


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
