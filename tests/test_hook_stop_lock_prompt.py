"""Stop hook session-end correction-lock prompt tests (#582).

Verifies:
  - `_belief_is_lock_candidate` filter rules (session, lock_level,
    type/origin gating).
  - `_collect_lock_candidates` walks all beliefs and returns only the
    matches.
  - `_format_stop_prompt` renders the stderr block with the
    `aelf lock` pre-fills and proper plural/empty handling.
  - `_autolock_enabled` env-var parsing.
  - `_autolock_candidates` mutates lock_level in place + writes back.
  - `stop()` end-to-end: empty / malformed / missing-session-id
    payloads return 0 and emit nothing; candidate payloads emit the
    block on stderr; AUTOLOCK env var auto-locks instead of prompting.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    AUTOLOCK_ENV_VAR,
    STOP_PROMPT_CLOSE_TAG,
    STOP_PROMPT_OPEN_TAG,
    _autolock_candidates,
    _autolock_enabled,
    _belief_is_lock_candidate,
    _collect_lock_candidates,
    _format_stop_prompt,
    stop,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_AGENT_REMEMBERED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    *,
    session_id: str | None = "sess-A",
    lock_level: str = LOCK_NONE,
    type_: str = BELIEF_FACTUAL,
    origin: str = "unknown",
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=type_,
        lock_level=lock_level,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        session_id=session_id,
        origin=origin,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def _payload(session_id: str | None) -> str:
    obj: dict[str, object] = {"cwd": "/tmp"}
    if session_id is not None:
        obj["session_id"] = session_id
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# _belief_is_lock_candidate
# ---------------------------------------------------------------------------


def test_candidate_correction_type_unlocked_in_session() -> None:
    b = _mk("B1", "x", type_=BELIEF_CORRECTION, session_id="sess-A")
    assert _belief_is_lock_candidate(b, "sess-A") is True


def test_candidate_agent_inferred_origin_in_session() -> None:
    b = _mk("B2", "x", origin=ORIGIN_AGENT_INFERRED, session_id="sess-A")
    assert _belief_is_lock_candidate(b, "sess-A") is True


def test_candidate_agent_remembered_origin_in_session() -> None:
    b = _mk("B3", "x", origin=ORIGIN_AGENT_REMEMBERED, session_id="sess-A")
    assert _belief_is_lock_candidate(b, "sess-A") is True


def test_not_candidate_when_locked_user() -> None:
    b = _mk("B4", "x", type_=BELIEF_CORRECTION,
            lock_level=LOCK_USER, session_id="sess-A")
    assert _belief_is_lock_candidate(b, "sess-A") is False


def test_not_candidate_other_session() -> None:
    b = _mk("B5", "x", type_=BELIEF_CORRECTION, session_id="sess-OTHER")
    assert _belief_is_lock_candidate(b, "sess-A") is False


def test_not_candidate_no_session_id_on_belief() -> None:
    b = _mk("B6", "x", type_=BELIEF_CORRECTION, session_id=None)
    assert _belief_is_lock_candidate(b, "sess-A") is False


def test_not_candidate_factual_user_origin() -> None:
    b = _mk("B7", "x", type_=BELIEF_FACTUAL, origin=ORIGIN_USER_STATED,
            session_id="sess-A")
    assert _belief_is_lock_candidate(b, "sess-A") is False


# ---------------------------------------------------------------------------
# _collect_lock_candidates
# ---------------------------------------------------------------------------


def test_collect_returns_only_session_unlocked_correction_class(
    tmp_path: Path,
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [
        _mk("KEEP1", "fix one", type_=BELIEF_CORRECTION, session_id="sess-A"),
        _mk("KEEP2", "fix two", origin=ORIGIN_AGENT_INFERRED,
            session_id="sess-A"),
        _mk("SKIP_OTHER_SESSION", "x", type_=BELIEF_CORRECTION,
            session_id="sess-B"),
        _mk("SKIP_LOCKED", "x", type_=BELIEF_CORRECTION,
            lock_level=LOCK_USER, session_id="sess-A"),
        _mk("SKIP_FACTUAL_USER", "x", type_=BELIEF_FACTUAL,
            origin=ORIGIN_USER_STATED, session_id="sess-A"),
    ])
    s = MemoryStore(str(db))
    try:
        cands = _collect_lock_candidates(s, "sess-A")
    finally:
        s.close()
    ids = {c.id for c in cands}
    assert ids == {"KEEP1", "KEEP2"}


def test_collect_empty_store_returns_empty(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        assert _collect_lock_candidates(s, "sess-A") == []
    finally:
        s.close()


# ---------------------------------------------------------------------------
# _format_stop_prompt
# ---------------------------------------------------------------------------


def test_format_empty_returns_empty() -> None:
    assert _format_stop_prompt([]) == ""


def test_format_has_open_close_tags_and_lock_command() -> None:
    b = _mk("B1", "atomic commits beat batched",
            type_=BELIEF_CORRECTION, session_id="sess-A")
    block = _format_stop_prompt([b])
    assert STOP_PROMPT_OPEN_TAG in block
    assert STOP_PROMPT_CLOSE_TAG in block
    assert "B1" in block
    assert "atomic commits beat batched" in block
    assert "aelf lock --statement" in block


def test_format_pluralizes_correctly() -> None:
    one = _format_stop_prompt(
        [_mk("B1", "x", type_=BELIEF_CORRECTION)])
    many = _format_stop_prompt([
        _mk("B1", "x", type_=BELIEF_CORRECTION),
        _mk("B2", "y", type_=BELIEF_CORRECTION),
    ])
    assert "1 correction" in one
    assert "2 corrections" in many


def test_format_truncates_long_content() -> None:
    long = "x" * 300
    block = _format_stop_prompt(
        [_mk("BL", long, type_=BELIEF_CORRECTION)])
    # Snippet capped at ~120 chars + ellipsis; full content still in
    # the lock command for the user to inspect.
    assert "..." in block


# ---------------------------------------------------------------------------
# _autolock_enabled env-var parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", ["1", "true", "True", "YES", "on"])
def test_autolock_enabled_truthy(v: str) -> None:
    assert _autolock_enabled({AUTOLOCK_ENV_VAR: v}) is True


@pytest.mark.parametrize("v", ["", "0", "false", "no", "off", "maybe"])
def test_autolock_disabled_falsy(v: str) -> None:
    assert _autolock_enabled({AUTOLOCK_ENV_VAR: v}) is False


def test_autolock_disabled_missing_env() -> None:
    assert _autolock_enabled({}) is False


# ---------------------------------------------------------------------------
# _autolock_candidates
# ---------------------------------------------------------------------------


def test_autolock_candidates_mutates_in_place(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [
        _mk("L1", "fix", type_=BELIEF_CORRECTION, session_id="sess-A"),
    ])
    s = MemoryStore(str(db))
    try:
        cands = _collect_lock_candidates(s, "sess-A")
        n = _autolock_candidates(s, cands, io.StringIO())
        assert n == 1
        # Re-read to confirm the persisted state.
        b = s.get_belief("L1")
        assert b is not None
        assert b.lock_level == LOCK_USER
        assert b.origin == ORIGIN_USER_STATED
        assert b.locked_at is not None
        assert b.demotion_pressure == 0
    finally:
        s.close()


# ---------------------------------------------------------------------------
# stop() integration
# ---------------------------------------------------------------------------


def test_stop_empty_stdin_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_db(monkeypatch, tmp_path / "memory.db")
    rc = stop(stdin=io.StringIO(""), stdout=io.StringIO(),
              stderr=io.StringIO())
    assert rc == 0


def test_stop_malformed_json_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_db(monkeypatch, tmp_path / "memory.db")
    rc = stop(stdin=io.StringIO("not json"),
              stdout=io.StringIO(), stderr=io.StringIO())
    assert rc == 0


def test_stop_missing_session_id_returns_zero_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("B1", "x", type_=BELIEF_CORRECTION, session_id="sess-A")])
    _set_db(monkeypatch, db)
    serr = io.StringIO()
    rc = stop(stdin=io.StringIO(_payload(None)),
              stdout=io.StringIO(), stderr=serr)
    assert rc == 0
    assert STOP_PROMPT_OPEN_TAG not in serr.getvalue()


def test_stop_no_candidates_emits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("B1", "x", type_=BELIEF_FACTUAL,
                    origin=ORIGIN_USER_STATED, session_id="sess-A")])
    _set_db(monkeypatch, db)
    serr = io.StringIO()
    rc = stop(stdin=io.StringIO(_payload("sess-A")),
              stdout=io.StringIO(), stderr=serr)
    assert rc == 0
    assert serr.getvalue() == ""


def test_stop_emits_lock_prompt_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("B1", "use uv not pip",
                    type_=BELIEF_CORRECTION, session_id="sess-A")])
    _set_db(monkeypatch, db)
    serr = io.StringIO()
    rc = stop(stdin=io.StringIO(_payload("sess-A")),
              stdout=io.StringIO(), stderr=serr,
              env={})  # explicitly no AUTOLOCK
    assert rc == 0
    out = serr.getvalue()
    assert STOP_PROMPT_OPEN_TAG in out
    assert "use uv not pip" in out
    assert "aelf lock --statement" in out
    # Belief should NOT be locked yet — prompt mode is informational only.
    s = MemoryStore(str(db))
    try:
        b = s.get_belief("B1")
        assert b is not None
        assert b.lock_level == LOCK_NONE
    finally:
        s.close()


def test_stop_autolock_env_locks_and_suppresses_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("B1", "use uv not pip",
                    type_=BELIEF_CORRECTION, session_id="sess-A")])
    _set_db(monkeypatch, db)
    serr = io.StringIO()
    rc = stop(stdin=io.StringIO(_payload("sess-A")),
              stdout=io.StringIO(), stderr=serr,
              env={AUTOLOCK_ENV_VAR: "1"})
    assert rc == 0
    out = serr.getvalue()
    # No lock-prompt block — autolock path emits a one-liner per locked belief
    assert STOP_PROMPT_OPEN_TAG not in out
    assert "auto-locked B1" in out
    s = MemoryStore(str(db))
    try:
        b = s.get_belief("B1")
        assert b is not None
        assert b.lock_level == LOCK_USER
        assert b.origin == ORIGIN_USER_STATED
    finally:
        s.close()
