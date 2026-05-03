"""#391 acceptance tests — unlock/promote/demote parity (CLI + MCP).

Covers:
- unlock() idempotency, error paths, audit row, field clearing
- demote lock-drop path writes audit row (regression)
- promote CLI parity with validate CLI
- tool_unlock / tool_promote MCP shape parity
- lock state machine round-trip: locked → unlocked → re-locked
"""
from __future__ import annotations

import argparse
import io

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.mcp_server import tool_demote, tool_promote, tool_unlock, tool_validate
from aelfrice.promotion import SOURCE_LOCK_UNLOCK, unlock
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    *,
    lock: str = LOCK_NONE,
    locked_at: str | None = None,
    origin: str = ORIGIN_AGENT_INFERRED,
    alpha: float = 1.0,
    beta: float = 1.0,
    demotion_pressure: int = 0,
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=locked_at,
        demotion_pressure=demotion_pressure,
        created_at="2026-05-01T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )


def _seed(*beliefs: Belief) -> MemoryStore:
    s = MemoryStore(":memory:")
    for b in beliefs:
        s.insert_belief(b)
    return s


def _locked_belief(bid: str, **kwargs) -> Belief:
    return _mk(
        bid,
        lock=LOCK_USER,
        locked_at="2026-05-01T01:00:00Z",
        alpha=9.0,
        beta=0.5,
        origin="user_stated",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# unlock() — idempotency
# ---------------------------------------------------------------------------


def test_unlock_clears_lock_level() -> None:
    s = _seed(_locked_belief("A"))
    result = unlock(s, "A")
    assert result.already_unlocked is False
    after = s.get_belief("A")
    assert after is not None
    assert after.lock_level == LOCK_NONE


def test_unlock_clears_locked_at() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A")
    after = s.get_belief("A")
    assert after is not None
    assert after.locked_at is None


def test_unlock_clears_demotion_pressure() -> None:
    b = _locked_belief("A", demotion_pressure=3)
    s = _seed(b)
    unlock(s, "A")
    after = s.get_belief("A")
    assert after is not None
    assert after.demotion_pressure == 0


def test_unlock_does_not_touch_origin() -> None:
    b = _locked_belief("A")
    s = _seed(b)
    original_origin = b.origin
    unlock(s, "A")
    after = s.get_belief("A")
    assert after is not None
    assert after.origin == original_origin


def test_unlock_does_not_touch_alpha_beta() -> None:
    b = _locked_belief("A")
    s = _seed(b)
    unlock(s, "A")
    after = s.get_belief("A")
    assert after is not None
    assert after.alpha == b.alpha
    assert after.beta == b.beta


def test_unlock_idempotent_second_call_no_op() -> None:
    s = _seed(_locked_belief("A"))
    first = unlock(s, "A")
    second = unlock(s, "A")
    assert first.already_unlocked is False
    assert second.already_unlocked is True
    assert second.audit_event_id is None


def test_unlock_idempotent_writes_only_one_audit_row() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A")
    unlock(s, "A")  # no-op
    assert s.count_feedback_events() == 1


def test_unlock_idempotent_on_never_locked_belief() -> None:
    s = _seed(_mk("A"))  # never locked
    result = unlock(s, "A")
    assert result.already_unlocked is True
    assert result.audit_event_id is None
    assert s.count_feedback_events() == 0


# ---------------------------------------------------------------------------
# unlock() — error path
# ---------------------------------------------------------------------------


def test_unlock_raises_value_error_on_missing_belief() -> None:
    s = _seed(_mk("A"))
    with pytest.raises(ValueError, match="belief not found"):
        unlock(s, "ghost")


# ---------------------------------------------------------------------------
# unlock() — audit row
# ---------------------------------------------------------------------------


def test_unlock_writes_lock_unlock_audit_row() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A")
    events = s.list_feedback_events()
    assert len(events) == 1
    assert events[0].source == SOURCE_LOCK_UNLOCK


def test_unlock_audit_row_source_prefix() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A")
    ev = s.list_feedback_events()[0]
    assert ev.source.startswith("lock:")


def test_unlock_audit_row_has_zero_valence() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A")
    ev = s.list_feedback_events()[0]
    assert ev.valence == 0.0


def test_unlock_audit_row_belief_id_is_subject() -> None:
    s = _seed(_locked_belief("A"), _locked_belief("B"))
    unlock(s, "A")
    ev = s.list_feedback_events()[0]
    assert ev.belief_id == "A"


def test_unlock_audit_row_carries_now_kwarg() -> None:
    s = _seed(_locked_belief("A"))
    unlock(s, "A", now="2026-05-01T12:00:00Z")
    ev = s.list_feedback_events()[0]
    assert ev.created_at == "2026-05-01T12:00:00Z"


def test_unlock_returns_audit_event_id_on_active_path() -> None:
    s = _seed(_locked_belief("A"))
    result = unlock(s, "A")
    assert isinstance(result.audit_event_id, int)
    assert result.audit_event_id > 0


# ---------------------------------------------------------------------------
# demote lock-drop path writes audit row (regression for #391)
# ---------------------------------------------------------------------------


def test_demote_lock_drop_writes_lock_unlock_row() -> None:
    """tool_demote lock-drop path now writes lock:unlock audit row via unlock()."""
    s = _seed(_locked_belief("A"))
    tool_demote(s, belief_id="A")
    events = s.list_feedback_events()
    assert len(events) == 1
    assert events[0].source == SOURCE_LOCK_UNLOCK


def test_demote_lock_drop_result_shape() -> None:
    s = _seed(_locked_belief("A"))
    result = tool_demote(s, belief_id="A")
    assert result["kind"] == "demote.demoted"
    assert result["id"] == "A"
    assert result["demoted"] is True


def test_demote_still_clears_lock_level() -> None:
    s = _seed(_locked_belief("A"))
    tool_demote(s, belief_id="A")
    after = s.get_belief("A")
    assert after is not None
    assert after.lock_level == LOCK_NONE


# ---------------------------------------------------------------------------
# promote CLI parity — aelf validate and aelf promote identical outcomes
# ---------------------------------------------------------------------------


def test_promote_and_validate_produce_same_origin_change(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """validate and promote share _cmd_validate; outcome must be identical.

    Uses file-backed stores so we can re-open after the CLI handler closes.
    """
    import os
    from aelfrice.cli import _cmd_promote, _cmd_validate
    import unittest.mock as mock

    db_v = str(tmp_path / "v.db")  # type: ignore[operator]
    db_p = str(tmp_path / "p.db")  # type: ignore[operator]

    for db in (db_v, db_p):
        s = MemoryStore(db)
        s.insert_belief(_mk("X", origin=ORIGIN_AGENT_INFERRED))
        s.close()

    def make_args() -> argparse.Namespace:
        ns = argparse.Namespace()
        ns.belief_id = "X"
        ns.source = "user_validated"
        return ns

    with mock.patch.dict(os.environ, {"AELFRICE_DB": db_v}):
        rc_v = _cmd_validate(make_args(), io.StringIO())
    with mock.patch.dict(os.environ, {"AELFRICE_DB": db_p}):
        rc_p = _cmd_promote(make_args(), io.StringIO())

    assert rc_v == rc_p == 0

    sv = MemoryStore(db_v)
    sp = MemoryStore(db_p)
    try:
        after_v = sv.get_belief("X")
        after_p = sp.get_belief("X")
        assert after_v is not None and after_p is not None
        assert after_v.origin == after_p.origin == ORIGIN_USER_VALIDATED
    finally:
        sv.close()
        sp.close()


def test_promote_and_validate_same_audit_row_source(
    tmp_path: pytest.TempPathFactory,
) -> None:
    import os
    from aelfrice.cli import _cmd_promote, _cmd_validate
    import unittest.mock as mock

    db_v = str(tmp_path / "v2.db")  # type: ignore[operator]
    db_p = str(tmp_path / "p2.db")  # type: ignore[operator]

    for db in (db_v, db_p):
        s = MemoryStore(db)
        s.insert_belief(_mk("X"))
        s.close()

    def make_args() -> argparse.Namespace:
        ns = argparse.Namespace()
        ns.belief_id = "X"
        ns.source = "user_validated"
        return ns

    with mock.patch.dict(os.environ, {"AELFRICE_DB": db_v}):
        _cmd_validate(make_args(), io.StringIO())
    with mock.patch.dict(os.environ, {"AELFRICE_DB": db_p}):
        _cmd_promote(make_args(), io.StringIO())

    sv = MemoryStore(db_v)
    sp = MemoryStore(db_p)
    try:
        evs_v = sv.list_feedback_events()
        evs_p = sp.list_feedback_events()
        assert len(evs_v) == len(evs_p) == 1
        assert evs_v[0].source == evs_p[0].source
    finally:
        sv.close()
        sp.close()


# ---------------------------------------------------------------------------
# MCP tool_unlock shape parity
# ---------------------------------------------------------------------------


def test_tool_unlock_active_path_shape() -> None:
    s = _seed(_locked_belief("A"))
    result = tool_unlock(s, belief_id="A")
    assert result["kind"] == "unlock.unlocked"
    assert result["id"] == "A"
    assert result["unlocked"] is True
    assert "audit_event_id" in result


def test_tool_unlock_already_unlocked_shape() -> None:
    s = _seed(_mk("A"))  # never locked
    result = tool_unlock(s, belief_id="A")
    assert result["kind"] == "unlock.already"
    assert result["id"] == "A"
    assert result["unlocked"] is False


def test_tool_unlock_not_found_shape() -> None:
    s = _seed(_mk("A"))
    result = tool_unlock(s, belief_id="ghost")
    assert result["kind"] == "unlock.not_found"
    assert result["unlocked"] is False
    assert "error" in result


def test_tool_unlock_clears_lock_in_store() -> None:
    s = _seed(_locked_belief("A"))
    tool_unlock(s, belief_id="A")
    after = s.get_belief("A")
    assert after is not None
    assert after.lock_level == LOCK_NONE


# ---------------------------------------------------------------------------
# MCP tool_promote shape parity with tool_validate
# ---------------------------------------------------------------------------


def test_tool_promote_active_path_shape_matches_validate() -> None:
    s_v = _seed(_mk("X"))
    s_p = _seed(_mk("X"))
    result_v = tool_validate(s_v, belief_id="X")
    result_p = tool_promote(s_p, belief_id="X")
    # Both should return same structure / kind
    assert result_v["kind"] == result_p["kind"]
    assert result_v.keys() == result_p.keys()


def test_tool_promote_changes_origin() -> None:
    s = _seed(_mk("X", origin=ORIGIN_AGENT_INFERRED))
    result = tool_promote(s, belief_id="X")
    assert result["kind"] == "validate.promoted"
    after = s.get_belief("X")
    assert after is not None
    assert after.origin == ORIGIN_USER_VALIDATED


def test_tool_promote_idempotent_already_path() -> None:
    s = _seed(_mk("X", origin=ORIGIN_USER_VALIDATED))
    result = tool_promote(s, belief_id="X")
    assert result["kind"] == "validate.already"


def test_tool_promote_writes_same_audit_source_as_validate() -> None:
    s_v = _seed(_mk("X"))
    s_p = _seed(_mk("X"))
    tool_validate(s_v, belief_id="X")
    tool_promote(s_p, belief_id="X")
    evs_v = s_v.list_feedback_events()
    evs_p = s_p.list_feedback_events()
    assert len(evs_v) == len(evs_p) == 1
    assert evs_v[0].source == evs_p[0].source


# ---------------------------------------------------------------------------
# Lock state machine round-trip: locked → unlocked → re-locked
# ---------------------------------------------------------------------------


def test_lock_unlock_relock_round_trip() -> None:
    """Full round-trip: lock a belief, unlock it, re-lock it."""
    from aelfrice.mcp_server import tool_lock

    s = MemoryStore(":memory:")

    # Lock via tool_lock (inserts new belief at lock priors)
    lock_result = tool_lock(s, statement="the sky is blue")
    bid = lock_result["id"]

    belief_after_lock = s.get_belief(bid)
    assert belief_after_lock is not None
    assert belief_after_lock.lock_level == LOCK_USER

    # Unlock
    result = unlock(s, bid)
    assert result.already_unlocked is False
    belief_after_unlock = s.get_belief(bid)
    assert belief_after_unlock is not None
    assert belief_after_unlock.lock_level == LOCK_NONE

    # Re-lock by calling tool_lock again — lock is idempotent
    relock_result = tool_lock(s, statement="the sky is blue")
    assert relock_result["action"] in ("locked", "upgraded")
    belief_after_relock = s.get_belief(bid)
    assert belief_after_relock is not None
    assert belief_after_relock.lock_level == LOCK_USER


def test_lock_unlock_relock_audit_trail() -> None:
    """Audit trail after round-trip: at least a lock:unlock row exists."""
    from aelfrice.mcp_server import tool_lock

    s = MemoryStore(":memory:")
    lock_result = tool_lock(s, statement="another locked fact")
    bid = lock_result["id"]

    unlock(s, bid)

    events = s.list_feedback_events()
    sources = [ev.source for ev in events]
    assert SOURCE_LOCK_UNLOCK in sources


def test_unlock_after_relock_clears_again() -> None:
    """unlock() after a re-lock cycle clears the lock a second time."""
    from aelfrice.mcp_server import tool_lock

    s = MemoryStore(":memory:")
    tool_lock(s, statement="re-lockable fact")
    bid = list(s.list_locked_beliefs())[0].id

    unlock(s, bid)
    tool_lock(s, statement="re-lockable fact")  # re-lock same content
    second_unlock = unlock(s, bid)
    assert second_unlock.already_unlocked is False

    final = s.get_belief(bid)
    assert final is not None
    assert final.lock_level == LOCK_NONE
