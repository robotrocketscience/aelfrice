"""#391 acceptance tests — unlock/promote/demote parity (CLI + MCP).

Covers:
- unlock() idempotency, error paths, audit row, field clearing
- demote lock-drop path writes audit row (regression)
- promote CLI parity with validate CLI
- lock state machine round-trip: locked → unlocked → re-locked
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
    Belief,
)
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


def _seed_file_backed(db: Path, *beliefs: Belief) -> None:
    """`_seed`, but on a file-backed store the CLI handlers can re-open."""
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _demote_via_cli(db: Path, belief_id: str) -> int:
    import os

    from aelfrice.cli import main as cli_main

    env_db = os.environ.get("AELFRICE_DB")
    os.environ["AELFRICE_DB"] = str(db)
    try:
        return cli_main(["demote", belief_id])
    finally:
        if env_db is None:
            os.environ.pop("AELFRICE_DB", None)
        else:
            os.environ["AELFRICE_DB"] = env_db


def test_demote_lock_drop_writes_lock_unlock_row(tmp_path: Path) -> None:
    """#391 regression: demote's lock-drop path writes a lock:unlock audit row.

    Asserted through `aelf demote` since #1422 removed the MCP surface this was
    previously driven through; the audit-row contract belongs to the demote
    operation, not to the entry point.
    """
    db = tmp_path / "demote.db"
    _seed_file_backed(db, _locked_belief("A"))

    assert _demote_via_cli(db, "A") == 0

    s = MemoryStore(str(db))
    try:
        events = s.list_feedback_events()
        assert len(events) == 1
        assert events[0].source == SOURCE_LOCK_UNLOCK
    finally:
        s.close()


def test_demote_still_clears_lock_level(tmp_path: Path) -> None:
    db = tmp_path / "demote_clear.db"
    _seed_file_backed(db, _locked_belief("A"))

    assert _demote_via_cli(db, "A") == 0

    s = MemoryStore(str(db))
    try:
        after = s.get_belief("A")
        assert after is not None
        assert after.lock_level == LOCK_NONE
    finally:
        s.close()


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
# Lock state machine round-trip: locked -> unlocked -> re-locked
# ---------------------------------------------------------------------------
#
# These drive `aelf lock` rather than seeding a locked belief with `_locked_belief`
# on purpose: what they exercise is the lock *ingest* path (derive -> record ->
# worker), which is what makes "re-lock" mean corroborate-the-canonical-belief
# rather than insert-a-duplicate. Seeding the row directly would still assert the
# lock levels while testing none of that. Previously driven through the MCP
# `tool_lock` surface, removed in #1422.


def _lock_via_cli(db: Path, statement: str) -> int:
    """Run `aelf lock <statement>` against `db`, returning the exit code."""
    import os

    from aelfrice.cli import main as cli_main

    env_db = os.environ.get("AELFRICE_DB")
    os.environ["AELFRICE_DB"] = str(db)
    try:
        return cli_main(["lock", statement])
    finally:
        if env_db is None:
            os.environ.pop("AELFRICE_DB", None)
        else:
            os.environ["AELFRICE_DB"] = env_db


def test_lock_unlock_relock_round_trip(tmp_path: Path) -> None:
    """Full round-trip: lock a belief, unlock it, re-lock it."""
    db = tmp_path / "roundtrip.db"
    statement = "the sky is blue"
    assert _lock_via_cli(db, statement) == 0

    s = MemoryStore(str(db))
    try:
        locked = list(s.list_locked_beliefs())
        assert len(locked) == 1
        bid = locked[0].id
        assert locked[0].lock_level == LOCK_USER

        result = unlock(s, bid)
        assert result.already_unlocked is False
        after_unlock = s.get_belief(bid)
        assert after_unlock is not None
        assert after_unlock.lock_level == LOCK_NONE
    finally:
        s.close()

    # Re-lock the same content: idempotent, and lands on the same belief.
    assert _lock_via_cli(db, statement) == 0
    s = MemoryStore(str(db))
    try:
        after_relock = s.get_belief(bid)
        assert after_relock is not None
        assert after_relock.lock_level == LOCK_USER
    finally:
        s.close()


def test_lock_unlock_relock_audit_trail(tmp_path: Path) -> None:
    """Audit trail after round-trip: at least a lock:unlock row exists."""
    db = tmp_path / "audit.db"
    assert _lock_via_cli(db, "another locked fact") == 0

    s = MemoryStore(str(db))
    try:
        bid = list(s.list_locked_beliefs())[0].id
        unlock(s, bid)
        sources = [ev.source for ev in s.list_feedback_events()]
        assert SOURCE_LOCK_UNLOCK in sources
    finally:
        s.close()


def test_unlock_after_relock_clears_again(tmp_path: Path) -> None:
    """unlock() after a re-lock cycle clears the lock a second time."""
    db = tmp_path / "relock.db"
    statement = "re-lockable fact"
    assert _lock_via_cli(db, statement) == 0

    s = MemoryStore(str(db))
    try:
        bid = list(s.list_locked_beliefs())[0].id
        unlock(s, bid)
    finally:
        s.close()

    assert _lock_via_cli(db, statement) == 0

    s = MemoryStore(str(db))
    try:
        second_unlock = unlock(s, bid)
        assert second_unlock.already_unlocked is False
        final = s.get_belief(bid)
        assert final is not None
        assert final.lock_level == LOCK_NONE
    finally:
        s.close()
