"""Tests for store additions introduced by the review workflow (#936).

Covers:
- last_confirmed_at round-trips through insert/get/update
- update_last_confirmed_at helper sets the field and commits
- list_review_candidates ordering (last_confirmed_at NULLS FIRST, then
  last_retrieved_at NULLS FIRST, then created_at ASC)
- list_review_candidates excludes soft-deleted and user-locked beliefs
"""
from __future__ import annotations

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


def _mk_belief(
    bid: str,
    content: str = "belief content",
    created_at: str = "2026-01-01T00:00:00Z",
    last_retrieved_at: str | None = None,
    last_confirmed_at: str | None = None,
    lock_level: str = LOCK_NONE,
    valid_to: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=last_retrieved_at,
        last_confirmed_at=last_confirmed_at,
        valid_to=valid_to,
    )


def test_last_confirmed_at_defaults_to_none() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief("b1")
    s.insert_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.last_confirmed_at is None


def test_last_confirmed_at_roundtrips_through_update() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief("b1")
    s.insert_belief(b)
    b.last_confirmed_at = "2026-06-01T12:00:00Z"
    s.update_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.last_confirmed_at == "2026-06-01T12:00:00Z"


def test_update_last_confirmed_at_sets_field() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief("b1")
    s.insert_belief(b)
    ts = "2026-06-04T10:00:00Z"
    s.update_last_confirmed_at("b1", ts)
    got = s.get_belief("b1")
    assert got is not None
    assert got.last_confirmed_at == ts


def test_update_last_confirmed_at_noop_on_soft_deleted() -> None:
    """update_last_confirmed_at must not touch soft-deleted rows."""
    s = MemoryStore(":memory:")
    b = _mk_belief("b1")
    s.insert_belief(b)
    s.soft_delete_belief("b1")
    s.update_last_confirmed_at("b1", "2026-06-04T10:00:00Z")
    # The belief is soft-deleted so get_belief still returns it
    # (get_belief does not filter on valid_to). Confirm the field
    # was NOT updated on the soft-deleted row.
    got = s.get_belief("b1")
    assert got is not None
    assert got.valid_to is not None  # still soft-deleted
    assert got.last_confirmed_at is None  # field not updated


def test_list_review_candidates_empty_store() -> None:
    s = MemoryStore(":memory:")
    assert s.list_review_candidates() == []


def test_list_review_candidates_ordering_nulls_first() -> None:
    """Ordering: last_confirmed_at NULL before non-NULL, then last_retrieved_at
    NULL before non-NULL, then created_at ASC."""
    s = MemoryStore(":memory:")
    # b_confirmed: confirmed recently — should appear last
    b_confirmed = _mk_belief(
        "confirmed",
        content="confirmed belief",
        created_at="2026-01-01T00:00:00Z",
        last_confirmed_at="2026-06-01T00:00:00Z",
    )
    # b_old: never confirmed, old retrieval
    b_old = _mk_belief(
        "old",
        content="old retrieved",
        created_at="2026-01-02T00:00:00Z",
        last_retrieved_at="2026-01-10T00:00:00Z",
    )
    # b_never: never confirmed, never retrieved, oldest created
    b_never = _mk_belief(
        "never",
        content="never retrieved",
        created_at="2026-01-01T00:00:00Z",
    )
    # b_newer: never confirmed, never retrieved, newer created
    b_newer = _mk_belief(
        "newer",
        content="newer belief",
        created_at="2026-02-01T00:00:00Z",
    )
    for b in (b_confirmed, b_old, b_never, b_newer):
        s.insert_belief(b)

    result = s.list_review_candidates()
    ids = [b.id for b in result]
    # never-confirmed, never-retrieved come before old-retrieved
    # and both of those before confirmed
    assert ids.index("never") < ids.index("old")
    assert ids.index("newer") < ids.index("old")
    # never (older created_at) before newer (newer created_at)
    assert ids.index("never") < ids.index("newer")
    assert ids.index("old") < ids.index("confirmed")


def test_list_review_candidates_excludes_soft_deleted() -> None:
    s = MemoryStore(":memory:")
    b1 = _mk_belief("b1", content="active belief")
    b2 = _mk_belief("b2", content="deleted belief")
    s.insert_belief(b1)
    s.insert_belief(b2)
    s.soft_delete_belief("b2")
    result = s.list_review_candidates()
    ids = [b.id for b in result]
    assert "b1" in ids
    assert "b2" not in ids


def test_list_review_candidates_excludes_user_locked() -> None:
    s = MemoryStore(":memory:")
    b_open = _mk_belief("open", content="open belief")
    b_locked = _mk_belief("locked", content="locked belief", lock_level=LOCK_USER)
    s.insert_belief(b_open)
    s.insert_belief(b_locked)
    result = s.list_review_candidates()
    ids = [b.id for b in result]
    assert "open" in ids
    assert "locked" not in ids


def test_list_review_candidates_limit() -> None:
    s = MemoryStore(":memory:")
    for i in range(15):
        s.insert_belief(_mk_belief(
            f"b{i:02d}",
            content=f"belief number {i:02d}",
            created_at=f"2026-01-{i+1:02d}T00:00:00Z",
        ))
    assert len(s.list_review_candidates(limit=10)) == 10
    assert len(s.list_review_candidates(limit=5)) == 5
    assert len(s.list_review_candidates(limit=20)) == 15
