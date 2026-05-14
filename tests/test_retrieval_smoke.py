"""Retrieval module smoke + basic L0/L1 wiring.

Confirms the two-layer flow runs end-to-end against an in-memory store:
locked beliefs come back even when they don't match the query, and FTS5
results come back when they do.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def test_retrieval_returns_l0_locked_beliefs_for_unrelated_query() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("L1", "user pinned a fact about cats",
                        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("F1", "an unrelated fact about elephants"))

    hits = retrieve(s, query="dogs")
    ids = [b.id for b in hits]
    # L0 fires regardless of query content.
    assert "L1" in ids
    # F1 doesn't match "dogs" so L1 produces nothing.
    assert "F1" not in ids


def test_retrieval_returns_l1_fts5_match_when_present() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.insert_belief(_mk("F2", "the garage is full of tools"))

    hits = retrieve(s, query="bananas")
    ids = {b.id for b in hits}
    assert ids == {"F1"}


def test_retrieval_empty_query_returns_locked_only() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("L1", "locked truth",
                        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("F1", "free-floating fact"))

    hits = retrieve(s, query="")
    ids = [b.id for b in hits]
    assert ids == ["L1"]


def test_retrieval_dedupes_locked_belief_appearing_in_fts5_match() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("L1", "the user pinned a banana fact",
                        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))

    hits = retrieve(s, query="banana")
    ids = [b.id for b in hits]
    # L1 first (locked), F1 second (FTS5), no duplicate.
    assert ids[0] == "L1"
    assert ids.count("L1") == 1
    assert "F1" in ids


def test_retrieval_returns_pure_belief_objects() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "a fact"))
    hits = retrieve(s, query="fact")
    assert len(hits) == 1
    assert isinstance(hits[0], Belief)
