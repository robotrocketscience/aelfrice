"""Soft-deleted beliefs must not be retrievable through any lane (#980).

A belief is soft-deleted (``valid_to`` set) by two production paths:
``wonder_gc`` (stale phantom GC) and ``aelf review`` with a ``remove``
verdict -- both call ``MemoryStore.soft_delete_belief``. Before #980 the
retrieval candidate lanes did not filter ``valid_to``, so a soft-deleted
belief still surfaced in keyword search, the default-on BM25F lane, the
entity index, the locked lane, and end-to-end ``retrieve()``.

These tests pin the invariant: once soft-deleted, a belief is absent from
every read path, while its audit rows (beliefs / belief_entities) survive.
"""
from __future__ import annotations

from aelfrice.bm25 import BM25Index
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    *,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-25T00:00:00Z",
        last_retrieved_at=None,
    )


def _store_with_pair() -> MemoryStore:
    """Two beliefs sharing the token 'alpha'; b2 is soft-deleted."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha factual statement active"))
    s.insert_belief(_mk("b2", "alpha factual statement removed"))
    s.soft_delete_belief("b2")
    return s


def test_search_beliefs_excludes_soft_deleted() -> None:
    s = _store_with_pair()
    ids = [b.id for b in s.search_beliefs("alpha")]
    assert "b1" in ids
    assert "b2" not in ids


def test_search_beliefs_scored_excludes_soft_deleted() -> None:
    s = _store_with_pair()
    ids = [b.id for b, _ in s.search_beliefs_scored("alpha")]
    assert "b1" in ids
    assert "b2" not in ids


def test_soft_delete_prunes_fts_row() -> None:
    s = _store_with_pair()
    # The derived FTS row for the soft-deleted belief is gone; the active
    # one remains.
    rows = s._conn.execute(
        "SELECT id FROM beliefs_fts ORDER BY id"
    ).fetchall()
    fts_ids = {r["id"] for r in rows}
    assert "b1" in fts_ids
    assert "b2" not in fts_ids


def test_soft_deleted_belief_row_survives_for_audit() -> None:
    s = _store_with_pair()
    # The belief itself is retained (valid_to set) so the audit trail and
    # fetch-by-id still work -- only retrieval excludes it.
    got = s.get_belief("b2")
    assert got is not None
    assert got.valid_to is not None


def test_list_beliefs_for_indexing_excludes_soft_deleted() -> None:
    # Covers the BM25/BM25F retrieval lane, dedup, and relationship
    # detection -- all three consume this method.
    s = _store_with_pair()
    ids = [bid for bid, _ in s.list_beliefs_for_indexing()]
    assert "b1" in ids
    assert "b2" not in ids


def test_bm25_index_excludes_soft_deleted() -> None:
    s = _store_with_pair()
    idx = BM25Index.build(s)
    assert "b2" not in idx.belief_ids
    scored_ids = {bid for bid, _ in idx.score("alpha")}
    assert "b1" in scored_ids
    assert "b2" not in scored_ids


def test_lookup_entities_excludes_soft_deleted() -> None:
    s = MemoryStore(":memory:")
    content = "uses aelfrice.retrieval and src/store.py daily"
    s.insert_belief(_mk("b1", content))
    s.insert_belief(_mk("b2", content))
    # A shared entity key exists for both; pick one actually extracted.
    keys = [k for k, _, _ in s.belief_entities_for("b2")]
    assert keys, "fixture must produce at least one entity"
    s.soft_delete_belief("b2")
    hit_ids = {bid for bid, _ in s.lookup_entities(keys, limit=10)}
    assert "b1" in hit_ids
    assert "b2" not in hit_ids
    # belief_entities rows for the soft-deleted belief are preserved (audit).
    assert s.belief_entities_for("b2") != []


def test_list_locked_beliefs_excludes_soft_deleted() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(
        _mk("L1", "locked active", lock_level=LOCK_USER,
            locked_at="2026-04-25T00:00:00Z")
    )
    s.insert_belief(
        _mk("L2", "locked removed", lock_level=LOCK_USER,
            locked_at="2026-04-25T00:00:00Z")
    )
    s.soft_delete_belief("L2")
    ids = [b.id for b in s.list_locked_beliefs()]
    assert "L1" in ids
    assert "L2" not in ids


def test_retrieve_excludes_soft_deleted_end_to_end() -> None:
    # The user-facing assertion: retrieve() (default lanes, BM25F default-on)
    # never surfaces a soft-deleted belief.
    s = _store_with_pair()
    ids = [b.id for b in retrieve(s, "alpha")]
    assert "b1" in ids
    assert "b2" not in ids
