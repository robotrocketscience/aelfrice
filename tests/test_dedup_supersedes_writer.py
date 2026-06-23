"""Unit tests for the #999 SUPERSEDES edge writer.

Covers ``write_supersedes_edges`` (directional member→oldest edges within
near-duplicate clusters — the dedup write-path documented at dedup.py:160
and deferred behind the #197 bench gate), idempotency, the per-belief
write-gate, and the determinism guarantee.

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.dedup import write_supersedes_edges
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore

# Three near-identical paraphrases — clear dedup's jaccard >= 0.8 AND
# levenshtein >= 0.85 thresholds, so they collapse into one cluster.
_BASE = "never push commits directly to the main branch"
_BASE_NOW = _BASE + " now"
_BASE_HERE = _BASE + " here"
_UNRELATED = "the harbor seals bask on the warm rocks at noon each day"


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    created_at: str,
) -> Belief:
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _supersedes_edges(store: MemoryStore) -> list[tuple[str, str]]:
    """Return all SUPERSEDES edges as sorted (src, dst) tuples."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_SUPERSEDES,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def _seed_cluster(store: MemoryStore) -> None:
    # b1 is the oldest → the SUPERSEDES target. ids sorted b1 < b2 < b3.
    _make_belief(store, belief_id="b1", content=_BASE, created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="b2", content=_BASE_NOW, created_at="2026-02-01T00:00:00Z")
    _make_belief(store, belief_id="b3", content=_BASE_HERE, created_at="2026-03-01T00:00:00Z")


# ---------------------------------------------------------------------------
# write_supersedes_edges — write / scope / idempotency
# ---------------------------------------------------------------------------


def test_writes_supersedes_edges_to_oldest(store: MemoryStore) -> None:
    _seed_cluster(store)
    report = write_supersedes_edges(store)
    assert report.n_clusters == 1
    assert report.n_edges_written == 2
    # Both newer members point at the oldest (b1); direction is member→oldest.
    assert _supersedes_edges(store) == [("b2", "b1"), ("b3", "b1")]


def test_idempotent_second_run_writes_nothing(store: MemoryStore) -> None:
    _seed_cluster(store)
    write_supersedes_edges(store)
    report = write_supersedes_edges(store)
    assert report.n_edges_written == 0
    assert report.n_edges_skipped_existing == 2
    assert _supersedes_edges(store) == [("b2", "b1"), ("b3", "b1")]


def test_no_cluster_writes_no_edges(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_BASE, created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="b2", content=_UNRELATED, created_at="2026-02-01T00:00:00Z")
    report = write_supersedes_edges(store)
    assert report.n_clusters == 0
    assert report.n_edges_written == 0
    assert _supersedes_edges(store) == []


# ---------------------------------------------------------------------------
# Determinism — byte-equal edge sets across two fresh stores
# ---------------------------------------------------------------------------


def test_determinism_byte_equal_edges() -> None:
    def build() -> list[tuple[str, str]]:
        s = MemoryStore(":memory:")
        _seed_cluster(s)
        write_supersedes_edges(s)
        return _supersedes_edges(s)

    assert build() == build()


# ---------------------------------------------------------------------------
# Write-gate — per-belief edge cap (Exp-48 dilution guard)
# ---------------------------------------------------------------------------


def test_write_gate_caps_per_belief_edges(store: MemoryStore) -> None:
    _seed_cluster(store)
    # The oldest target (b1) is the dst of both edges; cap=1 lets only the
    # first (b2→b1) through, gating b3→b1 once b1 hits its budget.
    report = write_supersedes_edges(store, max_edges_per_belief=1)
    assert report.n_edges_written == 1
    assert report.n_edges_skipped_gated == 1
    assert _supersedes_edges(store) == [("b2", "b1")]
