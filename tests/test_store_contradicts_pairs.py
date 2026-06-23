"""Tests for MemoryStore.list_contradicts_pairs (#980 signal c source).

Real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _belief(store: MemoryStore, belief_id: str, content: str) -> None:
    store.insert_belief(
        Belief(
            id=belief_id,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        )
    )


def _contradicts(store: MemoryStore, src: str, dst: str) -> None:
    store.insert_edge(Edge(src=src, dst=dst, type=EDGE_CONTRADICTS, weight=-0.5))


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_empty_store_no_pairs(store: MemoryStore) -> None:
    assert store.list_contradicts_pairs() == []


def test_returns_canonicalised_pair(store: MemoryStore) -> None:
    _belief(store, "b", "beta")
    _belief(store, "a", "alpha")
    # Insert in non-canonical direction; result must be (min, max).
    _contradicts(store, "b", "a")
    assert store.list_contradicts_pairs() == [("a", "b")]


def test_dedups_bidirectional_edges(store: MemoryStore) -> None:
    _belief(store, "a", "alpha")
    _belief(store, "b", "beta")
    _contradicts(store, "a", "b")
    _contradicts(store, "b", "a")
    assert store.list_contradicts_pairs() == [("a", "b")]


def test_multiple_pairs_sorted(store: MemoryStore) -> None:
    for bid in ("a", "b", "c", "d"):
        _belief(store, bid, f"content-{bid}")
    _contradicts(store, "c", "d")
    _contradicts(store, "a", "b")
    assert store.list_contradicts_pairs() == [("a", "b"), ("c", "d")]


def test_other_edge_types_ignored(store: MemoryStore) -> None:
    _belief(store, "a", "alpha")
    _belief(store, "b", "beta")
    store.insert_edge(Edge(src="a", dst="b", type="RELATES_TO", weight=0.5))
    assert store.list_contradicts_pairs() == []


def test_soft_deleted_endpoint_excluded(store: MemoryStore) -> None:
    _belief(store, "a", "alpha")
    _belief(store, "b", "beta")
    _contradicts(store, "a", "b")
    assert store.list_contradicts_pairs() == [("a", "b")]
    store.soft_delete_belief("b")
    # A contradiction touching a GC'd belief is not a live contradiction.
    assert store.list_contradicts_pairs() == []
