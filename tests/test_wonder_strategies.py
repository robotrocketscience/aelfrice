"""Unit tests for the three #228 generation strategies."""
from __future__ import annotations

import random

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.models import (
    STRATEGY_RW,
    STRATEGY_STS,
    STRATEGY_TC,
)
from aelfrice.wonder.strategies import (
    DEFAULT_RW_UNCERTAINTY_FLOOR,
    random_walk,
    span_topic_sampling,
    triangle_closure,
)


def _belief(bid: str, *, alpha: float = 1.0, beta: float = 1.0,
            session: str | None = None) -> Belief:
    return Belief(
        id=bid, content=f"c_{bid}", content_hash=f"h_{bid}",
        alpha=alpha, beta=beta, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-05-03T00:00:00Z", last_retrieved_at=None,
        session_id=session,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_random_walk_empty_store(store: MemoryStore) -> None:
    assert random_walk(store, rng=random.Random(0), n_walks=10) == []


def test_random_walk_no_edges_returns_empty(store: MemoryStore) -> None:
    # A seed with no outgoing edges produces a 1-atom bundle, which
    # is below the ≥2 minimum, so the strategy returns nothing.
    store.insert_belief(_belief("a", alpha=0.7, beta=0.7))
    assert random_walk(store, rng=random.Random(0), n_walks=5) == []


def test_random_walk_produces_phantoms_with_edges(store: MemoryStore) -> None:
    for bid in ("a", "b", "c"):
        store.insert_belief(_belief(bid, alpha=0.7, beta=0.7))
    store.insert_edge(Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0))
    store.insert_edge(Edge(src="b", dst="c", type=EDGE_SUPPORTS, weight=1.0))
    phantoms = random_walk(store, rng=random.Random(0), n_walks=10, depth=2)
    assert phantoms
    for p in phantoms:
        assert p.strategy == STRATEGY_RW
        assert p.seed_id is not None
        assert len(p.composition) >= 2
        assert tuple(sorted(p.composition)) == p.composition  # sorted
        assert p.construction_cost >= 1.0


def test_random_walk_floor_excludes_low_uncertainty(store: MemoryStore) -> None:
    # alpha=10, beta=10 gives strongly negative differential entropy;
    # with the default floor (-0.5) this belief is ineligible as a seed.
    store.insert_belief(_belief("low", alpha=10.0, beta=10.0))
    store.insert_belief(_belief("low2", alpha=10.0, beta=10.0))
    store.insert_edge(Edge(src="low", dst="low2", type=EDGE_SUPPORTS, weight=1.0))
    assert random_walk(
        store, rng=random.Random(0), n_walks=5,
        uncertainty_floor=DEFAULT_RW_UNCERTAINTY_FLOOR,
    ) == []


def test_random_walk_is_deterministic(store: MemoryStore) -> None:
    for bid in ("a", "b", "c", "d"):
        store.insert_belief(_belief(bid, alpha=0.7, beta=0.7))
    store.insert_edge(Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0))
    store.insert_edge(Edge(src="b", dst="c", type=EDGE_CITES, weight=1.0))
    store.insert_edge(Edge(src="c", dst="d", type=EDGE_RELATES_TO, weight=1.0))
    a = random_walk(store, rng=random.Random(42), n_walks=10, depth=2)
    b = random_walk(store, rng=random.Random(42), n_walks=10, depth=2)
    assert [p.composition for p in a] == [p.composition for p in b]


def test_triangle_closure_empty_store(store: MemoryStore) -> None:
    assert triangle_closure(store) == []


def test_triangle_closure_finds_pair(store: MemoryStore) -> None:
    for bid in ("a", "b", "c"):
        store.insert_belief(_belief(bid))
    store.insert_edge(Edge(src="a", dst="c", type=EDGE_SUPPORTS, weight=1.0))
    store.insert_edge(Edge(src="b", dst="c", type=EDGE_SUPPORTS, weight=1.0))
    phantoms = triangle_closure(store)
    assert len(phantoms) == 1
    p = phantoms[0]
    assert p.composition == ("a", "b")
    assert p.strategy == STRATEGY_TC
    assert p.construction_cost == 3.0
    assert p.seed_id is None


def test_triangle_closure_skips_non_eligible_edge_types(store: MemoryStore) -> None:
    from aelfrice.models import EDGE_CONTRADICTS
    for bid in ("a", "b", "c"):
        store.insert_belief(_belief(bid))
    store.insert_edge(Edge(src="a", dst="c", type=EDGE_CONTRADICTS, weight=1.0))
    store.insert_edge(Edge(src="b", dst="c", type=EDGE_CONTRADICTS, weight=1.0))
    assert triangle_closure(store) == []


def test_triangle_closure_dedups_unordered_pairs(store: MemoryStore) -> None:
    # Three sources sharing one target → C(3,2) = 3 unordered pairs.
    for bid in ("a", "b", "c", "t"):
        store.insert_belief(_belief(bid))
    for src in ("a", "b", "c"):
        store.insert_edge(Edge(src=src, dst="t", type=EDGE_SUPPORTS, weight=1.0))
    phantoms = triangle_closure(store)
    assert len(phantoms) == 3
    comps = {p.composition for p in phantoms}
    assert comps == {("a", "b"), ("a", "c"), ("b", "c")}


def test_span_topic_sampling_empty_store(store: MemoryStore) -> None:
    assert span_topic_sampling(store, rng=random.Random(0)) == []


def test_span_topic_sampling_needs_enough_sessions(store: MemoryStore) -> None:
    store.insert_belief(_belief("a", session="s1"))
    store.insert_belief(_belief("b", session="s1"))
    # only one real session — composition_size=2 → no phantoms
    assert span_topic_sampling(store, rng=random.Random(0), n_samples=5) == []


def test_span_topic_sampling_produces_diverse_compositions(
    store: MemoryStore,
) -> None:
    for i, bid in enumerate(("a", "b", "c", "d")):
        store.insert_belief(_belief(bid, session=f"sess_{i}"))
    phantoms = span_topic_sampling(store, rng=random.Random(0), n_samples=20)
    assert phantoms
    for p in phantoms:
        assert p.strategy == STRATEGY_STS
        assert len(p.composition) == 2
        assert p.construction_cost == 2.0
        assert p.seed_id is None


def test_span_topic_sampling_rejects_invalid_size(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        span_topic_sampling(store, rng=random.Random(0), composition_size=1)
