"""Every store mutator fires the invalidation callback registry.

Rescued from `tests/test_retrieval_cache.py`, deleted with
`RetrievalCache` under #1369. The registry itself is production-live and
outlived the cache: `bm25.BM25IndexCache`, `hrr_index`,
`graph_spectral` and `query_understanding.store_cache` all subscribe to
it, and each serves stale derived state if a mutator stops firing.

The old coverage reached the registry only through the cache's
`invalidate`, so deleting the cache would have taken the six-mutator
assertion with it. This asserts the registry directly, with a plain
counting callback, which is also what the live subscribers register.
"""
from __future__ import annotations

from aelfrice.feedback import apply_feedback
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RELATES_TO,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seeded_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.insert_belief(_mk("F2", "the garage is full of tools"))
    return s


def test_every_mutator_fires_the_invalidation_registry() -> None:
    """All six belief / edge mutators call `_fire_invalidation`.

    Counted per mutator rather than once at the end: a single tally
    would pass if one mutator fired twice and another not at all.
    """
    s = _seeded_store()
    fired: list[str] = []
    s.add_invalidation_callback(lambda: fired.append("x"))

    def assert_fires(label: str, mutate: object) -> None:
        before = len(fired)
        mutate()  # type: ignore[operator]
        assert len(fired) > before, f"{label}: did not fire invalidation"

    new_belief = _mk("F3", "the basement is full of crates")
    assert_fires("insert_belief", lambda: s.insert_belief(new_belief))

    fetched = s.get_belief("F3")
    assert fetched is not None
    fetched.content = "the basement holds crates and ropes"
    assert_fires("update_belief", lambda: s.update_belief(fetched))

    assert_fires("delete_belief", lambda: s.delete_belief("F3"))

    edge = Edge(src="F1", dst="F2", type=EDGE_RELATES_TO, weight=1.0)
    assert_fires("insert_edge", lambda: s.insert_edge(edge))

    edge2 = Edge(src="F1", dst="F2", type=EDGE_RELATES_TO, weight=0.5)
    assert_fires("update_edge", lambda: s.update_edge(edge2))

    assert_fires(
        "delete_edge",
        lambda: s.delete_edge("F1", "F2", EDGE_RELATES_TO),
    )


def test_apply_feedback_fires_invalidation_through_the_store() -> None:
    """`apply_feedback` must not special-case subscribers.

    It writes the posterior via `store.update_belief`, so the wipe
    travels the same registry every other mutation uses. A hand-rolled
    notification inside `apply_feedback` would reach whatever it knew
    about and silently miss the rest.
    """
    s = _seeded_store()
    fired: list[str] = []
    s.add_invalidation_callback(lambda: fired.append("x"))
    apply_feedback(s, "F1", valence=+1.0, source="test_invalidation")
    assert fired, "apply_feedback did not fire the store invalidation registry"


def test_callbacks_are_per_store_instance() -> None:
    """Two stores never share subscribers."""
    s1 = _seeded_store()
    s2 = _seeded_store()
    fired1: list[str] = []
    s1.add_invalidation_callback(lambda: fired1.append("x"))
    s2.insert_belief(_mk("F9", "extra fact"))
    assert fired1 == [], "a mutation on one store fired another store's callback"
    s1.insert_belief(_mk("F9", "extra fact"))
    assert fired1 != []
