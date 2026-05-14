"""DERIVED_FROM edge type — registered in vocabulary, persisted, propagated."""
from __future__ import annotations

from aelfrice.models import (
    EDGE_DERIVED_FROM,
    EDGE_TYPES,
    EDGE_VALENCE,
    Belief,
    Edge,
    LOCK_NONE,
)
from aelfrice.store import MemoryStore


def _b(id_: str, content: str = "x", alpha: float = 5.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_,
        alpha=alpha,
        beta=beta,
        type="factual",
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-27T00:00:00+00:00",
        last_retrieved_at=None,
    )


def test_derived_from_in_edge_types() -> None:
    assert EDGE_DERIVED_FROM == "DERIVED_FROM"
    assert EDGE_DERIVED_FROM in EDGE_TYPES


def test_derived_from_valence_is_half() -> None:
    assert EDGE_VALENCE[EDGE_DERIVED_FROM] == 0.5


def test_store_persists_derived_from() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("a"))
        store.insert_belief(_b("b"))
        e = Edge(src="b", dst="a", type=EDGE_DERIVED_FROM, weight=1.0)
        store.insert_edge(e)
        got = store.get_edge("b", "a", EDGE_DERIVED_FROM)
        assert got is not None
        assert got.type == EDGE_DERIVED_FROM
    finally:
        store.close()


def test_propagate_valence_walks_derived_from() -> None:
    """DERIVED_FROM walks like CITES: positive propagation, broker-attenuated."""
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("src", alpha=10.0, beta=1.0))
        store.insert_belief(_b("mid", alpha=10.0, beta=1.0))
        store.insert_edge(Edge(src="src", dst="mid", type=EDGE_DERIVED_FROM, weight=1.0))
        applied = store.propagate_valence("src", valence=1.0, max_hops=2)
        assert "mid" in applied
        assert applied["mid"] > 0.0
    finally:
        store.close()
