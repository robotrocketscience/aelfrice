"""Every belief/edge mutator fires the store's invalidation registry.

`MemoryStore.add_invalidation_callback` is how derived state learns the
store moved. Four production components subscribe — `BM25IndexCache`,
`graph_spectral.GraphEigenbasisCache`, `hrr_index` and
`query_understanding.store_cache` — so a mutator that commits without
firing lets all four keep serving results derived from a graph that has
since changed, with nothing red.

Until #1418 the only coverage of `delete_belief` and the three edge
mutators was
`tests/test_retrieval_cache.py::test_ac4_every_mutator_invalidates_cache`,
which observed the fire through `RetrievalCache`. That class is deleted;
the contract it observed is not, so it is re-pinned here against a probe
callback instead — one test per mutator, so a mutator that stops firing
names itself rather than hiding inside a loop.

`insert_belief`, `update_belief` and `bump_posterior` are covered by
`tests/test_entity_index.py` (AC4) and `tests/test_bayesian_ranking.py`
(AC7) and are not duplicated here.

Each test is falsifiable by replacing that one mutator's
`self._commit_mutation()` with `self._commit()` in `store.py`: the write
still commits, the durable generation bump and `_fire_invalidation()` do
not, and exactly this test goes red. Both halves are asserted — the
in-process registry and the durable counter that cross-process caches
revalidate against — because they are the two things `_commit_mutation`
does beyond committing.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RELATES_TO,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-08-09T00:00:00Z",
        last_retrieved_at=None,
    )


def _seeded(tmp_path: Path) -> MemoryStore:
    """Two beliefs and one edge between them, on a file-backed store."""
    store = MemoryStore(str(tmp_path / "invalidation.db"))
    store.insert_belief(_mk("A"))
    store.insert_belief(_mk("B"))
    store.insert_edge(
        Edge(src="A", dst="B", type=EDGE_RELATES_TO, weight=1.0)
    )
    return store


def _assert_fires(store: MemoryStore, mutate, label: str) -> None:
    """Run `mutate` and require it to fire the registry and bump the
    durable generation."""
    fired: list[int] = []
    store.add_invalidation_callback(lambda: fired.append(1))
    gen_before = store.store_generation()
    assert not fired, f"{label}: probe fired before the mutation"
    mutate()
    assert fired, f"{label}: invalidation callback did not fire"
    assert store.store_generation() > gen_before, (
        f"{label}: durable store generation did not advance"
    )


def test_delete_belief_fires_invalidation(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    try:
        _assert_fires(
            store, lambda: store.delete_belief("B"), "delete_belief"
        )
    finally:
        store.close()


def test_insert_edge_fires_invalidation(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    try:
        store.insert_belief(_mk("C"))
        edge = Edge(src="A", dst="C", type=EDGE_RELATES_TO, weight=1.0)
        _assert_fires(
            store, lambda: store.insert_edge(edge), "insert_edge"
        )
    finally:
        store.close()


def test_update_edge_fires_invalidation(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    try:
        edge = Edge(src="A", dst="B", type=EDGE_RELATES_TO, weight=0.25)
        _assert_fires(
            store, lambda: store.update_edge(edge), "update_edge"
        )
        reread = store.get_edge("A", "B", EDGE_RELATES_TO)
        assert reread is not None and reread.weight == 0.25, (
            "update_edge did not write the new weight"
        )
    finally:
        store.close()


def test_delete_edge_fires_invalidation(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    try:
        _assert_fires(
            store,
            lambda: store.delete_edge("A", "B", EDGE_RELATES_TO),
            "delete_edge",
        )
        assert store.get_edge("A", "B", EDGE_RELATES_TO) is None
    finally:
        store.close()
