# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportConstantRedefinition=false
"""Tests for the heat-kernel composition path in retrieve() — Slice 2 of #151.

These tests cover the wiring of `eigenbasis_cache` + `heat_kernel_enabled`
through `retrieve()` -> `_l1_hits` -> `_heat_by_id` -> `combine_log_scores`.
The pure spectral math is covered by `tests/test_heat_kernel.py`; this file
exercises the retrieval-side dispatch and graceful-degrade contract.

Test plan (per the design note):
1. heat-off byte-identical to the no-heat-kwarg call (Slice 1 contract).
2. heat-on with an unbuilt eigenbasis falls back byte-identically.
3. heat-on with a built eigenbasis re-orders an authority graph as expected.
4. cold belief (inserted after build) is rankable, gets HEAT_SCORE_FLOOR.
5. store mutation flips `cache.is_stale()` and the next retrieval degrades.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.graph_spectral import GraphEigenbasisCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _mk(bid: str, content: str, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-29T00:00:00Z",
        last_retrieved_at=None,
    )


def _build_authority_store() -> tuple[MemoryStore, str, str]:
    """Three beliefs sharing the same query terms.

    `hub` is supported by `s1` and `s2` (incoming SUPPORTS edges) so it
    accumulates positive heat through the signed Laplacian. `iso` is
    isolated with the same query-term overlap. Returns (store, hub_id,
    iso_id).
    """
    s = MemoryStore(":memory:")
    # `iso` has slightly higher BM25 score (extra term repetition) so at
    # baseline it outranks `hub`. With heat on, the two SUPPORTS edges
    # accumulating into `hub` should overcome the BM25 gap and lift it.
    s.insert_belief(_mk("hub", "alpha beta gamma delta epsilon"))
    s.insert_belief(_mk("s1", "alpha beta gamma delta epsilon"))
    s.insert_belief(_mk("s2", "alpha beta gamma delta epsilon"))
    s.insert_belief(_mk("iso", "alpha alpha beta beta gamma gamma delta epsilon"))
    # Two incoming SUPPORTS into `hub`. SUPPORTS carries +1 weight in the
    # signed-Laplacian convention so heat accumulates positively at `hub`.
    s.insert_edge(Edge(src="s1", dst="hub", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="s2", dst="hub", type=EDGE_SUPPORTS, weight=1.0))
    return s, "hub", "iso"


def _build_small_store() -> MemoryStore:
    """Three beliefs, no edges — enough for the byte-identical contract."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("a", "rust ownership borrow checker"))
    s.insert_belief(_mk("b", "rust ownership borrow checker"))
    s.insert_belief(_mk("c", "rust ownership borrow checker"))
    return s


# ---------------------------------------------------------------------------
# Test 1 — heat-off byte-identical to Slice 1 contract (AC4)
# ---------------------------------------------------------------------------


def test_heat_kernel_off_byte_identical_to_slice1() -> None:
    """`heat_kernel_enabled=False` must match the no-kwarg call exactly."""
    s = _build_small_store()
    query = "rust ownership"

    no_kwarg = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
    )
    explicit_off = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        heat_kernel_enabled=False,
    )

    assert [b.id for b in no_kwarg] == [b.id for b in explicit_off]
    s.close()


# ---------------------------------------------------------------------------
# Test 2 — empty / unbuilt eigenbasis falls back (AC4 fallback)
# ---------------------------------------------------------------------------


def test_heat_kernel_empty_eigenbasis_falls_back(tmp_path: Path) -> None:
    """heat_kernel_enabled=True with a never-built cache must match heat-off."""
    s = _build_small_store()
    query = "rust ownership"
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz")
    # Critically: do NOT call cache.build(). eigvals stays None, the
    # `_l1_hits` heat dispatch sees the None and degrades to the
    # partial_bayesian_score path.

    heat_off = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        heat_kernel_enabled=False,
    )
    heat_on_unbuilt = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        heat_kernel_enabled=True, eigenbasis_cache=cache,
    )

    assert [b.id for b in heat_off] == [b.id for b in heat_on_unbuilt]
    s.close()


# ---------------------------------------------------------------------------
# Test 3 — heat-on changes ranking on a connected authority graph
# ---------------------------------------------------------------------------


def test_heat_kernel_changes_ranking_on_authority_graph(tmp_path: Path) -> None:
    """With a built eigenbasis, the highly-supported `hub` should rank
    above the isolated belief at otherwise-similar BM25.

    Use posterior_weight=0.0 so the difference is purely the heat term.
    """
    s, hub_id, iso_id = _build_authority_store()
    query = "alpha beta gamma"

    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz")
    cache.build()
    assert cache.eigvals is not None  # sanity

    heat_off = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        posterior_weight=0.0, heat_kernel_enabled=False,
    )
    heat_on = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        posterior_weight=0.0, heat_kernel_enabled=True, eigenbasis_cache=cache,
    )

    off_ids = [b.id for b in heat_off]
    on_ids = [b.id for b in heat_on]

    # Both calls return the same belief set.
    assert set(off_ids) == set(on_ids)
    # Heat propagation should reorder: the hub (two incoming SUPPORTS) must
    # rank strictly above the isolated belief.
    assert hub_id in on_ids and iso_id in on_ids
    assert on_ids.index(hub_id) < on_ids.index(iso_id), (
        f"hub {hub_id} should outrank isolated {iso_id} with heat on; got {on_ids}"
    )
    # And the heat-on ordering should differ from heat-off in at least one
    # rank position (otherwise the term has no observable effect).
    assert off_ids != on_ids, (
        f"heat-on ordering identical to heat-off; expected at least one "
        f"rank shift. off={off_ids} on={on_ids}"
    )
    s.close()


# ---------------------------------------------------------------------------
# Test 4 — cold belief inserted after build is rankable (HEAT_SCORE_FLOOR)
# ---------------------------------------------------------------------------


def test_heat_kernel_cold_belief_neutral(tmp_path: Path) -> None:
    """A belief inserted after `cache.build()` is missing from
    `cache.belief_ids`. The retrieval path must fall through to the
    heat-off path (since `is_stale()` flips on insert) and still rank
    the cold belief without crashing.
    """
    s = MemoryStore(":memory:")
    for bid in ("a", "b", "c"):
        s.insert_belief(_mk(bid, "rust ownership borrow checker"))
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz")
    cache.build()
    assert cache.belief_ids == ["a", "b", "c"]

    # Insert AFTER build. Triggers the invalidation callback -> is_stale.
    s.insert_belief(_mk("d", "rust ownership borrow checker"))
    assert cache.is_stale() is True

    results = retrieve(
        s, "rust ownership", l1_limit=10,
        entity_index_enabled=False, bfs_enabled=False,
        heat_kernel_enabled=True, eigenbasis_cache=cache,
    )
    ids = [b.id for b in results]
    # All four beliefs must be present and rankable; no crash.
    assert set(ids) == {"a", "b", "c", "d"}
    s.close()


# ---------------------------------------------------------------------------
# Test 5 — store mutation flips is_stale() and next retrieval degrades
# ---------------------------------------------------------------------------


def test_heat_kernel_cache_invalidation_on_store_mutation(tmp_path: Path) -> None:
    """After `cache.build()` then any store mutation, `cache.is_stale()`
    must be True and the heat dispatch must degrade to the heat-off path
    (byte-identical to heat off).
    """
    s, hub_id, iso_id = _build_authority_store()
    query = "alpha beta gamma"

    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz")
    cache.build()
    assert cache.is_stale() is False

    # Mutate the store. Per `GraphEigenbasisCache.__post_init__` this
    # registers an invalidation callback that flips `_stale` and clears
    # eigvals/eigvecs/belief_ids.
    s.insert_belief(_mk("new", "alpha beta gamma delta epsilon"))
    assert cache.is_stale() is True
    assert cache.eigvals is None

    heat_off = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        posterior_weight=0.0, heat_kernel_enabled=False,
    )
    heat_on_stale = retrieve(
        s, query, l1_limit=10, entity_index_enabled=False, bfs_enabled=False,
        posterior_weight=0.0, heat_kernel_enabled=True, eigenbasis_cache=cache,
    )

    assert [b.id for b in heat_off] == [b.id for b in heat_on_stale]
    # Defensive: the freshly-inserted belief participates in both rankings.
    assert any(b.id == "new" for b in heat_on_stale)
    # Silence unused-binding warnings on hub_id / iso_id.
    assert hub_id and iso_id
    s.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
