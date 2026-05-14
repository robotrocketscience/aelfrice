# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportConstantRedefinition=false
"""Tests for the signed normalized Laplacian + eigenbasis builder (#149).

Acceptance map (#149 §Test plan):
- AC1: build_signed_adjacency → symmetric sparse matrix
- AC2: edge-type weight mapping is a module-level constant
- AC3: build_signed_normalized_laplacian → symmetric L per Kunegis 2010
- AC4: compute_eigenbasis returns smallest-K eigenpairs
- AC5: save/load round-trip is byte-identical (numerically lossless)
- AC6: invalidation callback fires on store mutation
- AC7: build is deterministic at fixed seed for the eigsolve
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from aelfrice.graph_spectral import (
    DEFAULT_K,
    EDGE_LAPLACIAN_WEIGHTS,
    GraphEigenbasisCache,
    build_signed_adjacency,
    build_signed_normalized_laplacian,
    compute_eigenbasis,
    load_eigenbasis,
    save_eigenbasis,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _has_run_perf(request: pytest.FixtureRequest) -> bool:
    try:
        return bool(request.config.getoption("--run-perf", default=False))
    except (AttributeError, ValueError):
        return False


def _mk(bid: str, content: str = "x") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _toy_store() -> MemoryStore:
    """Five-node toy graph mixing positive and negative edges.

    Topology:
        b1 -SUPPORTS-> b2     (+1)
        b2 -CITES-> b3        (+1)
        b1 -CONTRADICTS-> b3  (-1)
        b3 -SUPERSEDES-> b4   (-0.5)
        b4 -RELATES_TO-> b5   (+1)
    """
    s = MemoryStore(":memory:")
    for i in range(1, 6):
        s.insert_belief(_mk(f"b{i}"))
    s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b2", dst="b3", type=EDGE_CITES, weight=1.0))
    s.insert_edge(Edge(src="b1", dst="b3", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="b3", dst="b4", type=EDGE_SUPERSEDES, weight=1.0))
    s.insert_edge(Edge(src="b4", dst="b5", type=EDGE_RELATES_TO, weight=1.0))
    return s


# --- AC1: adjacency symmetry & ordering ---------------------------------------


def test_signed_adjacency_is_symmetric() -> None:
    s = _toy_store()
    W, ids = build_signed_adjacency(s)
    assert ids == ["b1", "b2", "b3", "b4", "b5"]
    Wd = W.toarray()
    np.testing.assert_array_equal(Wd, Wd.T)
    # b1 <-> b3 carries -1 (contradicts) plus 0 supportive => -1
    assert Wd[0, 2] == pytest.approx(-1.0)
    assert Wd[2, 0] == pytest.approx(-1.0)
    # b1 <-> b2 supports => +1
    assert Wd[0, 1] == pytest.approx(1.0)
    # b3 <-> b4 supersedes => -0.5
    assert Wd[2, 3] == pytest.approx(-0.5)


def test_signed_adjacency_empty_store() -> None:
    s = MemoryStore(":memory:")
    W, ids = build_signed_adjacency(s)
    assert ids == []
    assert W.shape == (0, 0)


# --- AC2: edge-type weight mapping is a module-level constant -----------------


def test_edge_laplacian_weights_constant() -> None:
    """Mapping is exposed and overridable per spec acceptance #2."""
    assert EDGE_LAPLACIAN_WEIGHTS[EDGE_SUPPORTS] == 1.0
    assert EDGE_LAPLACIAN_WEIGHTS[EDGE_CONTRADICTS] == -1.0
    assert EDGE_LAPLACIAN_WEIGHTS[EDGE_SUPERSEDES] == -0.5
    s = _toy_store()
    # Override: drop CONTRADICTS to zero — adjacency loses that entry.
    custom = dict(EDGE_LAPLACIAN_WEIGHTS)
    custom[EDGE_CONTRADICTS] = 0.0
    W, _ = build_signed_adjacency(s, weights=custom)
    Wd = W.toarray()
    assert Wd[0, 2] == pytest.approx(0.0)


# --- AC3: Laplacian symmetry & shape ------------------------------------------


def test_laplacian_symmetric_and_normalized() -> None:
    s = _toy_store()
    W, _ = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    Ld = L.toarray()
    np.testing.assert_allclose(Ld, Ld.T, atol=1e-12)
    # Diagonal entries are 1 for nodes with nonzero absolute degree.
    deg_abs = np.abs(W.toarray()).sum(axis=1)
    for i, d in enumerate(deg_abs):
        if d > 0:
            assert Ld[i, i] == pytest.approx(1.0, abs=1e-12)


def test_laplacian_empty_input() -> None:
    L = build_signed_normalized_laplacian(sp.csr_matrix((0, 0)))
    assert L.shape == (0, 0)


# --- AC4: eigenbasis -----------------------------------------------------------


def test_eigenbasis_smallest_k() -> None:
    s = _toy_store()
    W, _ = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    eigvals, eigvecs = compute_eigenbasis(L, k=3)
    # toy graph N=5, k=3 → falls back to dense eigh (k_eff = min(3,4) = 3)
    assert eigvals.shape == (3,)
    assert eigvecs.shape == (5, 3)
    # Ascending order
    assert np.all(np.diff(eigvals) >= -1e-12)
    # Reconstruction sanity: L @ v ≈ λ v for each pair
    Ld = L.toarray()
    for j in range(eigvals.shape[0]):
        lhs = Ld @ eigvecs[:, j]
        rhs = eigvals[j] * eigvecs[:, j]
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)


def test_eigenbasis_real_valued() -> None:
    """Symmetric real matrices yield real eigenvalues."""
    s = _toy_store()
    W, _ = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    eigvals, _ = compute_eigenbasis(L, k=3)
    assert eigvals.dtype.kind == "f"
    assert np.all(np.isfinite(eigvals))


# --- AC5: serialization round-trip --------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    s = _toy_store()
    W, ids = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    ev, vc = compute_eigenbasis(L, k=3)

    p = tmp_path / "subdir" / "graph_eigenbasis.npz"
    save_eigenbasis(p, ev, vc, ids)
    assert p.exists()

    ev2, vc2, ids2 = load_eigenbasis(p)
    np.testing.assert_array_equal(ev, ev2)
    np.testing.assert_array_equal(vc, vc2)
    assert ids2 == ids


def test_save_load_without_belief_ids(tmp_path: Path) -> None:
    p = tmp_path / "no_ids.npz"
    ev = np.array([0.1, 0.2, 0.3])
    vc = np.eye(3)
    save_eigenbasis(p, ev, vc, belief_ids=None)
    ev2, vc2, ids2 = load_eigenbasis(p)
    np.testing.assert_array_equal(ev, ev2)
    np.testing.assert_array_equal(vc, vc2)
    assert ids2 is None


# --- AC6: invalidation callback wiring ----------------------------------------


def test_eigenbasis_cache_invalidated_on_mutation(tmp_path: Path) -> None:
    s = _toy_store()
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz", k=3)
    assert cache.is_stale()
    cache.build()
    assert not cache.is_stale()
    assert cache.eigvals is not None
    assert (tmp_path / "eb.npz").exists()

    # Insert a new belief — fires _fire_invalidation
    s.insert_belief(_mk("b6"))
    assert cache.is_stale()
    assert cache.eigvals is None
    assert cache.eigvecs is None
    assert cache.belief_ids is None
    # Persisted file is removed alongside the in-memory wipe so a
    # crash-restart cannot re-load stale eigenpairs.
    assert not (tmp_path / "eb.npz").exists()


def test_eigenbasis_cache_invalidated_on_edge_mutation(tmp_path: Path) -> None:
    s = _toy_store()
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz", k=3)
    cache.build()
    assert not cache.is_stale()

    s.insert_edge(Edge(src="b1", dst="b5", type=EDGE_SUPPORTS, weight=1.0))
    assert cache.is_stale()


def test_rebuild_after_invalidation_diverges(tmp_path: Path) -> None:
    """Cached eigenbasis from before the mutation is not consumed
    after; the rebuild produces a fresh result that differs from
    the pre-mutation snapshot when the graph changed."""
    s = _toy_store()
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz", k=3)
    cache.build()
    ev_before = np.copy(cache.eigvals)  # type: ignore[arg-type]

    # Add a new node + supportive edge — changes the spectrum.
    s.insert_belief(_mk("b6"))
    s.insert_edge(Edge(src="b1", dst="b6", type=EDGE_SUPPORTS, weight=1.0))
    cache.build()
    ev_after = cache.eigvals
    assert ev_after is not None
    # Spectrum changed (different N, different topology).
    assert ev_before.shape != ev_after.shape or not np.allclose(
        ev_before, ev_after, atol=1e-9
    )


# --- AC7: determinism ---------------------------------------------------------


def test_build_deterministic(tmp_path: Path) -> None:
    """Same store, two builds → identical eigenvalues (numerically).
    Eigenvector signs may differ across eigsh runs; we test eigvals
    plus the absolute-value subspace projection norm, which is
    sign-invariant."""
    s = _toy_store()
    W, _ = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    ev1, vc1 = compute_eigenbasis(L, k=3)
    ev2, vc2 = compute_eigenbasis(L, k=3)
    np.testing.assert_allclose(ev1, ev2, atol=1e-10)
    # Subspace agreement: |vc1.T @ vc2| diagonal close to 1
    overlap = np.abs(vc1.T @ vc2)
    np.testing.assert_allclose(np.diag(overlap), np.ones(3), atol=1e-8)


# --- Perf gate ----------------------------------------------------------------


def test_eigsolve_under_budget_n10k(request: pytest.FixtureRequest) -> None:
    """Spec table: top-200 eigsolve at N=10k completes in ~4s. Gated
    behind --run-perf."""
    if not _has_run_perf(request):
        pytest.skip("perf benchmark; pass --run-perf to enable")
    rng = np.random.default_rng(0)
    n = 10_000
    # Sparse signed graph: ~5 edges/node, 20% negative.
    nnz = 5 * n
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    signs = rng.choice([-1.0, 1.0], size=nnz, p=[0.2, 0.8])
    W = sp.csr_matrix((signs, (rows, cols)), shape=(n, n))
    W = (W + W.T).tocsr()
    L = build_signed_normalized_laplacian(W)
    t0 = time.monotonic()
    eigvals, eigvecs = compute_eigenbasis(L, k=DEFAULT_K)
    elapsed = time.monotonic() - t0
    assert eigvals.shape == (DEFAULT_K,)
    assert eigvecs.shape == (n, DEFAULT_K)
    # Generous bound — spec says ~4s; allow 30s for noisy hosts.
    assert elapsed < 30.0
