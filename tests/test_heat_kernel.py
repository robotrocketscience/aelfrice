# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportConstantRedefinition=false
"""Tests for the heat-kernel authority scorer (#150).

Acceptance map (#150 § "Acceptance criteria"):
- AC1: heat_kernel_score matches dense expm reference at t=8, K=200
- AC2: seeds_from_bm25 returns L1-normalized vector with
       exactly min(top_k, len) non-zero entries
- AC3: combine_log_scores implements
       log(BM25F) + 1.0 * log(heat_safe) + 0.5 * log(posterior_or_1)
- AC4: broker attenuation produces ≥ 5% reduction at confidence 0.1,
       ~0% reduction at confidence 0.99
- AC5: per-query latency ≤ 10 ms at N=50k (perf-gated)
- AC6: eigenbasis loaded once per process — covered structurally by
       the existing GraphEigenbasisCache tests
- AC7: feature-flag default is False (use_heat_kernel)
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from aelfrice.graph_spectral import (
    DEFAULT_BM25_SEED_TOP_K,
    DEFAULT_HEAT_BANDWIDTH,
    HEAT_SCORE_FLOOR,
    apply_broker_attenuation,
    build_signed_adjacency,
    build_signed_normalized_laplacian,
    combine_log_scores,
    compute_eigenbasis,
    heat_kernel_safe,
    heat_kernel_score,
    seeds_from_bm25,
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


def _mk(bid: str, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid,
        content=bid,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _toy_store(broker_alpha: float = 1.0, broker_beta: float = 1.0) -> MemoryStore:
    """Same five-node topology as test_graph_spectral; broker per-belief
    overridable for AC4 attenuation tests."""
    s = MemoryStore(":memory:")
    for i in range(1, 6):
        s.insert_belief(_mk(f"b{i}", alpha=broker_alpha, beta=broker_beta))
    s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b2", dst="b3", type=EDGE_CITES, weight=1.0))
    s.insert_edge(Edge(src="b1", dst="b3", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="b3", dst="b4", type=EDGE_SUPERSEDES, weight=1.0))
    s.insert_edge(Edge(src="b4", dst="b5", type=EDGE_RELATES_TO, weight=1.0))
    return s


# --- AC1 -----------------------------------------------------------------


def test_heat_kernel_matches_dense_expm() -> None:
    s = _toy_store()
    W, ids = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W).toarray()

    # The toy graph has only 5 nodes; compute_eigenbasis with K=200
    # falls back to dense eigh — perfect for a reference comparison
    # because we get the FULL spectrum.
    eigvals, eigvecs = compute_eigenbasis(sp.csr_matrix(L), k=200)

    rng = np.random.default_rng(42)
    seeds = rng.uniform(size=(len(ids),))

    eigenbasis_result = heat_kernel_score(
        eigvals, eigvecs, seeds, t=DEFAULT_HEAT_BANDWIDTH,
    )
    dense_kernel = sla.expm(-DEFAULT_HEAT_BANDWIDTH * L)
    reference = dense_kernel @ seeds

    np.testing.assert_allclose(eigenbasis_result, reference, atol=1e-3)


def test_heat_kernel_zero_eigenbasis_returns_zeros() -> None:
    eigvals = np.zeros((0,), dtype=np.float64)
    eigvecs = np.zeros((0, 0), dtype=np.float64)
    seeds = np.zeros((0,), dtype=np.float64)
    out = heat_kernel_score(eigvals, eigvecs, seeds, t=8.0)
    assert out.shape == (0,)


# --- AC2 -----------------------------------------------------------------


def test_seeds_from_bm25_l1_normalized_with_top_k_nonzero() -> None:
    bm25 = np.array([3.0, 0.0, 1.0, 2.0, 0.0, 4.0])
    seeds = seeds_from_bm25(bm25, top_k=3)
    assert seeds.shape == (6,)
    assert np.count_nonzero(seeds) == 3
    assert pytest.approx(seeds.sum(), abs=1e-12) == 1.0
    # The three positive top-k entries are 4.0, 3.0, 2.0 → indices 5, 0, 3
    assert seeds[5] > 0 and seeds[0] > 0 and seeds[3] > 0
    assert seeds[1] == 0.0 and seeds[2] == 0.0 and seeds[4] == 0.0


def test_seeds_from_bm25_top_k_exceeds_positive_count() -> None:
    bm25 = np.array([1.0, 0.0, 2.0, 0.0])
    seeds = seeds_from_bm25(bm25, top_k=DEFAULT_BM25_SEED_TOP_K)
    # Only 2 positive scores; should clamp.
    assert np.count_nonzero(seeds) == 2
    assert pytest.approx(seeds.sum(), abs=1e-12) == 1.0


def test_seeds_from_bm25_all_zero_returns_zero_vector() -> None:
    bm25 = np.zeros(5)
    seeds = seeds_from_bm25(bm25, top_k=25)
    assert np.all(seeds == 0)


# --- AC3 -----------------------------------------------------------------


def test_combine_log_scores_log_additive() -> None:
    bm25 = 2.0
    heat = 0.5
    posterior = 0.8
    expected = (
        np.log(bm25) + 1.0 * np.log(heat) + 0.5 * np.log(posterior)
    )
    got = combine_log_scores(bm25, heat, posterior)
    assert pytest.approx(got, abs=1e-12) == expected


def test_combine_log_scores_no_posterior_defaults_to_one() -> None:
    bm25 = 2.0
    heat = 0.5
    no_post = combine_log_scores(bm25, heat, None)
    explicit_one = combine_log_scores(bm25, heat, 1.0)
    assert pytest.approx(no_post, abs=1e-12) == explicit_one
    # log(1) = 0, so the third term contributes nothing
    assert pytest.approx(no_post, abs=1e-12) == np.log(bm25) + np.log(heat)


def test_combine_log_scores_floors_heat() -> None:
    # heat below the floor must be clamped before log
    bm25 = 1.0
    out = combine_log_scores(bm25, heat=-1.0)
    expected = np.log(1.0) + np.log(HEAT_SCORE_FLOOR)
    assert pytest.approx(out, abs=1e-9) == expected


def test_combine_log_scores_rejects_nonpositive_bm25() -> None:
    with pytest.raises(ValueError):
        combine_log_scores(0.0, 0.5, 0.5)


def test_heat_kernel_safe_clamps_below_floor() -> None:
    scores = np.array([-1.0, 0.0, 1e-15, 0.5, 2.0])
    out = heat_kernel_safe(scores)
    assert np.all(out >= HEAT_SCORE_FLOOR)
    assert out[3] == 0.5
    assert out[4] == 2.0


# --- AC4 -----------------------------------------------------------------


def test_broker_attenuation_low_confidence_reduces_score() -> None:
    # Confidence = α / (α + β). α=0.1, β=0.9 → 0.1
    s = _toy_store(broker_alpha=0.1, broker_beta=0.9)
    ids = ["b1", "b2", "b3", "b4", "b5"]
    raw = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    out = apply_broker_attenuation(raw, s, ids)
    # Each score multiplied by 0.1 — that's a 90% reduction, well past
    # the 5% threshold the spec requires.
    assert np.all(out <= raw * 0.95)
    np.testing.assert_allclose(out, raw * 0.1, atol=1e-12)


def test_broker_attenuation_high_confidence_near_zero_reduction() -> None:
    # α=99, β=1 → broker = 0.99 → ~1% reduction (close to "0 reduction")
    s = _toy_store(broker_alpha=99.0, broker_beta=1.0)
    ids = ["b1", "b2", "b3", "b4", "b5"]
    raw = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    out = apply_broker_attenuation(raw, s, ids)
    np.testing.assert_allclose(out, raw * 0.99, atol=1e-12)
    # Reduction must be < 5% — distinguishes the "high confidence" arm.
    assert np.all(out >= raw * 0.95)


def test_broker_attenuation_missing_belief_passthrough() -> None:
    s = _toy_store()
    raw = np.array([1.0, 1.0])
    out = apply_broker_attenuation(raw, s, ["does_not_exist", "also_missing"])
    np.testing.assert_array_equal(out, raw)


# --- AC5 (perf-gated) ---------------------------------------------------


def test_heat_kernel_latency_at_n_50k_under_10ms(
    request: pytest.FixtureRequest,
) -> None:
    if not _has_run_perf(request):
        pytest.skip("perf-gated: pass --run-perf to run")
    n = 50_000
    k = 200
    rng = np.random.default_rng(0)
    eigvals = np.sort(rng.uniform(0.0, 2.0, size=k))
    eigvecs = rng.standard_normal((n, k)).astype(np.float64)
    seeds = np.zeros(n, dtype=np.float64)
    seeds[:25] = 1.0 / 25.0

    # Warm up BLAS / caches
    _ = heat_kernel_score(eigvals, eigvecs, seeds, t=8.0)
    t0 = time.perf_counter()
    _ = heat_kernel_score(eigvals, eigvecs, seeds, t=8.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 10.0, f"heat_kernel_score took {elapsed_ms:.2f} ms"


# --- AC7 -----------------------------------------------------------------


def test_use_heat_kernel_default_off() -> None:
    from aelfrice.retrieval import is_heat_kernel_enabled

    # No env, no kwarg, no toml → default False
    assert is_heat_kernel_enabled() is False


# --- Bandwidth sanity ----------------------------------------------------


def test_heat_kernel_bandwidth_smoothing() -> None:
    # Larger t should spread mass more globally; smaller t localizes.
    s = _toy_store()
    W, ids = build_signed_adjacency(s)
    L = build_signed_normalized_laplacian(W)
    eigvals, eigvecs = compute_eigenbasis(L, k=200)

    seeds = np.zeros(len(ids), dtype=np.float64)
    seeds[0] = 1.0  # all mass on b1

    local = heat_kernel_score(eigvals, eigvecs, seeds, t=0.5)
    smooth = heat_kernel_score(eigvals, eigvecs, seeds, t=20.0)

    # b1 (the seed) keeps a higher fraction of its mass under
    # localized t than under smoothed t.
    assert local[0] > smooth[0]
