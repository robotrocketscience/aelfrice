# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false
"""Acceptance tests for the Plate FFT HRR primitives (#216).

One test per acceptance criterion in the issue body. All
deterministic at fixed seed; per-test wall-clock under 200ms.
"""
from __future__ import annotations

import numpy as np
import pytest

from aelfrice.hrr import (
    DEFAULT_DIM,
    CleanupMemory,
    bind,
    cosine_similarity,
    random_vector,
    superpose,
    unbind,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- AC1: bind/unbind round-trip --------------------------------------------


def test_unbind_inverts_bind_high_similarity() -> None:
    """Plate 1995 with random unit keys: ``unbind(k, bind(k, v))``
    is ``v`` filtered per-frequency by ``|fft(k)|²``. With random
    Gaussian keys (non-unitary) the empirical cosine similarity sits
    at ~0.71 (range ~0.69–0.72 across seeds 0–9 at dim=2048), not
    the ~1.0 you'd get with unitary keys. Cleanup memory recovers
    the true filler from the degraded vector — that is the standard
    HRR recovery pipeline."""
    rng = _rng()
    k = random_vector(DEFAULT_DIM, rng)
    v = random_vector(DEFAULT_DIM, rng)
    recovered = unbind(k, bind(k, v))
    sim = cosine_similarity(recovered, v)
    assert sim >= 0.65


# --- AC2: random_vector determinism -----------------------------------------


def test_random_vector_deterministic_at_fixed_seed() -> None:
    a = random_vector(DEFAULT_DIM, _rng(0))
    b = random_vector(DEFAULT_DIM, _rng(0))
    np.testing.assert_array_equal(a, b)
    # Different seeds produce different vectors.
    c = random_vector(DEFAULT_DIM, _rng(1))
    assert not np.allclose(a, c)
    # Unit norm.
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-12)


# --- AC3: bind commutativity ------------------------------------------------


def test_bind_is_commutative() -> None:
    rng = _rng()
    a = random_vector(DEFAULT_DIM, rng)
    b = random_vector(DEFAULT_DIM, rng)
    np.testing.assert_allclose(bind(a, b), bind(b, a), atol=1e-12)


# --- AC4: noise-free single-pair recovery -----------------------------------


def test_unbind_single_pair_recovery_above_threshold() -> None:
    """Single-pair recovery cosine well above the dim=2048 noise floor."""
    rng = _rng()
    k = random_vector(DEFAULT_DIM, rng)
    v = random_vector(DEFAULT_DIM, rng)
    composite = bind(k, v)
    recovered = unbind(k, composite)
    sim = cosine_similarity(recovered, v)
    assert sim >= 0.65


# --- AC5: superposed-pair recovery ------------------------------------------


def test_superposed_recovery_above_capacity_threshold() -> None:
    """Two bound pairs in superposition: unbinding either key
    should recover the corresponding filler at cosine similarity
    well above the noise floor (1/sqrt(dim) ~ 0.022 at dim=2048)."""
    rng = _rng()
    k1, v1 = random_vector(DEFAULT_DIM, rng), random_vector(DEFAULT_DIM, rng)
    k2, v2 = random_vector(DEFAULT_DIM, rng), random_vector(DEFAULT_DIM, rng)
    s = superpose([bind(k1, v1), bind(k2, v2)])
    sim_v1 = cosine_similarity(unbind(k1, s), v1)
    sim_v2 = cosine_similarity(unbind(k2, s), v2)
    assert sim_v1 >= 0.5
    assert sim_v2 >= 0.5
    # Cross-talk should be close to noise floor (well below the
    # right-key recovery).
    sim_cross = cosine_similarity(unbind(k1, s), v2)
    assert abs(sim_cross) < sim_v1 / 2


# --- AC6: cleanup-memory ordering -------------------------------------------


def test_cleanup_memory_returns_descending_similarity() -> None:
    rng = _rng()
    cm = CleanupMemory()
    targets = [random_vector(DEFAULT_DIM, rng) for _ in range(5)]
    for i, t in enumerate(targets):
        cm.add(f"v{i}", t)
    # Probe = exact match of v2 → v2 should be #1.
    out = cm.query(targets[2], top_k=5)
    assert out[0][0] == "v2"
    assert out[0][1] == pytest.approx(1.0, abs=1e-9)
    # Descending order across the full result.
    sims = [s for _, s in out]
    assert sims == sorted(sims, reverse=True)


# --- AC7: cleanup-memory exact-match recall ---------------------------------


def test_cleanup_memory_exact_recall_of_stored_vector() -> None:
    rng = _rng()
    cm = CleanupMemory()
    v = random_vector(DEFAULT_DIM, rng)
    cm.add("only", v)
    out = cm.query(v, top_k=1)
    assert out == [("only", pytest.approx(1.0, abs=1e-9))]


# --- AC8: determinism + corner cases ----------------------------------------


def test_superpose_empty_list_returns_zero_vector() -> None:
    out = superpose([])
    assert out.shape == (0,)


def test_cosine_similarity_zero_norm_returns_zero() -> None:
    rng = _rng()
    v = random_vector(DEFAULT_DIM, rng)
    z = np.zeros(DEFAULT_DIM, dtype=np.float64)
    assert cosine_similarity(z, v) == 0.0
    assert cosine_similarity(v, z) == 0.0
    assert cosine_similarity(z, z) == 0.0


def test_cleanup_memory_empty_returns_empty() -> None:
    cm = CleanupMemory()
    rng = _rng()
    probe = random_vector(DEFAULT_DIM, rng)
    assert cm.query(probe, top_k=10) == []
    assert cm.size() == 0


def test_cleanup_memory_zero_probe_returns_empty() -> None:
    cm = CleanupMemory()
    rng = _rng()
    cm.add("a", random_vector(DEFAULT_DIM, rng))
    z = np.zeros(DEFAULT_DIM, dtype=np.float64)
    assert cm.query(z, top_k=1) == []
