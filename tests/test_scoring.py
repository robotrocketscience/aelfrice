"""Property and numeric-sanity tests for scoring.uncertainty_score.

Beta differential entropy H(α, β) — mathematical properties under test:
    1. Known numeric values (uniform, Jeffreys prior, symmetric peaked).
    2. Maximum over [1, 10] × [1, 10] at the uniform prior α = β = 1.
    3. Monotonic non-increase in α for fixed β ≥ 1, α ≥ 1 (more evidence
       → less uncertainty).
    4. Symmetry: H(α, β) == H(β, α).
    5. Finiteness for all tested inputs.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.scoring import uncertainty_score


# ---------------------------------------------------------------------------
# Numeric sanity
# ---------------------------------------------------------------------------

def test_uniform_prior_entropy_is_zero() -> None:
    """H(1, 1) — uniform on [0,1] — has differential entropy 0."""
    assert math.isclose(uncertainty_score(1.0, 1.0), 0.0, abs_tol=1e-9)


def test_jeffreys_prior_entropy() -> None:
    """H(0.5, 0.5) ≈ −0.2416 (Jeffreys prior, known value)."""
    assert math.isclose(uncertainty_score(0.5, 0.5), -0.2416, rel_tol=1e-3)


def test_symmetric_peaked_entropy() -> None:
    """H(2, 2) ≈ −0.1251 (symmetric unimodal, known value)."""
    assert math.isclose(uncertainty_score(2.0, 2.0), -0.1251, rel_tol=1e-3)


def test_all_tested_values_are_finite() -> None:
    """uncertainty_score is finite for a grid of representative inputs."""
    pairs = [
        (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (1.0, 5.0),
        (5.0, 1.0), (10.0, 10.0), (0.1, 0.1),
    ]
    for a, b in pairs:
        val = uncertainty_score(a, b)
        assert math.isfinite(val), f"Non-finite for alpha={a}, beta={b}: {val}"


# ---------------------------------------------------------------------------
# Maximum at α = β = 1 over [1, 10] × [1, 10]
# ---------------------------------------------------------------------------

def test_maximum_at_uniform_prior_over_unit_grid() -> None:
    """H(1,1) ≥ H(α,β) for all α,β in {1,2,...,10}."""
    h_uniform = uncertainty_score(1.0, 1.0)
    for a in range(1, 11):
        for b in range(1, 11):
            h = uncertainty_score(float(a), float(b))
            assert h <= h_uniform + 1e-9, (
                f"H({a},{b})={h:.6f} exceeds H(1,1)={h_uniform:.6f}"
            )


# ---------------------------------------------------------------------------
# Monotonicity: more evidence → less uncertainty
# ---------------------------------------------------------------------------

def test_monotonic_nonincreasing_in_alpha_with_beta_one() -> None:
    """H(α+1, 1) ≤ H(α, 1) for all α ≥ 1.

    Beta(α, 1) concentrates mass at 1 as α grows; entropy is strictly
    decreasing in α along this edge of the parameter space.
    """
    alphas = [float(k) for k in range(1, 20)]
    values = [uncertainty_score(a, 1.0) for a in alphas]
    for i in range(len(values) - 1):
        assert values[i + 1] <= values[i] + 1e-9, (
            f"Not non-increasing at alpha={alphas[i]}, beta=1.0: "
            f"H(α)={values[i]:.6f}, H(α+1)={values[i+1]:.6f}"
        )


def test_monotonic_nonincreasing_on_symmetric_diagonal() -> None:
    """H(k, k) is non-increasing for k = 1, 2, …, 15.

    Symmetric Beta concentrates near 0.5 as k grows; entropy decreases
    monotonically along the α = β diagonal.
    """
    ks = [float(k) for k in range(1, 16)]
    values = [uncertainty_score(k, k) for k in ks]
    for i in range(len(values) - 1):
        assert values[i + 1] <= values[i] + 1e-9, (
            f"Not non-increasing at k={ks[i]}: "
            f"H(k,k)={values[i]:.6f}, H(k+1,k+1)={values[i+1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,beta", [
    (1.0, 2.0), (3.0, 7.0), (0.5, 2.5), (10.0, 1.0),
])
def test_symmetry(alpha: float, beta: float) -> None:
    """H(α, β) == H(β, α) within float tolerance."""
    assert math.isclose(
        uncertainty_score(alpha, beta),
        uncertainty_score(beta, alpha),
        rel_tol=1e-12,
    ), f"Symmetry failed for ({alpha}, {beta})"
