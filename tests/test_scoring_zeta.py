"""Tests for #817 / #800 ζ rerank — zeta_posterior_score.

Properties under test:

1. **Posterior-neutral point.** ``zeta_posterior_score(bm, α, β, scale, 0.5)``
   equals ``log(max(-bm, EPS))`` exactly. The bounded sigmoid contribution
   collapses to zero when the posterior is uninformative — the same
   edge-case property ``partial_bayesian_score`` has at
   ``posterior_weight = 0.0``.
2. **σ-bound.** For any (α, β, scale) and any posterior_mean ∈ (0, 1),
   the posterior contribution lies strictly in
   ``(-α·scale/2, +α·scale/2)``. No belief is unboundedly leveraged
   by extreme posteriors — the structural difference from γ, where
   ``(1/T)·log(p)`` is unbounded as T → 0.
3. **Monotonicity in posterior_mean.** With (α, β, scale) fixed and
   bm25_raw fixed, the score is strictly increasing in posterior_mean
   on (0, 1).
4. **Floor clamp.** ``posterior_mean <= ZETA_POSTERIOR_FLOOR`` clamps
   upward rather than raising ``math domain error``. Negative posteriors
   (corrupted store row) are likewise clamped — they decode to the
   minimum contribution rather than NaN.
5. **Determinism.** Same inputs → bit-identical output across calls.
6. **Not byte-identical to γ or partial_bayesian.** ζ at the pinned
   defaults (α=1, β=0.25, scale=14.5) is a different function family
   from γ; the contracts of #817 §"Note re: cold-start byte-identity"
   say so explicitly. We assert *inequality* at non-trivial inputs to
   catch accidental coincidence-bugs (a future ζ refactor that
   collapses to log-additive shape would silently break the bench).
"""
from __future__ import annotations

import math

import pytest

from aelfrice.scoring import (
    PARTIAL_BAYESIAN_BM25_FLOOR,
    ZETA_ALPHA_DEFAULT,
    ZETA_BETA_DEFAULT,
    ZETA_POSTERIOR_FLOOR,
    ZETA_SCALE_DEFAULT,
    gamma_posterior_score,
    partial_bayesian_score,
    posterior_mean,
    zeta_posterior_score,
)


# ---------------------------------------------------------------------------
# Posterior-neutral point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bm25_raw,alpha,beta,scale",
    [
        (-1.5, 1.0, 0.25, 14.5),
        (-0.001, 0.5, 0.1, 5.0),
        (-10.0, 2.0, 1.0, 20.0),
        (0.0, 1.0, 0.25, 14.5),  # no-match: BM25 = 0; floor protects log
    ],
)
def test_posterior_half_is_log_bm25_only(
    bm25_raw: float, alpha: float, beta: float, scale: float,
) -> None:
    """At posterior_mean=0.5, the bracket (σ-0.5) is exactly 0 → ζ collapses
    to log(max(-bm25, EPS))."""
    relevance_pos = max(-bm25_raw, PARTIAL_BAYESIAN_BM25_FLOOR)
    expected = math.log(relevance_pos)
    got = zeta_posterior_score(bm25_raw, alpha, beta, scale, 0.5)
    assert got == expected, (got, expected)


# ---------------------------------------------------------------------------
# σ-bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("posterior", [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
def test_sigma_bound(posterior: float) -> None:
    """Contribution lies strictly in (-α·scale/2, +α·scale/2)."""
    bm25_raw = -1.5
    alpha, beta, scale = ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT
    half_range = alpha * scale / 2.0
    log_bm = math.log(-bm25_raw)
    s = zeta_posterior_score(bm25_raw, alpha, beta, scale, posterior)
    contribution = s - log_bm
    assert -half_range < contribution < +half_range, (
        f"posterior={posterior}: contribution={contribution} "
        f"outside (±{half_range})"
    )


def test_sigma_bound_at_extreme_posteriors() -> None:
    """As p → 1 or p → 0+, contribution approaches ±α·scale/2 but
    does not exceed it (open interval). Use very-extreme posteriors
    to stress the sigmoid saturation."""
    bm25_raw = -1.5
    alpha, beta, scale = ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT
    half_range = alpha * scale / 2.0
    log_bm = math.log(-bm25_raw)
    s_high = zeta_posterior_score(bm25_raw, alpha, beta, scale, 1.0 - 1e-15)
    s_low = zeta_posterior_score(bm25_raw, alpha, beta, scale, 1e-15)
    assert s_high - log_bm < half_range
    assert s_low - log_bm > -half_range
    # And the sigmoid does saturate close to the bound at β=0.25 even
    # with extreme p — verifies the parameter space we ship.
    assert s_high - log_bm > 0
    assert s_low - log_bm < 0


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_monotone_in_posterior() -> None:
    """Score is strictly increasing in posterior_mean on (0, 1)."""
    bm25_raw = -1.5
    alpha, beta, scale = ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT
    posteriors = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]
    scores = [
        zeta_posterior_score(bm25_raw, alpha, beta, scale, p) for p in posteriors
    ]
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1], (
            f"non-monotone at p={posteriors[i]} → {posteriors[i + 1]}: "
            f"{scores[i]} not < {scores[i + 1]}"
        )


# ---------------------------------------------------------------------------
# Floor clamp on degenerate posterior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_p", [0.0, -0.5, -1e9, ZETA_POSTERIOR_FLOOR])
def test_floor_clamp_finite(bad_p: float) -> None:
    """posterior_mean <= floor never raises; returns a finite score."""
    s = zeta_posterior_score(
        -1.5, ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT, bad_p,
    )
    assert math.isfinite(s)


def test_floor_clamp_matches_floor_value() -> None:
    """A posterior at the floor and a more-negative posterior produce
    the same score (both clamp to the floor)."""
    args = (-1.5, ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT)
    s_at_floor = zeta_posterior_score(*args, ZETA_POSTERIOR_FLOOR)
    s_below = zeta_posterior_score(*args, 0.0)
    s_negative = zeta_posterior_score(*args, -1.0)
    assert s_at_floor == s_below == s_negative


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_across_calls() -> None:
    """Same inputs → bit-identical output across calls."""
    args = (-1.5, 1.0, 0.25, 14.5, 0.7)
    for _ in range(3):
        assert zeta_posterior_score(*args) == zeta_posterior_score(*args)


# ---------------------------------------------------------------------------
# Not-byte-identical to γ / partial_bayesian (issue §"Note re cold-start")
# ---------------------------------------------------------------------------

def test_zeta_not_byte_identical_to_gamma_at_T_one() -> None:
    """ζ at the pinned defaults is not byte-identical to γ@T=1.0 (which
    is itself byte-identical to partial_bayesian_score(..., 1.0)). This
    asserts the structural difference noted in issue #817's 'Note re:
    cold-start byte-identity'."""
    bm25_raw, alpha_beta_a, alpha_beta_b = -1.5, 3.0, 2.0
    p = posterior_mean(alpha_beta_a, alpha_beta_b)  # = 0.6
    z = zeta_posterior_score(
        bm25_raw,
        ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT,
        p,
    )
    g = gamma_posterior_score(bm25_raw, alpha_beta_a, alpha_beta_b, 1.0)
    pb = partial_bayesian_score(bm25_raw, alpha_beta_a, alpha_beta_b, 1.0)
    assert g == pb  # already shipped contract; sanity
    assert z != g  # the load-bearing inequality
    assert z != pb


def test_zeta_collapses_to_log_bm_at_uniform_posteriors() -> None:
    """All-uniform-posterior=0.5 store: ζ ranks identically to log-BM25
    alone. The contribution is identically zero for every belief, so
    the entire posterior layer is inert. This is the same shape
    `partial_bayesian_score(..., 0.0)` has."""
    bm25_values = [-0.5, -1.0, -1.5, -2.0, -3.0]
    for bm in bm25_values:
        z = zeta_posterior_score(
            bm, ZETA_ALPHA_DEFAULT, ZETA_BETA_DEFAULT, ZETA_SCALE_DEFAULT, 0.5,
        )
        pb = partial_bayesian_score(bm, 1.0, 1.0, posterior_weight=0.0)
        # partial_bayesian at pw=0 short-circuits to log(-bm); ζ at p=0.5
        # also collapses there. Both should produce log(max(-bm, EPS)).
        assert z == pb
