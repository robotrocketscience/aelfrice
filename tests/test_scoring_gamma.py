"""Tests for #796 γ rerank — gamma_posterior_score.

Properties under test:

1. **T=1.0 byte-identity.** ``gamma_posterior_score(bm, α, β, 1.0)`` is
   bit-for-bit equal to ``partial_bayesian_score(bm, α, β, 1.0)`` —
   the operator-mandated reference. The bench panel uses this to
   anchor the γ surface against a known log-additive baseline.
2. **Reciprocal-temperature equivalence.** For any T > 0,
   ``gamma_posterior_score(bm, α, β, T)`` equals
   ``partial_bayesian_score(bm, α, β, 1/T)``. This is the load-bearing
   contract — γ is reparametrised log-additive, not a new function
   family.
3. **Floor clamp.** Non-positive temperatures clamp upward to
   ``GAMMA_TEMPERATURE_FLOOR`` rather than raising. Operationally this
   means a misconfigured meta-belief or env override degrades to a
   very-sharp posterior weighting instead of crashing retrieval.
4. **Determinism.** Same inputs → same float bits across calls.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.scoring import (
    GAMMA_TEMPERATURE_FLOOR,
    gamma_posterior_score,
    partial_bayesian_score,
)


@pytest.mark.parametrize(
    "bm25_raw,alpha,beta",
    [
        (-1.5, 4.0, 2.0),
        (-0.001, 0.5, 0.5),  # Jeffreys prior, weak BM25 hit
        (-10.0, 100.0, 1.0),  # strong evidence, strong match
        (0.0, 1.0, 1.0),  # no-match (BM25 = 0); floor protects log
    ],
)
def test_t_one_byte_identical_to_partial_bayesian_pw_one(
    bm25_raw: float, alpha: float, beta: float,
) -> None:
    """At T=1.0, γ collapses to partial_bayesian(posterior_weight=1.0)."""
    g = gamma_posterior_score(bm25_raw, alpha, beta, 1.0)
    p = partial_bayesian_score(bm25_raw, alpha, beta, 1.0)
    assert g == p, f"γ({bm25_raw},{alpha},{beta},T=1)={g!r} ≠ pb={p!r}"


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 1.5, 2.0, 5.0])
def test_reciprocal_temperature_equivalence(temperature: float) -> None:
    """γ(bm, α, β, T) == partial_bayesian(bm, α, β, 1/T) for any T > 0."""
    bm25_raw, alpha, beta = -1.5, 3.0, 2.0
    g = gamma_posterior_score(bm25_raw, alpha, beta, temperature)
    p = partial_bayesian_score(bm25_raw, alpha, beta, 1.0 / temperature)
    assert g == p, (
        f"T={temperature}: γ={g!r}, pb(pw={1.0 / temperature})={p!r}"
    )


def test_non_positive_temperature_clamps_to_floor() -> None:
    """T <= 0 clamps to GAMMA_TEMPERATURE_FLOOR; never raises."""
    bm25_raw, alpha, beta = -1.5, 3.0, 2.0
    expected = partial_bayesian_score(
        bm25_raw, alpha, beta, 1.0 / GAMMA_TEMPERATURE_FLOOR,
    )
    for bad_t in [0.0, -0.5, -1e9]:
        got = gamma_posterior_score(bm25_raw, alpha, beta, bad_t)
        assert got == expected, f"T={bad_t}: got {got}, expected {expected}"
        assert math.isfinite(got)


def test_higher_temperature_flattens_posterior_contribution() -> None:
    """Hot beliefs ranked above cold ones; effect shrinks with T.

    With identical BM25 and α_hot=10, β_hot=1 vs α_cold=1, β_cold=10,
    the score gap between the two should monotonically shrink as T
    grows — high T flattens the posterior log term.
    """
    bm = -1.5
    gaps = []
    for t in [0.5, 1.0, 2.0, 5.0]:
        hot = gamma_posterior_score(bm, 10.0, 1.0, t)
        cold = gamma_posterior_score(bm, 1.0, 10.0, t)
        gaps.append(hot - cold)
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)), gaps
    # Both arms still favour the hot belief at every T.
    for g in gaps:
        assert g > 0


def test_deterministic_across_calls() -> None:
    """Same inputs → bit-identical output across calls."""
    for _ in range(3):
        assert gamma_posterior_score(-1.5, 4.0, 2.0, 0.7) == (
            gamma_posterior_score(-1.5, 4.0, 2.0, 0.7)
        )
