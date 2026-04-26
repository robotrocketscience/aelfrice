"""Scoring primitives: Beta-Bernoulli posterior, type-specific decay, relevance.

Per [redacted] "In MVP (v1.0)" -> scoring.py:
- Beta-Bernoulli posterior mean (alpha / (alpha + beta))
- Type-specific decay using the 4 belief-type half-lives
- Basic relevance (no document-class multipliers; deferred to v1.5)

Half-life values are sourced from the v2.0 legacy scoring module:
    git ref: [redacted]:src/aelfrice/scoring.py:16-19
        factual:     336 hours  (14 days)
        preference:  2016 hours (12 weeks)
        correction:  4032 hours (24 weeks)
        requirement: 4032 hours (24 weeks)
Converted to seconds for the v0.2.0 API (time_seconds is the canonical unit
in this rewrite; CHANGELOG narrative talks weeks/days but the source-of-truth
constants live in legacy scoring.py).

Lock-floor ([redacted]): when a belief's lock_level is "user", decay() is a
no-op (zero work). Above the floor (lock_level "none"), decay is exponential
toward the Jeffreys prior (0.5, 0.5).
"""
from __future__ import annotations

from typing import Final

from aelfrice.models import LOCK_USER, Belief

# --- Half-lives in seconds, derived from legacy v2.0 scoring.py:16-19 ---
_HOUR: Final[float] = 3600.0
TYPE_HALF_LIFE_SECONDS: Final[dict[str, float]] = {
    "factual": 336.0 * _HOUR,         # 14 days
    "preference": 2016.0 * _HOUR,     # 12 weeks
    "correction": 4032.0 * _HOUR,     # 24 weeks
    "requirement": 4032.0 * _HOUR,    # 24 weeks
}

# Jeffreys prior -- decay target.
_PRIOR_ALPHA: Final[float] = 0.5
_PRIOR_BETA: Final[float] = 0.5


def posterior_mean(alpha: float, beta: float) -> float:
    """Beta-Bernoulli posterior mean: alpha / (alpha + beta).

    With the Jeffreys prior (0.5, 0.5), an unobserved belief reads 0.5.
    """
    total = alpha + beta
    if total <= 0.0:
        # Degenerate: fall back to prior. Should not occur in practice
        # since alpha,beta start at 0.5,0.5 and only grow.
        return 0.5
    return alpha / total


def type_half_life(belief_type: str) -> float:
    """Return the half-life (seconds) for the given belief type.

    Unknown types fall back to the factual half-life (most aggressive decay).
    """
    return TYPE_HALF_LIFE_SECONDS.get(belief_type, TYPE_HALF_LIFE_SECONDS["factual"])


def decay(
    alpha: float,
    beta: float,
    age_seconds: float,
    half_life_seconds: float,
    lock_level: str = "none",
) -> tuple[float, float]:
    """Exponentially decay (alpha, beta) toward the Jeffreys prior (0.5, 0.5).

    Lock-floor short-circuit ([redacted]): if lock_level == "user", returns
    (alpha, beta) unchanged regardless of age. Sharp step, not gradient.

    Otherwise both alpha and beta move toward the prior by the same factor
    f = 0.5 ** (age_seconds / half_life_seconds), so total evidence (alpha+beta)
    shrinks toward 1.0 (the prior mass) while the ratio alpha/(alpha+beta)
    is preserved when the deltas relative to prior are symmetric.

    Concretely: new_alpha = prior_alpha + (alpha - prior_alpha) * f
                new_beta  = prior_beta  + (beta  - prior_beta)  * f
    """
    if lock_level == LOCK_USER:
        return (alpha, beta)
    if half_life_seconds <= 0.0 or age_seconds <= 0.0:
        return (alpha, beta)
    factor = 0.5 ** (age_seconds / half_life_seconds)
    new_alpha = _PRIOR_ALPHA + (alpha - _PRIOR_ALPHA) * factor
    new_beta = _PRIOR_BETA + (beta - _PRIOR_BETA) * factor
    return (new_alpha, new_beta)


def relevance(belief: Belief, query_overlap_score: float) -> float:
    """Basic relevance: confidence * query overlap.

    query_overlap_score is supplied by retrieval (v0.3.0). Document-class
    multipliers and other layered weights are deferred to v1.5 per [redacted].
    """
    return posterior_mean(belief.alpha, belief.beta) * query_overlap_score
