"""Scoring primitives: Beta-Bernoulli posterior, type-specific decay, relevance.

Half-lives (in hours, converted to seconds below):
    factual     336   (14 days)
    requirement 720   (30 days)
    preference  2016  (12 weeks)
    correction  4032  (24 weeks)

Lock-floor: when a belief's lock_level is "user", decay() is a no-op
regardless of age (zero work, sharp step). Above the floor decay is
exponential toward the Jeffreys prior (0.5, 0.5).

v1.3.0 partial Bayesian-weighted ranking
-----------------------------------------

`partial_bayesian_score(bm25_raw, alpha, beta, posterior_weight)`
combines an FTS5 BM25 score (SQLite signs it non-positive: smaller
= better) with the existing Beta-Bernoulli posterior mean log-
additively, per `docs/design/bayesian_ranking.md` § Algorithm:

    score = log(max(-bm25_raw, EPS))
          + posterior_weight * log(posterior_mean(alpha, beta))

The first term flips SQLite's BM25 sign so `log()` is defined
(SQLite returns `0` for non-matches; an `EPS` floor keeps that
case finite without crashing). At `posterior_weight = 0.0` the
second term is zero and ranking collapses to `log(-bm25_raw)`,
which is monotone with `-bm25_raw` ascending — i.e., byte-
identical to the v1.0.x `ORDER BY bm25(beliefs_fts)` ordering. The
Jeffreys prior (0.5, 0.5) is preserved at the ranking layer; do
not introduce a Laplace `(α+1) / (α+β+2)` form here. See the spec
for the rejected-alternative analysis.
"""
from __future__ import annotations

import math
from typing import Final

from aelfrice.models import LOCK_USER, Belief

# Numerical floor for the BM25-side log term. SQLite FTS5 returns
# `0.0` for non-matches and very small magnitudes (~1e-6) for
# weak matches; the floor protects against `log(0)` while sitting
# well below any matched-document score on practical corpora.
PARTIAL_BAYESIAN_BM25_FLOOR: Final[float] = 1e-12

# v1.3.0 default weight on the posterior_mean log term. Picked
# from #151's synthetic-graph calibration (NDCG@10 ≈ 0.95 at
# λ=0.5; collapses to 0.91 at λ=1.0; minimal effect at λ=0.0).
DEFAULT_POSTERIOR_WEIGHT: Final[float] = 0.5

# #796 γ rerank — minimum temperature accepted by
# `gamma_posterior_score`. T must be strictly positive (division);
# values below this floor clamp upward so a misconfigured meta-belief
# or env override never raises at retrieval time.
GAMMA_TEMPERATURE_FLOOR: Final[float] = 1e-6

# #800 / #817 ζ rerank — pinned defaults from the R&D campaign
# verdict at `~/projects/aelfrice-lab/experiments/zeta-posterior/`
# (R0–R4, commits `2baac93` through `82ec453`). At
# (α=1.0, β=0.25, scale=14.5) ζ dominates γ on rank_biased_overlap
# for similar rank_changed_fraction (R2 head-to-head). Tunability
# via kwargs preserved for tests; env/TOML knobs for α / β / scale
# are deferred per #817 § "Out of scope".
ZETA_ALPHA_DEFAULT: Final[float] = 1.0
ZETA_BETA_DEFAULT: Final[float] = 0.25
ZETA_SCALE_DEFAULT: Final[float] = 14.5

# Floor on the `posterior_mean` argument to `zeta_posterior_score`.
# A corrupted store row reading `posterior_mean = 0.0` would otherwise
# raise `ValueError: math domain error` at the inner `log()` call;
# the floor clamps such inputs upward to `PARTIAL_BAYESIAN_BM25_FLOOR`
# so retrieval never crashes on degenerate posteriors.
ZETA_POSTERIOR_FLOOR: Final[float] = PARTIAL_BAYESIAN_BM25_FLOOR

# --- Half-lives in seconds ---
_HOUR: Final[float] = 3600.0
TYPE_HALF_LIFE_SECONDS: Final[dict[str, float]] = {
    "factual": 336.0 * _HOUR,         # 14 days
    "requirement": 720.0 * _HOUR,     # 30 days
    "preference": 2016.0 * _HOUR,     # 12 weeks
    "correction": 4032.0 * _HOUR,     # 24 weeks
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

    Lock-floor short-circuit: if lock_level == "user", returns (alpha, beta)
    unchanged regardless of age. Sharp step, not gradient.

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

    query_overlap_score is supplied by retrieval. Document-class multipliers
    and other layered weights are deferred to a later release.
    """
    return posterior_mean(belief.alpha, belief.beta) * query_overlap_score


def _digamma(x: float) -> float:
    """Digamma function ψ(x) via asymptotic series with recurrence shift.

    Recurrence ψ(x) = ψ(x+1) − 1/x shifts x ≥ 6, then applies:
        ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)
    Accuracy ~1e-10 for x ≥ 6 (adequate for entropy scoring).
    """
    # Shift via recurrence until x >= 6
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic series
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += (
        math.log(x)
        - 0.5 * inv_x
        - inv_x2 / 12.0
        + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0
    )
    return result


def uncertainty_score(alpha: float, beta: float) -> float:
    """Beta differential entropy H(α, β) as an uncertainty scalar.

    H(α, β) = ln B(α, β) − (α−1)·ψ(α) − (β−1)·ψ(β) + (α+β−2)·ψ(α+β)
    where ln B(α, β) = lgamma(α) + lgamma(β) − lgamma(α+β) and ψ is digamma.
    """
    ln_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        ln_beta
        - (alpha - 1.0) * _digamma(alpha)
        - (beta - 1.0) * _digamma(beta)
        + (alpha + beta - 2.0) * _digamma(alpha + beta)
    )


def partial_bayesian_score(
    bm25_raw: float,
    alpha: float,
    beta: float,
    posterior_weight: float = DEFAULT_POSTERIOR_WEIGHT,
) -> float:
    """v1.3 partial Bayesian-weighted retrieval score.

    `score = log(max(-bm25_raw, EPS)) + posterior_weight * log(posterior_mean)`

    `bm25_raw` is FTS5's signed score (non-positive: SQLite returns
    smaller-magnitude-negative for stronger matches). We negate to
    get a positive relevance magnitude before taking `log`. `EPS`
    (`PARTIAL_BAYESIAN_BM25_FLOOR`) prevents `log(0)` for non-
    matches without contaminating any real-match ordering.

    `posterior_weight = 0.0` collapses the second term to zero and
    makes the score a monotone function of `-bm25_raw` — byte-
    identical to v1.0.x `ORDER BY bm25(beliefs_fts)`.

    `posterior_mean` reuses the existing module-level helper, which
    returns `α / (α + β)` (Jeffreys prior, reads 0.5 for unobserved
    beliefs). Do not switch to Laplace at this layer — the prior
    must agree with `aelf stats`, the MCP, and `decay()`.

    Higher score = more relevant (matches the convention used by
    sort-descending callers).
    """
    relevance_pos = max(-bm25_raw, PARTIAL_BAYESIAN_BM25_FLOOR)
    log_bm25 = math.log(relevance_pos)
    if posterior_weight == 0.0:
        return log_bm25
    p = posterior_mean(alpha, beta)
    # `posterior_mean` returns 0.5 in the degenerate (alpha+beta<=0)
    # case, so `p > 0` is guaranteed; floor defensively for the
    # pathological `alpha = 0` operator-fed case to avoid `log(0)`.
    p_safe = p if p > 0.0 else PARTIAL_BAYESIAN_BM25_FLOOR
    return log_bm25 + posterior_weight * math.log(p_safe)


def gamma_posterior_score(
    bm25_raw: float,
    alpha: float,
    beta: float,
    temperature: float,
) -> float:
    """#796 γ rerank — Boltzmann temperature on the posterior log term.

    `score = log(max(-bm25_raw, EPS)) + (1 / T) * log(posterior_mean)`

    At `T = 1.0` this collapses to `partial_bayesian_score` with
    `posterior_weight = 1.0` (byte-identical). Lower `T` sharpens the
    posterior contribution (high-posterior beliefs pull harder);
    higher `T` flattens it toward BM25-only ranking.

    Temperatures at or below `GAMMA_TEMPERATURE_FLOOR` clamp upward —
    a misconfigured meta-belief value never raises at retrieval time.
    Negative temperatures are likewise clamped: the Boltzmann reading
    is undefined for `T <= 0` and the safest fall-back is the floor.

    γ is the load-bearing precursor to #758's adaptive
    `meta:retrieval.posterior_temperature`. Until that meta-belief is
    populated and learning, callers pin `T = 1.0` and the bench panel
    measures the γ vs log-additive surface (#796 R&D campaign verdict).
    """
    t_safe = temperature if temperature > GAMMA_TEMPERATURE_FLOOR else (
        GAMMA_TEMPERATURE_FLOOR
    )
    return partial_bayesian_score(
        bm25_raw, alpha, beta, posterior_weight=(1.0 / t_safe),
    )


def zeta_posterior_score(
    bm25_raw: float,
    alpha: float,
    beta: float,
    scale: float,
    posterior_mean: float,
) -> float:
    """#817 / #800 ζ rerank — bounded sigmoid posterior contribution.

    `score = log(max(-bm25_raw, EPS))
           + alpha * (sigmoid(beta * (log(p) - log(0.5))) - 0.5) * scale`

    Where `p = posterior_mean` (already computed by the caller, typically
    via `posterior_mean(α_beta, β_beta)`). `alpha`, `beta`, `scale` are
    the ζ shape parameters — not the Beta-Bernoulli `(α, β)` pair.

    Bounded contribution: the posterior term lies in
    `[-α·scale/2, +α·scale/2]`. No belief gets unboundedly leveraged
    by extreme posteriors — the structural difference from γ, where
    `(1/T)·log(p)` is unbounded as T → 0.

    Posterior-neutral point: at `p = 0.5`, `log(p) − log(0.5) = 0`, the
    sigmoid is 0.5, the bracket is 0, and the entire posterior term
    collapses. A store of all-uniform posteriors is rank-equivalent to
    log-BM25 alone — the same edge-case property `partial_bayesian_score`
    has at `posterior_weight = 0.0`.

    Floor clamp: `posterior_mean <= ZETA_POSTERIOR_FLOOR` clamps upward
    so a corrupted store row never raises `math domain error` at
    retrieval time.

    ζ is **not** byte-identical to `partial_bayesian_score` or
    `gamma_posterior_score` at any (α, β, scale) — the additive term
    is bounded around `log(0.5)`, not a free log of the posterior. ζ
    is the new long-term shape, not a continuous extension of γ. See
    `docs/feature-zeta-posterior-rerank.md` § "Contract" for the
    full derivation.
    """
    relevance_pos = max(-bm25_raw, PARTIAL_BAYESIAN_BM25_FLOOR)
    log_bm25 = math.log(relevance_pos)
    p_safe = posterior_mean if posterior_mean > ZETA_POSTERIOR_FLOOR else (
        ZETA_POSTERIOR_FLOOR
    )
    # `sigmoid(x) = 1 / (1 + exp(-x))`. Centred so p=0.5 → contribution=0.
    x = beta * (math.log(p_safe) - math.log(0.5))
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    return log_bm25 + alpha * (sigmoid - 0.5) * scale
