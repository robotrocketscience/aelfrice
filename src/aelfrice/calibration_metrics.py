"""Relevance-calibration metrics for the close-the-loop harness (#365 R1).

Pure-stdlib leaf module. Three functions:

- ``precision_at_k`` — precision at rank K (the gate metric per #365 Q3).
- ``roc_auc`` — ROC-AUC via Mann-Whitney U with average-rank tie handling.
- ``spearman_rho`` — Spearman rank correlation with average-rank ties.

All three are deterministic: same input list → bytes-identical float
return value across reruns. Used by:

- ``scripts/audit_rebuild_log.py --calibrate-corpus`` (R1, this PR).
- ``aelf eval`` subcommand (R4, future).
- The CI synthetic-corpus aggregate workflow (R5, future).

Imports nothing from the rest of ``aelfrice`` — keep it that way so the
metrics can be lifted into other tooling without dragging the package.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

__all__ = (
    "CalibrationReport",
    "precision_at_k",
    "roc_auc",
    "spearman_rho",
    "ordered_top_k_overlap",
    "rank_biased_overlap",
)


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregated calibration metrics for a single harness run.

    ``p_at_k`` averages per-query precision@k across queries; queries
    that returned fewer than ``k`` candidates contribute the count of
    relevant items in the truncated list divided by ``k`` (missing
    slots count as not-relevant — the standard P@K definition).

    ``roc_auc`` and ``spearman_rho`` are computed over the pooled
    (score, label) observations across all queries; both are ``None``
    when the metric is undefined (single-class labels for AUC; zero
    variance after ranking for ρ).
    """

    p_at_k: float
    k: int
    n_queries: int
    n_truncated_queries: int
    roc_auc: float | None
    spearman_rho: float | None
    n_observations: int


def precision_at_k(relevance_top_k: Sequence[bool], k: int) -> float:
    """Precision@K of a ranked list.

    ``relevance_top_k`` is the relevance label of each item in the
    top-K returned candidates, in rank order. Items beyond rank K are
    ignored. Missing slots (input shorter than K) count as
    not-relevant.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n_relevant = sum(1 for r in relevance_top_k[:k] if r)
    return n_relevant / k


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-indexed average-rank conversion. Ties get the mean of the
    spanning rank positions.
    """
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in range(i, j + 1):
            ranks[indexed[idx][0]] = avg
        i = j + 1
    return ranks


def roc_auc(
    scores: Sequence[float], labels: Sequence[bool],
) -> float | None:
    """ROC-AUC via Mann-Whitney U with average-rank tie handling.

    Higher score = more relevant. Returns ``None`` when the labels
    are single-class (AUC undefined).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    sum_ranks_pos = sum(r for r, label in zip(ranks, labels) if label)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman_rho(
    xs: Sequence[float], ys: Sequence[float],
) -> float | None:
    """Spearman rank correlation (Pearson on average-ranked vectors).

    Returns ``None`` when either ranked vector has zero variance, or
    when fewer than two observations are supplied.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    n = len(xs)
    if n < 2:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    sx = sum(rx)
    sy = sum(ry)
    sxx = sum(r * r for r in rx)
    syy = sum(r * r for r in ry)
    sxy = sum(a * b for a, b in zip(rx, ry))
    denom = sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def ordered_top_k_overlap(
    a: Sequence[object], b: Sequence[object], k: int,
) -> float:
    """Fraction of the top-k positions where ``a`` and ``b`` agree.

    Compares the first ``k`` elements of two ranked lists position-by-
    position; the score is the count of matching positions divided by
    ``k``. Returns 1.0 when both lists' prefixes are identical and 0.0
    when no top-k position matches.

    Items beyond rank ``k`` are ignored. Missing slots (an input shorter
    than ``k``) count as non-matches. Use this alongside
    `rank_biased_overlap` to discriminate top-of-list churn from
    middle-of-list reorderings — the #796 R4 finding was that PR@k and
    Spearman ρ alone cannot tell those apart.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    a_top = list(a[:k])
    b_top = list(b[:k])
    matches = sum(
        1
        for i in range(k)
        if i < len(a_top) and i < len(b_top) and a_top[i] == b_top[i]
    )
    return matches / k


def rank_biased_overlap(
    a: Sequence[object], b: Sequence[object], p: float = 0.9,
) -> float:
    """Rank-biased overlap (RBO) — top-weighted ranking similarity.

    Implements the extrapolated finite-list form (RBO_EXT) from Webber
    et al. (2010), "A Similarity Measure for Indefinite Rankings".
    Comparison runs up to depth ``D = min(len(a), len(b))`` and
    extrapolates a constant agreement rate beyond ``D``:

        RBO_EXT = (X_D / D) * p^D
                + (1 - p) * sum_{d=1}^{D} p^(d-1) * X_d / d

    where ``X_d = |A_d ∩ B_d|`` is the intersection size at depth d.
    This is the conventional "RBO score" used in IR practice and
    satisfies the unit-bound property: identical equal-length lists
    score 1.0, fully disjoint lists score 0.0. The non-extrapolated
    RBO_MIN underestimates identical lists by ``p^D`` and was
    rejected here because the tests in #796 R4 rely on identical →
    1.0 as a sanity gate.

    ``p`` controls top-weight: ``p → 0`` weights rank 1 only;
    ``p → 1`` weights the tail almost as much as the head. The #796
    R4 finding used ``p = 0.9`` to give the top-K meaningful weight
    without ignoring downstream rearrangements; that is the default.
    Lists of unequal length are compared on their common prefix
    length ``D``; tail items past ``D`` are ignored.

    Properties covered by the test suite:
      * Identical lists → 1.0.
      * Disjoint lists → 0.0.
      * Monotone in prefix agreement: extending a shared prefix never
        lowers the score.
      * Both empty → 1.0 (vacuous identity); one empty → 0.0.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    depth = min(la, lb)
    seen_a: set[object] = set()
    seen_b: set[object] = set()
    overlap_count = 0
    weighted_sum = 0.0
    final_agreement = 0.0
    for d in range(depth):
        x = a[d]
        if x in seen_b:
            overlap_count += 1
        seen_a.add(x)
        y = b[d]
        if y in seen_a:
            overlap_count += 1
        seen_b.add(y)
        agreement = overlap_count / (d + 1)
        weighted_sum += (p ** d) * agreement
        final_agreement = agreement
    # RBO_EXT extrapolation term: assume agreement stays at the depth-D
    # rate for ranks beyond D. For identical equal-length lists this
    # term equals p^D so the total converges to 1.0.
    return final_agreement * (p ** depth) + (1.0 - p) * weighted_sum
