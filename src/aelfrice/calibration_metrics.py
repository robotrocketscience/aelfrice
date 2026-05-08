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
