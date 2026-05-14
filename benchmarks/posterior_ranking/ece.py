"""Expected Calibration Error (ECE) scorer for the posterior-ranking eval harness.

Implements the ECE contract from docs/design/v2_posterior_ranking_residual.md § Slice 1.

For each (query, retrieved_belief, rank) triple in the eval set, treat
posterior_mean(b) as the predicted probability that the user will rate b
positive.

Bucketing: 10 equal-width buckets [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
Per bucket: (mean_predicted, mean_actual) where mean_actual is the empirical
positive-feedback rate from the synthetic feedback stream replayed in the eval.

ECE = sum_b (|bucket_b| / N) * |mean_predicted_b - mean_actual_b|

Pass criterion: ECE <= 0.10
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from aelfrice.scoring import posterior_mean as _posterior_mean

N_BUCKETS: int = 10
DEFAULT_ECE_THRESHOLD: float = 0.10


@dataclass
class BucketStat:
    """Statistics for one equal-width probability bucket."""

    bucket_idx: int
    bucket_lo: float
    bucket_hi: float
    count: int
    mean_predicted: float
    mean_actual: float
    weight: float  # count / N_total


@dataclass
class ECEResult:
    """ECE calibration result."""

    ece: float
    buckets: list[BucketStat]
    n_total: int
    pass_threshold: float
    passed: bool


def _bucket_index(predicted: float) -> int:
    """Map a predicted probability in [0, 1] to bucket index in [0, N_BUCKETS-1]."""
    idx = int(predicted * N_BUCKETS)
    # Clamp 1.0 exactly into the last bucket.
    return min(idx, N_BUCKETS - 1)


def compute_ece(
    triples: list[tuple[float, float, float]],
    threshold: float = DEFAULT_ECE_THRESHOLD,
) -> ECEResult:
    """Compute ECE from a list of (alpha, beta, actual_positive_rate) triples.

    Each triple represents one (query, retrieved_belief, rank) observation:
      - alpha, beta: the belief's current posterior parameters
      - actual_positive_rate: 1.0 if this belief received positive feedback
        in the synthetic stream, 0.0 otherwise

    Returns an ECEResult with per-bucket statistics and overall ECE.
    """
    n_total = len(triples)
    if n_total == 0:
        # Degenerate: no observations; ECE is 0, pass trivially.
        buckets = [
            BucketStat(
                bucket_idx=i,
                bucket_lo=i / N_BUCKETS,
                bucket_hi=(i + 1) / N_BUCKETS,
                count=0,
                mean_predicted=0.0,
                mean_actual=0.0,
                weight=0.0,
            )
            for i in range(N_BUCKETS)
        ]
        return ECEResult(ece=0.0, buckets=buckets, n_total=0, pass_threshold=threshold, passed=True)

    # Accumulate per bucket.
    bucket_predicted: list[list[float]] = [[] for _ in range(N_BUCKETS)]
    bucket_actual: list[list[float]] = [[] for _ in range(N_BUCKETS)]

    for alpha, beta, actual in triples:
        pred = _posterior_mean(alpha, beta)
        idx = _bucket_index(pred)
        bucket_predicted[idx].append(pred)
        bucket_actual[idx].append(actual)

    buckets: list[BucketStat] = []
    ece = 0.0

    for i in range(N_BUCKETS):
        preds = bucket_predicted[i]
        acts = bucket_actual[i]
        count = len(preds)
        weight = count / n_total

        if count > 0:
            mean_pred = sum(preds) / count
            mean_act = sum(acts) / count
        else:
            mean_pred = (i + 0.5) / N_BUCKETS  # midpoint for empty bucket
            mean_act = 0.0

        ece += weight * abs(mean_pred - mean_act)

        buckets.append(BucketStat(
            bucket_idx=i,
            bucket_lo=i / N_BUCKETS,
            bucket_hi=(i + 1) / N_BUCKETS,
            count=count,
            mean_predicted=mean_pred,
            mean_actual=mean_act,
            weight=weight,
        ))

    passed = ece <= threshold
    return ECEResult(ece=ece, buckets=buckets, n_total=n_total, pass_threshold=threshold, passed=passed)


def compute_ece_from_stores(
    fixture_observations: list[dict[str, object]],
    threshold: float = DEFAULT_ECE_THRESHOLD,
) -> ECEResult:
    """Compute ECE from a list of observation dicts.

    Each dict must have keys:
      "alpha": float
      "beta": float
      "received_positive": bool  (True if this observation received positive feedback)

    This is the interface used by run.py after replaying the synthetic feedback stream.
    """
    triples: list[tuple[float, float, float]] = [
        (
            float(obs["alpha"]),
            float(obs["beta"]),
            1.0 if obs["received_positive"] else 0.0,
        )
        for obs in fixture_observations
    ]
    return compute_ece(triples, threshold=threshold)
