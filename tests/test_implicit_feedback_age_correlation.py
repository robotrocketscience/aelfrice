"""Regression guard: belief age vs alpha-drift correlation (#555, revised spec).

Verifies that the deferred-feedback sweeper does not silently become a clock
(i.e., older beliefs do not accumulate disproportionate alpha bumps purely
because they are older, independent of retrieval frequency).

Two correlation measures are checked because Pearson r only catches *linear*
relationships — a non-linear "becomes a clock" failure mode (U-shape, plateau,
threshold) would slip past it.  This test uses:

  * Chatterjee's xi coefficient
    Chatterjee, S. (2021). "A New Coefficient of Correlation."
    Journal of the American Statistical Association, 116(536), 2009–2022.
    https://doi.org/10.1080/01621459.2020.1758115

  * Distance correlation (dCor)
    Szekely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
    "Measuring and Testing Dependence by Correlation of Distances."
    The Annals of Statistics, 35(6), 2769–2794.
    https://doi.org/10.1214/009053607000000505

Synthetic workload — 1 week, N=200 beliefs
-------------------------------------------
- N=200 beliefs created uniformly across a 7-day window.
- Retrieval-event counts drawn from Poisson(lambda=2) per belief, capped at 10.
  Each retrieval is placed at a random time within the belief's lifetime.
  Retrieval frequency is decorrelated from age *by construction*: the Poisson
  draw is independent of creation order.
- ~15 % of beliefs also receive one explicit positive-feedback event inside
  the grace window; the sweeper cancels those rows (no alpha change).
- All retrieval exposures are enqueued with enqueued_at = retrieval_time.
  The sweeper runs at T_end + 2 * T_grace so every grace window has elapsed.
- epsilon=0.05, T_grace=1800 s (defaults).

Thresholds
----------
Calibrated 2026-05-10 via _calibrate_thresholds() at N=200, 30 seeds
(seeds 0-29); values are 99th-percentile + 0.05 margin.

  Raw 99th-percentile (seeds 0-29):
    xi   p99 = 0.1313
    dCor p99 = 0.1933

  T_XI  = 0.1813   (= 0.1313 + 0.05)
  T_DCOR = 0.2433  (= 0.1933 + 0.05)

Asserted run (seed 42):
  xi   = -0.0335
  dCor =  0.1012

RNG seed: 42.  Change only when adjusting workload shape; document why.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from aelfrice.deferred_feedback import (
    DEFAULT_EPSILON,
    DEFAULT_T_GRACE_SECONDS,
    enqueue_retrieval_exposures,
    sweep_deferred_feedback,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Workload constants
# ---------------------------------------------------------------------------

RNG_SEED: int = 42
N_BELIEFS: int = 200
WINDOW_DAYS: int = 7
EPSILON: float = DEFAULT_EPSILON
T_GRACE: int = DEFAULT_T_GRACE_SECONDS
POISSON_LAMBDA: float = 2.0
RETRIEVAL_CAP: int = 10
EXPLICIT_FEEDBACK_FRACTION: float = 0.15
ALPHA_INITIAL: float = 1.0

# ---------------------------------------------------------------------------
# Calibrated thresholds
# Calibrated 2026-05-10 via _calibrate_thresholds() at N=200, 30 seeds;
# values are 99th-percentile + 0.05.
#   Raw p99: xi = 0.1313, dCor = 0.1933
# ---------------------------------------------------------------------------

T_XI: float = 0.1813
T_DCOR: float = 0.2433

_EPOCH = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
_WINDOW_SECONDS: int = WINDOW_DAYS * 86_400


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Pure-Python Poisson sampler (Knuth method)
# ---------------------------------------------------------------------------


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Exact Poisson draw via Knuth's multiplicative method."""
    l_ = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Correlation statistics — implemented directly to avoid heavy native deps
# ---------------------------------------------------------------------------


def _chatterjee_xi(x: list[float], y: list[float]) -> float:
    """Chatterjee's xi coefficient (JASA 2021, Eq. 1.1).

    xi_n(X, Y) = 1 - 3 * sum_i |r_{i+1} - r_i| / (n^2 - 1)

    where r_i are the ranks of Y_i sorted by X_i (ties in X broken
    randomly; ties in Y handled by average-rank).  xi is in [-0.5, 1];
    near 0 means Y is approximately independent of X; near 1 means Y is
    a measurable function of X.

    Reference:
      Chatterjee, S. (2021). A New Coefficient of Correlation.
      JASA, 116(536), 2009-2022.
      https://doi.org/10.1080/01621459.2020.1758115
    """
    n = len(x)
    if n < 2:
        return 0.0
    xarr = np.array(x, dtype=float)
    yarr = np.array(y, dtype=float)
    # Sort by X (stable sort preserves insertion order for ties).
    order = np.argsort(xarr, kind="stable")
    y_sorted = yarr[order]
    # Rank Y in the X-sorted order (1-indexed; average rank for ties).
    # scipy is a runtime dep so we compute ranks manually.
    temp_order = np.argsort(y_sorted, kind="stable")
    y_rank = np.empty(n, dtype=float)
    y_rank[temp_order] = np.arange(1, n + 1, dtype=float)
    # Handle ties: replace each tied group with the group's mean rank.
    # argsort of argsort gives a rank vector; we need to detect ties.
    sort_idx = np.argsort(y_sorted, kind="stable")
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_sorted[sort_idx[i]] == y_sorted[sort_idx[j]]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        y_rank[sort_idx[i:j]] = mean_rank
        i = j
    diffs = np.abs(np.diff(y_rank))
    return float(1.0 - 3.0 * float(np.sum(diffs)) / (n * n - 1))


def _distance_correlation(x: list[float], y: list[float]) -> float:
    """Distance correlation (Szekely-Rizzo-Bakirov 2007, Annals of Statistics).

    dCor(X, Y) = sqrt(dCov(X,Y) / sqrt(dVar(X) * dVar(Y)))

    where dCov(X, Y) is the squared distance covariance, computed via
    double-centering of the pairwise-distance matrices.  dCor is in [0, 1];
    near 0 means statistical independence; 1 means X is a function of Y
    (or vice versa).  Unlike Pearson r, dCor catches *any* dependence
    structure (nonlinear, non-monotone, multi-modal).

    Reference:
      Szekely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
      Measuring and Testing Dependence by Correlation of Distances.
      Annals of Statistics, 35(6), 2769-2794.
      https://doi.org/10.1214/009053607000000505
    """
    xarr = np.array(x, dtype=float)
    yarr = np.array(y, dtype=float)

    def _dcov2(u: np.ndarray, v: np.ndarray) -> float:
        """Squared distance covariance between two 1-D arrays."""
        a = np.abs(u[:, None] - u[None, :])
        b = np.abs(v[:, None] - v[None, :])
        # Double-center each distance matrix.
        A = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
        B = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
        return float(np.mean(A * B))

    dcov_xy = _dcov2(xarr, yarr)
    dcov_xx = _dcov2(xarr, xarr)
    dcov_yy = _dcov2(yarr, yarr)
    if dcov_xx <= 0.0 or dcov_yy <= 0.0:
        return 0.0
    denom = math.sqrt(dcov_xx * dcov_yy)
    return float(math.sqrt(max(0.0, dcov_xy) / denom))


# ---------------------------------------------------------------------------
# Synthetic workload builder
# ---------------------------------------------------------------------------


def _build_workload(rng: random.Random) -> tuple[MemoryStore, list[str], datetime]:
    """Build store + enqueued events; return (store, belief_ids, sweep_time)."""
    store = MemoryStore(":memory:")
    belief_ids: list[str] = []

    creation_offsets_s = [
        int(i * _WINDOW_SECONDS / N_BELIEFS) for i in range(N_BELIEFS)
    ]

    for i, offset_s in enumerate(creation_offsets_s):
        bid = f"b{i:04d}"
        created_at_dt = _EPOCH + timedelta(seconds=offset_s)
        belief = Belief(
            id=bid,
            content=f"synthetic belief {i}",
            content_hash=f"hash_{bid}",
            alpha=ALPHA_INITIAL,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at=_fmt(created_at_dt),
            last_retrieved_at=None,
            session_id=None,
            origin="synthetic",
            corroboration_count=0,
        )
        store.insert_belief(belief)
        belief_ids.append(bid)

    t_end = _EPOCH + timedelta(seconds=_WINDOW_SECONDS)
    sweep_at = t_end + timedelta(seconds=2 * T_GRACE)

    # Retrieval counts drawn from Poisson(POISSON_LAMBDA), capped, and
    # scheduled at random times within each belief's lifetime (decorrelated
    # from age by construction: same lambda regardless of creation order).
    retrieval_counts = [
        min(RETRIEVAL_CAP, _poisson_sample(rng, POISSON_LAMBDA))
        for _ in range(N_BELIEFS)
    ]

    for i, bid in enumerate(belief_ids):
        creation_offset_s = creation_offsets_s[i]
        lifetime_s = _WINDOW_SECONDS - creation_offset_s
        for _ in range(retrieval_counts[i]):
            eligible_s = max(0, lifetime_s - T_GRACE)
            offset_in_lifetime = rng.randint(0, max(0, eligible_s))
            enqueued_dt = _EPOCH + timedelta(
                seconds=creation_offset_s + offset_in_lifetime
            )
            enqueue_retrieval_exposures(store, [bid], now=_fmt(enqueued_dt))

    # Explicit feedback on a fraction of beliefs cancels their implicit rows.
    n_explicit = int(N_BELIEFS * EXPLICIT_FEEDBACK_FRACTION)
    explicit_targets = rng.sample(belief_ids, n_explicit)
    for bid in explicit_targets:
        idx = belief_ids.index(bid)
        creation_offset_s = creation_offsets_s[idx]
        fb_offset_s = creation_offset_s + (_WINDOW_SECONDS - creation_offset_s) // 2
        fb_dt = _EPOCH + timedelta(seconds=fb_offset_s)
        store.insert_feedback_event(
            bid,
            valence=1.0,
            source="user",
            created_at=_fmt(fb_dt),
        )

    return store, belief_ids, sweep_at


# ---------------------------------------------------------------------------
# Age helper
# ---------------------------------------------------------------------------


def _age_days(created_at_iso: str, reference: datetime) -> float:
    created = datetime.strptime(created_at_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (reference - created).total_seconds() / 86_400.0


# ---------------------------------------------------------------------------
# Calibration helper (not a test — does not match def test_*)
# ---------------------------------------------------------------------------


def _calibrate_thresholds(
    n_seeds: int = 30,
    margin: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute empirical 99th-percentile thresholds across ``n_seeds`` seeds.

    Returns ``(p99_xi, p99_dcor, t_xi, t_dcor)`` where
    ``t_* = p99_* + margin``.  Invoke from a REPL or a one-off script;
    not called during the normal test run.
    """
    xi_vals: list[float] = []
    dcor_vals: list[float] = []

    for seed in range(n_seeds):
        rng = random.Random(seed)
        store, belief_ids, sweep_at = _build_workload(rng)
        sweep_deferred_feedback(
            store,
            now=_fmt(sweep_at),
            grace_seconds=T_GRACE,
            epsilon=EPSILON,
        )
        ages = [
            _age_days(store.get_belief(bid).created_at, sweep_at)  # type: ignore[union-attr]
            for bid in belief_ids
        ]
        drifts = [
            store.get_belief(bid).alpha - ALPHA_INITIAL  # type: ignore[union-attr]
            for bid in belief_ids
        ]
        xi_vals.append(_chatterjee_xi(ages, drifts))
        dcor_vals.append(_distance_correlation(ages, drifts))

    p99_xi = float(np.percentile(xi_vals, 99))
    p99_dcor = float(np.percentile(dcor_vals, 99))
    return p99_xi, p99_dcor, p99_xi + margin, p99_dcor + margin


# ---------------------------------------------------------------------------
# Asserted test
# ---------------------------------------------------------------------------


def test_age_alpha_correlation_below_threshold() -> None:
    """Both xi and dCor(age_days, alpha - alpha_initial) must be below threshold.

    The synthetic workload ensures retrieval frequency is decorrelated from
    belief age *by construction*: Poisson(lambda=2) retrieval counts are
    drawn independently of creation order, so any correlation between age
    and alpha-drift would indicate the sweeper has become a clock.

    Chatterjee's xi catches monotone *and* non-monotone dependence (U-shape,
    plateau, threshold), closing the gap that a Pearson-only guard leaves.
    Distance correlation is independent of functional form and detects *any*
    statistical dependence between the two variables.

    Both thresholds were calibrated via _calibrate_thresholds() at N=200
    across 30 seeds (0-29) as the 99th-percentile + 0.05 margin.
    """
    rng = random.Random(RNG_SEED)
    store, belief_ids, sweep_at = _build_workload(rng)

    sweep_deferred_feedback(
        store,
        now=_fmt(sweep_at),
        grace_seconds=T_GRACE,
        epsilon=EPSILON,
    )

    ages: list[float] = []
    drifts: list[float] = []

    for bid in belief_ids:
        belief = store.get_belief(bid)
        assert belief is not None, f"belief {bid} missing after sweep"
        ages.append(_age_days(belief.created_at, sweep_at))
        drifts.append(belief.alpha - ALPHA_INITIAL)

    xi = _chatterjee_xi(ages, drifts)
    dcor = _distance_correlation(ages, drifts)

    assert xi < T_XI, (
        f"Chatterjee xi(age_days, alpha_drift) = {xi:.4f} >= T_XI={T_XI:.4f}. "
        "A value above the threshold suggests the sweeper is accumulating "
        "alpha bumps in an age-correlated pattern (monotone or non-linear). "
        "Inspect epsilon / T_grace / workload shape before re-tuning T_XI."
    )

    assert dcor < T_DCOR, (
        f"Distance correlation(age_days, alpha_drift) = {dcor:.4f} >= T_DCOR={T_DCOR:.4f}. "
        "A value above the threshold detects *any* dependence structure between "
        "age and alpha-drift, including non-linear relationships xi may miss. "
        "Inspect epsilon / T_grace / workload shape before re-tuning T_DCOR."
    )
