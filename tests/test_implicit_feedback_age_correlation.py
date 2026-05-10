"""Regression guard: belief age vs alpha-drift Pearson correlation (#555).

Verifies that the deferred-feedback sweeper does not silently become a clock
(i.e., older beliefs do not accumulate disproportionate alpha bumps purely
because they are older, independent of retrieval frequency).

Synthetic workload -- 1 week, N=200 beliefs
--------------------------------------------
- N=200 beliefs created uniformly across a 7-day window (one per ~50 min).
- Retrieval-event counts drawn from Poisson(lam=2) per belief, capped at 10.
  Each retrieval is placed at a random time within the belief's lifetime.
  The Poisson draw uses the seeded RNG and is independent of creation order,
  so retrieval frequency is decorrelated from age by construction.
- ~15 % of beliefs also receive one explicit positive-feedback event inside
  the grace window; the sweeper cancels those rows (no alpha change).
- All retrieval exposures are enqueued with enqueued_at = retrieval_time.
  The sweeper runs at T_end + 2 * T_grace so every grace window has elapsed.
- epsilon=0.05, T_grace=1800 s (defaults).

What the correlation bound means
----------------------------------
Pearson r in [-0.2, 0.5]:
  - True correlation is ~0 when retrieval frequency is decorrelated from age.
    Sampling noise with N=200 gives sigma ~0.07, so -0.2 is ~3 sigma below
    zero -- a very conservative lower bound that will only fire if the sweeper
    is systematically penalising older beliefs.
  - r > 0.5 means the sweeper is a clock: age drives alpha bumps more than
    retrieval frequency explains. A large increase to epsilon or a dramatic
    decrease in T_grace relative to the window can push r above 0.5.
  - The lower bound is intentionally -0.2 (not 0.0): with a decorrelated
    workload, r will hover near 0 and occasionally go slightly negative due
    to sampling variation. Clamping at 0.0 would produce spurious failures.

RNG seed: 42.  Change only when adjusting workload shape; document why.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from scipy.stats import pearsonr

from aelfrice.deferred_feedback import (
    DEFAULT_EPSILON,
    DEFAULT_T_GRACE_SECONDS,
    enqueue_retrieval_exposures,
    sweep_deferred_feedback,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
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

R_LOW: float = -0.2
R_HIGH: float = 0.5

_EPOCH = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
_WINDOW_SECONDS = WINDOW_DAYS * 86_400


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth exact Poisson draw using the provided RNG instance."""
    l_ = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l_:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Workload builder
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
            demotion_pressure=0,
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
# Correlation guard
# ---------------------------------------------------------------------------


def test_age_vs_alpha_drift_correlation_in_bounds() -> None:
    """Pearson r(age_days, alpha - alpha_initial) must be in [-0.2, 0.5].

    r < -0.2: sweeper systematically penalises older beliefs (unexpected).
    r > 0.5: sweeper has become a clock -- age drives alpha bumps more than
    retrieval frequency alone explains.
    r near 0: expected; retrieval frequency is decorrelated from age by design.
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

    r, _p = pearsonr(ages, drifts)

    assert R_LOW <= r <= R_HIGH, (
        f"Pearson r(age_days, alpha_drift) = {r:.4f} is outside [{R_LOW}, {R_HIGH}]. "
        "r > 0.5 suggests the sweeper became a clock; "
        "r < -0.2 suggests it is penalising older beliefs. "
        "Check epsilon / T_grace / workload shape before re-tuning."
    )
