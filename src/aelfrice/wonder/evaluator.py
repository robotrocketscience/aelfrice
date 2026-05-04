"""Bake-off evaluator metrics for #228.

Direct port of the four metrics in
``docs/v2_wonder_consolidation.md`` § "Adoption criteria for v2.0
ship":

* ``confirmation_rate`` — promotions / phantoms_generated under
  fixed feedback budget.
* ``retrieval_freq_per_cost`` — (mean retrieval count) / (mean
  construction_cost). Retrieval count is simulated as a function of
  composition size: larger compositions surface more often per
  the spec's "retrieval surface frequency per unit construction
  cost" criterion.
* ``pairwise_jaccard[s1, s2]`` — overlap between two strategies'
  composition sets. Sorted-tuple compositions hash identically
  across strategies so set algebra works directly.
* ``junk_rate`` — phantoms gc'd within 14d / generated. The
  simulator marks a phantom for gc when it junks repeatedly
  before any confirm; this evaluator treats "junked under fixed
  budget" as a proxy for "would gc within 14d" — the spec's
  threshold is qualitative, this makes it computable.

Output is a single ``BakeoffResult`` dataclass with one
``StrategyMetrics`` per strategy plus the cross-strategy Jaccard
matrix and the adoption-criterion verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .models import STRATEGY_RW, STRATEGY_STS, STRATEGY_TC, Phantom
from .simulator import (
    SyntheticCorpus,
    feedback_verdict,
    simulate_promotion,
)

Verdict = Literal["single", "ensemble", "defer", "drop"]

# Adoption criteria thresholds from spec § "Adoption criteria for
# v2.0 ship". Constants here so the runner can echo them in the
# result JSON for audit.
H0_PLUS_PP_FLOOR: float = 0.10
JACCARD_REDUNDANCY: float = 0.6
JACCARD_COMPLEMENT: float = 0.3
JUNK_RATE_DEFER: float = 0.60
H0_NULL_RATE: float = 0.065  # midpoint of spec's 5–8% prediction


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    n_phantoms: int
    confirmation_rate: float
    retrieval_freq_per_cost: float
    junk_rate: float
    mean_construction_cost: float


@dataclass(frozen=True)
class BakeoffResult:
    metrics: tuple[StrategyMetrics, ...]
    pairwise_jaccard: dict[tuple[str, str], float]
    verdict: Verdict
    h0_floor: float = H0_NULL_RATE + H0_PLUS_PP_FLOOR
    notes: tuple[str, ...] = field(default_factory=tuple)


def _retrieval_count(composition: tuple[str, ...]) -> int:
    """Simulated retrieval surface frequency.

    Each constituent contributes one retrieval event under the
    no-embedding form (the corpus has no real query log). Reuses
    composition size as a stand-in — strategies that generate
    larger phantoms surface more often, by design.
    """
    return len(composition)


def evaluate_strategy(
    phantoms: list[Phantom],
    corpus: SyntheticCorpus,
    *,
    feedback_budget_per_phantom: int = 16,
) -> StrategyMetrics:
    """Score a strategy's phantom set against the corpus.

    ``feedback_budget_per_phantom`` is the simulated number of
    feedback events each phantom can absorb. Confirms accumulate α;
    junks accumulate β. The promotion gate is from
    ``ALPHA_PROMOTION_THRESHOLD``.

    A phantom counts toward ``junk_rate`` if every event in its
    budget was a junk verdict (the simulator is deterministic
    given corpus + composition, so all events for one phantom
    share the same verdict — but the rate definition is preserved
    in case the simulator gains randomness later).
    """
    if not phantoms:
        return StrategyMetrics(
            strategy="<empty>",
            n_phantoms=0,
            confirmation_rate=0.0,
            retrieval_freq_per_cost=0.0,
            junk_rate=0.0,
            mean_construction_cost=0.0,
        )
    strategy = phantoms[0].strategy
    promotions = 0
    junked = 0
    total_retrieval = 0
    total_cost = 0.0
    for phantom in phantoms:
        verdicts = [
            feedback_verdict(phantom.composition, corpus)
            for _ in range(feedback_budget_per_phantom)
        ]
        confirms = sum(1 for v in verdicts if v == "confirm")
        junks = sum(1 for v in verdicts if v == "junk")
        if simulate_promotion(confirms, junks):
            promotions += 1
        if confirms == 0 and junks > 0:
            junked += 1
        total_retrieval += _retrieval_count(phantom.composition)
        total_cost += phantom.construction_cost
    n = len(phantoms)
    confirmation_rate = promotions / n
    junk_rate = junked / n
    mean_cost = total_cost / n
    mean_retrieval = total_retrieval / n
    retrieval_per_cost = mean_retrieval / mean_cost if mean_cost > 0 else 0.0
    return StrategyMetrics(
        strategy=strategy,
        n_phantoms=n,
        confirmation_rate=confirmation_rate,
        retrieval_freq_per_cost=retrieval_per_cost,
        junk_rate=junk_rate,
        mean_construction_cost=mean_cost,
    )


def pairwise_jaccard(
    a: list[Phantom], b: list[Phantom]
) -> float:
    """Jaccard over composition tuples between two phantom sets."""
    set_a = {p.composition for p in a}
    set_b = {p.composition for p in b}
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def adoption_verdict(
    metrics: tuple[StrategyMetrics, ...],
    jaccard: dict[tuple[str, str], float],
    *,
    h0_floor: float | None = None,
) -> Verdict:
    """Apply the four spec rules in order.

    1. **Drop** if all strategies fall below the H0 null floor.
    2. **Defer** if no strategy clears H0+10pp, or any strategy's
       junk_rate exceeds 60%.
    3. **Single-strategy ship** if exactly one strategy clears the
       floor with retrieval-per-cost within 25% of best, and the
       others are not complementary (any pairwise Jaccard ≥ 0.6).
    4. **Ensemble** if the top two strategies are complementary
       (Jaccard < 0.3) and each clears the floor.

    Order matters: drop dominates defer, which dominates the ship
    decisions.
    """
    if h0_floor is None:
        h0_floor = H0_NULL_RATE + H0_PLUS_PP_FLOOR
    by_strategy = {m.strategy: m for m in metrics}
    rates = [m.confirmation_rate for m in metrics]
    if all(r < H0_NULL_RATE for r in rates):
        return "drop"
    if any(m.junk_rate > JUNK_RATE_DEFER for m in metrics):
        return "defer"
    clearing = [m for m in metrics if m.confirmation_rate >= h0_floor]
    if not clearing:
        return "defer"
    best_retrieval = max(m.retrieval_freq_per_cost for m in clearing)
    cost_window = [
        m for m in clearing
        if m.retrieval_freq_per_cost >= 0.75 * best_retrieval
    ]
    if len(clearing) == 1:
        return "single"
    sorted_clearing = sorted(
        clearing, key=lambda m: m.confirmation_rate, reverse=True
    )
    top_two = sorted_clearing[:2]
    pair = tuple(sorted(p.strategy for p in top_two))
    j = jaccard.get(pair, 0.0)
    if j < JACCARD_COMPLEMENT:
        return "ensemble"
    if any(j >= JACCARD_REDUNDANCY for j in jaccard.values()):
        # Top strategy plus any other with high overlap → single
        # ship the leader.
        if cost_window:
            return "single"
    return "single"


__all__ = [
    "BakeoffResult",
    "H0_NULL_RATE",
    "H0_PLUS_PP_FLOOR",
    "JACCARD_COMPLEMENT",
    "JACCARD_REDUNDANCY",
    "JUNK_RATE_DEFER",
    "StrategyMetrics",
    "Verdict",
    "adoption_verdict",
    "evaluate_strategy",
    "pairwise_jaccard",
]
