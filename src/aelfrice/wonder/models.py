"""Wonder-package dataclasses (#228).

Kept out of top-level ``aelfrice.models`` because ``Phantom`` is a
v2.0 research-only concept and carries strategy-specific fields
(``construction_cost``) that don't belong alongside the
load-bearing ``Belief`` / ``Edge`` dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

STRATEGY_RW: Final[str] = "RW"
STRATEGY_TC: Final[str] = "TC"
STRATEGY_STS: Final[str] = "STS"

STRATEGIES: Final[frozenset[str]] = frozenset({
    STRATEGY_RW,
    STRATEGY_TC,
    STRATEGY_STS,
})


@dataclass(frozen=True)
class Phantom:
    """A speculative composition produced by a wonder generation strategy.

    ``composition`` is the sorted tuple of belief ids the strategy
    bundled together. Sorted so two strategies that produce the
    same belief set hash identically — required for the Jaccard
    metric to detect overlap.

    ``construction_cost`` is in ``atoms-touched`` units (Decision E
    in the planning memo) — every belief the strategy read while
    producing this phantom counts as one. Reproducible across
    machines and substrate-neutral.

    ``seed_id`` records the random-walk start atom; ``None`` for TC
    and STS which have no single-seed anchor.
    """

    composition: tuple[str, ...]
    strategy: str
    construction_cost: float
    seed_id: str | None = None
