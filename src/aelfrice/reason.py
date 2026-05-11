"""Reason verdict + impasse classifiers (#645 R1).

Pure-derivation module: takes the BFS walk evidence (seeds + ScoredHops)
plus the underlying MemoryStore and returns a typed verdict together
with the list of impasses observed. No graph traversal of its own and
no mutation — the walk has already happened in :func:`aelfrice.bfs_multihop.expand_bfs`.

Ports the four agentmemory diagnostic outputs into aelfrice while
preserving the deterministic, stdlib-only retrieval contract (PHILOSOPHY
#605):

- ``Verdict`` enum: ``SUFFICIENT`` / ``INSUFFICIENT`` / ``CONTRADICTORY``
  / ``UNCERTAIN`` / ``PARTIAL``.
- ``ImpasseKind`` enum: ``TIE`` / ``GAP`` / ``CONSTRAINT_FAILURE`` /
  ``NO_CHANGE``.

The derivation is a pure function of (seeds, hops, store-leaf-lookup)
with two named thresholds (:data:`CONFIDENT_TRIALS_MIN`,
:data:`CLOSE_MEAN_DELTA`). Two runs with the same inputs return the
same verdict + impasse list byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    EDGE_CONTRADICTS,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


class Verdict(str, Enum):
    """Top-level classification of the walk's evidence sufficiency."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"
    UNCERTAIN = "UNCERTAIN"
    PARTIAL = "PARTIAL"


class ImpasseKind(str, Enum):
    """Categorical label for one observed reasoning impasse."""

    TIE = "TIE"
    GAP = "GAP"
    CONSTRAINT_FAILURE = "CONSTRAINT_FAILURE"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class Impasse:
    """One reasoning impasse observed over the walk.

    ``belief_ids`` are the loci of the impasse: the leaf for ``GAP``,
    the locked endpoint for ``CONSTRAINT_FAILURE``, both sides for
    ``TIE``, and every visited belief for ``NO_CHANGE``. Tuples (not
    lists) so the impasse is hashable and stable across JSON round-
    trips.
    """

    kind: ImpasseKind
    belief_ids: tuple[str, ...]
    note: str


CONFIDENT_TRIALS_MIN: Final[int] = 4
"""Posterior trials (``alpha + beta``) floor above which a belief
has enough evidence to be treated as confident for verdict / impasse
purposes. A fresh ``Beta(1, 1)`` belief has ``alpha + beta == 2`` (the
uninformative prior); two additional observations (total = 4) is the
smallest sample that meaningfully narrows the credible interval over
the prior."""

CLOSE_MEAN_DELTA: Final[float] = 0.15
"""Two posterior means whose absolute difference is below this delta
count as "similar" for ``TIE`` detection. Empirical: a 0.15 gap in
posterior mean is roughly the noise floor of a ``Beta(2, 4)`` vs
``Beta(3, 3)`` comparison (both ~6 trials)."""


def _trials(b: Belief) -> float:
    return b.alpha + b.beta


def _mean(b: Belief) -> float:
    t = _trials(b)
    if t <= 0:
        return 0.5
    return b.alpha / t


def _is_low_evidence(b: Belief) -> bool:
    return _trials(b) < CONFIDENT_TRIALS_MIN


def classify(
    seeds: list[Belief],
    hops: list[ScoredHop],
    store: MemoryStore,
) -> tuple[Verdict, list[Impasse]]:
    """Derive ``(verdict, impasses)`` from walk evidence.

    Pure derivation — does not traverse the graph. The store is only
    consulted for :meth:`MemoryStore.edges_from` (leaf detection on
    ``GAP``); all other inputs are already materialised on ``seeds``
    and ``hops``.

    Determinism: impasse list order is fixed by construction order in
    this function (``NO_CHANGE`` first, then ``CONSTRAINT_FAILURE``,
    then ``TIE``, then ``GAP``); within each kind impasses are emitted
    in hop-list order (which itself is deterministic per
    :func:`aelfrice.bfs_multihop.expand_bfs`'s ordering contract).
    """
    impasses: list[Impasse] = []

    if not hops:
        return Verdict.INSUFFICIENT, [
            Impasse(
                kind=ImpasseKind.NO_CHANGE,
                belief_ids=tuple(sorted(s.id for s in seeds)),
                note="no expansions from seeds",
            )
        ]

    all_low = all(_is_low_evidence(h.belief) for h in hops) and all(
        _is_low_evidence(s) for s in seeds
    )
    if all_low:
        impasses.append(
            Impasse(
                kind=ImpasseKind.NO_CHANGE,
                belief_ids=tuple(sorted(h.belief.id for h in hops)),
                note=(
                    f"every visited belief has alpha+beta < {CONFIDENT_TRIALS_MIN}"
                ),
            )
        )

    contradiction_hops = [
        h for h in hops if h.path and h.path[-1] == EDGE_CONTRADICTS
    ]

    for h in contradiction_hops:
        if h.belief.lock_level == LOCK_USER:
            impasses.append(
                Impasse(
                    kind=ImpasseKind.CONSTRAINT_FAILURE,
                    belief_ids=(h.belief.id,),
                    note="path blocked by a locked belief via CONTRADICTS",
                )
            )

    for i in range(len(contradiction_hops)):
        for j in range(i + 1, len(contradiction_hops)):
            a = contradiction_hops[i].belief
            b = contradiction_hops[j].belief
            if abs(_mean(a) - _mean(b)) < CLOSE_MEAN_DELTA:
                impasses.append(
                    Impasse(
                        kind=ImpasseKind.TIE,
                        belief_ids=tuple(sorted([a.id, b.id])),
                        note="contradicting beliefs at similar posterior means",
                    )
                )

    for h in hops:
        if _is_low_evidence(h.belief) and not store.edges_from(h.belief.id):
            impasses.append(
                Impasse(
                    kind=ImpasseKind.GAP,
                    belief_ids=(h.belief.id,),
                    note="path ends at high-uncertainty leaf",
                )
            )

    has_tie = any(i.kind == ImpasseKind.TIE for i in impasses)
    has_cf = any(i.kind == ImpasseKind.CONSTRAINT_FAILURE for i in impasses)
    has_gap = any(i.kind == ImpasseKind.GAP for i in impasses)
    has_no_change = any(i.kind == ImpasseKind.NO_CHANGE for i in impasses)
    has_confident_hop = any(not _is_low_evidence(h.belief) for h in hops)

    if has_tie:
        verdict = Verdict.CONTRADICTORY
    elif has_no_change:
        verdict = Verdict.INSUFFICIENT
    elif has_cf:
        verdict = Verdict.PARTIAL
    elif has_gap and not has_confident_hop:
        verdict = Verdict.UNCERTAIN
    elif impasses and has_confident_hop:
        verdict = Verdict.PARTIAL
    else:
        verdict = Verdict.SUFFICIENT

    return verdict, impasses
