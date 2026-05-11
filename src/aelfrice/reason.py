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
class ConsequencePath:
    """One root-to-leaf consequence chain over the belief graph (#645 R2).

    A ``ConsequencePath`` is the path-centric view of a BFS expansion
    that R1's hop-centric :class:`~aelfrice.bfs_multihop.ScoredHop`
    encodes endpoint-first. The fields:

    - ``belief_ids`` — ordered ``root → leaf`` belief ids; first element
      is the originating seed, last is the path's terminal belief.
    - ``edge_kinds`` — ordered list of edge-type strings; length is
      always ``len(belief_ids) - 1``. Length-zero edge_kinds means the
      path is a seed-only "path" (the seed itself before any expansion).
    - ``compound_confidence`` — ``∏ posterior_mean(belief) for belief in
      belief_ids``. Multiplicative decay along the chain: a single
      weak intermediate belief attenuates the whole path.
    - ``weakest_link_belief_id`` — id of the belief with the lowest
      posterior mean over ``belief_ids``. Ties broken by hop index,
      deepest wins (later occurrence in the trail).
    - ``fork_from`` — when this path was forked because its terminal
      edge is ``EDGE_CONTRADICTS``, the belief id at the
      parent-path's terminal (i.e. ``belief_ids[-2]``). ``None`` on
      non-forked paths.

    Frozen + hashable so two derivations from the same evidence are
    byte-identical and the path can be used as a dict key.
    """

    belief_ids: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    compound_confidence: float
    weakest_link_belief_id: str
    fork_from: str | None = None


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


class SubagentRole(str, Enum):
    """Role label for one subagent dispatch produced by R3.

    The slash skill maps each impasse to a role; the host agent fans
    out one subagent per ``DispatchItem`` with the matching prompt
    scaffold. Determinism: role is a pure function of impasse kind.
    """

    VERIFIER = "Verifier"
    GAP_FILLER = "Gap-filler"
    FORK_RESOLVER = "Fork-resolver"


_ROLE_BY_IMPASSE: Final[dict[ImpasseKind, SubagentRole]] = {
    ImpasseKind.GAP: SubagentRole.GAP_FILLER,
    ImpasseKind.NO_CHANGE: SubagentRole.GAP_FILLER,
    ImpasseKind.TIE: SubagentRole.FORK_RESOLVER,
    ImpasseKind.CONSTRAINT_FAILURE: SubagentRole.VERIFIER,
}


@dataclass(frozen=True)
class DispatchItem:
    """One subagent the host should spawn to address an impasse.

    ``role`` is the prompt scaffold the slash skill picks. ``belief_ids``
    is the loci the subagent should investigate (passed through from
    :class:`Impasse`). ``note`` carries the impasse's original note so
    the prompt has the diagnostic phrase already attached.
    """

    role: SubagentRole
    belief_ids: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class SuggestedUpdate:
    """One ``(belief_id, direction)`` row for the SUGGESTED UPDATES section.

    ``direction`` is ``+1`` (helpful — on a confident, non-impasse hop),
    ``-1`` (rejected — on a fork-losing path; deferred until R2's fork
    data lands, never emitted by the R3 minimal surface), or ``"?"``
    (uncertain — on an impasse path).
    """

    belief_id: str
    direction: str
    note: str


def dispatch_policy(
    verdict: Verdict, impasses: list[Impasse]
) -> list[DispatchItem]:
    """Map ``(verdict, impasses)`` to the ordered subagent dispatch.

    Verdict drives whether to dispatch at all; impasse kind drives the
    role for each dispatched subagent.

    - :attr:`Verdict.SUFFICIENT`: no dispatch (the chain already answers).
    - :attr:`Verdict.PARTIAL`, :attr:`Verdict.UNCERTAIN`,
      :attr:`Verdict.INSUFFICIENT`, :attr:`Verdict.CONTRADICTORY`:
      one :class:`DispatchItem` per impasse, role derived from
      :class:`ImpasseKind` via :data:`_ROLE_BY_IMPASSE`.

    Output order matches impasse list order — itself deterministic per
    :func:`classify`'s ordering contract — so two runs with the same
    inputs produce the same dispatch sequence byte-for-byte.
    """
    if verdict == Verdict.SUFFICIENT:
        return []
    return [
        DispatchItem(
            role=_ROLE_BY_IMPASSE[imp.kind],
            belief_ids=imp.belief_ids,
            note=imp.note,
        )
        for imp in impasses
    ]


def suggested_updates(
    verdict: Verdict,
    impasses: list[Impasse],
    hops: list[ScoredHop],
) -> list[SuggestedUpdate]:
    """Derive the SUGGESTED UPDATES rows the slash skill emits.

    Direction rules (R3 minimal surface):

    - ``+1`` — hop belief is confident
      (``alpha + beta >= CONFIDENT_TRIALS_MIN``) and not the locus of
      any impasse. Emitted for every verdict except
      :attr:`Verdict.INSUFFICIENT` (which has no expansions to vote on).
    - ``"?"`` — belief appears in any impasse's ``belief_ids``. Emitted
      once per impasse-locus belief, deduplicated across impasses.
    - ``-1`` — deferred to a follow-up after R2 ships the fork-path data;
      this function never emits ``-1`` rows.

    A belief that is both confident-on-chain *and* on an impasse path
    resolves to ``"?"`` (impasse evidence wins; the caller's manual
    review is the right path).

    Order: confident-positive rows in hop order first, then
    impasse-locus rows in impasse order. Within each block, deduped
    by first occurrence.
    """
    impasse_locus: set[str] = set()
    impasse_note_by_id: dict[str, str] = {}
    for imp in impasses:
        for bid in imp.belief_ids:
            impasse_locus.add(bid)
            impasse_note_by_id.setdefault(bid, imp.note)

    rows: list[SuggestedUpdate] = []
    seen: set[str] = set()

    if verdict != Verdict.INSUFFICIENT:
        for h in hops:
            bid = h.belief.id
            if bid in seen or bid in impasse_locus:
                continue
            if _is_low_evidence(h.belief):
                continue
            seen.add(bid)
            rows.append(
                SuggestedUpdate(
                    belief_id=bid,
                    direction="+1",
                    note="confident hop on the answer chain",
                )
            )

    for imp in impasses:
        for bid in imp.belief_ids:
            if bid in seen:
                continue
            seen.add(bid)
            rows.append(
                SuggestedUpdate(
                    belief_id=bid,
                    direction="?",
                    note=impasse_note_by_id[bid],
                )
            )

    return rows


def derive_paths(
    seeds: list[Belief],
    hops: list[ScoredHop],
) -> list[ConsequencePath]:
    """Derive :class:`ConsequencePath` records from walk evidence.

    Pure function — no graph traversal, no store lookups. Reconstructs
    each path from each hop's ``belief_id_trail`` (populated by
    :func:`aelfrice.bfs_multihop.expand_bfs`) and the posterior means
    of beliefs reachable through ``seeds`` and ``hops``.

    Emits:

    1. One length-1 ``ConsequencePath`` per seed (the "no-expansion"
       baseline path; ``edge_kinds == ()``).
    2. One ``ConsequencePath`` per hop, mirroring its trail.

    Fork detection: a hop whose terminal edge is ``EDGE_CONTRADICTS``
    surfaces with ``fork_from`` set to the parent-path's terminal
    belief (``belief_id_trail[-2]``). Both the parent and the forked
    branch are present in the returned list — the parent is just the
    earlier hop (or seed) whose terminal id matches.

    Order is preserved from the input ``hops`` (which is itself
    deterministic per :func:`expand_bfs`'s sort contract). Seeds are
    emitted first, in seed-input order; hops after, in
    sorted-result order.

    Skips hops whose ``belief_id_trail`` is empty (test fixtures that
    don't bother to populate it) so the function is defensive against
    older callers without crashing.
    """
    belief_by_id: dict[str, Belief] = {s.id: s for s in seeds}
    for h in hops:
        belief_by_id[h.belief.id] = h.belief

    paths: list[ConsequencePath] = []

    for s in seeds:
        m = _mean(s)
        paths.append(
            ConsequencePath(
                belief_ids=(s.id,),
                edge_kinds=(),
                compound_confidence=m,
                weakest_link_belief_id=s.id,
                fork_from=None,
            )
        )

    for h in hops:
        trail = h.belief_id_trail
        if not trail:
            continue
        beliefs_along: list[Belief] = []
        for bid in trail:
            b = belief_by_id.get(bid)
            if b is None:
                break
            beliefs_along.append(b)
        if len(beliefs_along) != len(trail):
            continue
        compound = 1.0
        for b in beliefs_along:
            compound *= _mean(b)
        weakest_id = beliefs_along[0].id
        weakest_mean = _mean(beliefs_along[0])
        for b in beliefs_along[1:]:
            m = _mean(b)
            if m <= weakest_mean:
                weakest_mean = m
                weakest_id = b.id
        fork_from: str | None = None
        if h.path and h.path[-1] == EDGE_CONTRADICTS and len(trail) >= 2:
            fork_from = trail[-2]
        paths.append(
            ConsequencePath(
                belief_ids=tuple(trail),
                edge_kinds=tuple(h.path),
                compound_confidence=compound,
                weakest_link_belief_id=weakest_id,
                fork_from=fork_from,
            )
        )

    return paths
