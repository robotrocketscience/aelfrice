"""Edge-type-keyed rerank consumer for BFS expansion results (#421).

Problem: `BFS_EDGE_WEIGHTS` in `bfs_multihop` are non-negative path
multipliers — they bias what is *reached*, not how reached results
are *ranked*. Marker edges like ``POTENTIALLY_STALE`` need a
separate demotion pass that downgrades reachable beliefs after BFS,
not by path-multiplication during BFS expansion.

This module is that pass. It runs downstream of BFS / lane fusion,
takes the `ScoredHop` list, examines each hop's belief's incoming
edges, and applies a configurable multiplicative penalty per
matching edge type. The result is a new `list[ScoredHop]` with
rescored scores, re-sorted by ``(-score, belief.id)`` — the same
tie-breaking rule used by `expand_bfs` so two passes compose without
order surprises.

The producer for ``POTENTIALLY_STALE`` edges is `aelf doctor` (#387);
this module is its consumer-side substrate.

Multi-edge composition: when more than one penalty-keyed incoming
edge type fires on the same belief, penalties compose
**multiplicatively** (a belief reached via two penalty-keyed edge
types has its score multiplied by *both* factors). Same edge type
firing multiple times collapses to one factor — "at least one
matching incoming edge" is the trigger, not edge count.

Stdlib only.
"""
from __future__ import annotations

from typing import Final, Mapping

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import EDGE_POTENTIALLY_STALE
from aelfrice.store import MemoryStore

# Default penalty factor applied when at least one ``POTENTIALLY_STALE``
# incoming edge targets a surfaced belief. Conservative starting point:
# 0.5 halves the score, which preserves the belief's relative position
# below non-stale equivalents but does not drop it below the BFS floor
# (the rerank pass is a re-rank, not a hard filter). Operator-tunable
# per call via the `penalties` kwarg on `apply_edge_type_rerank`.
DEFAULT_STALE_PENALTY: Final[float] = 0.5

# Default penalty config — only ``POTENTIALLY_STALE`` is keyed today.
# Future marker edges that need rerank-time demotion add their entries
# here; non-marker (positive-weight) edges should never be in this
# table — those bias retrieval through `BFS_EDGE_WEIGHTS` at expansion
# time, not at rerank time.
EDGE_TYPE_PENALTIES_DEFAULT: Final[Mapping[str, float]] = {
    EDGE_POTENTIALLY_STALE: DEFAULT_STALE_PENALTY,
}


def apply_edge_type_rerank(
    hops: list[ScoredHop],
    store: MemoryStore,
    *,
    penalties: Mapping[str, float] | None = None,
) -> list[ScoredHop]:
    """Rerank `hops` by applying per-edge-type penalties.

    For each hop's belief, query incoming edges via `store.edges_to`.
    If any incoming edge type appears in `penalties`, multiply the
    hop's score by the corresponding penalty factor. Multiple distinct
    matching edge types compose multiplicatively. The same matching
    edge type appearing on multiple incoming edges fires once — the
    presence test is "at least one matching edge of this type."

    Determinism contract: the returned list is sorted by
    ``(-score, belief.id)``, matching `expand_bfs`'s tie-breaking
    rule. The same `(hops, store, penalties)` input produces
    byte-identical output.

    Args:
      hops: BFS expansion results from `expand_bfs`. Empty list is
        a no-op (returns empty list).
      store: MemoryStore providing `edges_to(dst)`.
      penalties: per-edge-type penalty factors, typically in
        ``[0.0, 1.0]`` for demotion semantics. ``None`` selects
        `EDGE_TYPE_PENALTIES_DEFAULT` (``POTENTIALLY_STALE`` @ 0.5).
        An explicit ``{}`` is identity (re-sort only).

    Returns:
      A new `list[ScoredHop]` with rescored `score` fields, sorted
      by ``(-score, belief.id)``.
    """
    if not hops:
        return []
    cfg: Mapping[str, float] = (
        EDGE_TYPE_PENALTIES_DEFAULT if penalties is None else penalties
    )
    if not cfg:
        return sorted(hops, key=lambda h: (-h.score, h.belief.id))
    rescored: list[ScoredHop] = []
    for hop in hops:
        incoming = store.edges_to(hop.belief.id)
        firing = {e.type for e in incoming if e.type in cfg}
        new_score = hop.score
        for edge_type in firing:
            new_score *= cfg[edge_type]
        rescored.append(
            ScoredHop(
                belief=hop.belief,
                score=new_score,
                depth=hop.depth,
                path=hop.path,
            )
        )
    rescored.sort(key=lambda h: (-h.score, h.belief.id))
    return rescored
