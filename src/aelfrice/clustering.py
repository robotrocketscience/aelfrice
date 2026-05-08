"""Intentional clustering (#436).

Retrieval-time pass that biases the top-K output toward cluster-diverse
beliefs — when a multi-fact query needs more than one belief to answer,
the existing rank+pack returns K beliefs from the highest-scoring graph
neighbourhood and a complementary cluster never makes the cut.
Clustering replaces the pack loop with a diversity-aware greedy fill.

Spec: ``docs/feature-intentional-clustering.md``.

This module owns the pure-library half of the contract:

- ``cluster_candidates`` — union-find pass over the candidate-induced
  edge subgraph. Returns one ``RetrievalCluster`` per connected
  component.
- ``pack_with_clusters`` — diversity-aware greedy fill. Stage 1 picks
  one representative per cluster up to ``cluster_diversity_target``
  distinct clusters; Stage 2 fills the remaining budget by score.

The retrieval-side wiring (flag resolution, integration with
``retrieve_v2``) lands separately so this module can ship + bench
without a hot-path edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from aelfrice.models import Belief, Edge

# Default edge-weight floor: 0.4. Picked to include `EDGE_CITES` (0.5
# in `EDGE_VALENCE`) but exclude `EDGE_RELATES_TO` (0.3) — beliefs that
# only relate are too weak a signal to be considered the same cluster.
# Tunable via `[retrieval] cluster_edge_weight_floor`.
DEFAULT_CLUSTER_EDGE_FLOOR: Final[float] = 0.4

# Default diversity target: 3 distinct clusters in the top-K. Three
# covers most multi-fact queries without crowding out the score-ranked
# tail. Tunable via `[retrieval] cluster_diversity_target`.
DEFAULT_CLUSTER_DIVERSITY_TARGET: Final[int] = 3

_CHARS_PER_TOKEN: Final[float] = 4.0


def _belief_tokens(b: Belief) -> int:
    """Char-based token estimate, conservative (rounds up).

    Mirrors `retrieval._belief_tokens`. Duplicated here rather than
    imported to keep this module free of a `retrieval`-side dependency
    (the wiring direction is retrieval → clustering, not vice versa).
    """
    if not b.content:
        return 0
    n = len(b.content)
    return int((n + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class RetrievalCluster:
    """One connected-component cluster within the post-rank candidate pool.

    ``cluster_id`` is dense (zero-indexed in deterministic insertion
    order). ``member_ids`` is sorted by descending rank score so
    ``member_ids[0]`` is the representative — the highest-scoring member
    that Stage 1 of the pack picks first.
    """

    cluster_id: int
    member_ids: tuple[str, ...]
    representative_id: str
    seed_score: float


class _UnionFind:
    """Path-compressed, union-by-size DSU. Mirrors `dedup._UnionFind`.

    Duplicated rather than imported so a future refactor can promote
    one of the two to a shared primitive; today neither owns it.
    """

    __slots__ = ("_parent", "_size")

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def make(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: str) -> str:
        path: list[str] = []
        while self._parent[x] != x:
            path.append(x)
            x = self._parent[x]
        for p in path:
            self._parent[p] = x
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]


def cluster_candidates(
    candidates: list[Belief],
    candidate_scores: dict[str, float],
    *,
    edges: Iterable[Edge],
    edge_weight_floor: float = DEFAULT_CLUSTER_EDGE_FLOOR,
) -> list[RetrievalCluster]:
    """Group ``candidates`` into connected components on the
    candidate-induced edge subgraph.

    The subgraph's vertex set is ``{c.id for c in candidates}``; edges
    are the ones in ``edges`` whose ``weight >= edge_weight_floor`` AND
    both endpoints are in the vertex set (candidate-induced — non-
    candidate beliefs are out of consideration per spec § Open question 1).

    ``candidate_scores`` is the per-belief rank score; clusters'
    ``seed_score`` is the max over the component, ``member_ids`` is
    sorted by descending score with ties broken by id ASC for determinism.

    Singletons (candidates with no in-pool neighbours above the floor)
    are returned as size-1 clusters.

    Cluster ordering in the returned list is by descending ``seed_score``;
    ties broken by ``representative_id`` ASC. ``cluster_id`` reflects
    that order.
    """
    if not candidates:
        return []

    candidate_ids = {c.id for c in candidates}
    uf = _UnionFind()
    for cid in candidate_ids:
        uf.make(cid)
    for e in edges:
        if e.weight < edge_weight_floor:
            continue
        if e.src not in candidate_ids or e.dst not in candidate_ids:
            continue
        uf.union(e.src, e.dst)

    groups: dict[str, list[str]] = {}
    for cid in candidate_ids:
        groups.setdefault(uf.find(cid), []).append(cid)

    raw_clusters: list[tuple[float, str, tuple[str, ...]]] = []
    for members in groups.values():
        ranked = sorted(
            members,
            key=lambda mid: (-candidate_scores.get(mid, 0.0), mid),
        )
        seed = candidate_scores.get(ranked[0], 0.0)
        raw_clusters.append((seed, ranked[0], tuple(ranked)))

    raw_clusters.sort(key=lambda t: (-t[0], t[1]))
    return [
        RetrievalCluster(
            cluster_id=i,
            member_ids=members,
            representative_id=members[0],
            seed_score=seed,
        )
        for i, (seed, _rep, members) in enumerate(raw_clusters)
    ]


def pack_with_clusters(
    clusters: list[RetrievalCluster],
    belief_by_id: dict[str, Belief],
    *,
    token_budget: int,
    cluster_diversity_target: int = DEFAULT_CLUSTER_DIVERSITY_TARGET,
    fallback_to_score: bool = True,
) -> list[Belief]:
    """Diversity-aware greedy fill at fixed ``token_budget``.

    Stage 1: walk clusters in descending ``seed_score``; pick each
    cluster's representative until ``cluster_diversity_target`` distinct
    clusters are covered or the budget is exhausted. ``fallback_to_score=True``
    (default) abandons Stage 1 the first time a representative does not
    fit the remaining budget; ``False`` skip-but-continues for strict-
    diversity benchmarks.

    Stage 2: fill the remaining budget from the score-ranked tail
    (members across all clusters in descending seed_score), skipping
    beliefs already in the output.

    ``belief_by_id`` must have an entry for every member id in every
    cluster; missing ids are silently skipped (treated as "deleted
    between rank and pack", same race-handling pattern as the existing
    L2.5 pack loop).
    """
    out: list[Belief] = []
    used_tokens = 0
    seen: set[str] = set()
    covered_clusters: set[int] = set()

    sorted_clusters = sorted(clusters, key=lambda c: -c.seed_score)

    # Stage 1: representatives.
    for cluster in sorted_clusters:
        if len(covered_clusters) >= cluster_diversity_target:
            break
        rep_id = cluster.representative_id
        if rep_id in seen:
            continue
        rep = belief_by_id.get(rep_id)
        if rep is None:
            continue
        cost = _belief_tokens(rep)
        if used_tokens + cost > token_budget:
            if fallback_to_score:
                break
            continue
        out.append(rep)
        seen.add(rep_id)
        used_tokens += cost
        covered_clusters.add(cluster.cluster_id)

    # Stage 2: score-ranked tail. Cluster traversal in descending seed
    # order; within a cluster, member_ids[0] is the representative
    # (already considered) and member_ids[1:] is the rest in score
    # order. Across clusters this is approximately score-order overall.
    for cluster in sorted_clusters:
        for mid in cluster.member_ids:
            if mid in seen:
                continue
            b = belief_by_id.get(mid)
            if b is None:
                continue
            cost = _belief_tokens(b)
            if used_tokens + cost > token_budget:
                continue
            out.append(b)
            seen.add(mid)
            used_tokens += cost

    return out
