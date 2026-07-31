"""Intentional clustering (#436).

Retrieval-time pass that biases the top-K output toward cluster-diverse
beliefs — when a multi-fact query needs more than one belief to answer,
the existing rank+pack returns K beliefs from the highest-scoring graph
neighbourhood and a complementary cluster never makes the cut.
Clustering replaces the pack loop with a diversity-aware greedy fill.

Spec: ``docs/design/feature-intentional-clustering.md``.

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
from typing import Callable, Final, Iterable

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
    cost_fn: Callable[[Belief], int] | None = None,
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

    ``cost_fn`` (#878 compose-reconciliation): per-belief token cost
    callable. Defaults to the raw ``_belief_tokens`` estimator. Callers
    composing with ``use_type_aware_compression`` pass a callable that
    returns the compressed ``rendered_tokens`` so the cluster pack
    accounts in the same currency as the outer pack loop.
    """
    cost = cost_fn or _belief_tokens
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
        rep_cost = cost(rep)
        if used_tokens + rep_cost > token_budget:
            if fallback_to_score:
                break
            continue
        out.append(rep)
        seen.add(rep_id)
        used_tokens += rep_cost
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
            b_cost = cost(b)
            if used_tokens + b_cost > token_budget:
                continue
            out.append(b)
            seen.add(mid)
            used_tokens += b_cost

    return out


# --- Budgeted maximum coverage pack selector (#1176 proposal 2) --------


def pack_max_coverage(
    candidates: list[Belief],
    *,
    token_budget: int,
    coverage: dict[str, frozenset[str]],
    term_weights: dict[str, float],
    cost_fn: Callable[[Belief], int] | None = None,
) -> list[Belief]:
    """Budgeted maximum coverage over query terms (Khuller-Moss-Naor 1999).

    Pack selection is a budgeted maximum-coverage problem, not a ranked
    fill: the value of adding a belief is the query-term mass it brings
    that nothing already selected covers. `pack_with_clusters` cannot see
    that — measured on 523 replayed prompts it leaves ~9 near-duplicate
    pairs per user pack among the beliefs it chooses, and on harness
    prompts it produces 79% *more* near-duplicate pairs (4-gram Jaccard
    >= 0.25) than plain rank-greedy, so its diversity pass is
    neutral-to-harmful on the redundancy axis it exists to address.

    `coverage` maps belief id -> the query terms that belief contains;
    `term_weights` maps term -> weight (idf). Both are supplied by the
    caller so this stays a pure function with no BM25 or store coupling.
    A belief absent from `coverage` covers nothing and can still be
    selected only by the tie-break, which is the correct handling for a
    belief that matched on a lane other than term overlap.

    Objective ``f(S) = sum of term_weights[t] for t in union of cov(b)``
    is monotone and submodular, so the cost-benefit greedy paired with
    the best single feasible element carries the standard (1 - 1/e)
    guarantee. Both halves are computed and the better is returned;
    omitting the single-element arm is what breaks the bound in the
    pathological case where one belief covers nearly everything at a cost
    just under budget.

    A relevance floor multiplies each marginal gain by a linear rank
    weight ``(n - i) / n``. Without it the greedy will spend budget on a
    low-ranked belief that happens to carry one rare term. The rank proxy
    rather than the composite rerank score matches the convention
    `retrieve_with_tiers` already uses for `cluster_scores`; the score
    itself is not threaded out of `_l1_hits` today, and using it is a
    follow-up rather than a silent reinterpretation of this constant.

    Deterministic: a fixed sequence of argmaxes over an explicit total
    order ``(-gain_ratio, rank, belief_id)``. No clock, no randomness, no
    reliance on set iteration order.
    """
    cost = cost_fn or _belief_tokens
    if not candidates or token_budget <= 0:
        return []

    n = len(candidates)
    rank_of: dict[str, int] = {b.id: i for i, b in enumerate(candidates)}
    # Linear rank weight in (0, 1]; candidates are rerank-sorted.
    rank_weight: dict[str, float] = {
        b.id: (n - i) / n for i, b in enumerate(candidates)
    }

    def gain(b: Belief, covered: frozenset[str]) -> float:
        new = coverage.get(b.id, frozenset()) - covered
        raw = sum(term_weights.get(t, 0.0) for t in new)
        return raw * rank_weight[b.id]

    # --- arm 1: cost-benefit greedy with CELF lazy evaluation ---------
    # The heap holds an upper bound on each element's gain/cost ratio.
    # Submodularity makes a stale bound a valid upper bound, so popping
    # and re-checking yields exactly the eager greedy's choices while
    # evaluating far fewer marginals.
    import heapq

    covered: frozenset[str] = frozenset()
    chosen: list[Belief] = []
    used = 0
    by_id: dict[str, Belief] = {b.id: b for b in candidates}
    heap: list[tuple[float, int, str, int]] = []
    for b in candidates:
        c = cost(b)
        if c <= 0:
            # A zero-cost belief is free; ratio is undefined. Give it the
            # raw gain so it sorts on value rather than dividing by zero.
            heapq.heappush(heap, (-gain(b, covered), rank_of[b.id], b.id, -1))
        else:
            heapq.heappush(
                heap, (-gain(b, covered) / c, rank_of[b.id], b.id, 0)
            )
    while heap:
        _neg_ratio, rank, bid, _flag = heapq.heappop(heap)
        b = by_id[bid]
        c = cost(b)
        if used + c > token_budget:
            # `used` only grows, so an element that does not fit now can
            # never fit later. Dropping it is not an approximation.
            continue
        true_gain = gain(b, covered)
        true_ratio = true_gain / c if c > 0 else true_gain
        key = (-true_ratio, rank, bid)
        # Compare the FULL total-order key against the next bound, not the
        # ratio alone. On a ratio tie the eager greedy takes the
        # lower-ranked element; a ratio-only comparison would take
        # whichever happened to be popped, which diverges from the eager
        # result on ~20% of random inputs (caught by the equality test).
        if heap and key > heap[0][:3]:
            heapq.heappush(heap, (-true_ratio, rank, bid, 0))
            continue
        chosen.append(b)
        covered |= coverage.get(bid, frozenset())
        used += c

    def f(sel: list[Belief]) -> float:
        u: frozenset[str] = frozenset()
        for b in sel:
            u |= coverage.get(b.id, frozenset())
        return sum(term_weights.get(t, 0.0) for t in u)

    # --- arm 2: best single feasible element by raw coverage ----------
    best_single: list[Belief] = []
    best_val = -1.0
    for b in candidates:
        if cost(b) > token_budget:
            continue
        v = sum(
            term_weights.get(t, 0.0) for t in coverage.get(b.id, frozenset())
        )
        if v > best_val:
            best_val, best_single = v, [b]

    winner = chosen if f(chosen) >= best_val else best_single
    # Emit in rerank order rather than selection order: the pack is
    # consumed as a ranked list downstream, and selection order is an
    # artefact of the greedy.
    return sorted(winner, key=lambda b: rank_of[b.id])
