"""BFS multi-hop graph traversal — v1.3.0 L3 retrieval tier.

Pure-expansion module: walks outbound edges from a set of seed beliefs,
scoring visited beliefs by the multiplicative product of edge-type
weights along the path that reached them.

Constants and behaviour follow `docs/bfs_multihop.md` exactly:

  - `BFS_EDGE_WEIGHTS` — module-level dict (monkey-patchable in tests),
    biases the frontier toward decisional edges (SUPERSEDES 0.90,
    CONTRADICTS 0.85) over informational edges (RELATES_TO 0.30).
  - `expand_bfs()` — pure function. Cycle detection via per-call
    visited-set initialised from seed ids. Bounded by `max_depth`,
    `nodes_per_hop`, `total_budget_nodes`. Pruned by `min_path_score`.
  - `ScoredHop` — dataclass result with `belief`, `score`, `depth`,
    `path` (list of edge-type strings).

Determinism: every tie (edge ranking, result ordering) breaks on
belief id ascending. Two `expand_bfs()` runs with the same store
contents and same seeds produce identical output. This is a
load-bearing property — see PHILOSOPHY § Determinism.

Stdlib only. No third-party dependencies. Wired into `retrieval.py`
behind a default-off flag at v1.3.0 (`AELFRICE_BFS=1` or
`[retrieval] bfs_enabled = true` in `.aelfrice.toml`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

# Default knobs — see docs/bfs_multihop.md § Depth cap and budget.
DEFAULT_MAX_DEPTH: Final[int] = 2
DEFAULT_NODES_PER_HOP: Final[int] = 16
DEFAULT_TOTAL_BUDGET_NODES: Final[int] = 32
DEFAULT_MIN_PATH_SCORE: Final[float] = 0.10

# Edge-type weight table. Biases the frontier toward decisional edges
# (SUPERSEDES, CONTRADICTS) over informational ones (RELATES_TO). Spec
# § Edge-type weight table is the source of truth; see also
# § Why these are not the EDGE_VALENCE numbers.
#
# Mutable dict (not frozen) so tests can monkeypatch it. Producers
# treat unknown edge types as weight 0.0 (skipped).
BFS_EDGE_WEIGHTS: dict[str, float] = {
    EDGE_SUPERSEDES: 0.90,
    EDGE_CONTRADICTS: 0.85,
    EDGE_DERIVED_FROM: 0.70,
    EDGE_IMPLEMENTS: 0.65,
    EDGE_SUPPORTS: 0.60,
    EDGE_TESTS: 0.55,
    EDGE_CITES: 0.40,
    EDGE_RELATES_TO: 0.30,
    EDGE_TEMPORAL_NEXT: 0.25,
    # Marker edge — skipped during BFS expansion. Demotion happens in
    # the rerank pass (`aelfrice.edge_rerank`), not here. Pinned at 0.0
    # explicitly so the contract is reviewable rather than implicit
    # via the `BFS_EDGE_WEIGHTS.get(..., 0.0)` default. See #421.
    EDGE_POTENTIALLY_STALE: 0.0,
}


@dataclass(frozen=True)
class ScoredHop:
    """One belief surfaced by BFS expansion.

    `score` is the multiplicative product of `BFS_EDGE_WEIGHTS` over
    the edges of the path that reached this belief, capped above at
    1.0 (no edge weight exceeds 1.0). `depth` is the number of edges
    in the path (1 for a direct neighbour, 2 for a two-hop expansion,
    etc.). `path` is the ordered list of edge-type strings.

    `belief_id_trail` is the ordered tuple of belief ids the BFS
    walked through to reach this hop, starting with the seed and
    ending with this hop's belief id. Length is always ``depth + 1``
    (seed + one id per hop). Empty default for backwards-compat with
    callers that construct ``ScoredHop`` directly in tests; the
    production ``expand_bfs`` always emits a populated trail. Added
    for #645 R2 (#658) — compound-confidence + fork-on-CONTRADICTS
    derivation needs the per-hop trail of beliefs, not just the
    terminal endpoint.

    ``owning_scope`` is the federation scope that owns this hop's
    belief: ``None`` for local-DB hops (the pre-federation default),
    a peer ``name`` (matching the peer's ``knowledge_deps.json``
    entry) when the hop walked into a peer DB. Added for #690 —
    peer-aware BFS walks step into a peer's edge table via
    ``MemoryStore.edges_from_in_scope`` and materialise neighbours
    via ``get_belief_in_scope``; the scope propagates from parent
    frontier entry to child so subsequent hops route to the right
    DB.
    """

    belief: Belief
    score: float
    depth: int
    path: list[str]
    belief_id_trail: tuple[str, ...] = ()
    owning_scope: str | None = None


def expand_bfs(
    seeds: list[Belief],
    store: MemoryStore,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    nodes_per_hop: int = DEFAULT_NODES_PER_HOP,
    total_budget: int = DEFAULT_TOTAL_BUDGET_NODES,
    min_path_score: float = DEFAULT_MIN_PATH_SCORE,
    seed_scopes: dict[str, str | None] | None = None,
) -> list[ScoredHop]:
    """Walk outbound edges from `seeds`, returning ranked expansions.

    Pseudocode + properties: see `docs/bfs_multihop.md § Algorithm`.

    Determinism contract:
      - Edges at each frontier expansion are ranked by
        (-edge_type_weight, -edge.weight, dst_id_ascending). Any
        ranking tie thus breaks on belief id ascending.
      - Final results are sorted by (-score, belief.id) so two
        identical inputs always produce byte-identical output.

    Cycle detection: visited-set initialised from seed ids. A belief
    cannot re-enter the frontier as an expansion result, and seeds'
    cross-edges are not double-counted as expansion nodes.

    Budget bookkeeping:
      - `nodes_per_hop` caps fanout per frontier entry (top-k after
        edge-type ranking).
      - `total_budget` caps the cumulative number of expanded
        beliefs across all hops.
      - `min_path_score` prunes paths whose multiplicative score has
        decayed below the noise floor.
      - `max_depth` is a hard ceiling on path length.

    Returns expansions only — seeds are NOT included in the output
    (the visited-set initialisation prevents that, and the L3 tier
    contract is "tier-0 seeds first, BFS expansions after").

    Federation (#690): ``seed_scopes`` is an optional ``{belief_id:
    owning_scope}`` mapping. When a seed id appears in the dict with
    a non-None scope, the walk follows that peer's edges (via
    ``store.edges_from_in_scope`` / ``get_belief_in_scope``) instead
    of local. The scope propagates from each frontier entry to its
    children — once the walk enters a peer, subsequent hops stay
    inside that peer's edge graph. Seeds not in the dict (and the
    default ``seed_scopes=None`` case) walk local edges only, so
    pre-federation callers see identical behaviour byte-for-byte.
    """
    if not seeds or max_depth < 1 or total_budget < 1:
        return []

    scopes_in: dict[str, str | None] = seed_scopes or {}
    visited: set[str] = {b.id for b in seeds}
    # Frontier entries: (belief_id, path_score, depth, path_edge_types,
    # belief_id_trail, owning_scope). The trail tracks every belief id
    # the BFS has walked through to reach `belief_id`, starting from
    # the seed; consumers downstream (compound-confidence + fork-on-
    # CONTRADICTS, #645 R2) reconstruct paths from this without
    # re-walking the graph. `owning_scope` is the federation scope
    # whose edges the next hop reads (None = local; #690).
    frontier: list[
        tuple[str, float, int, list[str], tuple[str, ...], str | None]
    ] = [
        (b.id, 1.0, 0, [], (b.id,), scopes_in.get(b.id))
        for b in seeds
    ]
    expanded: list[ScoredHop] = []
    nodes_used: int = 0

    while frontier and nodes_used < total_budget:
        next_frontier: list[
            tuple[str, float, int, list[str], tuple[str, ...], str | None]
        ] = []
        for current_id, score, depth, path, trail, scope in frontier:
            if depth >= max_depth:
                continue
            if nodes_used >= total_budget:
                break
            edges: list[Edge] = store.edges_from_in_scope(current_id, scope)
            # Determinism: rank by (-edge-type-weight, -edge.weight,
            # dst id). Filter already-visited dsts BEFORE ranking so
            # the top-k slice is over genuinely-novel candidates.
            candidates = [e for e in edges if e.dst not in visited]
            ranked = sorted(
                candidates,
                key=lambda e: (
                    -BFS_EDGE_WEIGHTS.get(e.type, 0.0),
                    -e.weight,
                    e.dst,
                ),
            )[:nodes_per_hop]
            for edge in ranked:
                if nodes_used >= total_budget:
                    break
                edge_w = BFS_EDGE_WEIGHTS.get(edge.type, 0.0)
                if edge_w == 0.0:
                    # Unknown / zero-weighted edge type — skip,
                    # don't mark visited (a future hop might still
                    # reach this dst on a higher-weight path).
                    continue
                new_score = score * edge_w
                if new_score < min_path_score:
                    continue
                # Mark visited BEFORE the materialisation guard so a
                # missing-belief race doesn't re-queue the same id
                # later in this same call.
                visited.add(edge.dst)
                belief = store.get_belief_in_scope(edge.dst, scope)
                if belief is None:
                    # Race: belief was deleted between edges_from
                    # and get_belief. Skip; the next mutation cycle
                    # will fire the cache invalidation that re-runs
                    # this query.
                    continue
                new_path = path + [edge.type]
                new_trail = trail + (edge.dst,)
                expanded.append(
                    ScoredHop(
                        belief=belief,
                        score=new_score,
                        depth=depth + 1,
                        path=new_path,
                        belief_id_trail=new_trail,
                        owning_scope=scope,
                    )
                )
                next_frontier.append(
                    (edge.dst, new_score, depth + 1, new_path, new_trail, scope)
                )
                nodes_used += 1
        frontier = next_frontier

    expanded.sort(key=lambda s: (-s.score, s.belief.id))
    return expanded
