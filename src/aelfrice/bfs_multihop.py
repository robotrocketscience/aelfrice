"""BFS multi-hop graph traversal — v1.3.0 L3 retrieval tier.

Pure-expansion module: walks outbound edges from a set of seed beliefs,
scoring visited beliefs by the multiplicative product of edge-type
weights along the path that reached them.

Constants and behaviour follow `docs/design/bfs_multihop.md` exactly:

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

# Default knobs — see docs/design/bfs_multihop.md § Depth cap and budget.
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
    # the rerank pass (`aelfrice.edge_rerank`), which `expand_bfs` now
    # actually calls (#1207); until then this comment described a pass
    # with no importer. Pinned at 0.0 explicitly so the contract is
    # reviewable rather than implicit via the
    # `BFS_EDGE_WEIGHTS.get(..., 0.0)` default. See #421.
    #
    # The pin does not neuter the rerank: this weight governs traversal
    # *through* a marker edge, while the rerank keys off marker edges
    # *incoming to* a surfaced belief. The two sets are disjoint.
    EDGE_POTENTIALLY_STALE: 0.0,
}

# Edge types the walk follows AGAINST their stored direction (#1170).
#
# Producers write SUPERSEDES as src=winner(new) -> dst=loser(old):
# `contradiction.resolve_contradiction` (src=winner.id, dst=loser.id) and
# `triple_extractor` parsing "X supersedes Y" as src=X. That direction is
# the one the edge type's name means, and it is kept.
#
# But the spec's justification for the 0.90 weight is "'B replaces A' — the
# most actionable adjacency. If the query hit A, the user almost certainly
# wants B", which needs an old -> new hop. Walking outbound delivered the
# exact opposite: a hit on the *current* belief surfaced its stale
# predecessor at the highest available path score, and a hit on the stale
# one surfaced nothing — so the case the weight was chosen for never fired.
#
# Following SUPERSEDES in reverse fixes the direction without a migration
# that would leave the edge type reading backwards. Types listed here are
# NOT also followed outbound; that is what produced the inversion.
REVERSE_TRAVERSED_EDGE_TYPES: frozenset[str] = frozenset({EDGE_SUPERSEDES})


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
    """Walk the edge graph from `seeds`, returning ranked expansions.

    Pseudocode + properties: see `docs/design/bfs_multihop.md § Algorithm`.

    Traversal direction (#1170). Most edge types are followed
    **outbound** — a hop from `current_id` reaches each edge's `dst`,
    read via ``edges_from_in_scope``. The types in
    ``REVERSE_TRAVERSED_EDGE_TYPES`` are followed **inbound** instead:
    they are read via ``edges_to_in_scope`` and the hop reaches each
    edge's `src`. Those types are never also followed outbound. Today
    that set is ``{SUPERSEDES}``, whose producers write `src` = the new
    belief and `dst` = the one it retires, so an inbound read is what
    steps a hit on a retired belief forward to its replacement. The two
    reads are merged into one candidate list before ranking, so
    "neighbour" below means either kind.

    Determinism contract:
      - Candidates at each frontier expansion are ranked by
        (-edge_type_weight, -edge.weight, neighbour_id_ascending). Any
        ranking tie thus breaks on belief id ascending.
      - Candidates are then deduplicated by neighbour id, keeping the
        strongest edge to each, BEFORE the `nodes_per_hop` slice — two
        edges in one hop can name the same neighbour, and letting a
        duplicate consume a slot would underfill the hop and drop an
        otherwise-eligible belief.
      - Final results are sorted by (-score, belief.id) so two
        identical inputs always produce byte-identical output.

    Cycle detection: visited-set initialised from seed ids. A belief
    cannot re-enter the frontier as an expansion result, and seeds'
    cross-edges are not double-counted as expansion nodes.

    Budget bookkeeping:
      - `nodes_per_hop` caps fanout per frontier entry (top-k after
        edge-type ranking and neighbour dedup, so the cap counts
        distinct beliefs rather than distinct edges).
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
    ``store.edges_from_in_scope`` / ``edges_to_in_scope`` /
    ``get_belief_in_scope``) instead of local. The scope propagates from each frontier entry to its
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
            # Neighbours reachable from `current_id`, normalised to
            # (neighbour_id, edge_type, edge.weight). Outbound edges give
            # their `dst`; the reverse-traversed types (#1170) are read
            # from the inbound side and give their `src`, so a hit on a
            # superseded belief steps forward to its replacement rather
            # than the other way round.
            neighbours: list[tuple[str, str, float]] = [
                (e.dst, e.type, e.weight)
                for e in store.edges_from_in_scope(current_id, scope)
                if e.type not in REVERSE_TRAVERSED_EDGE_TYPES
            ]
            neighbours += [
                (e.src, e.type, e.weight)
                for e in store.edges_to_in_scope(current_id, scope)
                if e.type in REVERSE_TRAVERSED_EDGE_TYPES
            ]
            # Determinism: rank by (-edge-type-weight, -edge.weight,
            # neighbour id). Filter already-visited ids BEFORE ranking so
            # the top-k slice is over genuinely-novel candidates.
            candidates = [n for n in neighbours if n[0] not in visited]
            ordered = sorted(
                candidates,
                key=lambda n: (
                    -BFS_EDGE_WEIGHTS.get(n[1], 0.0),
                    -n[2],
                    n[0],
                ),
            )
            # Deduplicate by neighbour id BEFORE the top-k slice, keeping
            # the strongest edge to each. Two edges in one hop can name
            # the same neighbour — different types between one pair are
            # permitted by the `(src, dst, type)` PK, and since #1170 an
            # outbound edge and a reverse-traversed inbound one can also
            # collide. Slicing first would let the duplicate consume a
            # slot and drop an otherwise-eligible neighbour, underfilling
            # the hop. That is not exotic: `resolve_contradiction` writes
            # SUPERSEDES between a pair that already carries CONTRADICTS,
            # and those are the two highest weights in the table, so the
            # duplicate reliably lands at the top of the ranking.
            ranked: list[tuple[str, str, float]] = []
            seen_this_hop: set[str] = set()
            for cand in ordered:
                if cand[0] in seen_this_hop:
                    continue
                seen_this_hop.add(cand[0])
                ranked.append(cand)
                if len(ranked) >= nodes_per_hop:
                    break
            for neighbour_id, edge_type, _edge_weight in ranked:
                if nodes_used >= total_budget:
                    break
                if neighbour_id in visited:
                    # Defence in depth. `candidates` is filtered against
                    # `visited` before ranking and `ranked` is deduped
                    # within the hop, so nothing should reach here —
                    # emitting a duplicate would return the same belief
                    # twice and charge the node budget twice.
                    continue
                edge_w = BFS_EDGE_WEIGHTS.get(edge_type, 0.0)
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
                visited.add(neighbour_id)
                belief = store.get_belief_in_scope(neighbour_id, scope)
                if belief is None:
                    # Race: belief was deleted between the edge read
                    # and get_belief. Skip; the next mutation cycle
                    # will fire the cache invalidation that re-runs
                    # this query.
                    continue
                new_path = path + [edge_type]
                new_trail = trail + (neighbour_id,)
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
                    (
                        neighbour_id, new_score, depth + 1,
                        new_path, new_trail, scope,
                    )
                )
                nodes_used += 1
        frontier = next_frontier

    expanded.sort(key=lambda s: (-s.score, s.belief.id))
    # Marker-edge demotion (#1207). `BFS_EDGE_WEIGHTS` pins
    # POTENTIALLY_STALE at 0.0 on the grounds that demotion happens in
    # this pass — which had no importer, so nothing demoted a stale
    # belief anywhere. Wired unconditionally rather than behind a lane
    # flag because the *producer* is already the opt-in: `aelf doctor
    # --detect-stale` defaults off, and a store with no marker edges
    # makes this an identity (no penalty fires, and the re-sort uses
    # the same `(-score, belief.id)` key already applied above).
    #
    # Deferred import: `edge_rerank` imports `ScoredHop` from this
    # module, so a top-level import here would be circular.
    from aelfrice.edge_rerank import apply_edge_type_rerank

    return apply_edge_type_rerank(expanded, store)
