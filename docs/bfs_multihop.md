# BFS multi-hop graph traversal

**Status:** spec.
**Target milestone:** v1.3.0.
**Tracking issue:** [#144](https://github.com/robotrocketscience/aelfrice/issues/144).
**Dependencies:** stdlib only. Reads the v1.2 edge schema (`anchor_text`,
`session_id`, `DERIVED_FROM`). Soft dependency on the v1.3.0
[entity index](#) (#143) for richer tier-0 hits.
**Risk:** medium. New retrieval tier behind a default-off toggle.
Pure read path — no schema change, no ingest change.

## Summary

Edge-type-weighted bounded BFS expansion layered on top of the L0
locked-belief auto-load + L1 BM25 + (concurrently shipping) L2.5
entity-index hits. From each tier-0 belief that survives the prior
tiers' budget cuts, walk outward along graph edges up to a bounded
depth and a bounded node-expansion budget, scoring each visited
belief by the product of edge weights along the path that reached it.
Return the top-scoring expansion beliefs as a new L3 tier, packed
into the unified retrieval token budget below the L0/L1/L2.5 tiers.

The aim is **decisional adjacency**: when the user's query keyword-
matches a leaf belief, the actually-relevant decision (a supersession,
a contradiction, a derivation root) often sits 1–2 hops away on the
graph. v1.0 retrieval cannot reach it; v1.3 can.

## Motivation

Per [ROADMAP.md § v1.3.0](ROADMAP.md), the v1.3 retrieval wave moves
beyond BM25-only. BFS is the structural retrieval layer of that wave:
neither HRR (parked for v2.0) nor embeddings (out of scope per
[PHILOSOPHY § Determinism is the property](PHILOSOPHY.md)) — instead,
the typed-edge graph that v1.2 ingest now densely populates is walked
directly.

The v1.2.0 release made this viable. Before v1.2 the edge population
was sparse: the v1.0 onboard scanner produces beliefs with content
but few typed edges. v1.2 added the commit-ingest hook, the
transcript-ingest hook, and the regex triple extractor, all of which
densely populate `SUPPORTS`, `DERIVED_FROM`, `CONTRADICTS`,
`SUPERSEDES`, `RELATES_TO`, and `CITES` edges with `anchor_text` from
the citing prose. A real corpus by mid-v1.2.x has enough graph
structure for BFS to find non-trivial chains.

## Scope

### In scope

- New L3 retrieval tier: edge-weighted BFS over outbound edges from
  tier-0 seeds.
- Edge-type weight table (below) calibrated to bias toward
  decisional edges.
- Concrete depth and budget caps.
- Cycle detection via per-query visited-set.
- Integration with the unified retrieval token budget.
- Cache invalidation rule for `RetrievalCache` extended to cover
  edge mutations (already covered by the v1.0.1 wipe-on-write
  policy — see [§ Cache invalidation](#cache-invalidation)).
- Default-off toggle (`use_bfs=True`) in `retrieve_v2`. Default-on
  candidate at v2.0.0 once benchmark uplift is confirmed.
- Regression test on a fixture graph with known multi-hop chains.

### Out of scope

- **Temporal-coherence fix.** Each hop resolves to the globally
  latest serial of its target belief independently. This is a known
  limitation of v1.3. See [§ Open question: temporal coherence](#open-question-temporal-coherence)
  for the decision and [LIMITATIONS § BFS multi-hop temporal coherence](LIMITATIONS.md#bfs-multi-hop-temporal-coherence)
  for the user-facing carry-forward.
- **Inbound edge traversal.** BFS only walks outbound edges
  (`edges WHERE src = ?`). Reverse-direction walks (e.g. "what
  beliefs cite this one?") are a separate retrieval path; they are
  not unified into the same BFS frontier at v1.3.
- **Cross-graph / cross-store walks.** One DB at a time, per
  [LIMITATIONS § Multi-project query](LIMITATIONS.md#multi-project-query).
- **Edge-weight calibration against MAB.** v1.3 ships the literature-
  default weights below; calibration delta is a v1.3.x patch if the
  benchmark indicates one is needed.
- **New edge types.** The issue body listed `IMPLEMENTS` and
  `THREADS_TO`; `THREADS_TO` is a v1.1.0 user-facing rename of the
  `edges` tab/key, not an edge type, and remains out of scope.
  `IMPLEMENTS` has now landed as a v2.0 Track A edge (#385).

## Algorithm

### Pseudocode

```python
def bfs_expand(
    store: MemoryStore,
    seeds: list[Belief],          # tier-0 + tier-1 + tier-2.5 hits
    max_depth: int = 2,
    nodes_per_hop: int = 16,
    total_budget_nodes: int = 32,
    min_path_score: float = 0.10,
) -> list[ScoredBelief]:
    visited: set[str] = {b.id for b in seeds}
    # Frontier entries: (belief_id, path_score, depth, path_edges)
    frontier: list[tuple[str, float, int, list[str]]] = [
        (b.id, 1.0, 0, []) for b in seeds
    ]
    expanded: list[ScoredBelief] = []
    nodes_used: int = 0

    while frontier and nodes_used < total_budget_nodes:
        next_frontier: list[tuple[str, float, int, list[str]]] = []
        for current_id, score, depth, path in frontier:
            if depth >= max_depth:
                continue
            edges = store.edges_from(current_id)
            # Rank candidate edges by edge-type weight DESC, then
            # by stored edge.weight DESC, then by dst id ASC for
            # determinism. Take the top nodes_per_hop after filtering
            # already-visited dsts.
            ranked = sorted(
                (e for e in edges if e.dst not in visited),
                key=lambda e: (
                    -BFS_EDGE_WEIGHTS.get(e.type, 0.0),
                    -e.weight,
                    e.dst,
                ),
            )[:nodes_per_hop]
            for e in ranked:
                if nodes_used >= total_budget_nodes:
                    break
                edge_w = BFS_EDGE_WEIGHTS.get(e.type, 0.0)
                if edge_w == 0.0:
                    continue
                new_score = score * edge_w
                if new_score < min_path_score:
                    continue
                visited.add(e.dst)
                belief = store.get_belief(e.dst)
                if belief is None:
                    continue
                expanded.append(
                    ScoredBelief(
                        belief=belief,
                        score=new_score,
                        depth=depth + 1,
                        path=path + [e.type],
                    )
                )
                next_frontier.append(
                    (e.dst, new_score, depth + 1, path + [e.type])
                )
                nodes_used += 1
        frontier = next_frontier

    expanded.sort(key=lambda s: (-s.score, s.belief.id))
    return expanded
```

### Properties

- **Determinism.** All ties (in edge ranking, in result ordering)
  break on belief id ascending. Two BFS runs with the same store
  contents and same seeds produce identical output. Required by
  [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md).
- **Budget-bounded.** Hard caps on depth (default 2), per-hop fanout
  (default 16), and total expanded nodes (default 32). The walk
  cannot run away even on a densely-connected component.
- **Outbound only.** Reads `edges_from(src)`. Reverse traversal is
  out of scope (see [§ Out of scope](#out-of-scope)).
- **Path score is multiplicative.** A two-hop path scores
  `w(edge_1) * w(edge_2)`, capped above at 1.0 (no edge weight
  exceeds 1.0). Decisional chains (`SUPERSEDES * SUPERSEDES = 0.81`)
  beat informational chains (`RELATES_TO * RELATES_TO = 0.09`)
  decisively.
- **Pruning.** A path falls below `min_path_score=0.10` either by
  accumulating low-weight edges or by going too deep. The default
  cuts off `RELATES_TO`-only chains at depth 1 (0.30) and depth 2
  (0.09 — pruned). Decisional chains survive to depth 2.

## Edge-type weight table

The weight table biases the BFS frontier toward **decisional**
edges (a supersession or contradiction is almost always the
relevant context for a query that hits the superseded belief)
over **informational** edges (`RELATES_TO` is the catch-all and
is the noisiest signal in the v1.2 ingest output).

| Edge type      | Weight | Class          | Rationale |
|----------------|--------|----------------|-----------|
| `SUPERSEDES`   | 0.90   | decisional     | "B replaces A" — the most actionable adjacency. If the query hit A, the user almost certainly wants B. Highest weight. |
| `CONTRADICTS`  | 0.85   | decisional     | "B disagrees with A" — surfacing it lets the agent flag the conflict instead of acting on a contradicted belief. Slightly below SUPERSEDES because contradictions are not always resolved (the v1.0.1 contradiction tie-breaker may not have fired yet). |
| `DERIVED_FROM` | 0.70   | provenance     | "B's content depends on A" — strong contextual coupling, per the v1.2 ingest enrichment spec ("sibling becomes stale if A is superseded"). Following it surfaces parent decisions. Triple extractor produces `DERIVED_FROM` from "X is derived from Y" / "X is based on Y" / "X extends Y". **Retroactive ship-gate (#388):** shipped pre-bench-gate at v1.2; now must clear the same ≥+5pp BFS multi-hop hit@k uplift bar as the other Track A edges per #382 ratification; gate harness at `tests/bench_gate/test_bfs_multihop_derived_from.py`. Below-floor closes #388 as `wontfix`. |
| `IMPLEMENTS`   | 0.65   | provenance     | "B implements A" — source is an implementation, target is the spec/claim being implemented. Slightly below DERIVED_FROM (0.70) because IMPLEMENTS is a more specific kind of derivation, but the dependency is almost as strong: an implementation becomes stale when its spec is superseded. Triple extractor produces `IMPLEMENTS` from "X implements Y" / "X is an implementation of Y" / "X realizes Y" / "X fulfills Y". **v2.0 ship-gate (#385):** the edge stays at weight 0.65 only while it clears a ≥+5pp BFS multi-hop hit@k uplift on the labeled `implements_edge/` corpus vs. the same fixture run with this entry zeroed; gate harness lives at `tests/bench_gate/test_bfs_multihop_implements.py`. Below-floor closes #385 as `wontfix`. |
| `SUPPORTS`     | 0.60   | evidential     | "B argues for A" — supporting evidence is useful adjacent context but lower-priority than provenance or supersession. |
| `TESTS`        | 0.55   | evidential     | "B is a test of A" — source is a test belief, target is the spec/claim under test. Placed just below SUPPORTS (0.60) because a test asserts coverage of a claim, slightly weaker than direct argumentation. Triple extractor produces `TESTS` from "X tests Y" / "X is a test for Y" / "X is test of Y" / "X covers Y". **v2.0 ship-gate (#384):** the edge stays at weight 0.55 only while it clears a ≥+5pp BFS multi-hop hit@k uplift on the labeled `tests_edge/` corpus vs. the same fixture run with this entry zeroed; gate harness lives at `tests/bench_gate/test_bfs_multihop_tests.py`. Below-floor closes #384 as `wontfix`. |
| `CITES`        | 0.40   | referential    | "B mentions A" — weakest of the explicitly-relational edges. Per v1.0 `EDGE_VALENCE`, CITES is half of SUPPORTS for valence propagation; mirror that here. |
| `RELATES_TO`      | 0.30   | informational  | The catch-all. Triple extractor produces `RELATES_TO` from "X relates to Y" / "X is related to Y" — the loosest relational verbs. Low weight, but non-zero so densely-connected `RELATES_TO` neighborhoods can still surface the *single* highest-`bm25` hit they reach. **v2.0 ship-gate (#383):** the edge stays at weight 0.30 only while it clears a ≥+5pp BFS multi-hop hit@k uplift on the labeled `bfs_relates_to/` corpus vs. the same fixture run with this entry zeroed; gate harness lives at `tests/bench_gate/test_bfs_multihop_relates_to.py`. Per #382 Decision A2 (operator ratification 2026-05-04) the universal +5pp bar replaced the umbrella's proposed +3pp floor — `RELATES_TO` is the most generic catch-all and has the highest over-fit risk among Track A edges. Below-floor closes #383 as `wontfix`. |
| `TEMPORAL_NEXT`   | 0.25   | structural     | "B is the chronological successor of A" — pure structural adjacency with no evidential or argumentative content. Placed just below RELATES_TO (0.30) because RELATES_TO at least implies topical connection; temporal ordering does not. Non-zero so chains of temporally adjacent beliefs can still reach the single closest content hit. **v2.0 ship-gate (#386):** the edge stays at weight 0.25 only while it clears a ≥+5pp BFS multi-hop hit@k uplift on the labeled `temporal_next_edge/` corpus vs. the same fixture run with this entry zeroed; gate harness lives at `tests/bench_gate/test_bfs_multihop_temporal_next.py`. Per #382 Decision A2 the universal +5pp bar applies. Below-floor closes #386 as `wontfix`. |

**Retroactive bench gate for `DERIVED_FROM` (#388).** `DERIVED_FROM` shipped
at v1.2.0 as part of the ingest enrichment wave, before the #382 Track A
ship-gate ratification. Per #382 Decision A2 (operator ratification
2026-05-04), all Track A edges must demonstrate ≥+5pp BFS multi-hop hit@k
uplift — no audit-only escape hatch (Decision A3). The retroactive gate
harness lives at `tests/bench_gate/test_bfs_multihop_derived_from.py` and
reads labeled fixtures from `tests/corpus/v2_0/derived_from_edge/`. The
edge remains at weight 0.70 while the gate is pending; below-floor closes
#388 as `wontfix` and the weight entry drops to 0.0.

Constants live in `src/aelfrice/retrieval.py` next to the BFS
implementation, not in `models.py` — the BFS weights are a
retrieval-tier hyperparameter, distinct from `EDGE_VALENCE` (which
governs feedback propagation through `propagate_valence`). Keeping
them separate prevents a benchmark-driven retrieval re-tune from
silently changing valence semantics.

```python
# src/aelfrice/bfs_multihop.py
BFS_EDGE_WEIGHTS: dict[str, float] = {
    EDGE_SUPERSEDES:    0.90,
    EDGE_CONTRADICTS:   0.85,
    EDGE_DERIVED_FROM:  0.70,
    EDGE_IMPLEMENTS:    0.65,
    EDGE_SUPPORTS:      0.60,
    EDGE_CITES:         0.40,
    EDGE_RELATES_TO:    0.30,
    EDGE_TEMPORAL_NEXT: 0.25,
}
```

### Why these are not the `EDGE_VALENCE` numbers

`EDGE_VALENCE` (in `models.py`) is calibrated for **valence
propagation** through `propagate_valence`: positive values amplify a
feedback signal in the direction of the edge, negative values invert
it (a `CONTRADICTS` edge propagates *negative* valence — penalizing
the contradicted belief), and `SUPERSEDES` deliberately propagates
0.0 (the superseded belief should not gain or lose confidence merely
because its replacement was reinforced).

For BFS retrieval the calculus is different: `CONTRADICTS` is a
**high-relevance** adjacency (we want to surface it), not a negative-
valence carrier. `SUPERSEDES` is the **highest-relevance** adjacency,
not a structural-zero. Reusing `EDGE_VALENCE` would therefore actively
mis-score retrieval. Two tables, two purposes.

## Depth cap and budget

Defaults, with the rationale that gives them concrete numbers
(the issue body says "bounded depth, bounded budget" without
specifying):

| Knob                  | Default | Rationale |
|-----------------------|---------|-----------|
| `max_depth`           | **2**   | Empirically, the research-line MAB chain-valid baseline finds nearly all decisional chains within 2 hops; depth-3 doubles cost for marginal recall on the v1.2 corpus density. |
| `nodes_per_hop`       | **16**  | Caps fanout per frontier entry. A v1.2 belief with 100+ outbound `RELATES_TO` edges (densely-connected hub) would otherwise explode the frontier. 16 is the top-k after edge-weight ranking. |
| `total_budget_nodes`  | **32**  | Hard cap on total expanded beliefs across all hops. At depth 2 with seeds=8 and per-hop=16 the worst case without this cap is 8 + 8\*16 + 8\*16\*16 = 2,184; the cap brings worst-case wall time predictable. |
| `min_path_score`      | **0.10**| Prunes chains whose multiplicative score has decayed below the noise floor. Default cuts `RELATES_TO`-only depth-2 chains (0.09) and forces decisional adjacency to dominate. |
| `token_budget_share`  | **shared** | BFS pulls from the same `token_budget` as L0/L1/L2.5; no separate budget. See [§ Pipeline integration](#integration-with-l1-and-l25). |

All five are kwargs on `retrieve_v2`. Defaults are conservative —
the toggle ships off; users opting in get the values above. The
v1.3.0 release notes will publish the latency band the defaults
produce on the v1.0.0 baseline corpus.

## Cycle detection

**Decision: visited-set per query.**

Two options were considered:

**(a) Per-query visited-set.** Maintain `visited: set[str]` for the
duration of one BFS call; before adding a dst to the next frontier,
check membership. O(1) amortized lookup; trivial cycle prevention;
matches the existing pattern in `MemoryStore.propagate_valence`
(`store.py:617`), which already uses this approach in the codebase.

**(b) LIMIT-based truncation.** Rely on `total_budget_nodes` to
implicitly terminate any cycle: a cycle would be discovered as a
self-loop in the frontier and would simply consume budget until the
cap fires. Simpler code, no set bookkeeping, but produces
non-deterministic and surprising output: the same query against the
same store could surface different beliefs depending on which path
into the cycle was taken first, and a cycle on a high-weight edge
type would crowd out genuinely-distinct beliefs.

**Picking (a)** for three reasons:

1. **Determinism.** Visited-set + path-score-tie-broken-by-id
   guarantees one BFS run = one output. Option (b) does not.
2. **Precedent.** `propagate_valence` already uses option (a).
   Picking the same pattern keeps the graph-walk surface coherent
   across the codebase; a future reader sees one cycle-prevention
   idiom, not two.
3. **Cost.** A `set[str]` of belief ids capped at `total_budget_nodes
   + len(seeds)` (worst case ~64 entries) is sub-microsecond
   overhead. Not a budget concern.

The visited-set is initialized from the seed belief ids so a seed
cannot re-enter the frontier as an expansion result (and so a
seed's outbound edges to another seed are not counted as expansion
nodes — they are already in tier 0).

## Integration with L1 and L2.5

BFS is **L3**, sitting after the prior tiers in the unified pipeline:

```
                          ┌──────────────────────────────────────┐
  query ──> retrieve() ──>│ L0: list_locked_beliefs()            │
                          │ L1: store.search_beliefs(query, ...) │
                          │ L2.5: entity_index.lookup(query, ...)│  (#143)
                          │ L3: bfs_expand(seeds=L0+L1+L2.5, ...) │  (this spec)
                          └──────────────────────────────────────┘
                                          │
                                  token-budget pack
                                          │
                                          ▼
                                 list[Belief]
```

### Seed selection

L3 seeds are the union of the L0, L1, and L2.5 result sets that
survived their own tier's filtering — i.e., the seeds are the
beliefs the user is *about to receive* from the prior tiers. This
choice has three consequences:

- **No double counting.** The visited-set is initialized from seed
  ids, so a seed never appears in the L3 expansion list.
- **Tier ordering preserved.** L0 sits above L1 sits above L2.5 sits
  above L3 in the final returned list — locks first, ground-truth
  retrieval next, structural expansion last.
- **L3 quality scales with prior tiers' quality.** A weak L1 hit
  becomes a weak BFS seed; the chain may still surface useful
  context, but the seed selection is the lever for "did BFS get the
  right starting point". This is by design — BFS is an expansion
  layer, not a recall layer.

### Budget allocation

**Decision: shared token budget, no separate L3 cap.** L3 results
are appended to the same packed-output list and consume the same
`token_budget` as the prior tiers, in the same order: L0 first
(never trimmed), then L1, then L2.5, then L3 expansions packed in
score-descending order until `token_budget` is reached. A query
where L0+L1+L2.5 already saturate the budget gets zero L3 output;
that is the correct conservative behavior (the user asked for
2,000 tokens, the user gets 2,000 tokens of the highest-tier hits).

Alternative considered: separate L3 budget (e.g., reserve 25% of
`token_budget` for BFS). Rejected because it can crowd out higher-
tier hits when the locks alone fit cleanly under budget, and it
adds a knob the user has to reason about. The simple "L3 fills
remaining budget" rule is what the v1.0 retrieval pipeline already
does for L1 (everything below L0 fills remaining budget); L3
inherits that contract.

### `retrieve_v2` plumbing

The existing `retrieve_v2` wrapper in `src/aelfrice/retrieval.py`
(currently a no-op for `use_bfs`) becomes the public surface. The
free-function `retrieve()` stays unchanged — calls without
`use_bfs=True` see no behavior change at v1.3.

```python
def retrieve_v2(
    store: MemoryStore,
    query: str,
    budget: int = DEFAULT_TOKEN_BUDGET,
    include_locked: bool = True,
    use_hrr: bool = False,
    use_bfs: bool = False,
    use_entity_index: bool = False,    # added in #143
    l1_limit: int = DEFAULT_L1_LIMIT,
    bfs_max_depth: int = 2,
    bfs_nodes_per_hop: int = 16,
    bfs_total_budget_nodes: int = 32,
    bfs_min_path_score: float = 0.10,
) -> RetrievalResult:
    ...
```

`RetrievalResult.bfs_chains` (already a placeholder list field on the
v1.0.x `RetrievalResult`) is populated by L3 with the edge-type path
that reached each expansion belief, suitable for downstream auditors
and for benchmark scoring.

## Cache invalidation

The v1.0.1 `RetrievalCache` is keyed on `(canonicalized_query,
token_budget, l1_limit)` and is wiped on any belief or edge
mutation. BFS expands the cached value (the result list) but does
**not** change the key, because:

- The seeds are determined by the query + budget + l1 limit (already
  in the key).
- The expansion is a pure function of (seeds, current edge graph,
  current belief contents).
- The current edge graph is mutated through `insert_edge`,
  `update_edge`, `delete_edge` — all three already fire
  `_fire_invalidation()` (verified in `store.py:559,576,584`).
- The current belief contents are mutated through `insert_belief`,
  `update_belief`, `delete_belief` — all three already fire
  invalidation.

**Conclusion: no cache-key change required.** The wipe-on-write
policy already covers BFS correctness. A v1.3 store that adds a new
edge between two cached beliefs will wipe the cache, the next
identical query will re-run the full pipeline including BFS, and
the new edge will participate.

This was the load-bearing reason `RetrievalCache` was specified
with edge mutators in the invalidation set at v1.0.1 — see
[lru_query_cache.md § Invalidation](lru_query_cache.md#invalidation).
The v1.0.1 spec said: "A finer-grained policy ... is a later
optimization. v1.0.1 ships the wipe-on-write version." v1.3.0
inherits that decision.

The v1.3 BFS kwargs (`bfs_max_depth`, `bfs_nodes_per_hop`,
`bfs_total_budget_nodes`, `bfs_min_path_score`) **are not added to
the cache key**: callers that toggle them per-call would defeat the
cache anyway, and the default-off `use_bfs` toggle means a single
process either uses BFS for every retrieval or none. If a future
caller wants per-call BFS knobs and cache hits across them, that's
a finer-grained-key follow-up; v1.3 ships with the simple key.

## Open question: temporal coherence

> Per ROADMAP / issue: each hop currently resolves to the globally
> latest serial of its target belief independently; this misses
> chains where intermediate hops should follow earlier serials.

**Decision: option (a) — v1.3 ships with the limitation documented;
fix targeted at v2.0.**

### What "temporal coherence" means here

When a belief A is superseded by A', and A' is superseded by A'',
the BFS frontier walking outbound from a tier-0 hit on A naively
surfaces A'' as the latest. That is correct *for that one hop*. But
when the seed is itself a session-scoped belief from session S₁ and
the chain has crossed `SUPERSEDES` boundaries that postdate S₁, the
"latest serial" the agent receives may not be the one that was
canonical at S₁'s timestamp. For audit-shaped queries ("what did the
agent believe at the time it made this decision?") this is a
fidelity loss.

### Why defer

1. **The data is barely there yet.** Session-coherent supersession
   reads `Belief.session_id` plus `SUPERSEDES` edges. The session
   column landed in v1.2.0; the dense edge population landed in
   v1.2.0. v1.3 retrieval-side code would be reading a corpus that
   has at most one minor-version of dense ingest history. The
   fix-now case fights data sparsity.
2. **The fix is non-trivial.** Correctly resolving "what was the
   canonical chain at timestamp T" requires walking the
   `SUPERSEDES` chain *backwards* from each hop's target until the
   `created_at` is ≤ T, *or* requires materializing per-session
   supersession views. Both are larger scope than a v1.3.0 retrieval
   wave that already includes entity-index, BFS, LLM-classification,
   and partial posterior-weighted ranking.
3. **The default mode is recall, not audit.** The vast majority of
   retrieval calls ask "what's the current best context?" — the
   globally-latest-serial behavior is the *correct* answer for that
   query class. The audit-shaped query is the minority case.
4. **Regression risk.** Adding a temporal filter to BFS could mask
   genuinely-relevant later context for the common-case query, and
   the benchmark to detect that masking does not yet exist.
5. **No small fix is obvious.** A "small" fix candidate would be
   "filter expansion beliefs to those with `created_at` ≤ seed's
   `created_at`," which is wrong: it would prune supersessions that
   are *the whole point* of the BFS expansion (the agent wants to
   see "this belief was later replaced by X"). There is no
   one-liner; the correct fix needs design.

### What lands

- v1.3.0 ships option (a): documented limitation, BFS resolves each
  hop to the globally latest serial.
- A new entry in [LIMITATIONS.md § BFS multi-hop temporal
  coherence](LIMITATIONS.md#bfs-multi-hop-temporal-coherence)
  pointing at this spec.
- The fix target moves to v2.0.0, scoped under "feature parity and
  reproducibility." The v2.0 work that lands HRR, type-aware
  compression, and the full feedback-into-ranking eval is the
  natural milestone for a temporal-coherence rework, because v2.0 is
  also where the benchmark suite reproduces published numbers — the
  measurement instrument that lets a temporal-coherence change be
  evaluated for regression.
- An optional `as_of_session_id` kwarg on `retrieve_v2` is **not**
  added at v1.3 — adding the kwarg without the implementation would
  be dishonest API surface. v2.0 lands both together.

### Acceptance signal

The v1.3 BFS regression test fixture (below) explicitly includes a
chain whose latest-serial hop is *correct* under the documented
behavior. The test's pass/fail criterion is "BFS finds the chain
under the v1.3 contract"; a future v2.0 test with a temporal-
coherence fixture will exercise the new contract on top.

## Validation

### Regression test: `tests/test_bfs_multihop.py`

Fixture graph with three known multi-hop chains:

1. **Decisional chain (depth 2).** Seed S₀ → `RELATES_TO` → S₁ →
   `SUPERSEDES` → S₂. The test asserts S₂ surfaces with score
   ≥ 0.30 \* 0.90 = 0.27 (above `min_path_score`), and S₁ surfaces
   above S₂.
2. **Pruned informational chain (depth 2).** Seed P₀ → `RELATES_TO`
   → P₁ → `RELATES_TO` → P₂. The test asserts P₂ does NOT surface
   (score 0.30 \* 0.30 = 0.09 < 0.10 floor), and P₁ does (score
   0.30).
3. **Contradiction surfacing (depth 1).** Seed C₀ → `CONTRADICTS` →
   C₁. The test asserts C₁ surfaces with score 0.85 and is ranked
   above any depth-2 expansion from a different seed.

### Latency budget regression band

The v1.3 release publishes the L0+L1+L2.5+L3 retrieval latency
against the v1.0.0 baseline corpus and the v1.2.0 ingest-enriched
corpus, on the same hardware as previous releases. Acceptance band:
**p50 ≤ 25 ms, p95 ≤ 100 ms** for a 10k-belief / 25k-edge store
with `use_bfs=True`. p50/p95 with `use_bfs=False` (the default)
must not regress against the v1.2.0 numbers at all.

### Determinism test

Two BFS runs against the same fixture, same seeds, same kwargs,
must produce identical output (id-tied ordering, same scores, same
expansion paths). Verifies the determinism property explicitly.

### Cache-invalidation test

1. Run `retrieve_v2(..., use_bfs=True)` on a fixture, assert the
   chain surfaces.
2. Insert a new `SUPERSEDES` edge into the fixture.
3. Re-run the same retrieval; assert the cache was wiped and the
   new edge participates in the L3 expansion.

### Toggle test

`retrieve_v2(..., use_bfs=False)` produces the v1.2.0 baseline
output exactly — same beliefs, same order. BFS is gated.

## Acceptance criteria for the implementation PR

1. `BFS_EDGE_WEIGHTS` constant lands in `src/aelfrice/retrieval.py`
   with the values in this spec.
2. `bfs_expand()` function implemented per the pseudocode above,
   stdlib-only.
3. `retrieve_v2` accepts `use_bfs` and the four BFS kwargs; routes
   through `bfs_expand` when `use_bfs=True`; populates
   `RetrievalResult.bfs_chains` with the expansion edge-type paths.
4. `tests/test_bfs_multihop.py` covers the three fixture chains
   (decisional, pruned informational, contradiction), determinism,
   cache invalidation, and toggle off.
5. `tests/test_bfs_multihop.py` includes the temporal-coherence
   fixture asserting the v1.3 contract (latest-serial-per-hop).
   Not the v2.0 contract.
6. Latency benchmark added to `benchmarks/` matching the bands in
   [§ Latency budget regression band](#latency-budget-regression-band).
7. `LIMITATIONS.md` updated with the temporal-coherence entry per
   [§ Open question: temporal coherence](#open-question-temporal-coherence).
8. `RetrievalCache` test re-asserts wipe-on-edge-mutation behavior
   end-to-end with `use_bfs=True`.
9. `docs/ROADMAP.md § v1.3.0` cross-links this spec.
10. CI green on the staging-gate pytest matrix; no new dependencies.

## Dependencies

- **v1.2.0 ingest enrichment** (`anchor_text`, `session_id`,
  `DERIVED_FROM`) — already shipped. Required: BFS reads `edges`
  and benefits from dense edge population, but does not read
  `anchor_text` directly at v1.3 (anchor-aware re-ranking is
  follow-up work).
- **v1.0.1 RetrievalCache** — already shipped. The wipe-on-write
  policy is what makes the v1.3 cache correctness story
  zero-effort.
- **#143 (entity-index) — soft.** BFS layers on top of L2.5 hits.
  If #143 lands first, BFS seeds include L2.5 results; if it lands
  after, BFS seeds are L0+L1 only and the L2.5 path joins later.
  Order-independent.
- **stdlib only.** No new third-party dependencies. BFS is pure
  Python over the existing `MemoryStore.edges_from()` API.

## Limitations carried forward

- **Temporal coherence.** Resolved above; carry-forward entry in
  [LIMITATIONS.md](LIMITATIONS.md#bfs-multi-hop-temporal-coherence).
  Targeted at v2.0.0.
- **Outbound-only traversal.** Reverse-direction walks are out of
  scope. A query whose only useful adjacency is "what cites this?"
  is not served by v1.3 BFS. Targeted: undecided; opens at v2.0
  for review against benchmark recall.
- **Edge-weight calibration is literature-default, not corpus-
  calibrated.** v1.3 ships the table above. A v1.3.x patch can
  re-tune against the MAB benchmark if the headline number
  benefits.
- **No anchor-aware re-ranking.** `Edge.anchor_text` is populated
  but BFS does not use it. The straightforward anchor-aware
  augmentation (boost an expansion belief's score if its incoming
  edge's `anchor_text` substring-matches the query) is parked for
  follow-up; v1.3 spec is structural-only.
- **One DB at a time** — the project-scope rule from
  [LIMITATIONS § Multi-project query](LIMITATIONS.md#multi-project-query)
  applies unchanged.

## Open questions resolved by this spec

| Question (from issue / ROADMAP)                  | Resolution |
|--------------------------------------------------|------------|
| Edge-type weights — concrete table?              | Yes, six edges, decisional > provenance > evidential > informational. See [§ Edge-type weight table](#edge-type-weight-table). |
| Depth cap?                                       | `max_depth = 2`. |
| Budget cap (per-hop fanout)?                     | `nodes_per_hop = 16`. |
| Budget cap (total expansion)?                    | `total_budget_nodes = 32`. |
| Path-score floor?                                | `min_path_score = 0.10`. |
| Cycle detection — visited-set or LIMIT?          | Visited-set, per-query. See [§ Cycle detection](#cycle-detection). |
| Temporal coherence — fix at v1.3 or document?    | Document, fix at v2.0. See [§ Open question: temporal coherence](#open-question-temporal-coherence). |
| BFS budget shared with L0/L1/L2.5 or separate?   | Shared. See [§ Budget allocation](#budget-allocation). |
| Cache invalidation rule?                         | No cache-key change; v1.0.1 wipe-on-write covers it. See [§ Cache invalidation](#cache-invalidation). |
| Default-on or default-off at v1.3?               | Default-off. Default-on candidate at v2.0 with benchmark uplift. |
| `IMPLEMENTS` edge type?                          | Shipped as a v2.0 Track A edge (#385) at weight 0.65. `THREADS_TO` remains out of scope. |
