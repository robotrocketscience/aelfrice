# Feature spec: Intentional clustering (#436)

**Status:** wired into `retrieve_v2` behind `use_intentional_clustering`; default-ON since v3.0 after the #436 R6 A4 latency gate cleared 60/60 PASS at p99 0.328ms on the multi-store production sweep
**Issue:** #436
**Recovery-inventory line:** [`docs/concepts/ROADMAP.md`](../concepts/ROADMAP.md) — *"Intentional clustering | v2.0.0"*
**Substrate prereqs:** edge graph (foundation), `dedup.DuplicateCluster` union-find pattern (`src/aelfrice/dedup.py:155-185`, shipped #197), heat kernel authority (#150, shipped v1.7.0), BFS multi-hop (#143, shipped v1.3.0)

---

## Purpose

When a query needs more than one belief to answer (e.g. *"how do we deploy + what are the prerequisites?"*), the existing rank+pack pipeline can return K beliefs that all sit in the same graph neighbourhood — the highest BM25F + heat-kernel scores cluster around one topic, and a complementary cluster covering the second half of the query never makes the cut. Intentional clustering is a **retrieval-time pass** that biases the top-K output toward **distinct graph-connected clusters**, paying a small per-belief score cost in exchange for multi-fact coverage.

This is **not** the same mechanism as the heat kernel (#150). Heat kernel propagates authority *along* the graph from BM25 seeds — beliefs near a hot belief get a score boost. Clustering operates on the **already-ranked** candidate set: it changes which beliefs survive the pack, not their individual scores.

---

## Contract

```python
from aelfrice.clustering import RetrievalCluster, cluster_candidates, pack_with_clusters

@dataclass(frozen=True)
class RetrievalCluster:
    cluster_id: int                     # zero-indexed, dense
    member_ids: tuple[str, ...]         # belief ids, ordered by descending rank score
    representative_id: str              # member_ids[0] — the highest-ranked member
    seed_score: float                   # representative's pre-clustering rank score

# Step 1: identify clusters in the ranked candidate pool.
def cluster_candidates(
    candidates: list[Belief],
    candidate_scores: dict[str, float],
    *,
    edges: Iterable[Edge],
    edge_weight_floor: float = DEFAULT_CLUSTER_EDGE_FLOOR,
) -> list[RetrievalCluster]: ...

# Step 2: cluster-diverse pack at fixed token_budget.
def pack_with_clusters(
    clusters: list[RetrievalCluster],
    *,
    token_budget: int,
    cluster_diversity_target: int = DEFAULT_CLUSTER_DIVERSITY_TARGET,
    fallback_to_score: bool = True,
) -> list[Belief]: ...
```

The two halves are split so the union-find pass is testable in isolation. Production retrieval calls them in sequence:

```
candidates = lane_fan_out + rank      # existing behaviour
clusters    = cluster_candidates(candidates, candidate_scores, edges=...)
out         = pack_with_clusters(clusters, token_budget=token_budget, ...)
```

`out` replaces the existing pack-loop output (`retrieval.py:1048-1085, :1197-1232`) when the flag is on.

---

## Cluster definition

**Graph-connected components within the post-rank candidate pool, scoped to the edge subgraph induced by the candidates.**

Concretely: take the top-N ranked candidates (where N is the existing pack-loop's seed pool size). Build a subgraph where vertices are the candidate belief IDs and edges are the elements of `store.edges_for_beliefs(candidate_ids)` whose `weight ≥ edge_weight_floor`. Run union-find over the edge list (the same primitive `dedup.py:248` calls path-compressed union-find on). Each connected component is one `RetrievalCluster`; isolated candidates are singletons.

### Why graph-connected, not topic-coherent or co-occurrence-derived

Three options were on the table per the issue acceptance:

| Option | Reasoning | Decision |
|---|---|---|
| **graph-connected** | Reuses the existing edge graph (the union of `SUPPORTS`, `CONTRADICTS`, `RELATES_TO`, `CITES`, `DERIVED_FROM` edges plus their tuned weights at `models.py:55-66`). No new substrate. Deterministic. | **chosen** |
| topic-coherent | Requires an embedding model or a topic classifier. The project's posture is no embeddings in retrieval — feedback memory `feedback_avoid_embeddings_nondeterminism`. Out. | rejected |
| co-occurrence-derived | "Belief A and belief B were both ranked in the top-K for query Q in the past" needs a query-event log indexed by belief co-rank. The `ingest_log` doesn't index that, and the `rebuild_logs` (#288) are not yet a queryable derived table. Could ship in v2.x once #288 grows a queryable view. | deferred |

The graph-connected definition is the cheapest, most deterministic, and uses substrate already on `main`.

### Edge-weight floor

`DEFAULT_CLUSTER_EDGE_FLOOR = 0.4` — picked to include `CITES` (`EDGE_VALENCE[EDGE_CITES] = 0.5` per `models.py:66`) but exclude `RELATES_TO` (`0.3` per `bfs_multihop.md` § Edge weights). Beliefs that only "relate to" each other are too weak a signal to be considered the same cluster; beliefs that cite each other are.

The floor is a tunable. Per-store override via `[retrieval] cluster_edge_weight_floor = 0.4` in `.aelfrice.toml`. Bench-evidence is the gate for changing the default.

---

## Pack algorithm

`pack_with_clusters` replaces the existing tail-trim with a **diversity-aware greedy fill**:

```
out: list[Belief] = []
used_tokens: int = 0
covered_clusters: set[int] = set()

# Stage 1: cluster representatives, in descending seed_score.
for cluster in sorted(clusters, key=lambda c: -c.seed_score):
    if len(covered_clusters) >= cluster_diversity_target:
        break
    rep = cluster.member_ids[0]    # representative
    cost = _belief_tokens(rep)
    if used_tokens + cost > token_budget:
        if fallback_to_score:
            break
        continue
    out.append(rep)
    used_tokens += cost
    covered_clusters.add(cluster.cluster_id)

# Stage 2: remaining budget — fill from the score-ranked tail.
score_ranked_remaining = [b for b in flattened_score_order(clusters)
                          if b.id not in {x.id for x in out}]
for b in score_ranked_remaining:
    cost = _belief_tokens(b)
    if used_tokens + cost > token_budget:
        break
    out.append(b)
    used_tokens += cost

return out
```

- **Stage 1** guarantees the top-K hits at least `cluster_diversity_target` distinct clusters (or as many as exist), one representative per cluster, paid for at the budget cost of the representatives.
- **Stage 2** fills the remaining budget by score, skipping already-included beliefs. This recovers single-fact behaviour on queries where the candidate pool has fewer clusters than the diversity target — single-fact recall is preserved by acceptance #3.
- **`fallback_to_score=True`** (default) — when a cluster representative does not fit in the remaining budget, abandon Stage 1 and let Stage 2 finish the pack from the score-ranked tail. This keeps the pack non-degenerate on tight budgets. Setting `False` causes the pack to skip-but-continue Stage 1, which is the strict-diversity mode for benchmarks; default keeps backward-compatible recall behaviour on single-cluster queries.

`DEFAULT_CLUSTER_DIVERSITY_TARGET = 3` — ships as the default. Three clusters covers most multi-fact queries without crowding out the score-ranked tail. Tunable via `[retrieval] cluster_diversity_target` in `.aelfrice.toml`.

---

## Where clustering sits

**Post-rank, pre-pack.** After lane fan-out + score composition, before the budget pack:

```
query
  -> lane fan-out (BM25F + heat kernel + HRR-structural + BFS)
  -> score composition + rank
  -> [cluster_candidates → pack_with_clusters] if use_intentional_clustering
  -> compose
```

When the flag is OFF, the existing pack loop runs unchanged. When ON, both `cluster_candidates` and `pack_with_clusters` run; `cluster_candidates` is a single union-find pass over the candidate edge subgraph, `pack_with_clusters` is the diversity-aware greedy fill above.

### Configuration

`use_intentional_clustering` follows the convention at `retrieval.py:118-131`:

1. `retrieve(..., use_intentional_clustering=True|False)` kwarg.
2. `AELFRICE_INTENTIONAL_CLUSTERING=1|0` env var.
3. `[retrieval] use_intentional_clustering = true|false` in `.aelfrice.toml`.
4. Default ON since v3.0 (#436 R6, 60/60 PASS at p99 0.328ms — ~15-30x margin under the 5ms A4 budget). Opt out via any of the three paths above for v2.0.x ranking parity.

Two adjacent knobs (also TOML-only, no env var):

- `[retrieval] cluster_edge_weight_floor` — default `0.4`.
- `[retrieval] cluster_diversity_target` — default `3`.

---

## Latency budget

Issue acceptance #4: *"Cluster pass runs inside the retrieve() budget; no separate query."* The cost decomposes into:

- **One `store.edges_for_beliefs(candidate_ids)` call.** Indexed lookup; existing pattern from BFS multi-hop. Cost is dominated by the `IN (...)` clause; at typical candidate-pool size (≤200) this is sub-ms in SQLite-backed numpy.
- **One union-find pass.** Path-compressed union-by-size, the same primitive `dedup.py:248` uses. O(α(N)·E) where α is the inverse Ackermann. At N=200 candidates and E ≤ 1000 inter-candidate edges, this is microseconds.
- **One score-ranked tail iteration.** Same as the existing pack loop, no algorithmic change.

Total expected overhead at N=200, E=500: **<1 ms**, well below the existing per-call budget (~7-8 ms heat kernel at N=50k per `graph_spectral.py:227`). The latency-budget acceptance is met without a benchmark — the algorithmic profile is bounded.

---

## Reconciliation

### vs. #150 (heat kernel)

Heat kernel **changes scores** by propagating authority along the graph from BM25 seeds. Beliefs near hot beliefs get score-boosted. Clustering **leaves scores untouched** but biases the **selection** of which K of the score-ranked candidates survive the pack.

The two compose. Heat kernel's input is the BM25 lane; its output is a per-belief authority score that participates in the rank. Clustering's input is the post-rank candidate pool; its output is the packed top-K. Different stages, different inputs, different outputs.

### vs. BFS multi-hop (#143)

BFS multi-hop **expands** the candidate pool: starting from a seed, walk N hops to surface beliefs that BM25 didn't find. Clustering **prunes** the candidate pool to a diverse subset. BFS adds candidates; clustering chooses among them.

When both are on (`bfs_enabled=True` AND `use_intentional_clustering=True`), BFS-discovered beliefs become candidates that the clusterer then groups. The two compose with no shared code path.

### vs. #197 (dedup) `DuplicateCluster`

`dedup.DuplicateCluster` (`dedup.py:155-185`) and this spec's `RetrievalCluster` are **different relations**:

| | `DuplicateCluster` (#197) | `RetrievalCluster` (#436) |
|---|---|---|
| input | Jaccard-prefiltered duplicate **pairs** | post-rank candidate edges |
| edge type | implicit (duplicate-pair predicate) | explicit (`SUPPORTS`/`CITES`/etc., weight ≥ floor) |
| consumer | dedup audit / SUPERSEDES write-path | retrieval pack |
| timing | offline audit | retrieval-time |

Both use union-find. The implementation PR may share the path-compressed union-find primitive — that is a refactor opportunity, not a requirement.

### vs. type-aware compression (#434)

Both are pre-pack transforms in the same stage. The order is:

```
rank → cluster_candidates → compress → pack_with_clusters
```

Compression reduces per-belief token cost; clustering biases selection toward cluster-diverse top-K. They compose: a `transient`-class belief in cluster X gets stub-compressed before being considered a representative; if its stub fits the budget, the cluster gets covered cheaply.

### vs. doc linker (#435)

Doc linker projects metadata onto returned beliefs (anchors). It does not change which beliefs are returned; clustering does. They sit at orthogonal stages: clustering at pack-time, linker at output-projection time. They compose with no interaction.

---

## Acceptance

### A1 — multi-fact corpus (issue acceptance #2)

A new labeled `multi_fact` corpus lives under `tests/corpus/v2_0/multi_fact/`. Each row encodes:

```jsonl
{"query": "...",
 "expected_belief_ids": ["b/...", "b/..."],
 "expected_clusters": [["b/..."], ["b/..."]],
 "n_clusters_required": 2,
 "tag": "complementary|conjunctive|sequential"}
```

`expected_clusters` partitions `expected_belief_ids` into the cluster groupings the labeller saw — this lets the bench distinguish "found both clusters" from "found two beliefs from the same cluster." Public CI runs with `AELFRICE_CORPUS_ROOT` unset and skips via the autouse `bench_gated` marker; labeled content lives in the lab repo only, per the published corpus policy.

### A2 — multi-fact recall uplift (issue acceptance #3a)

On the `multi_fact` fixture:

```
recall@k(use_intentional_clustering=ON)  >  recall@k(use_intentional_clustering=OFF)
cluster_coverage@k(ON)                    >  cluster_coverage@k(OFF)
```

`cluster_coverage@k` is a new metric: number of `expected_clusters` represented in the top-K, divided by `n_clusters_required`. Both inequalities must hold for ship; the second is the multi-fact-coherence claim, the first is the recall-doesn't-degrade backstop.

Threshold: **strictly positive uplift** on both. Magnitudes are bench-evidence-derived, not pre-set.

### A3 — single-fact non-regression (issue acceptance #3b)

On the existing `retrieve_uplift` fixture (the v0.1 single-fact fixture #154 used):

```
NDCG@k(use_intentional_clustering=ON, BM25F=ON) ≥ NDCG@k(use_intentional_clustering=OFF, BM25F=ON) - 0.005
```

Tolerance band matches the BM25F bench-gate band at #154 — half-point of NDCG noise floor.

### A4 — latency (issue acceptance #4)

A microbench under `tests/bench_gate/` measures `cluster_candidates + pack_with_clusters` wall-time at N ∈ {50, 100, 200} candidates. Assertion: **<5 ms p99** at N=200. The latency-budget claim under §"Latency budget" is the analytical bound; the microbench is the empirical confirmation.

### A5 — composition tracker

The #154 composition tracker doc gains a row for `use_intentional_clustering`: input shape, output shape, where it sits, bench verdict. Like the doc linker (#435) and type-aware compression (#434), this is **not a lane** — it is a pack-stage selection transform.

---

## Bench-gate / ship-or-defer policy

`needs-spec` → `bench-gated` once this spec lands. Implementation is the next gate. **The implementation PR ships only on positive bench evidence per A2 and non-regression per A3.** A mechanically-correct implementation that fails either ships merged-but-default-OFF or gets reverted; it does not ship default-ON without a benchmark cut.

A2 is the ship gate; A3 is the don't-break-existing-things backstop; A4 is a latency assertion that should pass by construction.

---

## Out of scope at v2.0.0

- **Topic-coherent clustering** (embedding-derived). No embeddings in retrieval; rejected.
- **Co-occurrence-derived clustering** (history-derived). Requires a queryable rebuild-logs view (#288 follow-up). Deferred to v2.x.
- **Cluster-aware ranking** (per-cluster score adjustment). Out of scope; this spec changes selection, not ranking. Per-cluster ranking is a different spec.
- **Streaming / incremental clustering** (online union-find as candidates arrive). The pack runs over a finished candidate list. Streaming is a future concern, not a v2.0 ship gate.
- **Cluster surface in `RetrievalResult`.** Returning the cluster assignment to consumers is not in this spec — only the packed beliefs are returned. If a consumer needs to know "which cluster did this belief come from," that is a follow-up surface (additional `cluster_assignments` parallel list, the way doc-anchors lands at #435).

---

## Implementation prereqs

- `src/aelfrice/dedup.py:248` — path-compressed union-find primitive to reuse or duplicate.
- `src/aelfrice/models.py:55-66` — `EDGE_VALENCE` weights informing the default `cluster_edge_weight_floor = 0.4`.
- `src/aelfrice/store.py` — needs `edges_for_beliefs(candidate_ids)` (or equivalent batched edge fetch). If the method does not yet exist by name, the impl PR adds it; the existing `edges_from(belief_id)` is the unbatched precedent.
- `src/aelfrice/retrieval.py:118-131` — flag-resolution convention.
- `src/aelfrice/retrieval.py:1048-1085, :1197-1232` — pack loops to replace.
- `tests/corpus/v2_0/`, `tests/bench_gate/` — corpus + harness.

All substrate is on `main` as of `e646383`. No new dependencies. No schema changes.

---

## Open questions for review

1. **Edge subgraph scope.** `cluster_candidates` builds the subgraph from the candidate pool's outgoing edges. Should it also include incoming edges from non-candidate beliefs (e.g. a candidate's citer, even if the citer isn't a candidate)? Spec says no — keep the subgraph candidate-induced. The candidate pool is the universe; non-candidates are out of consideration. Confirm at impl-PR review.
2. **Singleton handling.** A candidate with no in-pool edges is a singleton cluster of size 1. The Stage 1 loop visits it like any other cluster — it gets a representative (itself), it consumes one diversity slot. Is that correct, or should singletons be Stage-2-only? Spec keeps singletons in Stage 1 because demoting them would penalise queries whose hits are intentionally graph-isolated (e.g. lock-asserted policy beliefs). Confirm at impl-PR review.
3. **Diversity-target tuning.** Default `3`. If a typical multi-fact query in the wild needs 4-5 clusters, the default under-shoots. Tune from bench evidence, not pre-set.
4. **Locked beliefs and clustering.** L0 (locked) beliefs are never trimmed (`retrieval.py:950`). The pack-with-clusters output should pre-include locked beliefs ahead of Stage 1, exactly like the existing pack does (`retrieval.py:1015`). The spec says so implicitly via the phrase "replaces the existing pack-loop output"; the impl PR makes the L0-skip explicit at the top of the algorithm.
