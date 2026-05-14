# Edge-type-keyed rerank consumer (#421)

`aelfrice.edge_rerank.apply_edge_type_rerank` is a post-BFS rescore
pass that demotes surfaced beliefs based on the **edge types of their
incoming edges**. It is the demotion-half of the retrieval pipeline,
orthogonal to `BFS_EDGE_WEIGHTS` (which biases reachability during
expansion).

## Where it lives in the pipeline

```
seeds (L0 + L2.5 + L1)
   │
   ▼
expand_bfs                ← BFS_EDGE_WEIGHTS bias what is reached
   │  list[ScoredHop]
   ▼
apply_edge_type_rerank    ← per-edge-type penalties demote what was reached  (#421)
   │  list[ScoredHop] (rescored, re-sorted)
   ▼
caller (e.g., retrieve_with_tiers token-budget pack)
```

The pass is pure: same `(hops, store, penalties)` produces
byte-identical output. It uses `MemoryStore.edges_to(dst)` to query
incoming edges per surfaced belief.

## Skip-during-BFS contract

`POTENTIALLY_STALE` is a **marker** edge type: it tags a target belief
as suspected stale, with no relational/propagation semantics. BFS must
not walk through it, so it is pinned at weight 0.0:

```python
BFS_EDGE_WEIGHTS[POTENTIALLY_STALE] = 0.0
```

The `BFS_EDGE_WEIGHTS.get(t, 0.0)` default would do this implicitly,
but the contract is pinned explicitly so the intent is reviewable.
The actual *demotion* of beliefs reached via other edges that happen
to have stale incoming markers occurs only in the rerank pass.

## Config knobs

```python
DEFAULT_STALE_PENALTY: float = 0.5
EDGE_TYPE_PENALTIES_DEFAULT: Mapping[str, float] = {
    EDGE_POTENTIALLY_STALE: 0.5,
}
```

Pass-level override:

```python
apply_edge_type_rerank(
    hops, store,
    penalties={EDGE_POTENTIALLY_STALE: 0.25, ...},
)
```

- `penalties=None` → `EDGE_TYPE_PENALTIES_DEFAULT`.
- `penalties={}` → identity (re-sort only; no per-hop edge query).
- Penalty values are typically in `[0.0, 1.0]` for demotion semantics;
  `> 1.0` would amplify and is allowed but contradicts "demotion".

## Multi-edge interaction

When a single surfaced belief has incoming edges of more than one
penalty-keyed type, the penalties **compose multiplicatively**:

```
score_after = score_before × ∏ penalties[t]   for t in firing_types
```

`firing_types` is a *set*: the same edge type appearing on multiple
incoming edges fires once. The trigger is "at least one matching
incoming edge of this type," not edge count.

## Determinism

The returned list is sorted by `(-score, belief.id)` — the same
tie-break used by `expand_bfs`. Two passes compose without
ordering surprises.

## Bench gate

`tests/bench_gate/test_edge_rerank_potentially_stale.py` enforces
#421 acceptance #3 / #387 acceptance #3: **≥1pp@k drop** in
stale-tagged retrieval after the rerank pass vs. before, on the
`bfs_potentially_stale` corpus module. Skips cleanly on public CI
when the corpus is unmounted.

## Producer side

`POTENTIALLY_STALE` edges are produced by `aelf doctor --detect-stale`
(#387), implemented in `aelfrice.relationship_detector.write_potentially_stale_edges`.

**Staleness signal**: sub-confidence `contradicts` pairs from
`relationship_detector.relationships_audit`. A pair qualifies when its
`score < confidence_min` (default 0.5); high-confidence contradicting
pairs are excluded — those belong to the deferred CONTRADICTS write-path
(R2, #201).

**Edge direction**: for each qualifying pair `(a, b)`, one POTENTIALLY_STALE
edge is emitted with `src = newer belief`, `dst = older belief`. "Newer"
is the belief with the lexicographically greater `created_at` ISO string;
ties break on lex `id` order. Semantics: the newer belief casts doubt on
the older one.

**Idempotency**: a pre-insert `get_edge(src, dst, POTENTIALLY_STALE)` check
skips pairs whose edge already exists. Repeated invocations on the same
store are safe.

**Tuning**: the same `--relationships-jaccard`, `--relationships-confidence`,
and `--relationships-max-pairs` flags that tune `--relationships` apply to
`--detect-stale` as well.
