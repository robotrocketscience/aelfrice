# Precomputed-neighbor vocabulary bridge

Spec for issue [#227](https://github.com/robotrocketscience/aelfrice/issues/227).

Status: spec, no implementation. Targets a v1.x slice ahead of the
full HRR retrieval lane (#152).

## What this ships

A narrow `find_similar(belief_id, k)` lane backed by a precomputed
per-belief top-K nearest-neighbor list. Materialized offline as a
sibling table; queried as an O(K) array slice.

This is **not** the full HRR retrieval lane. It does not do
structural composition (NOT/AND/CHAINS/IMPLICATION), bind/probe, or
edge-typed queries. Those remain on the v1.7.0 track at #152 and
its prereqs (#216 Plate FFT, #149 Laplacian, #150 heat-kernel).

## Why narrower than #152 — why ship now

The full HRR lane has three hard prereqs (#216, #149, #150). The
neighbor list has none. The most common "find similar" query —
"what else in memory looks like this belief" — can be served
without any of the structural primitives, against substrate the
v1.5.0 BM25F index (#148) already materializes.

Splitting it forward:

- delivers user-visible `find_similar` at v1.6 instead of v1.7
- gives downstream features (e.g. dedup #197, contradiction
  detection #201) a stable similarity primitive earlier
- does not displace the structural lane — that lane handles
  query types this one cannot answer

## Substrate

The BM25F index from #148 (`src/aelfrice/bm25.py`,
`class BM25Index`) already materializes:

- per-belief sparse term-frequency rows (`tf`, CSR matrix)
- per-term IDF vector (`idf`)
- per-document length and the BM25 length-normalization terms

Doc-doc similarity is one transform away from this index. No new
embedding model, no new primitives, no new hyperparameters beyond
`K`. The neighbor builder consumes `BM25Index` and writes a
sibling table.

## Schema

New table:

```sql
CREATE TABLE belief_neighbors (
    belief_id    TEXT NOT NULL,
    neighbor_id  TEXT NOT NULL,
    score        REAL NOT NULL,
    rank         INTEGER NOT NULL,           -- 1..K, dense, no ties
    computed_at  REAL NOT NULL,              -- unix timestamp
    PRIMARY KEY (belief_id, neighbor_id),
    FOREIGN KEY (belief_id)   REFERENCES beliefs(id) ON DELETE CASCADE,
    FOREIGN KEY (neighbor_id) REFERENCES beliefs(id) ON DELETE CASCADE
);
CREATE INDEX idx_belief_neighbors_lookup
    ON belief_neighbors (belief_id, rank);
```

Notes:

- `rank` is materialized so `retrieve_similar(id, k)` is a single
  range scan, not an order-by sort.
- Self-pairs (`belief_id == neighbor_id`) are excluded from the
  builder output.
- `ON DELETE CASCADE` on both columns: when a belief is dropped,
  every row mentioning it disappears. This is the right default
  here; `belief_neighbors` is a derived table, not an audit trail.
  Audit-trail tables (`feedback_history`, `belief_corroborations`)
  follow a different policy (see #223).
- Migration is forward-only (`migrate.py`); empty table on first
  open after upgrade. The first index build populates it.

## Score function — design call #1

**Adopted:** cosine over BM25-saturated tf-idf row vectors from the
existing `BM25Index`.

For belief `i`, let `s_i[j] = sat(tf[i, j]) * idf[j]` where `sat`
is the BM25 saturation transform already implemented in
`BM25Index.score` (Robertson 2004, with parameters `k1` and `b`
matching the query path). Similarity:

```
sim(i, j) = (s_i · s_j) / (||s_i|| * ||s_j||)
```

Three reasons:

1. **Substrate reuse.** `BM25Index` already materializes every
   term in this expression. The builder is a sparse-matrix-self-
   product against precomputed data; no new index.
2. **Symmetry.** Cosine is symmetric, so `top_k(i)` and `top_k(j)`
   answer the same question consistently. Asymmetric forms
   (e.g. BM25 query-doc with doc i as query against doc j) are
   length-biased and would produce non-mutual neighbor links.
3. **Same vocabulary as the BM25 retrieval path.** Beliefs that
   are similar under the existing keyword retrieval will be
   similar here. No vocabulary divergence between
   `retrieve(query)` and `retrieve_similar(belief_id)`.

Rejected alternatives:

- **Raw cosine over `tf` alone (no idf, no saturation).** Loses
  IDF rarity weighting — every common-word-heavy belief looks
  similar to every other. Hard fail on the "find non-trivial
  near-neighbors" criterion.
- **Jaccard over token sets.** Compatible with stdlib but throws
  away frequency information; loses the same ranking signal the
  BM25 index already encodes.
- **Embedding cosine (e.g. sentence-transformers).** Adds a
  learned model dependency. Violates the determinism property
  (`docs/concepts/PHILOSOPHY.md`). Defer to a future v2.x lane if needed.

The BM25-derived score is not normalized to `[0, 1]`. The cosine
form above is, naturally, in `[0, 1]` for nonnegative vectors.
Stored `score` values are cosine.

## K default and storage — design call #2

**Adopted:** `K = 10` default, configurable via
`[retrieval.vocabulary_bridge].k` in `.aelfrice.toml`.

Storage envelope (current store sizes from #219):

| store size | rows in `belief_neighbors` | bytes (~) |
|---|---|---|
| 4,000 distinct beliefs | 40,000 | ~1.5 MB |
| 20,000 belief rows | 200,000 | ~7 MB |
| 100,000 (projected) | 1,000,000 | ~35 MB |

Per-row size estimate: 2× ULID (`TEXT`, ~26 chars) + REAL + INT +
REAL + index overhead ≈ 35 bytes serialized in SQLite. The
projection assumes a future post-#219 dedup state where rows ≈
distinct hashes; pre-dedup numbers are 5× higher and still bounded.

`K = 10` is the sweet spot for the use cases this lane serves:
dedup wants ~5 candidates; "show me similar beliefs" UIs want
~10; downstream consumers (#197 dedup, #201 contradiction) can
pull `k <= K` cheaply via the materialized rank index. Larger
`K` would be available by rebuild only.

Out-of-band query for `k > K`: caller falls through to
`BM25Index.score(belief.content)` with `top_k = k`. This is the
v1.x escape hatch; documented in the API contract below.

## Refresh trigger — design call #3

**Adopted:** piggyback on the BM25F index rebuild lifecycle, with
an explicit `aelf neighbors rebuild` command for the operator
escape hatch.

The BM25F index already has an invalidation contract
(`BM25IndexCache.invalidate()` in `src/aelfrice/bm25.py`) tied to
ingest events. The neighbor builder hooks the same lifecycle:

1. **On BM25 index rebuild:** the neighbor builder runs once after
   the new index is materialized, before the cache returns the new
   index to callers. Synchronous; this is already the slow path.
2. **On batch backstop:** `aelf neighbors rebuild` (new CLI
   subcommand) forces a full rebuild. Wired into `aelf doctor` as
   an operator-callable repair.
3. **No per-write rebuild.** The BM25 index itself is rebuilt
   lazily, not per-`ingest_turn`. Per-belief incremental updates
   to the neighbor list would require recomputing every other
   belief's top-K (insertion can displace any existing neighbor),
   which is O(N) per write — worse than the lazy full rebuild.

Build cost envelope:

- Sparse self-product `S @ S.T` over the CSR `tf-idf-sat` matrix.
  At 20K beliefs × 50K vocab terms × ~30 nonzeros/row, that's
  ~600K nonzeros total. Sparse matvec is O(nnz × n_docs) ≈
  ~10⁹ ops; with scipy this fits in seconds, not minutes.
- Memory peak is the dense `(n_docs, K)` int + float arrays plus
  the CSR. Bounded.

If build time exceeds 30s on representative stores, the fallback
is a per-row top-K scan that streams instead of materializing the
full self-product. That's an implementation tactic, not a design
question; spec'ed separately if reached.

## Lock interaction — design call #4

**Adopted:** locks do not short-circuit the neighbor list.
`belief_neighbors` is content-similarity-only. Locks are a
retrieval-time signal, not a storage-time one.

Concretely:

- The builder treats locked and unlocked beliefs identically.
- `retrieve_similar(belief_id, k)` returns the top-K by cosine
  regardless of lock state.
- Downstream callers that want lock-aware reordering (e.g. the
  context rebuilder) apply it on top of the returned list, the
  same way they do today against `retrieve()`.

Rationale:

1. **Orthogonality.** "What is similar to X" and "what is locked
   ground truth" are independent questions. Conflating them at
   storage time pushes lock semantics into a derived index, where
   they are hard to evolve without a full rebuild.
2. **Existing precedent.** `retrieve()` (`src/aelfrice/retrieval.py`)
   does not short-circuit on lock state at the BM25 stage either;
   locks are applied as a tier-promotion layer above L1/L2.5/L3.
   The neighbor lane should match this convention so a future
   `retrieve_similar` consumer can layer the same logic.
3. **Audit clarity.** A locked belief that has no near-neighbors
   is informative ("this lock is novel"). Suppressing those
   candidates at the storage layer destroys the signal.

## API contract

```python
# src/aelfrice/vocabulary_bridge.py  (new module)

def build_neighbors(
    store: MemoryStore,
    bm25_index: BM25Index,
    k: int = 10,
) -> int:
    """Materialize the top-K neighbor table from a BM25 index.

    Truncates `belief_neighbors`, recomputes, and writes in a
    single transaction. Returns the row count written.
    """

def retrieve_similar(
    store: MemoryStore,
    belief_id: str,
    k: int = 10,
) -> list[tuple[Belief, float]]:
    """Return up to `k` precomputed neighbors of `belief_id`.

    Reads from `belief_neighbors` ordered by `rank ASC`. Returns
    fewer than `k` rows if the belief has fewer materialized
    neighbors (e.g. a freshly inserted belief before the next
    rebuild). Empty list for unknown `belief_id`.

    For `k > K_built`, callers fall through to
    `BM25Index.score(belief.content)`; this function does not
    perform that fallback itself.
    """
```

CLI:

- `aelf neighbors rebuild` — operator rebuild trigger.
- `aelf neighbors stats` — rows, mean/median rank coverage,
  last `computed_at`. Reuses the `aelf doctor`-style report
  surface.

## Acceptance

1. Schema migration adds `belief_neighbors` and its index. No-op
   on stores that already have it. Forward-only.
2. `build_neighbors` produces deterministic output for a fixed
   `BM25Index` snapshot (deterministic tie-break by `neighbor_id`
   ASC, matching `BM25Index.score`).
3. `retrieve_similar` returns rows in ascending `rank` order,
   with `score` matching the builder's stored value.
4. The BM25 cache rebuild path triggers `build_neighbors` exactly
   once per rebuild; concurrent reads see the old table until the
   new one is written.
5. Existing FTS5/BM25/posterior retrieval paths show no
   regression on the v1.5 bench fixtures (`benchmarks/`).
6. `aelf neighbors rebuild` rebuilds end-to-end on a populated
   store in bounded time (acceptance threshold: 30s on a 20K-row
   store; alternative streaming path documented if exceeded).
7. `aelf neighbors stats` reports nonzero rank coverage for every
   non-orphan belief after a fresh rebuild.

## Test plan

`tests/test_vocabulary_bridge.py` (new):

- **Determinism:** same `BM25Index` snapshot → same neighbor
  table, byte-equal across two runs.
- **Schema:** migration is idempotent; second open of an upgraded
  store is a no-op.
- **Self-exclusion:** `belief_neighbors` never contains
  `belief_id == neighbor_id`.
- **Symmetry sanity:** for a small fixture, every (i, j) pair in
  the table satisfies `score(i, j) == score(j, i)` to within
  float tolerance.
- **K bound:** every `belief_id` has at most `k` rows.
- **Rank density:** ranks for any `belief_id` are exactly
  `1..n_neighbors`, no gaps.
- **Cascade delete:** dropping a belief removes all its rows from
  both sides of `belief_neighbors`.
- **Refresh integration:** writing a new belief, then triggering
  the BM25 cache rebuild, populates rows for the new belief and
  may shift existing top-K entries.
- **Locked beliefs:** a locked belief and an unlocked belief with
  identical content produce identical neighbor lists.
- **Empty / cold start:** `retrieve_similar` on a freshly opened
  store with no neighbor rows returns `[]`, no error.

## Out of scope

- Structural composition queries (NOT/AND/CHAINS/IMPLICATION) —
  covered by #152.
- Bind/probe / FFT primitives — covered by #216.
- Edge-typed similarity (e.g. "find beliefs related via
  SUPERSEDES") — covered by graph signal lane (#149/#150).
- Embedding-model-based similarity — out of v1.x scope.
- Backfilling neighbor lists onto historical snapshots — the
  table is derived; first build after upgrade is the only
  one-shot.

## Dependencies

- BM25F index (#148) — shipped at v1.5.0.
- Anchor-text BM25F (#148) — same.
- No structural-lane dependency.

## References

- Issue #227 (this).
- Issue #152 — full HRR retrieval lane (companion track).
- `docs/design/bayesian_ranking.md` — example of the v2-style spec
  format this doc follows.
- `src/aelfrice/bm25.py` — `BM25Index` class consumed by the
  builder.
- `docs/concepts/PHILOSOPHY.md` — determinism property the score-function
  decision rests on.
