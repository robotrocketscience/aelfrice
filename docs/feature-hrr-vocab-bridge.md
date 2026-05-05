# Feature spec: HRR vocabulary bridge (#433)

**Status:** spec, no implementation
**Issue:** #433
**Recovery-inventory line:** [`docs/ROADMAP.md`](ROADMAP.md) — *"HRR vocabulary bridge | v2.0.0"*
**Substrate prereqs:** #216 (Plate FFT primitives, shipped v1.7.0), #152 (HRR bind/probe + struct index, shipped v1.7.0)
**Forward-compat slot:** `retrieve_v2(..., use_hrr=False)` already accepted as a no-op kwarg at `src/aelfrice/retrieval.py:1251`; the comment block at `retrieval.py:1271` is the public placeholder this spec fills.

---

## Purpose

Close the **vocabulary-gap-recovery** claim from the original research line: a query whose surface form diverges from the corpus's canonical vocabulary should still recover the canonical-entity beliefs. Today the BM25F anchor-text augmentation (#148, default-on at v1.7.0) closes the gap on the **document side** — a belief whose internal jargon differs from its citers' descriptions becomes recoverable through the citers' anchor text. The HRR vocabulary bridge closes the gap on the **query side** — a query token that does not appear in either belief content or anchor text gets bridged to the canonical entity vocabulary before any retrieval lane fires.

The two mechanisms are orthogonal. BM25F anchor text helps when the citer pool already uses the query's vocabulary. The vocabulary bridge helps when the query uses a surface form that no document — including its citers — uses verbatim, but the operator's mental model identifies it with a canonical entity that is in the corpus.

---

## Contract

```python
from aelfrice.vocab_bridge import VocabBridge

bridge = VocabBridge(dim=2048)
bridge.build(store, *, store_path=None, seed=None)

# Single-call query rewriting; called by retrieve()/retrieve_v2() before
# the per-lane fan-out when use_vocab_bridge=True.
augmented: str = bridge.rewrite(query, top_k=3, min_score=None)
```

Inputs:

- `store: MemoryStore` — the per-project store the canonical-entity vocabulary is harvested from.
- `query: str` — the raw query string passed by `retrieve()` / `retrieve_v2()`.
- `top_k: int` — maximum number of canonical-entity rewrites appended per surface-form token. Default `3`.
- `min_score: float | None` — confidence floor on cleanup-memory cosine. Default = `bridge.noise_floor()` (i.e. `1.0 / sqrt(dim)`, the same threshold the HRR struct index uses at `hrr_index.py:206-210`).

Output: a string of the form `<original_query> <space> <appended canonical tokens>`. The original query is preserved verbatim so downstream lanes see no regression on already-canonical tokens; bridged candidates are appended, not substituted.

The bridge is a pure query-rewriting stage. It does not touch the lane composition (BM25F + heat kernel + HRR structural + BFS), it does not change ranking, and it does not insert beliefs. It only widens the input string.

---

## Algorithm

### Build

For each canonical entity `e` discovered in the store (definition below), the bridge materialises a single HRR composite vector:

```
bridge_vec[e] = sum_{s in surface_forms(e)} bind(token_vec[s], canonical_vec[e])
```

Where:

- `canonical_vec[e]` is a deterministic random unit vector keyed off the entity ID, drawn the same way `HRRStructIndex.id_vecs` are drawn at `hrr_index.py:148-151` (`np.random.default_rng(seed)` over a path-derived seed).
- `token_vec[s]` is a deterministic random unit vector keyed off the surface-form token string. A second `Generator` (`role_rng = np.random.default_rng(seed ^ 0xVOCAB_SALT)`) provides the stream, mirroring the role/id stream split at `hrr_index.py:148-149`. Cross-build determinism falls out of the same seed convention.
- `surface_forms(e)` is the set of token strings the corpus has used to refer to `e`. The minimum viable set comes from three sources, in this priority order:
  1. **Anchor text under `incoming_anchors(e)`** — the same field BM25F (#148) consumes, harvested from `belief_corroborations.anchor_text`.
  2. **Belief content tokens** for beliefs whose `entity_id` (or equivalent) is `e`.
  3. **Lock-asserted statements** referencing `e` in `Belief.lock_state == LOCK_USER`.

The build walks the store once, materialises a `(N_entities, dim)` matrix, and a parallel cleanup-memory of `(canonical_token_string, canonical_vec)` pairs, exactly the `aelfrice.hrr.CleanupMemory` shape at `hrr.py:106-148`.

### Rewrite

For each token `t` in the input query:

```
probe = unbind(token_vec[t], bridge_matrix.T)         # (N_entities,) recovered-canonical vector per entity
candidates = cleanup_memory.query(probe, top_k)       # [(canonical_token, score), ...]
```

The bridge appends `[c for c, score in candidates if score >= min_score]` to the rewritten query. Tokens that already appear in the canonical vocabulary cleanly self-recover (cosine ≈ 1) and are appended once; tokens with no match score below `noise_floor()` and are dropped.

### Why HRR composition rather than a flat anchor-token table

A flat surface-form → canonical lookup (e.g. a hash table of `{"sqlite": "SQLite", "sqlite3": "SQLite"}`) requires either an authored alias list or an embedding model to populate. The HRR composition is built directly from the corpus the bridge is rewriting against, with no additional inputs and no learned components — it is pure linear algebra over the surface-form token universe the corpus already exposes.

Capacity: per Plate (1995, §6) and the documented bound at `hrr.py:30-34`, retrievable bound pairs scale as `dim/9`. At `dim=2048` that is ~227 surface forms per canonical entity before noise dominates. For a typical aelfrice store (≤ a few hundred entities, ≤ a few dozen surface forms per entity) this is comfortably above water.

---

## Where the bridge sits

The bridge is a **query-rewriting stage** that runs **before** the lane fan-out in `retrieve()` / `retrieve_v2()`. The `retrieve_v2` `use_hrr=False` placeholder at `retrieval.py:1251` is the slot it fills.

```
query
  -> [bridge.rewrite(query) if use_vocab_bridge else query]
  -> retrieve() lane fan-out: BM25F + heat-kernel + HRR-structural + BFS
  -> compose
  -> rank
  -> pack
```

The bridge is **not** a retrieval lane. It does not contribute scores; it only widens the string the lanes consume. This keeps it composable with every existing lane and the #154 composition tracker — the bridge is an input transform, not a candidate source, so per-lane gating is unaffected.

### Configuration

A new `use_vocab_bridge` flag follows the established convention at `retrieval.py:118-131`:

1. `retrieve(..., use_vocab_bridge=True)` kwarg (highest precedence).
2. `AELFRICE_VOCAB_BRIDGE=1` env var.
3. `[retrieval] use_vocab_bridge = true` in `.aelfrice.toml`.
4. Default OFF at v2.0.0 until bench-gate clears.

The `use_hrr` parameter at `retrieve_v2()` becomes a deprecated alias for `use_vocab_bridge`. Lab v2.0.0 adapters that pass `use_hrr=True` continue to work; `retrieval.py:1271` is updated from "has not yet ported" to "deprecated alias for `use_vocab_bridge`."

---

## Storage

### In-memory only at v2.0.0

The bridge index is rebuilt at process start, parallel to `BM25Index` and `HRRStructIndex`. Persistence is deferred. This matches the existing pattern: `BM25Index` is in-memory (`bm25.py`), `HRRStructIndex` has explicit `save()`/`load()` but no auto-persistence (`hrr_index.py:214-264`).

A `BridgeCache` wrapper analogous to `BM25IndexCache` is in scope for the implementation PR; it keys on `(store_path, mtime)` and rebuilds when the underlying corpus changes. **No new SQLite table is added at v2.0.0.** If a future revision needs persistence, the `.npz` round-trip pattern at `hrr_index.py:214-264` is the precedent.

### Schema impact

None. The bridge consumes existing schema:

- `belief_corroborations.anchor_text` (#148, shipped v1.5.0) — surface-form source #1.
- `beliefs.content` (foundation) — surface-form source #2.
- `beliefs.lock_state` (foundation) — surface-form source #3.

No migrations.

---

## Reconciliation

### vs. #227 — precomputed-neighbor vocabulary bridge (`docs/vocabulary_bridge.md`)

#227 closed 2026-04-29. Its `docs/vocabulary_bridge.md` ships a narrow `find_similar(belief_id, k)` lane backed by a precomputed-neighbor table over the BM25F index. It answers "what else in memory looks like this belief?" Different I/O shape:

| dimension | #227 (`docs/vocabulary_bridge.md`) | #433 (this spec) |
|---|---|---|
| input | `belief_id` | query string |
| output | top-K nearest beliefs | rewritten query string |
| persistence | new SQLite table `belief_neighbors` | in-memory, no new schema |
| primitive | BM25F sparse-matvec doc-doc cosine | HRR bind/cleanup-memory |
| invocation | `store.retrieve_similar(id, k)` | inside `retrieve()` before lane fan-out |

The two are not substitutes. #227 is a similarity primitive consumed by dedup (#197), contradiction detection (#201), etc. #433 is a query-side rewriter consumed only by retrieval. Both can coexist; they share no code and no table.

This spec lives at `docs/feature-hrr-vocab-bridge.md` precisely so the slug does not collide with the shipped `docs/vocabulary_bridge.md`.

### vs. #148 — BM25F anchor-text augmentation

Both close the same headline claim ("vocabulary-gap-recovery"). They close it on different sides of the lane interface:

- **#148** rewrites the **document field** (anchor text augments the indexed-document side).
- **#433** rewrites the **query string** (canonical-entity tokens augment the query side).

Composability: both can be on simultaneously. The bench-gate (below) verifies the bridge does not regress queries that BM25F anchor text already handles. Per the v1.7.0 evidence row in the README roadmap, BM25F anchor text contributes `+0.6650 NDCG@k` on the v0.1 retrieve_uplift fixture; the bridge is gated on **strictly additive** uplift over that baseline, not replacement.

### vs. #152 — HRR structural-query lane

`HRRStructIndex` (shipped v1.7.0, default-OFF behind `use_hrr_structural`) routes structural-marker queries (e.g. `"CONTRADICTS:b/abc"`) to a probe over the per-belief edge-structure index. It answers structural questions ("what beliefs contradict X?"). The vocabulary bridge sits one layer earlier — it does not route to a lane; it widens the query before any lane sees it. Both share the `aelfrice.hrr` algebra and the `dim=2048` default; they are cousins, not alternatives.

---

## Acceptance

### A1 — corpus

A labeled `vocab_bridge` corpus lives under `tests/corpus/v2_0/vocab_bridge/`, mirroring the v2.0 corpus scaffold from #307 / #311 (`tests/corpus/v2_0/README.md`). Public CI runs with `AELFRICE_CORPUS_ROOT` unset and skips the bench-gate cleanly via the existing autouse `bench_gated` marker; the labeled corpus content lives in the lab repo only, per the published corpus policy.

The corpus encodes vocab-shifted query/expected pairs of the form:

```jsonl
{"query": "<surface-form query>", "expected_belief_ids": ["b/...", "b/..."], "tag": "synonym|abbreviation|paraphrase|..."}
```

### A2 — bench-gate

The `tests/bench_gate/` harness (#319) is extended with a `vocab_bridge` consumer. Two assertions, both required for ship:

1. **Strictly additive uplift.** `NDCG@k(vocab_bridge=ON, BM25F=ON) > NDCG@k(vocab_bridge=OFF, BM25F=ON)` on the `vocab_bridge` fixture. The threshold is **positive uplift**, not a fixed magnitude — the bridge is gated on doing better than the v1.7.0 default-on baseline, not on hitting a research-line headline number.
2. **No regression.** `NDCG@k(vocab_bridge=ON, BM25F=ON) ≥ NDCG@k(BM25F=ON)` on the existing `retrieve_uplift` fixture (the v0.1 fixture #154 used). Tolerance band: ≥ baseline − 0.005 (a half-point of NDCG noise floor).

### A3 — composition tracker

`docs/RETRIEVAL_COMPOSITION.md` (or wherever the #154 tracker doc lands by ship-time) gains a row for `use_vocab_bridge` with: input shape, output shape, where it sits, and the bench-gate verdict. The bridge is **not a lane** in the composition matrix; it is a pre-lane transform. The tracker row is present for operator clarity.

### A4 — documentation

`docs/COMMANDS.md` and `docs/CONFIG.md` gain `use_vocab_bridge` entries describing the kwarg, env var, and TOML key. `retrieval.py:1271` placeholder comment is rewritten from "has not yet ported" to the deprecated-alias note above.

---

## Bench-gate / ship-or-defer policy

`needs-spec` → `bench-gated` once this spec lands. Implementation is the next gate. **The implementation PR ships only on positive bench evidence per A2.** A mechanically-correct implementation that fails A2 stays merged-but-default-OFF or gets reverted; it does not ship default-ON without a benchmark cut.

---

## Out of scope at v2.0.0

- **Persistence.** Rebuild on process start. `.npz` round-trip is precedent if a future revision needs it.
- **Cross-store bridges.** Each per-project store builds its own bridge; no federation.
- **Online updates.** The bridge is rebuilt, not incrementally maintained, on store mutation. The v1.5.0 BM25F path is the same; this is consistent.
- **Surface-form mining beyond the three sources above.** Query-log mining, embedding-model alias generation, and authored alias tables are all explicitly out of scope. The bridge is a function of the corpus, full stop.
- **Renaming `use_hrr`.** `retrieve_v2(use_hrr=...)` survives as a deprecated alias for one minor version; the rename to `use_vocab_bridge` is the canonical name for new callers.

---

## Implementation prereqs

- `src/aelfrice/hrr.py` — primitives (`bind`, `unbind`, `random_vector`, `CleanupMemory`). Shipped v1.7.0.
- `src/aelfrice/hrr_index.py` — seed convention, dual-`Generator` pattern, `.npz` round-trip pattern. Shipped v1.7.0.
- `src/aelfrice/bm25.py` — anchor-text harvesting from `belief_corroborations`. Shipped v1.5.0.
- `src/aelfrice/retrieval.py:118-131` — flag-resolution convention. Shipped v1.5.0.
- `tests/corpus/v2_0/` — corpus scaffold + autouse `bench_gated` marker. Shipped v1.6.0 (#307 / #311).
- `tests/bench_gate/` — harness. Shipped v1.6.0 (#319 / #320).

All substrate is on `main` as of `68dafc0`. No new dependencies. No schema changes.

---

## Open questions for review

1. **Surface-form harvesting priority.** The three sources listed under *Build* are the minimum viable set. If `entity_id` (or equivalent canonical-entity column) is not yet on `beliefs`, source #2 collapses to the entity-index lane (`entity_index.py`) and we harvest from there instead. This is a ship-time question; the spec accommodates either.
2. **`use_hrr` deprecation horizon.** One minor version at v2.x, removal at v2.y? Or keep indefinitely as the lab-adapter alias? Pick at impl-PR review.
3. **Cleanup-memory population scope.** Cleanup-memory at `hrr.py:106-148` is materialised lazily and invalidated on every `add()`. Whether it is rebuilt per-query or held in a `BridgeCache` between calls is an impl-PR detail; the contract is unchanged.
