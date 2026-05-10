# Feature spec: HRR integration — v2.1 ship plan

**Status:** spec (ready for implementation)
**Issue:** #553
**Target milestone:** v2.1
**Recovery-inventory line:** [`docs/ROADMAP.md`](ROADMAP.md) — v2.1 row (HRR persistence + format migration)
**Substrate prereqs:** `src/aelfrice/{hrr,hrr_index,retrieval}.py` at v1.7+ (Plate primitives shipped #216; `HRRStructIndex` for `KIND:target_id` queries shipped #152).

**Phase question.** Can the HRR stack ship with persistence default-ON in v2.1 without regressing latency, memory, or the FTS5 baseline NDCG?

**Risk:** medium — touches retrieval-surface defaults, persistence format, and the disk-cost story; mitigations are bench-gated.

---

## Problem

v1.7+ ships HRR machinery still default-OFF behind a flag awaiting bench evidence:

- `hrr.py` — Plate (1995) primitives.
- `hrr_index.py` — `HRRStructIndex` for `KIND:target_id` structural-marker queries. Default-OFF behind `use_hrr_structural`.

Three blocking items stand between the current state and a default-ON ship:

1. **Cold-start build cost dominates.** At N=50k, full-rank HRR build is 38 s. Warm-probe latency is fine (24.7 ms p95). Without persistence, every `aelf search` cold start pays the rebuild.
2. **`hrr_index.py:save()` format blocks `mmap_mode='r'`.** `np.load(.npz, mmap_mode='r')` is silently ignored per numpy docs. Mmap requires a separate `.npy` file. The bundled `.npz` format v1.7 currently writes is incompatible with the persistence-default-ON architecture this spec proposes.
3. **Disk overhead per persisted store is non-trivial.** 328 MB at N=10k, 1.64 GB at N=50k. Federation × multiple stores per user amplifies. A disk-cost opt-out flag is required.

A fourth finding is recorded for v2.2+ and is **not load-bearing for v2.1**:

4. **A possible third HRR use case (`_hrr_expand`-style FTS5-seed graph expansion via typed-edge neighborhoods).** Synthetic eval is currently negative; heat-kernel may substitute. Real-corpus eval needed before any port decision. Out of scope here.

   *Why an alternative encoding is not a substitute for the structural lane (#152).* The current `HRRStructIndex` per-belief composite holds a small number of bound terms per belief, giving SNR an order of magnitude better than a per-partition encoding for "find-many" queries (e.g. *find all sources of edges into target X with kind KIND*). A per-partition encoding is shape-correct for **find-one** queries (top-3 neighbors of a seed); the per-belief composite is shape-correct for **find-many**. The two are complementary, not substitutes. The v2.1 spec keeps the structural lane on the v1.7 encoding and treats `_hrr_expand` as a separate v2.2+ port question.

---

## Hypotheses

### H₀ — null (the perceived problem is illusory)

The current v1.7 ship state is acceptable. CLI users tolerate the 38 s build, daemon users amortize it, the disk cost is theoretical because no one runs at N=50k, and the bench gates that motivated this spec are over-cautious.

**Predicted outcome if H₀ holds.** Per-user telemetry (when collected) shows median store size ≤ 10⁴ beliefs **and** ≥ 90% of `aelf search` invocations are warm (the same daemon process). Then v1.7 stays as-is; no v2.1 spec needed.

**If refuted.** The 244× speedup from persistence at N=10k is unconditionally a win — even at the median N, persistence beats rebuild.

### H₁ — persistence default-ON is a clean win

Writing `.npy` + `.npz` to the per-store path on first build, loading via `mmap_mode='r'` on subsequent process starts, retains the warm-probe latency profile while collapsing cold-start to ≤ 1 s.

**Predicted outcome if H₁ holds.** Cold-start `aelf search` at N=50k completes in ≤ 1 s (vs 38 s without persistence). Warm-probe p95 stays ≤ 30 ms. NDCG round-trips byte-identical.

**Evidence on hand.** `.npz` load 0.20 s at N=50k; mmap load near-instant (page-cache caveat — true cold-disk retest pending). NDCG identical across rebuild / load paths.

### H₂ — format migration is small + cheap

Migrating `hrr_index.py:save()` from bundled `.npz` to split `struct.npy` + `meta.npz` is a localized change; v1.7 readers stay compatible by falling back to bundled-format detection.

**Predicted outcome if H₂ holds.** Migration touches ≤ 100 LoC in `hrr_index.py`; existing tests in `tests/test_hrr_index.py` require ≤ 5 LoC of changes (just the file-shape assertions); rollback is a single revert.

### H₃ — disk-cost opt-out is operationally necessary

A non-trivial fraction of users will have N=50k stores or multiple federated stores totaling ≥ 1 GB of persistence overhead. They need an env / TOML opt-out.

**Predicted outcome if H₃ holds.** First production data shows ≥ 1% of users hitting `AELFRICE_HRR_PERSIST=0` within 30 days of v2.1 ship.

---

## Falsification plan

- **Instrumentation required (pre-impl):** add `aelf doctor` rows reporting (a) HRR persistence on/off, (b) on-disk `.npy` size, (c) build vs load wallclock on the most recent rebuild.
- **Acceptance threshold per hypothesis:**
  - H₁: cold-start `aelf search` p95 ≤ 1 s at the user's actual N (measured via `aelf bench`).
  - H₂: migration PR diff ≤ 100 LoC in `hrr_index.py`; test diff ≤ 5 LoC; existing v1.7 `.npz` files load without re-encode.
  - H₃: a synthetic regression test that exercises both persistence-on and persistence-off codepaths and asserts NDCG byte-equality.
- **Sample size / duration:** persistence wallclocks already exercise the relevant N range. Production telemetry is post-ship.

---

## Decision rule

| Outcome | Action | Bucket |
|---|---|---|
| H₀ holds (telemetry shows daemon-dominant + small N) | Withdraw spec; keep v1.7 ship state. | drop |
| H₁ + H₂ hold | Ship default-ON persistence + format migration as the v2.1 cut. Default-flip flag stays separate (labeled-corpus gate). | adopt |
| H₁ holds, H₂ refuted (migration is bigger than 100 LoC) | Ship persistence default-ON in bundled format; defer mmap until split-format follow-up. | adopt + split |
| H₃ refuted (no disk-cost concern) | Drop the opt-out flag from the design; persistence is unconditional. | adopt + simplify |

---

## Design

### Persistence

**Format.** Per-store directory `<store_root>/.aelfrice/hrr/` containing:

```
struct.npy        — (N, dim) float64; mmap-able
meta.npz          — id_vecs, role_vecs, inv_role_vecs (small),
                    belief_ids (object array), seed (int64),
                    dim (int64), version (int32)
```

Old single-file `<store_root>/.aelfrice/hrr.npz` is read-only fallback; if found, `HRRStructIndex.load` returns the v1.7 format and a deprecation warning logs once. New writes always go to the split format.

**Load policy.** `np.load("struct.npy", mmap_mode="r")` for the struct matrix; `np.load("meta.npz", allow_pickle=True)` for the small metadata blob. Warm probes operate on the mmap'd array directly; the OS page cache handles working-set residency.

**Save policy.** On `HRRStructIndex.build()` completion, write both files atomically (write to temp paths, `os.replace` to final names). Writes happen on the build path, not on every probe.

**Cache invalidation.** Extend `HRRStructIndexCache.invalidate` (`hrr_index.py:308-310` at v1.7) to:

1. Drop the in-memory `_index` reference (existing behavior).
2. `os.unlink` the on-disk `struct.npy` and `meta.npz` if they exist. Concurrent reader processes' mmaps stay valid because the page-cache pages are reference-counted by the kernel until the last `madvise(MADV_DONTNEED)`.

The store mutation hook (`store.add_invalidation_callback`) wires to this on `HRRStructIndexCache.__post_init__` already (v1.7 line 295-298); no new wiring needed.

**Pre-implementation audit.** All mutation paths in `store.py` (verified at v1.7) fire `_fire_invalidation()`:

| Method | def line | invalidation line |
|---|---|---|
| `insert_belief` | 1323 | 1355 |
| `update_belief` | 1394 | 1434 |
| `delete_belief` | 1436 | 1447 |
| `set_retention_class` | 2437 | 2457 |
| `insert_edge` | 2535 | 2543 |
| `update_edge` | 2553 | 2561 |
| `delete_edge` | 2563 | 2569 |

`feedback.apply_feedback` routes through `update_belief`. Schema-metadata and onboarding-session methods correctly skip invalidation. No mutation path is missed; the proposed extension hangs off this existing wiring without coverage gaps. Implementation should re-verify these line numbers against current main before extending the callback.

**Filed v2.2+ optimization (not v2.1 blocker).** Posterior-only updates (alpha/beta nudges via `apply_feedback`) currently fire invalidation, but the HRR struct depends on edges, not posterior. A finer-grained signal (`_fire_invalidation(graph_changed=False)`) would let the HRR cache survive feedback events. Worth filing as a separate optimization spec; not required for v2.1.

### Configuration surface

```toml
[retrieval]
hrr_persist = true                # default; auto-disabled on ephemeral paths
hrr_persist_path = ".aelfrice/hrr"  # relative to store root
hrr_dim = 512                      # post-#538 default; configurable up to 2048 for memory-rich deployments
```

```bash
AELFRICE_HRR_PERSIST=0       # opt-out for disk-constrained envs
AELFRICE_HRR_PERSIST=1       # force-on, overrides ephemeral-path auto-disable
```

Resolution precedence is the existing `retrieval.py` convention: explicit kwarg → env var → TOML → default. The ephemeral-path heuristic (below) sits between the default and the env-var: an unset env var on an ephemeral path resolves to OFF; an explicit `AELFRICE_HRR_PERSIST=1` overrides.

### Ephemeral-path auto-disable

Persistence default-ON wastes a save cycle (~0.5 s at N=50k) in deployments where `<store_root>` is on a non-persistent filesystem — ephemeral containers, serverless runtimes, CI on fresh checkouts, `/tmp`-rooted test setups. Auto-detect via prefix match against:

```
/tmp/, /var/tmp/, /dev/shm/, /run/
```

When the store root resolves to a path under one of these prefixes, the resolver:

1. Logs once per process: `aelfrice: HRR persistence disabled on ephemeral path <path>; set AELFRICE_HRR_PERSIST=1 to force.`
2. Treats `hrr_persist` as if it were explicitly OFF.

`AELFRICE_HRR_PERSIST=1` (explicit force) overrides the auto-disable. The TOML key cannot override (TOML lives at the store root which is itself the path being checked); env var is the only override.

False negatives: legitimate test setups under `/tmp/` get the auto-disable. Mitigation: explicit `AELFRICE_HRR_PERSIST=1` for those tests.

### Operator surface

- `aelf doctor` gains three rows:
  - `hrr.persist_enabled: true|false`
  - `hrr.on_disk_bytes: <N>`
  - `hrr.last_build_seconds: <X>` (from the most recent `HRRStructIndex.build` call)
- `aelf status` adds an `hrr.persist_state` summary line.

### Default-flip flag (gated by labeled-corpus evidence, not this spec's blocker)

The persistence work above is the **substrate** that makes default-ON viable. The actual **default-flip** of `use_hrr_structural` (`retrieve_v2(..., use_hrr_structural=...)`) stays gated on labeled-corpus evidence per the existing #154 composition tracker policy. v2.1 ships persistence + format migration regardless. The flag ships default-OFF until the bench clears; once it does, a separate atomic PR flips it.

### What v2.1 does NOT change

- The structural lane's encoding (`HRRStructIndex` per-belief composite) — verified as the right shape for `KIND:target_id` queries.
- A potential third HRR use case (`_hrr_expand`) — deferred to v2.2+ pending real-corpus eval.

---

## Acceptance criteria

- [ ] `HRRStructIndex.save()` writes split format (`struct.npy` + `meta.npz`); existing tests pass against the new shape.
- [ ] `HRRStructIndex.load(path)` accepts the split format AND the legacy bundled `.npz` (with deprecation log).
- [ ] `HRRStructIndexCache` reads via `mmap_mode='r'` when the split format is present.
- [ ] `aelf doctor` reports `hrr.persist_enabled`, `hrr.on_disk_bytes`, `hrr.last_build_seconds` rows.
- [ ] Cold-start `aelf search` at N=50k synthetic store completes in ≤ 1 s when persisted (rebuilds in ≤ 38 s when not persisted; verifies `AELFRICE_HRR_PERSIST=0` codepath).
- [ ] NDCG byte-equality test: build → save → load → probe vs in-memory probe must produce bit-identical scores.
- [ ] `[retrieval] hrr_persist = false` in `.aelfrice.toml` disables persistence entirely; subsequent runs show no `.aelfrice/hrr/` directory created.
- [ ] Ephemeral-path auto-disable: a store rooted at `/tmp/<x>/store` logs the documented message exactly once per process and creates no `.aelfrice/hrr/` directory; `AELFRICE_HRR_PERSIST=1` overrides this and creates the directory normally.
- [ ] Cache invalidation: mutating a belief that participates in any HRR encoding drops both the in-memory index AND the on-disk blob.
- [ ] `docs/CONFIG.md` and `docs/COMMANDS.md` document the new flags and `aelf doctor` rows.

---

## Test plan

- **Unit (`tests/test_hrr_index.py`):**
  - Existing build / probe / save / load tests pass against split format.
  - New: `test_hrr_index_legacy_bundled_npz_loads_with_warning`.
  - New: `test_hrr_index_mmap_load_byte_identical_to_in_memory`.
  - New: `test_hrr_index_cache_invalidation_unlinks_blob`.
- **Integration (`tests/test_retrieval.py`):**
  - New: `test_retrieve_v2_cold_start_under_one_second_at_50k` (gated by `bench_gated` autouse marker).
  - New: `test_retrieve_v2_persist_off_no_disk_blob`.
- **Bench-gate (`tests/bench_gate/`):** the existing `bench_gated` marker keeps the cold-start threshold test in the bench-gated tier; non-bench CI skips cleanly.
- **Determinism:** all new tests use a fixed seed.

---

## Out of scope

- **`_hrr_expand` port** (the speculative third HRR use case). Real-corpus eval is the right gate. Deferred to v2.2+.
- **`hrr_dim` further reduction.** dim=512 is the post-#538 default; further reduction without a capacity-bounded encoding is deferred.
- **BM25-cascade for the structural lane.** The cascade finding originally applied to a query-rewrite path that has since been removed (#540).
- **LSH at N > 10⁵.** Prefilter cost projects to ~100 ms at N=10⁶; LSH or a sublinear posting-list prefilter is the v2.2+ scaling story.
- **Default-flip of `use_hrr_structural`.** Persistence ships unconditionally; the flag stays gated on labeled-corpus evidence per the existing #154 composition tracker. A separate atomic PR flips it once the bench gate clears.
- **Federation-aware persistence.** Per-store persistence; no cross-store sharing of `.npy` matrices. Federation spec separately.

---

## Coordination with other v2.1 work

- `docs/ROADMAP.md` v2.1 row gains an HRR persistence + format-migration line, paired with the existing #474 milestone tracker.
- `CHANGELOG.md` — Unreleased entry summarizes the persistence + format migration once the implementation PR lands.
- Tracking issue (this spec's #553) sub-issues the implementation as separate PRs (split-format save/load, mmap loader, ephemeral-path resolver, doctor rows, opt-out flag), per the umbrella-issue convention.
