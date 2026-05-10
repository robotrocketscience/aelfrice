# Partial Bayesian-weighted ranking (v1.3.0)

Spec for issue [#146](https://github.com/robotrocketscience/aelfrice/issues/146). Closely related: issue [#151](https://github.com/robotrocketscience/aelfrice/issues/151) (full log-additive Beta-Bernoulli ranking).

Status: shipped. v1.3.0 wired the partial Bayesian re-rank (`DEFAULT_POSTERIOR_WEIGHT = 0.5` in `src/aelfrice/scoring.py`). v1.7.0 added heat-kernel and HRR-structural primitives behind opt-in flags. v2.1.0 (#154) flipped both default-on after the #437 reproducibility-harness gate cleared 11/11.

## Motivation

Through v1.2.x, `apply_feedback` updates a belief's `(α, β)` counters and writes an audit row, but `retrieve()` orders L1 hits by `bm25(beliefs_fts)` alone. The posterior signal is computed and persisted; no retrieval path consumes it. This is the launch gap [`docs/LIMITATIONS.md`](LIMITATIONS.md) calls *"the big one: feedback doesn't drive ranking"*.

v1.3.0 closes that gap **partially**: posterior contributes to L1 ordering with a fixed log-additive weight, on a stable scoring contract that v2.0 extends rather than replaces. The full feedback-into-ranking eval — 10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — lands at v2.0.0.

## Relationship to issue #151

Two viable paths were considered. **Path B is adopted.**

- **Path A — different forms.** v1.3 ships a posterior-mean multiplier `score = bm25 * (1 + λ * posterior_mean)`. v2.0 ships #151's full log-additive form `score = log(BM25F) + 0.5 * log(posterior_mean) + log(heat_kernel)`. Two formulations, two calibrations.
- **Path B — same form, fewer terms (adopted).** v1.3 ships the log-additive form #151 specifies, with weight `λ = 0.5`, against `log(BM25)` only. v2.0 composes the additional terms (BM25F field-weighted, heat-kernel authority) into the same equation. One scoring contract from v1.3 onward.

Rationale for Path B:

1. **Single derivation.** No re-derivation between releases. The contract `score = log(BM25_term) + 0.5 * log(posterior_mean)` ships in v1.3 and only gains terms in v2.0.
2. **Calibration carries forward.** The synthetic-graph calibration that picked `λ = 0.5` (issue #151's numbers) is the same calibration that runs at v1.3. Re-running at v2.0 measures composition effects, not the basic posterior contribution.
3. **No silent re-ranking surprise at v2.0.** Users see the same monotone direction (more positive feedback → higher rank) starting at v1.3.0; v2.0 adds magnitude, not direction.
4. **Cold-belief neutrality is preserved across releases.** A belief with no feedback contributes a constant `log(0.5)` term and does not skew its rank relative to other unobserved beliefs. This holds whether the score has one extra term (v1.3) or three (v2.0).

**Disposition recommendation for #151.** After this spec lands, #151 should be re-scoped as the v2.0.0 follow-up that adds BM25F + heat-kernel terms to the contract this spec defines. The Beta-Bernoulli posterior term itself ships at v1.3 under #146; #151's other claims (BM25F, heat-kernel, real-feedback retest) become the v2.0 work. An orchestrator note proposing this consolidation is filed at PR-open time.

## Scope

In scope at v1.3.0:

- Add a `posterior_weight` parameter to `retrieve()` and `retrieve_v2()`. Default value matches the v1.3 ROADMAP slot ([§ "Posterior-weighted ranking (partial)"](ROADMAP.md#v130--retrieval-wave)) — see § Defaults below.
- Compute `score(b) = log(bm25_score(b)) + posterior_weight * log(posterior_mean(b))` for L1 candidates.
- L0 (`lock_level = "user"`) bypasses scoring entirely — locks are user-asserted ground truth and ranking does not move them. This is unchanged from v1.0.x.
- Extend `RetrievalCache`'s key tuple to include `posterior_weight` so two callers passing different weights do not collide on a shared cache.
- A regression test demonstrating that one `apply_feedback(belief, valence=+1.0, source=...)` reorders the next retrieval.
- `aelf bench` gains a partial-MRR-uplift metric: same fixture corpus, one round of synthetic feedback applied, MRR delta reported. Honest "partial" claim, not the full v2.0 eval.
- `docs/LIMITATIONS.md § "The big one"` is rewritten to reflect that ranking now consumes posterior — partially.

Out of scope at v1.3.0 (deferred to v2.0.0 unless noted):

- BM25F field-weighted scoring (separate retrieval-wave issue).
- Heat-kernel authority signal (graph-walk issue).
- 10-round MRR uplift evaluation against fixture corpus (v2.0).
- ECE calibration scorer (v2.0).
- Confidence-weighted posterior, e.g. variance-aware shrinkage (v2.0+).
- Per-corpus weight tuning. v1.3 ships the synthetic-graph optimum constant; v2.x can sweep.
- Retrieval-time `(α, β)` decay (separate issue).
- Real-feedback retest. Synthetic numbers ship at v1.3; user-utility-driven delta is measured separately once v1.3 has captured enough live feedback.

## Algorithm

### `f(posterior)` choice

For belief `b` with current counters `(α, β)`:

```
posterior_mean(b) = α / (α + β)
f(b)              = log(posterior_mean(b))
score(b)          = log(bm25_score(b)) + posterior_weight * f(b)
```

Two implementation notes follow from existing code:

1. **Use the existing `scoring.posterior_mean(α, β)`.** Per [`src/aelfrice/scoring.py`](../src/aelfrice/scoring.py), `posterior_mean(α, β) = α / (α + β)` already returns 0.5 for unobserved beliefs (which start at `(α, β) = (0.5, 0.5)`, the Jeffreys prior, per the existing decay target). **Do not switch to the Laplace form `(α + 1) / (α + β + 2)`** that issue #151 sketches: aelfrice's prior is Jeffreys, not Laplace, and conditioning the existing posterior on a different prior at the ranking layer would silently disagree with `aelf stats`, the MCP, and `decay()`. Cold-belief neutrality holds either way (both forms read 0.5 at the prior); use the form that matches the rest of the codebase.

2. **Numerical safety.** `α` and `β` are `float`. Both start at `0.5` and only grow under feedback; `decay()` shrinks the deltas-from-prior factor toward zero, so post-decay `(α, β)` asymptote at `(0.5, 0.5)` from above. `posterior_mean` is therefore in the open interval `(0, 1)` for any observable belief, and `log(posterior_mean(b))` is finite. No clamp needed; assert `posterior_mean(b) > 0` in dev builds.

### Defaults

- v1.3.0 ships `posterior_weight = 0.5` as default. This is the synthetic-graph optimum from #151's calibration table (NDCG@10 ≈ 0.95 at λ=0.5; collapses to 0.91 at λ=1.0; minimal effect at λ=0.0). Identical default to what #151 proposes at v2.0.
- v1.x patch releases prior to v1.3.0 ship `posterior_weight = 0.0` — exact v1.0.x ordering preserved. The flag is **introduced** in v1.3.0; there is no flip-the-default release transition. Behaviour at the v1.2.x patch line is unchanged.
- Callers may override per-call. `aelf retrieve --posterior-weight=0.0` reproduces v1.0.x ordering for diff-tooling and bisection.

### L0 lock bypass

Locked beliefs (`lock_level = "user"`) bypass scoring entirely. They are returned first, in lock-time order, and are never trimmed by the token budget — same contract as v1.0.x. The posterior of a locked belief is irrelevant to its retrieval position; the lock contract supersedes it.

This matters for the cold-belief case: a fresh user-lock has `(α, β) = (9.0, 0.5)` (per [`docs/LIMITATIONS.md § Sharp edges`](LIMITATIONS.md#sharp-edges)), giving `posterior_mean ≈ 0.947`. If lock bypass were forgotten, that posterior would still rank the lock above L1 hits — but the contract is "locks always come first regardless of math", so the bypass is what users expect, not an emergent consequence of the prior.

## Cache invalidation

The v1.0.1 `RetrievalCache` is keyed on `(canonicalize_query(query), token_budget, l1_limit)`. Two changes:

### 1. Extend the key tuple to include `posterior_weight`

Two callers passing different weights against the same store must not collide. Add `posterior_weight` as the fourth element of the key tuple:

```python
key = (canonicalize_query(query), token_budget, l1_limit, posterior_weight)
```

This is a structural fix, not a correctness fix for the staleness problem below. It prevents *cross-caller* collisions; it does not address *within-caller* staleness from posterior writes.

### 2. Posterior-write staleness — already handled

Three options were considered:

- **(a)** Invalidate cache on every `apply_feedback` call (explicit hook).
- **(b)** Include a `feedback_serial` counter in the cache key.
- **(c)** TTL the cache.

**Recommended: rely on the existing invalidation path. No new hook needed.**

The store already exposes `add_invalidation_callback`, and `RetrievalCache.__init__` already subscribes. Every store mutation that changes a belief — `insert_belief`, `update_belief`, `delete_belief`, the three callable mutators in `store.py` — calls `_fire_invalidation()`, which clears the cache. `apply_feedback` writes the new posterior via `store.update_belief(b)` (see [`src/aelfrice/feedback.py:150`](../src/aelfrice/feedback.py)), which triggers the wipe.

In other words: option (a) is what already happens, just one indirection deeper. Option (b) is unnecessary because the existing wipe is finer-grained than a serial counter would be (it also invalidates on edge mutations, lock changes, demotion-pressure increments, and content updates — anything that could reorder retrieval). Option (c) is wrong on its own because it permits stale results within the TTL window.

The acceptance test for cache invalidation is therefore: after `apply_feedback`, an in-flight `RetrievalCache` instance returns a different result list (or at minimum a different ordering) for the same `(query, budget, limit, weight)` tuple, **without** the implementation calling `cache.invalidate()` directly. The wipe must come through the store-level callback, not through a new hook in `apply_feedback`.

### Edge case: in-process concurrent retrieval during feedback

If two threads share a `RetrievalCache` and one calls `apply_feedback` while the other is mid-`retrieve()`, the second can either see pre-feedback or post-feedback ordering depending on lock interleaving. v1.3 does not introduce a stronger guarantee than v1.0.1 — `RetrievalCache` is documented as agent-loop scoped, single-thread per store, and `MemoryStore` already documents WAL + `busy_timeout=5000` as the cross-process concurrency contract. Multi-threaded sharing of a single cache is not supported.

## Calibration on synthetic harness

`aelf bench` (per [`src/aelfrice/benchmark.py`](../src/aelfrice/benchmark.py)) currently runs 16 queries against 16 beliefs, BM25-only. The v1.3 implementation extends it as follows:

- **Phase A** (`bench --baseline`): no feedback applied. Posterior weight = 0.0 (v1.0.x ordering). Reports hit@1 / hit@3 / hit@5 / MRR — current floor `hit_at_5 >= 0.75` preserved.
- **Phase B** (`bench --partial-uplift`): one round of feedback applied. For each query, the harness applies `apply_feedback(expected_belief, valence=+1.0, source="bench-synthetic")` once before re-running retrieval at the v1.3 default `posterior_weight = 0.5`. Reports the same metrics, plus the delta from Phase A.

### Calibration target (regression-asserted)

For the size-16 corpus, after Phase B:

- **MRR uplift ≥ +0.02** over the Phase A baseline. Small in absolute terms (the corpus is hand-authored to be high-precision under BM25), but signed positive and reproducible.
- **At least one query shows a strict rank promotion**: a belief that ranked at position R under Phase A ranks at position ≤ R-1 under Phase B for the same query, where R ≥ 2 in the baseline. This is the existence proof — not "the math moves," but "the math moves the rank."
- **No regression on Phase A queries**: for every query that hit at rank 1 in Phase A, the Phase B run also hits at rank 1. Posterior boost should never demote a correct top hit.

The single-query-promotion target is the v1.3 minimum; the regression test asserts it as `assert any(b_rank < a_rank and a_rank >= 2 for ...)`. The MRR-delta target is the *aggregate* version of the same claim, and is the aelf-bench-published number.

These thresholds intentionally undersell what real feedback can do. They guard against a regression that silently disables the posterior contribution; they do not claim v2.0's published uplift number.

## README copy update

The README (post-1.2 invisibility-led rewrite) does not currently carry the BM25-only caveat in the headline section — it points at LIMITATIONS for that. The implementation PR for v1.3.0 should:

1. Update [`docs/LIMITATIONS.md § "The big one: feedback doesn't drive ranking"`](LIMITATIONS.md#the-big-one-feedback-doesnt-drive-ranking) to:

   > **Feedback drives ranking, partially (v1.3.0+).** `apply_feedback` updates `(α, β)`, and L1 retrieval consumes the posterior log-additively at weight 0.5. The full feedback-into-ranking eval — 10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — lands at v2.0.0. See [`docs/bayesian_ranking.md`](bayesian_ranking.md) for the partial contract.

2. Update [`docs/ROADMAP.md § v1.3.0`](ROADMAP.md#v130--retrieval-wave) bullet to link this spec.

3. Leave the README headline copy alone unless a marketing pass reopens it. The README's claim is "memory the agent uses;" v1.3 makes that more true without needing a copy change at the top of the file.

The README copy update is **not** part of this spec PR. It is part of the implementation PR.

## Acceptance criteria for the implementation PR

The implementation PR must satisfy all of:

1. `retrieve(store, query, *, posterior_weight=0.5)` and `retrieve_v2(..., posterior_weight=0.5)` accept and apply the new parameter.
2. With `posterior_weight=0.0`, retrieval results are byte-identical to v1.0.x ordering. Verified by reusing `tests/test_retrieval_smoke.py` cases under the new flag.
3. With `posterior_weight > 0`, two beliefs at equal BM25 are ordered by posterior mean (descending).
4. With `posterior_weight > 0`, a high-BM25-low-posterior belief can be ranked below a lower-BM25-high-posterior belief once the posterior gap is large enough. Demonstrated by a fixture in `tests/test_posterior_ranking.py`.
5. After one `apply_feedback(b, valence=+1.0, source="test")` on a belief that ranks at position R ≥ 2 in the baseline, the same query returns `b` at rank ≤ R-1. Demonstrated in `tests/test_posterior_ranking.py`.
6. `RetrievalCache` keys include `posterior_weight`. Two `cache.retrieve(...)` calls with different weights produce different lookups; same weight is a hit.
7. After `apply_feedback`, an in-flight `RetrievalCache` produces fresh results on the next `retrieve()` call **without** the implementation explicitly calling `cache.invalidate()` from `apply_feedback`. The wipe must come through `store._fire_invalidation()`.
8. Locked beliefs (`lock_level = "user"`) are unaffected by `posterior_weight`. Verified by adding a lock to a smoke fixture and confirming its position is identical at every weight in `{0.0, 0.5, 1.0}`.
9. Cold-belief neutrality: with `posterior_weight = 0.5` and a corpus where every belief has `(α, β) = (0.5, 0.5)`, ranking is identical to `posterior_weight = 0.0`. (The `log(0.5)` constant shifts every score by the same amount and does not reorder.)
10. `aelf bench --partial-uplift` runs to completion against the v1 corpus and reports MRR delta ≥ +0.02 over baseline. Asserted by `tests/test_benchmark.py`.
11. Per-query overhead from the posterior term ≤ 1ms at N=10⁵. Latency regression test asserts within `tests/test_retrieval_*` budget.
12. `docs/LIMITATIONS.md § "The big one"` is rewritten per the copy in this spec.
13. `docs/ROADMAP.md § v1.3.0` links this spec.
14. CI green: `uv run pytest -x -q` passes.

## Dependencies

- **v1.0.1 `RetrievalCache`** — already shipped. Key tuple extends; invalidation path is reused unchanged.
- **v1.0 `apply_feedback` audit** — already shipped. Feedback rows already exist; the regression test for criterion 5 just needs to call `apply_feedback` and re-`retrieve()`.
- **`scoring.posterior_mean`** — already shipped. Implementation imports and uses it directly. Do not duplicate the formula.

No schema changes. No new tables. No new columns.

## Soft-blocks lifted at v1.3.0 ship

- **v1.4 rebuilder quality.** Per ROADMAP, the rebuilder ships either way; with v1.3 partial Bayesian ranking landed, the rebuilder's belief-selection step consumes the same posterior-aware ordering rather than BM25-only. Continuation-fidelity numbers are correspondingly higher. The rebuilder issue does not need to wait on v2.0.0.

## What v2.0.0 still owes

This spec ships partial. The remaining v2.0.0 work, tracked separately:

- BM25F field-weighted scoring as the first term (replacing plain BM25).
- Heat-kernel authority signal as a third additive term.
- 10-round MRR uplift evaluation against the synthetic corpus, with reproducibility band documented.
- ECE (Expected Calibration Error) scorer for the posterior signal as a probability estimate.
- Real-feedback retest once captured telemetry passes the corpus-size threshold.
- Per-corpus weight sweep (currently `λ = 0.5` is hardcoded; v2.x exposes calibration).

These are the items #151's full claim covers. After this spec PR opens, #151 should be re-scoped to the above set, with the v1.3 posterior term itself acknowledged as shipped under #146.

## Heat-kernel composition (Slice 2)

Slice 2 of #151 adds a third log-additive term — a graph authority signal computed from the signed-Laplacian heat kernel `exp(-tL)`. The composed score is:

```
score(q, b) = log(BM25(q, b))
            + heat_weight    * log(heat_kernel_safe(q, b))
            + posterior_weight * log(posterior_mean(b))
```

where `heat_kernel_safe(q, b)` is the per-belief heat propagation seeded from the top-K BM25 hits, clamped at `HEAT_SCORE_FLOOR = 1e-9`. Default mixing weights are `heat_weight = 1.0` and `posterior_weight = 0.5` (`graph_spectral.combine_log_scores`).

### Feature flag (default-OFF)

The composition is gated behind `is_heat_kernel_enabled()`, which resolves with precedence env > kwarg > TOML > False:

- `AELFRICE_HEAT_KERNEL=1` (env override)
- `retrieve(..., heat_kernel_enabled=True)` (kwarg)
- `[retrieval] use_heat_kernel = true` in `.aelfrice.toml`

The flag stays default-OFF in v1.x. When the flag is OFF, the heat term is not constructed — `retrieve()` is byte-identical to the Slice 1 (BM25 + posterior) path.

### Cost

The dominant per-query cost is the eigenbasis matvec `eigvecs.T @ seeds` followed by `eigvecs @ filt`, which is O(N · K). At N=50k, K=200 the wall-clock is ~7-8 ms in BLAS-backed numpy on commodity hardware. With the flag OFF the per-query overhead is unmeasurable (≤ 1 ms — no spectral work happens). AC6 of #151 was renegotiated to ≤ 10 ms heat-on / ≤ 1 ms heat-off to reflect this real cost.

### Graceful degrade

Three graceful-degrade paths fall back byte-identically to the heat-off contract:

1. No `eigenbasis_cache` passed (caller didn't construct one).
2. `cache.is_stale()` — store mutation invalidated the cache; offline rebuild is a separate (#149) entry point.
3. `cache.eigvals is None` — never built, or every L1 hit was inserted after the last build.

In each case the rerank reverts to `partial_bayesian_score(bm25, alpha, beta, posterior_weight)`. Newly-inserted beliefs missing from `cache.belief_ids` get `HEAT_SCORE_FLOOR` for the heat term — the floor preserves rankability without invented authority signal.

### Bench wedge

`python -m benchmarks.posterior_ranking --heat-kernel` runs the MRR + ECE harness with the flag flipped on. Each per-seed `retrieve()` gets a fresh `GraphEigenbasisCache` built against that seed's in-memory store and rebuilt on stale (the synthetic feedback stream mutates the store after every round). Without `--heat-kernel`, output is byte-identical to today.

### What Slice 2 still doesn't ship

- Eigenbasis offline build CLI (#149).
- Heat-weight sweep — `heat_weight = 1.0` is the spec default, no calibration exposed yet.
- Real-feedback retest once captured telemetry passes the corpus-size threshold.

## References

- Issue [#146](https://github.com/robotrocketscience/aelfrice/issues/146) — partial Bayesian-weighted ranking, v1.3 milestone.
- Issue [#151](https://github.com/robotrocketscience/aelfrice/issues/151) — full log-additive Beta-Bernoulli, no milestone (proposed: re-scope to v2.0.0 follow-up).
- [`docs/ROADMAP.md`](ROADMAP.md) § v1.3.0 and § v2.0.0.
- [`docs/LIMITATIONS.md`](LIMITATIONS.md) § "The big one: feedback doesn't drive ranking".
- [`src/aelfrice/retrieval.py`](../src/aelfrice/retrieval.py), [`src/aelfrice/scoring.py`](../src/aelfrice/scoring.py), [`src/aelfrice/feedback.py`](../src/aelfrice/feedback.py), [`src/aelfrice/store.py`](../src/aelfrice/store.py).
- Robertson 1977, *The Probability Ranking Principle in IR* — canonical reference for mixing IR scores with prior probabilities log-additively.
- Croft & Lafferty (eds.) 2003, *Language Modeling for Information Retrieval* — Bayesian smoothing background.
