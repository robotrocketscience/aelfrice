# v2.0 spec: posterior-weighted ranking — residual scope

Spec for issue [#151](https://github.com/robotrocketscience/aelfrice/issues/151). Cascade addendum to [`bayesian_ranking.md`](bayesian_ranking.md). Defines the v2.0 work that closes the residual after the v1.3 partial shipped under [#146](https://github.com/robotrocketscience/aelfrice/issues/146).

Status: spec, no implementation. Recommendation included; decision is the user's.

## What's being decided

Five calls left open by the v1.3 spec's "out of scope at v1.3.0" list and the [`#151` re-scope comment](https://github.com/robotrocketscience/aelfrice/issues/151#issuecomment) that handed v2.0 the residual:

1. **Composition order** for the BM25F field-weighted term (#148, shipped) and the heat-kernel authority term (#150, not started) into the contract `score = log(BM25_term) + 0.5 * log(posterior_mean)`.
2. **MRR uplift evaluator** — what fixture, how many rounds, how reproducibility is reported.
3. **ECE calibration scorer** — bucketing strategy and what counts as a "predicted probability" for the posterior signal.
4. **Real-feedback retest** — when the live-telemetry corpus is large enough, what the retest asserts, and what failure means.
5. **Per-corpus weight sweep** — surface (CLI flag vs config), search range, and stop condition.

After this issue ships, the v1.3 partial-Bayesian claim becomes a full posterior-driven ranking claim: the contract composes three terms (BM25F + posterior + heat-kernel), the eval reports both ranking quality (MRR uplift) and probability calibration (ECE), and the weight `λ = 0.5` becomes a calibratable knob with a documented default.

## Substrate / chain dependency

- **Depends on:** #148 (BM25F field-weighted, **shipped** — see `scoring.partial_bayesian_score` already accepting BM25F-derived inputs).
- **Depends on:** #150 (heat-kernel via eigenbasis K=200, **not started**). Heat-kernel composition cannot land until #150 ships its scorer.
- **Depends on:** #154 (v1.7 pipeline-composition tracker) for the unified `retrieve()` entry point that consumes all three terms.
- **Independent of:** #196 (v2.0 substrate decision). The Beta-Bernoulli posterior shipped at v1.3; this spec extends it without re-opening single-axis vs multi-axis. If #196 lands Option A (multi-axis), `posterior_mean` projects to scalar via the existing `α = sum(α_i)` / `β = sum(β_i)` reduction and this spec is unaffected.

## Recommendation

**Ship at v2.0 in three independent slices, in order: (1) eval harness first, (2) heat-kernel composition once #150 lands, (3) weight sweep last.** The eval harness is a docs-and-bench change with no production-path risk; the composition is a one-line addition to the score equation; the sweep ships only after the eval harness can measure the deltas it produces.

### Slice 1 — eval harness (`benchmarks/posterior_ranking/`)

Two scorers and one runner. All deterministic, fixture-driven, no live store dependency.

```
benchmarks/posterior_ranking/
  fixtures/                        # known-item fixtures, hand-curated
  mrr_uplift.py                    # 10-round MRR uplift evaluator
  ece.py                           # ECE calibration scorer
  run.py                           # CLI: aelf bench --posterior-residual
```

**MRR uplift contract:**
- Round 0: baseline retrieve, no feedback applied. Record `mrr_0`.
- Rounds 1..10: each round, apply one synthetic feedback event per query against the top-1 result if it matches the known item, against the top-K relevant set otherwise. Re-retrieve. Record `mrr_i`.
- Output: `mrr_uplift = mrr_10 - mrr_0`, plus the full per-round series.
- Reproducibility band: 5 seeds; report `(mean, ±2σ)`.
- Pass criterion: `mrr_uplift ≥ +0.05` on the v2.0 fixture corpus, no round shows regression below `mrr_0 - 0.01`.

**ECE contract:**
- For each `(query, retrieved_belief, rank)` triple in the eval set, treat `posterior_mean(b)` as the predicted probability that the user will rate `b` positive.
- Bucket beliefs by predicted probability into 10 equal-width buckets `[0, 0.1), ..., [0.9, 1.0]`.
- For each bucket: compute `(mean_predicted, mean_actual)` where `mean_actual` is the empirical positive-feedback rate from the synthetic feedback stream.
- ECE = `sum_b (|bucket_b| / N) * |mean_predicted_b - mean_actual_b|`.
- Pass criterion: `ECE ≤ 0.10` on the synthetic fixture; report ECE on the real-feedback corpus once it exists, no pass/fail gate there until enough data accumulates.

**Why this slice ships first:** it produces the numbers every later decision needs. Without it the heat-kernel composition lands blind and the weight sweep has nothing to optimize against.

### Slice 2 — heat-kernel composition (`scoring.py`, `retrieval.py`)

One change to the score equation:

```python
score(q, b) = log(BM25F(q, b))
            + heat_kernel_weight * log(heat_kernel_safe(q, b))
            + posterior_weight   * log(posterior_mean(b))
```

`heat_kernel_weight` defaults to `1.0` per the [#154 pipeline contract](https://github.com/robotrocketscience/aelfrice/issues/154). `posterior_weight` keeps the `0.5` ship default from v1.3.

Cold-belief neutrality survives because `log(heat_kernel_safe)` is bounded and `log(posterior_mean) = log(0.5)` for unobserved beliefs — both contribute constants to a belief with no graph-authority signal and no feedback, so its rank relative to other unobserved beliefs is unchanged.

**Out of scope for slice 2:** the actual heat-kernel scorer (#150). This slice is the integration; it lands in the same release as #150 or in the release after, gated by whether #150 has shipped.

### Slice 3 — per-corpus weight sweep (`benchmarks/posterior_ranking/sweep.py`)

CLI: `aelf bench --posterior-residual --sweep λ=0.0:1.0:0.1`.

Outputs a `(λ, mrr_uplift, ece)` table for every λ in the range. Default range `[0.0, 1.0]` step `0.1`; configurable. The sweep does **not** auto-flip the production default — it produces a recommendation that the user ratifies via `posterior_weight` in `.aelfrice.toml`. Per-corpus tuning is opt-in.

**Why last:** there is no point sweeping a knob whose objective function (MRR uplift, ECE) doesn't yet have an evaluator. Slice 1 provides the objective; this slice optimizes against it.

### Real-feedback retest — out of scope for the v2.0 ship

Filed as a follow-up. The retest pulls real `feedback_history` rows from a user store, replays them, and compares MRR uplift + ECE against the synthetic-fixture numbers. Threshold for "enough data": `≥ 1000 feedback events` across `≥ 100 distinct beliefs`, per the synthetic-bench corpus shape. No user store has this volume yet; filing a follow-up to hold the retest under a `--from-store <path>` flag rather than racing it into v2.0.

## Decision asks

- [ ] **Confirm three-slice sequencing.** Alternative: bundle the eval harness with the heat-kernel composition in one PR. Recommendation: split — the eval harness lands the moment #154's pipeline ships; the composition waits on #150. Splitting unblocks the harness against ~6 weeks of #150 dev time.
- [ ] **Confirm MRR uplift threshold `+0.05`.** v1.3 ships against `+0.02`; v2.0 with the additional terms should clear a higher bar. Alternative: keep `+0.02` to make the gate easier. Recommendation: `+0.05` — if the additional terms can't move MRR an extra 3pp on a synthetic corpus, they aren't paying for the per-query latency cost.
- [ ] **Confirm ECE threshold `0.10`.** ECE convention in the calibration literature treats `0.05` as "well-calibrated" and `0.10` as "acceptable". Recommendation: `0.10` for synthetic; defer real-feedback ECE until sample size supports a tighter gate.
- [ ] **Confirm weight sweep default range `[0.0, 1.0]` step `0.1`.** Tighter step (0.05) doubles the runtime; coarser (0.2) misses the optimum. Recommendation: 0.1 for the default, configurable for users who want finer.
- [ ] **Confirm the real-feedback retest is held outside v2.0.** Alternative: gate v2.0 on a real-feedback retest, which would push v2.0 by months. Recommendation: ship synthetic at v2.0, real-feedback in a v2.x point release once corpus size supports it.

## Why queen

The five calls above are the queen work. Once thresholds and sequencing are settled, implementation is mechanical (~300 LOC harness + ~100 LOC composition + ~150 LOC sweep + ~400 LOC tests). Without the calls settled, the eval harness grows a flag matrix to support multiple thresholds and the sweep grows multiple objectives.

## Downstream impact

- **Public claim alignment:** [`docs/LIMITATIONS.md § "The big one"`](LIMITATIONS.md), [`docs/CONFIG.md`](CONFIG.md), and [`docs/bayesian_ranking.md`](bayesian_ranking.md) all currently caveat "10-round MRR uplift, ECE calibration, BM25F + heat-kernel composition — lands at v2.0.0". This issue is what makes that caveat true.
- **`aelf bench` surface grows:** new `--posterior-residual` mode, new `--sweep` mode. Both opt-in; existing `aelf bench` behavior unchanged.
- **`scoring.py` contract:** one line added to `partial_bayesian_score` (or its successor); existing v1.3 callers see no behavior change at default weights.
- **README claim:** the "feedback drives retrieval" claim already shipped at v1.3. v2.0 strengthens it from "partial" to "full" with the full eval published alongside.
- **`benchmarks/`:** new subdirectory; no impact on the existing synthetic-graph harness.

## Risk and mitigations

- **Synthetic-real divergence.** Synthetic feedback is correlated with graph centrality; real feedback is user-utility driven and may have a different signature. Mitigation: real-feedback retest is filed as a follow-up issue with a hard threshold (≥1000 events / ≥100 beliefs) before it runs. v2.0's claim is explicitly synthetic-fixture-bounded.
- **Heat-kernel availability.** Slice 2 cannot land until #150 ships. Mitigation: the eval harness (Slice 1) and weight sweep (Slice 3) both work with `heat_kernel_weight = 0.0`, so they ship independently of #150.
- **Weight sweep over-tuning.** A user who runs the sweep daily and pushes the optimum each time will overfit to recent feedback. Mitigation: sweep output is a recommendation, never auto-applied. Document this in the CLI help text.

## Out of scope (deferred)

- **Confidence-weighted posterior** (variance-aware shrinkage). Filed as separate v2.x issue.
- **Multi-axis posterior** (depends on #196 Option A). If multi-axis lands, this spec's recommendation is to compose per-axis posteriors via the existing scalar projection rather than re-derive the score equation.
- **Retrieval-time `(α, β)` decay.** Separate issue, separate cohort of failure modes (stale feedback) that this spec doesn't address.
- **HRR structural lane composition** (#152). The structural lane is its own scoring path; this spec covers only the textual lane.

## Provenance

- v1.3 partial spec: [`bayesian_ranking.md`](bayesian_ranking.md). This memo extends rather than replaces.
- Pipeline contract: [#154](https://github.com/robotrocketscience/aelfrice/issues/154) defines `retrieve()` composing the three terms.
- Re-scope comment: [#151 closing comment](https://github.com/robotrocketscience/aelfrice/issues/151#issuecomment) handed v2.0 the residual after #146 shipped the core.
- Substrate: [`substrate_decision.md`](substrate_decision.md) — independent of this spec; either substrate option preserves the scalar `posterior_mean` interface this spec consumes.
- BM25F dependency: #148 (shipped). Heat-kernel dependency: #150 (not started).
- Adjacent specs: [`v2_replay.md`](v2_replay.md), [`v2_view_flip.md`](v2_view_flip.md), [`v2_derivation_worker.md`](v2_derivation_worker.md) — all v2.x cascade addenda; this memo follows the same pattern.
