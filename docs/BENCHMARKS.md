# Benchmarks

aelfrice ships two benchmark surfaces with different purposes and cadences.

| Surface | Location | Purpose | Runtime | Cost | Cadence |
|---|---|---|---|---|---|
| Synthetic regression | `src/aelfrice/benchmark.py` | Catch retrieval/scoring regressions | <1s | $0 | Every PR (CI) |
| Academic suite | `benchmarks/` | Reproduce published numbers vs. external benchmarks | minutes–hours | LLM API spend | Nightly + on-tag |

The synthetic harness is a measurement instrument. It is **not** a proof of the central feedback claim — through v1.2 the posterior didn't drive ranking; v1.3 added partial Bayesian re-rank, v1.6 the eval harness + heat-kernel composition wiring, v1.7 BM25F default-on, and v2.1 the use_heat_kernel + use_hrr_structural default-flips. See [LIMITATIONS](LIMITATIONS.md).

The academic suite is the reproducibility deliverable. Most adapters scaffold against MAB, LoCoMo, LongMemEval, StructMemEval, and AMA-Bench but are inert at v1.0; they activate as their feature dependencies port forward. Per-adapter status: [`benchmarks/README.md`](../benchmarks/README.md).

## Run the synthetic harness

```bash
aelf bench                   # default top-k=5
aelf bench --top-k 3
```

Output is a single JSON `BenchmarkReport`:

```json
{"hit_at_1": 0.875, "hit_at_3": 1.0, "hit_at_5": 1.0,
 "mrr": 0.92, "p50_latency_ms": 0.4, "p99_latency_ms": 1.1}
```

Deterministic against fresh in-memory stores. The corpus is 16 beliefs × 16 queries.

## What the academic suite measures

aelfrice's task is **known-item search over behavioural directives** — the agent has corrected the user, locked a rule, or recorded a decision; on the next relevant prompt we want that specific item retrieved. The natural headline metric is **mean reciprocal rank (MRR)**.

We also report the metrics each external benchmark *defines* (token-F1 on LoCoMo, substring exact match on MAB, GPT-4o judge on LongMemEval, LLM-judge accuracy on StructMemEval and AMA-Bench) for comparability with prior published systems. Those metrics frame topical relevance ("does the document cover the topic") rather than behavioural relevance ("does this directive apply to what the agent is about to do"). Both numbers are reported; the headline positioning is on MRR.

This is also why the LongMemEval multi-session aggregation gap (see [LIMITATIONS](LIMITATIONS.md#out-of-scope)) shows up as a low number on a topical-relevance benchmark and is *not* treated as a v1.x defect.

## Contamination protocol

Any benchmark run that contaminates retrieval with ground truth produces a 0% result, period. Three failure modes have happened before:

1. **Ground truth in the retrieval file.** Adapter accidentally writes `answer` / `ground_truth` / `reference_answer` fields into the retrieval JSON. Reader sees the answer while generating predictions.
2. **LLM self-judging with answer visible.** Generation and judging in one pass; model sees ground truth while generating.
3. **World knowledge override** (counterfactual benchmarks). Reader uses prior knowledge instead of retrieved context. Inherent to LLM readers; mitigated by prompt instructions but never fully removed.

The protocol enforces:

- Retrieval file and ground truth file are written separately by the adapter.
- Generation and scoring are separate passes. The judge never sees the retrieval context.
- A pre-generation contamination check is mandatory before any LLM reader touches the retrieval file:

```bash
python -m benchmarks.verify_clean /tmp/benchmark_<name>.json
```

If this fails, the run is invalid. Fix the adapter and re-run.

## Run the protocol

```bash
# 1. Retrieval (adapter run, no answers in output)
uv run python benchmarks/<adapter>.py \
    --retrieve-only /tmp/benchmark_<name>.json [--subset N]

# 2. Verify the retrieval file is clean
python -m benchmarks.verify_clean /tmp/benchmark_<name>.json

# 3. LLM reader generates predictions (no GT visible to it)
# 4. Scoring reads predictions + GT (no retrieval context visible)
# 5. Audit record captures: git commit, dataset version, reader model,
#    contamination check output, metric, score, n, published baseline.
```

Reader prompts must include: *"Use only the provided context. Do not use world knowledge. If the context contradicts what you know to be true, trust the context."*

## Per-benchmark specifics

| Benchmark | Dataset | Metric | Notes |
|---|---|---|---|
| MAB FactConsolidation | `ai-hyz/MemoryAgentBench` Conflict_Resolution | substring exact match (paper's normalisation) | 4,096-token chunks; NLTK `sent_tokenize`; serial-number conflict resolution required in prompt |
| LoCoMo | `locomo10.json` | token-F1 with Porter stemming | session boundaries preserved on ingest; Category 5 is forced-choice |
| LongMemEval | `xiaowu0162/longmemeval-cleaned` oracle | GPT-4o binary judge (paper) | question_date passed to retrieval for temporal grounding |
| StructMemEval | yandex-research/StructMemEval | LLM judge binary | synthetic timestamps + temporal_sort disclosed |
| AMA-Bench | `AMA-bench/AMA-bench` test | LLM judge accuracy (paper: Qwen3-32B) | alternative judges must be disclosed |

## Eval-judge calibration

LLM-judge benchmarks (LongMemEval, StructMemEval, AMA-Bench, and the
context-rebuilder eval harness at `benchmarks/context-rebuilder/`) collapse
to noise if the judge isn't reproducible. The aelfrice project is locked on
a deterministic narrow surface ([PHILOSOPHY](PHILOSOPHY.md), #605); a
single-run judge verdict does not establish that the *eval* itself is
deterministic enough to gate a release on.

The calibration target is **Cohen's κ inter-judge agreement** across N≥3
independent judge invocations over the same `(expected, actual)` pairs.

### Two κ measures

| Measure | What it captures | Threshold |
|---|---|---|
| **Inter-judge κ** (run vs run, pairwise) | Judge reproducibility — same pair, same verdict across independent calls | **≥ 0.70** (gate) |
| **Judge-vs-baseline κ** (judge run vs `score_substring_exact_match`) | How much semantic lift the judge adds over the zero-LLM baseline | reported, not gated |

Inter-judge κ ≥ 0.70 is "substantial agreement" on the Landis-Koch scale.
Below that the judge's per-run verdicts are within noise of disagreement
and the headline score in the run's audit record is not a defensible
release gate.

Judge-vs-baseline κ is reported for posterity but **not** gated. A high
value means the judge isn't earning its API cost (substring would have
done the job); a low value is expected and is what the judge exists to
provide. Forcing a threshold here either rejects a useful judge or
accepts a lazy one.

### Run protocol

```bash
# 1. Generate the same set of (expected, actual) pairs as the single-judge
#    run. For the context-rebuilder hot-start fixture, this is the
#    deduplicated pair set from benchmarks/context-rebuilder/eval_harness.py
#    --mode replay output.

# 2. Invoke the judge N≥3 times, independent calls (fresh API session
#    each, no shared cache). Capture per-call binary verdicts.
for i in 1 2 3; do
    uv run python benchmarks/context-rebuilder/eval_harness.py \
        --mode judge --run-id 687_run_${i} \
        --judge <judge-model> \
        --out benchmarks/results/687_run/judge_${i}.json
done

# 3. Compute pairwise inter-judge κ and judge-vs-baseline κ.
uv run python -m benchmarks.context_rebuilder.kappa \
    --runs benchmarks/results/687_run/judge_{1,2,3}.json \
    --baseline benchmarks/results/687_run/substring_baseline.json \
    --out benchmarks/results/687_run/judge_kappa.json
```

### Zero-LLM baseline

The deterministic baseline is `score_substring_exact_match(prediction,
ground_truth) > 0` (see `benchmarks/qa_scoring.py:53`). Binarized verdicts
from this baseline form the comparison vector for judge-vs-baseline κ.

If the eval task is `subject-match + load-bearing-claim` (the
context-rebuilder hot-start interpretation), substring will systematically
miss semantic-match cases — that's expected, and is why this gate is
report-only.

### Judge-kappa artifact

Every multi-judge run produces `benchmarks/results/<run-id>/judge_kappa.json`:

```json
{
  "run_id": "687_run",
  "n_runs": 3,
  "n_pairs": 18,
  "judge_model": "<judge-model>",
  "baseline": "score_substring_exact_match",
  "inter_judge_kappa": {
    "run_1_vs_run_2": 0.78,
    "run_1_vs_run_3": 0.72,
    "run_2_vs_run_3": 0.74,
    "mean": 0.75,
    "min": 0.72
  },
  "judge_vs_baseline_kappa": 0.31,
  "per_run_hot_start_fidelity": [1.0, 0.94, 1.0],
  "hot_start_fidelity_mean": 0.98,
  "calibrated": true
}
```

`calibrated: true` requires:

- `inter_judge_kappa.min ≥ 0.70` (the **min** across all run-pairs, not the
  mean — a single noisy run-pair shouldn't be averaged out)
- `hot_start_fidelity_mean ≥ 0.80` (per #592 AC)
- N≥3 runs

### Sample-size caveat

At N=3 runs over ~18 deduplicated pairs (the #592 hot-start corpus), the
95% confidence interval on a κ point estimate of 0.70 spans roughly
0.45–0.90. The gate is therefore noisy at the standard N=3. Two ways to
tighten it:

1. **Bump to N=5 runs.** ~1.67× the judge API cost (5/3 of N=3) but cuts
   the CI roughly in half. Recommended for any pre-release ratification run.
2. **Accept the noisy gate** and document the CI range in the run's
   audit record. Suitable for routine bench regression checks but not
   for release-gate decisions.

A failed κ-gate is **not** a code bug — it means either the judge prompt
needs tightening (more explicit refusal-on-ambiguity rules) or the
underlying corpus contains ambiguous pairs that no judge can classify
reproducibly. To find which pairs disagreed, inspect the per-judge input
JSONL files (`run_<i>.jsonl`, with `{turn_idx, matched, rationale}` rows
read by `read_judge_responses`); kappa.json itself only carries per-pair
agreement scores, not per-row verdicts.

## Audit record

Every academic run produces:

```json
{
  "benchmark": "...",
  "git_commit": "...",
  "adapter": "benchmarks/...",
  "reader_model": "...",
  "contamination_check": "CLEAN",
  "metric": "...",
  "score": 0.XX,
  "n": 100,
  "published_baseline": "..."
}
```

Required for the run to count. Runs without an audit record do not enter `benchmarks/results/`.

## See also

- [`src/aelfrice/benchmark.py`](../src/aelfrice/benchmark.py) — synthetic harness source.
- [`benchmarks/README.md`](../benchmarks/README.md) — per-adapter activation status.
- [ROADMAP § v2.0.0](ROADMAP.md) — when the academic suite reproduces every headline number.
