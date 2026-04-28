# Benchmarks

aelfrice ships two benchmark surfaces with different purposes and cadences.

| Surface | Location | Purpose | Runtime | Cost | Cadence |
|---|---|---|---|---|---|
| Synthetic regression | `src/aelfrice/benchmark.py` | Catch retrieval/scoring regressions | <1s | $0 | Every PR (CI) |
| Academic suite | `benchmarks/` | Reproduce published numbers vs. external benchmarks | minutes–hours | LLM API spend | Nightly + on-tag |

The synthetic harness is a measurement instrument. It is **not** a proof of the central feedback claim — at v1.0–v1.2 the posterior doesn't drive ranking yet. See [LIMITATIONS](LIMITATIONS.md).

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
aelf bench verify-clean /tmp/benchmark_<name>.json
```

If this fails, the run is invalid. Fix the adapter and re-run.

## Run the protocol

```bash
# 1. Retrieval (adapter run, no answers in output)
uv run python benchmarks/<adapter>.py \
    --retrieve-only /tmp/benchmark_<name>.json [--subset N]

# 2. Verify the retrieval file is clean
aelf bench verify-clean /tmp/benchmark_<name>.json

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
