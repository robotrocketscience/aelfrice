# aelfrice external benchmark suite

Reproducibility harness for the headline numbers published at
`www.robotrocketscience.com/projects/agentmemory/`.

This directory is the **academic suite** — external benchmarks
(MAB, LoCoMo, LongMemEval, StructMemEval, AMA-Bench) that run
against published datasets. It is distinct from the
**synthetic regression harness** at `src/aelfrice/benchmark.py`,
which is a small in-tree corpus that runs in CI on every PR.

| Surface | Location | Purpose | Runtime | Cost |
|---|---|---|---|---|
| Synthetic regression | `src/aelfrice/benchmark.py` | Catch retrieval/scoring regressions | <1s | $0 |
| Academic suite | `benchmarks/` (here) | Reproduce website headline numbers | minutes–hours | LLM API spend |

## Activation status (aelfrice v3.x)

All five academic adapters run end-to-end under
`aelfrice.retrieval.retrieve_v2` + the `[benchmarks]` extra. v3.x adapters target the canonical post-#605 substrate (HRR
structural lane + posterior-weighted ranking default-on since v2.1
per #154/#437; BFS multi-hop remains default-off — adapters opt in
via `use_bfs=True`, default-flip bench-gated under #739; type-aware
compression default-on per v3.3 #769). The reproducibility harness (#437) is the canonical entry
point — `aelf bench all` runs the full academic suite.

| File | Imports | Retrieve-only | Notes |
|---|---|---|---|
| `verify_clean.py` | OK | runs | stdlib only |
| `mab_adapter.py` | OK with `[benchmarks]` extras | runs | needs `nltk` + `tiktoken` |
| `mab_reader.py` | needs the `[onboard-llm]` extra's SDK dependency (not covered by `[benchmarks]`) | runs | LLM reader; operator-run only — never invoked by CI |
| `locomo_adapter.py` | OK with `[benchmarks]` extras | runs | needs `nltk` |
| `locomo_generate.py` | OK | runs | stdlib only |
| `locomo_score.py` | OK with `[benchmarks]` extras | runs | scoring after adapter |
| `locomo_score_protocol.py` | OK with `[benchmarks]` extras | runs | scoring after adapter |
| `longmemeval_adapter.py` | OK | runs (full 500-Q retrieve-only cut in `results/v2.0.0.json` + nightly bench-canonical) | first 15-Q smoke preserved in `results/v1.2.0-pre.json`; reader/judge passes still operator-driven |
| `longmemeval_budget_sweep.py` | OK | runs | depends on adapter |
| `longmemeval_score.py` | OK | runs | stdlib only |
| `structmemeval_adapter.py` | OK | runs | retrieve-only path |
| `amabench_adapter.py` | OK with `[benchmarks]` extras | runs | needs `datasets` |

Two additional adapters from the lab (`mab_triple_adapter.py` and
`mab_llm_entity_adapter.py`) remain unported. `mab_entity_index_adapter.py`
shipped and is in tree.

### Substrate at v3.x

Retrieve-only runs under the current substrate exercise the full
post-#605 retrieval stack:

- Public `retrieve()` covers L0 (locked) + L2.5 (entity-index,
  default-on since v1.3) + L1 (FTS5 BM25/BM25F) + L3 (BFS
  multi-hop, shipped v1.3, default-off — adapters opt in with
  `use_bfs=True`) with Bayesian log-additive reranking;
  `retrieve_v2` additionally routes structural-marker queries to
  the HRR lane (default-on since v2.1 #154).
- Adapters call `retrieve_v2(...)` directly; the legacy `use_hrr`
  alias was retired (see #536). The live kwarg is
  `use_hrr_structural`.
- LongMemEval session-level Recall@5 ≥ 0.634 (paper baseline) is
  the floor; benchmarks/results/v2.0.0.json onward records
  retrieve-only telemetry (latency, beliefs-per-query) — Recall@5
  against that floor is not yet captured in the results files.

### Running benchmarks locally

```bash
# Install with the benchmarks extras:
uv pip install -e ".[benchmarks]"

# Run a tiny LongMemEval smoke (3 questions, retrieve-only):
PYTHONPATH=. uv run python benchmarks/longmemeval_adapter.py \
    --subset 3 --retrieve-only /tmp/lme_smoke.json

# Verify no contamination in the retrieval file:
uv run python -m benchmarks.verify_clean /tmp/lme_smoke.json

# Or use the canonical entry point:
uv run aelf bench all --smoke --out /tmp/bench-smoke.json
```

## What these harnesses measure, and what they do not

**Agreement is not quality.** Read this before you quote a number out of any
harness in this directory.

Some of these instruments report how much two rankers agree: top-10 movement
rate, Jaccard overlap at k, Kendall tau, "N of M queries changed". Every one of
those is a **change** statistic. It tells you the two configurations return
different results. It tells you nothing about which of them is right.

A movement rate becomes a **quality** statistic only against labelled
relevance — a judgement, external to both rankers, about which documents the
query should have returned. Without labels, a large movement and a small
movement are equally consistent with an improvement, a regression, and noise.

Two failure modes follow, and both have occurred here:

1. **Reporting a movement rate as if it were an uplift.** "This change moved
   47% of top-10s" reads as an effect. It is not one.
2. **Treating agreement between two approximations as validation.** If neither
   arm scores what production scores, their agreement measures the shared
   approximation, not the production behaviour.

The second is why this section exists. Commit
[`848dbf83`](https://github.com/robotrocketscience/aelfrice/commit/848dbf83aab6e2c5198b1bc5b0f70ebcc7f11ea9)
— *"revert(bench): stop calling the recorded query the production population"* —
retracted a published table for exactly this reason. The recorded
`extracted_query` was handed to `retrieve()` by **neither** caller: 97.7% of
recorded rows come from `user_prompt_submit`, which scores a conversation-aware
composition built in `hook.py` that the context rebuilder never sees. Both arms
were relabelled as approximations, and neither is quoted as production.

So, when you run anything here:

- Say whether your figure is a change statistic or a quality statistic. If you
  cannot name the labels, it is a change statistic.
- Name the population the arm actually scored, not the population you wanted it
  to score.
- Stamp the corpus identity with the result, following the
  `benchmarks/scan_admission_funnel.py` precedent.
- If the population is too small to support the claim, report that and stop.
  Measuring anyway and qualifying it in prose is how a retracted number gets
  quoted a second time.

## Protocol

See [`docs/concepts/BENCHMARKS.md`](../docs/concepts/BENCHMARKS.md) for the run protocol (retrieval-only, contamination check, reader generation, scoring, audit record). The protocol is stable across phases; only the activation status of individual adapters changes.

## Datasets

Adapters pull from upstream sources rather than vendoring data. Each adapter pins its upstream dataset identifier (HuggingFace dataset id or expected local path) in its module header; no immutable revision/SHA is pinned.

| Benchmark | Source (actual, per adapter pin) | License |
|---|---|---|
| MAB | HuggingFace `ai-hyz/MemoryAgentBench` | MIT |
| LoCoMo | github.com/snap-research/locomo | CC BY-NC 4.0 |
| LongMemEval | HuggingFace `xiaowu0162/longmemeval-cleaned` | MIT |
| StructMemEval | github.com/yandex-research/StructMemEval | no LICENSE file (default copyright) |
| AMA-Bench | (TBD on activation) | check upstream |

License verification happens at activation time per benchmark, not
at scaffold time. The MAB and LongMemEval entries match the actual
`HF_DATASET` pins in `mab_adapter.py` and `longmemeval_adapter.py`;
earlier versions of this table pointed at related-but-different
upstreams (`huangchaoyi/MemoryAgentBench`, `xiaowu0162/longmemeval`)
that the adapters do not actually load.

PR-smoke fixtures under `tests/fixtures/bench_smoke/` are
schema-matching synthetic data — not derived from any of these
upstreams — per the activation-time license review on #476.
Real-data shape coverage continues to come from the nightly
`bench-canonical` cron.
