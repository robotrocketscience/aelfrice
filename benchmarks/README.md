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

The adapters were ported from the private lab repo
(`aelfrice-lab/benchmarks/`). All five academic adapters run
end-to-end under `aelfrice.retrieval.retrieve_v2` + the `[benchmarks]`
extra. v3.x adapters target the canonical post-#605 substrate (HRR
structural lane + BFS multi-hop + posterior-weighted ranking all
default-on per v2.1; type-aware compression default-on per v3.0
#769). The reproducibility harness (#437) is the canonical entry
point — `aelf bench all` runs the full academic suite.

| File | Imports | Retrieve-only | Notes |
|---|---|---|---|
| `verify_clean.py` | OK | runs | stdlib only |
| `mab_adapter.py` | OK with `[benchmarks]` extras | runs | needs `nltk` + `tiktoken` |
| `mab_reader.py` | OK | runs | LLM reader; gated behind workflow_dispatch in CI |
| `locomo_adapter.py` | OK with `[benchmarks]` extras | runs | needs `nltk` |
| `locomo_generate.py` | OK | runs | stdlib only |
| `locomo_score.py` | OK with `[benchmarks]` extras | runs | scoring after adapter |
| `locomo_score_protocol.py` | OK with `[benchmarks]` extras | runs | scoring after adapter |
| `longmemeval_adapter.py` | OK | **smoked** (15 Q, baseline in `results/v1.2.0-pre.json`) | retrieve-only path validated; reader/judge passes deferred |
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

- Public `retrieve()` covers L0 (locked) + L1 (FTS5 BM25/BM25F) +
  L2 (BFS multi-hop, default-on since v1.3) + L2.5 (HRR structural
  lane, default-on since v2.1 #154) with Bayesian log-additive
  reranking.
- Adapters call `retrieve_v2(...)` directly; the legacy `use_hrr`
  alias was retired (see #536). The live kwarg is
  `use_hrr_structural`.
- LongMemEval session-level Recall@5 ≥ 0.634 (paper baseline) is
  the floor; benchmarks/results/v2.0.0.json onward records actual
  measurements.

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

## Protocol

See [`docs/concepts/BENCHMARKS.md`](../docs/concepts/BENCHMARKS.md) for the 5-step run protocol (env check, data acquisition, retrieval-only, contamination check, reader generation, scoring). The protocol is stable across phases; only the activation status of individual adapters changes.

## Datasets

Adapters pull from upstream sources rather than vendoring data. Each adapter pins its own dataset SHA in its module header.

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
