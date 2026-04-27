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

## Activation status (aelfrice v1.0.0)

The adapters were ported from the private lab repo
(`aelfrice-lab/benchmarks/`) where they target lab v2.0.0. They
require modules that have not yet ported into the public v1.0.0
release. They are present in the public tree as **scaffold for
future activation**, not because they currently run.

| File | Status | Activates in |
|---|---|---|
| `verify_clean.py` | **runnable** | v1.1.0 (P1) |
| `mab_adapter.py` | inert — needs `aelfrice.ingest`, `MemoryStore` | v1.2.0 (P2) |
| `mab_reader.py` | runnable (stdlib + `anthropic` only) | v1.1.0 (P1) |
| `locomo_adapter.py` | inert — needs `aelfrice.ingest`, `MemoryStore` | v1.2.0 (P2) |
| `locomo_generate.py` | runnable (stdlib only) | v1.1.0 (P1) |
| `locomo_score.py` | inert — depends on `locomo_adapter` symbols | v1.2.0 (P2) |
| `locomo_score_protocol.py` | inert — depends on `locomo_adapter` symbols | v1.2.0 (P2) |
| `longmemeval_adapter.py` | inert — needs `aelfrice.ingest`, `MemoryStore` | v1.2.0 (P2) |
| `longmemeval_budget_sweep.py` | inert — depends on adapter | v1.2.0 (P2) |
| `longmemeval_score.py` | runnable (stdlib only) | v1.1.0 (P1) |
| `structmemeval_adapter.py` | inert — needs `aelfrice.ingest`, `MemoryStore` | v1.2.0 (P2) |
| `amabench_adapter.py` | inert — needs `aelfrice.ingest`, `MemoryStore` | v1.2.0 (P2) |

Three additional adapters from the lab (`mab_triple_adapter.py`,
`mab_entity_index_adapter.py`, `mab_llm_entity_adapter.py`) are
**not yet present**; they require triple-extraction (P2) and
entity-index retrieval (P3) and will land alongside those features.

## Missing public-surface dependencies

The adapters import the following lab-side modules that have not
yet ported to the public v1.0.0 release:

- `aelfrice.ingest` — `ingest_turn()` end-to-end ingest pipeline
- (transitively) `aelfrice.extraction`, `aelfrice.relationship_detector`,
  `aelfrice.supersession`, `aelfrice.triple_extraction`

The `Store` → `MemoryStore` rename landed in P2 (`feat/p2-ingest-port`)
ahead of the ingest port itself; adapters now find the class they
expect.

These ports are scheduled in P2/P3/P4 of the v2.0.0 milestone plan
(see private `aelfrice-lab/docs/V2_BENCHMARK_MILESTONE_PLAN.md`).

## What runs today (v1.0.0)

```bash
# Contamination gate (verifies a retrieval file has no answer/gt leakage)
uv run aelf bench verify-clean path/to/retrieval.json

# Score a LongMemEval predictions file (no aelfrice imports needed)
uv run aelf bench longmemeval-score preds.json gt.json
```

Anything else exits 2 with a friendly pointer back to this file.

## Protocol

See `docs/BENCHMARKS.md` for the 5-step run protocol (env check,
data acquisition, retrieval-only, contamination check, reader
generation, scoring). The protocol is stable across phases; only
the activation status of individual adapters changes.

## Datasets

When adapters activate, they pull from upstream sources rather
than vendoring data. See `benchmarks/datasets.toml` (ships with
P2) for pinned commit SHAs and sha256 checksums.

| Benchmark | Source | License |
|---|---|---|
| MAB | HuggingFace `huangchaoyi/MemoryAgentBench` | check upstream |
| LoCoMo | github.com/snap-research/locomo | check upstream |
| LongMemEval | HuggingFace `xiaowu0162/longmemeval` | check upstream |
| StructMemEval | github.com/yandex-research/StructMemEval | check upstream |
| AMA-Bench | (TBD on activation) | check upstream |

License verification happens at activation time per benchmark, not
at scaffold time.
