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

## Activation status (aelfrice v1.0.x post-retrieve_v2 wrapper)

The adapters were ported from the private lab repo
(`aelfrice-lab/benchmarks/`) where they target lab v2.0.0. P2
landed the ingest-pipeline shim. The follow-up wrapper PR adds
`aelfrice.retrieval.retrieve_v2` and `[benchmarks]` extras; all
five academic adapters now run end-to-end in `--retrieve-only`
mode. Numbers produced are the L0+FTS5 baseline.

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

Three additional adapters from the lab (`mab_triple_adapter.py`,
`mab_entity_index_adapter.py`, `mab_llm_entity_adapter.py`) are
**not yet present**; they require triple-extraction (P3) and
entity-index retrieval (P4) and will land alongside those features.

### Baseline behavior at v1.0.x

Retrieve-only runs produce **avg_beliefs ≈ 0** on LongMemEval at
this version. This is the expected, documented behavior:

- Public `retrieve()` is L0 (locked) + L1 (FTS5 BM25). No HRR
  vocabulary bridge, no BFS multi-hop chaining.
- Adapters call `retrieve_v2(use_bfs=True)`. The structural HRR
  lane is on by default since v2.1; the legacy `use_hrr` alias was
  removed in #536 and now raises `TypeError` if passed.
- The LongMemEval paper itself reports BM25 session-level
  Recall@5 = 0.634 — the BM25-only failure mode is precedented.

Real numbers (matching the website's headline 59.0% on
LongMemEval) appear once HRR (P4) and BFS (P3) port from lab.

### Running benchmarks locally

```bash
# Install with the benchmarks extras:
pip install -e ".[benchmarks]"

# Or with uv:
uv pip install -e ".[benchmarks]"

# Run a tiny LongMemEval smoke (3 questions, retrieve-only):
PYTHONPATH=. python benchmarks/longmemeval_adapter.py \
    --subset 3 --retrieve-only /tmp/lme_smoke.json

# Verify no contamination in the retrieval file:
python -m benchmarks.verify_clean /tmp/lme_smoke.json
```

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
uv run python -m benchmarks.verify_clean path/to/retrieval.json

# Score a LongMemEval predictions file (no aelfrice imports needed)
uv run python -m benchmarks.longmemeval_score preds.json gt.json
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
