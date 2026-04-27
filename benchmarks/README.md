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

## Activation status (aelfrice v1.0.x post-P2 ingest shim)

The adapters were ported from the private lab repo
(`aelfrice-lab/benchmarks/`) where they target lab v2.0.0. P2
landed an ingest-pipeline shim (`aelfrice.ingest.ingest_turn`,
`aelfrice.extraction`, `MemoryStore.create_session`) so all five
academic adapters now **import successfully**. End-to-end runs
remain blocked on per-adapter issues:

| File | Imports | End-to-end | Notes |
|---|---|---|---|
| `verify_clean.py` | OK | runs | stdlib only |
| `mab_adapter.py` | needs `nltk` + `tiktoken` | blocked | retrieval kwargs `budget`/`use_hrr`/`use_bfs` differ from public `retrieve(token_budget=...)` |
| `mab_reader.py` | needs `anthropic` (already a dep) | runs | LLM reader; not an aelfrice ingest path |
| `locomo_adapter.py` | needs `nltk` | blocked | same retrieval-kwarg gap |
| `locomo_generate.py` | OK | runs | stdlib only |
| `locomo_score.py` | needs `nltk` (via adapter) | blocked | scoring depends on adapter symbols |
| `locomo_score_protocol.py` | same | blocked | same |
| `longmemeval_adapter.py` | OK | blocked | retrieval-kwarg gap |
| `longmemeval_budget_sweep.py` | OK | blocked | depends on adapter |
| `longmemeval_score.py` | OK | runs | stdlib only |
| `structmemeval_adapter.py` | OK | blocked | retrieval-kwarg gap |
| `amabench_adapter.py` | needs `datasets` (HuggingFace) | blocked | retrieval-kwarg gap |

Three additional adapters from the lab (`mab_triple_adapter.py`,
`mab_entity_index_adapter.py`, `mab_llm_entity_adapter.py`) are
**not yet present**; they require triple-extraction (P3) and
entity-index retrieval (P4) and will land alongside those features.

### Remaining blockers for end-to-end runs

1. **Optional dependencies.** `nltk`, `tiktoken`, `datasets`. Add to
   `[project.optional-dependencies] benchmarks` and require
   `pip install aelfrice[benchmarks]` for end-to-end use.
2. **Retrieval-kwarg gap.** Adapters call:

   ```python
   result = retrieve(store=store, query=q, budget=N,
                     include_locked=False, use_hrr=True, use_bfs=True)
   parts = [b.content for b in result.beliefs]
   ```

   Public v1.0.x `retrieve` returns `list[Belief]` directly and
   accepts `token_budget` (not `budget`); has no `include_locked`,
   `use_hrr`, or `use_bfs`. Two paths to close this:
   - Patch each adapter to use the public signature (intrusive).
   - Add `aelfrice.retrieval.retrieve_v2(...)` wrapper with the
     lab signature; HRR/BFS flags no-op until ported. (Recommended.)

   This wrapper is the next chunk of P2 (or rolled into P3).

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
