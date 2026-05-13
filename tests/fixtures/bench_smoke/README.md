# bench_smoke fixtures — synthetic, schema-matching, not derived

These four fixtures drive `tests/test_bench_smoke.py` (and the CI job
`bench-smoke`). They exist so the four benchmark adapters
(`benchmarks/{locomo,mab,longmemeval,structmemeval}_adapter.py`) can
be exercised offline on every PR without a HuggingFace download or an
external clone — a 2-minute dispatcher-shape sanity check, distinct
from the nightly `bench-canonical` cron that runs real corpora.

## What they are

- `locomo_micro.json` — 1 LoCoMo conversation, 2 sessions, 9 turns, 3 QA pairs.
- `mab_micro.json` — 4 MAB rows, 7 questions across all four splits.
- `longmemeval_micro.json` — 3 LongMemEval questions, retrieve-only.
- `structmemeval_micro/` — 1 case per task type (location, accounting, recommendations, tree).

Each file matches its adapter's expected schema exactly. No
real-data shape coverage — the conversations, contexts, and answers
were authored from scratch for this fixture.

## Why they are synthetic

Per the activation-time license review on issue #476, the decision was
to **synthesize all four** rather than excerpt from upstream:

| Upstream | License | Why not excerpt |
|---|---|---|
| `snap-research/locomo` | CC BY-NC 4.0 | NonCommercial clause copylefts to any derivative; would constrain future commercial use of aelfrice. |
| `ai-hyz/MemoryAgentBench` | MIT | Permissive, but bundled with the others to keep a single clean redistribution story. |
| `xiaowu0162/longmemeval-cleaned` | MIT | Same — uniform synthesis simplifies the audit trail. |
| `yandex-research/StructMemEval` | (no LICENSE file) | Upstream not verifiably licensed; cannot redistribute derivatives. |

Real-data shape coverage lives in the nightly `bench-canonical` cron
(see `benchmarks/README.md` § "Datasets"). The smoke path covers
the dispatcher contract: that each adapter's `load → ingest → retrieve`
sequence completes against schema-shaped input.

## What changes if the smoke job catches a regression

A real adapter break (changed entrypoint, broken `--retrieve-only` path,
schema parse error) trips the smoke job and blocks the PR. A
real-data-only regression (e.g. F1 drop on a category we don't
synthesize for) is invisible here and only shows up in the nightly run.
