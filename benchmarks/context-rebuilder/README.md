# context-rebuilder eval harness

Skeleton for the eval that backs [`docs/context_rebuilder.md`](../../docs/context_rebuilder.md).

## What this measures

**Continuation fidelity at fixed token budget.** Given a real working
session captured as a `turns.jsonl` (per the v1.2.0 transcript-ingest
spec), fork at midpoint turn `T`, force a clear, run the rebuilder,
replay turns `T+1..end`, and ask: does the agent continue as it did
originally, on a smaller context budget?

Three numbers per run:

1. **fidelity** — fraction of post-clear turns where the agent's
   answer matches the original session's answer. Judge: deterministic
   string-match for explicit recall questions; LLM-judge for
   open-ended turns.
2. **token_cost_ratio** — rebuild block size / pre-clear context size.
3. **rebuild_latency_ms** — wall-time of the PreCompact hook.

## Layout

- `eval_harness.py` — entry point.
- `eval_corpus/` — gitignored. Holds scrubbed `turns.jsonl` files
  for replay. Populate from real working sessions; PII-scrub each
  before adding to the corpus.
- `judges/` — fidelity scorers (string-match + LLM-judge). To be
  added in v1.3.0.
- `results/` — per-run JSON output. Gitignored except for
  `calibration_*.json` files referenced from the spec.

## Run modes

- `--mode threshold-sweep` — sweep PreCompact trigger thresholds
  (50/60/70/80/90%) per task type, output the fidelity-vs-trigger
  curve. Default for calibration.
- `--mode dynamic` — evaluate the dynamic-threshold heuristic
  against fixed thresholds. Gate for shipping dynamic mode.
- `--mode budget-sweep` — sweep token budgets (500/1000/2000/4000)
  at a fixed trigger. Drives the default budget choice.
- `--mode regression` — fixed config, pass/fail against the v1.0.0
  baseline numbers. Run on every release.

## Open questions answered by the eval, not the spec

1. The default trigger threshold for v1.4.0.
2. The default token budget for v1.4.0.
3. Whether dynamic-mode improves over a fixed threshold.
4. Whether augment-mode loses fidelity vs. suppress-mode (matters
   for v2.x suppress-mode promotion decision).

## Status

This is a skeleton. The four core integration points
(`replay_to_fork`, `run_rebuilder`, `replay_post_fork`,
`measure_token_cost`) raise `NotImplementedError` and depend on the
v1.2.0 transcript-ingest module and the v1.4.0 rebuilder
implementation. The skeleton ships now to lock the metric names,
JSON output format, and run-mode list against the spec's
acceptance criteria.
