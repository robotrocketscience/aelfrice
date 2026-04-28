# context-rebuilder eval harness

Eval harness that backs [`docs/context_rebuilder.md`](../../docs/context_rebuilder.md).
Two layers ship in this directory:

1. **v1.2.0 skeleton** at [`eval_harness.py`](eval_harness.py). Locks
   metric names, JSON output format, and run-mode list against the
   spec's acceptance criteria. Four integration points
   (`replay_to_fork`, `run_rebuilder`, `replay_post_fork`,
   `measure_token_cost`) raise `NotImplementedError` -- they fill
   in as the rebuilder implementation completes.
2. **v1.4.0 scaffolding** at [`../context_rebuilder/`](../context_rebuilder/)
   (note: underscore -- importable Python package). End-to-end
   replay loader + midpoint-clear injection + token-cost / latency
   measurement that runs today against synthetic fixtures and emits
   the per-turn output schema the v1.4.0 acceptance criteria
   require. Closes [#136][i136].

[i136]: https://github.com/robotrocketscience/aelfrice/issues/136

## What v1.4.0 scaffolding measures (and what it does NOT)

The scaffolding is a *harness shape*, not a *fidelity scorer*. Two
metrics are captured per replay turn:

* **`token_budget_delta`** -- signed cumulative-token delta:
  `rebuilt_cumulative - full_cumulative`. Pre-clear: 0. At the clear
  turn: `rebuild_block_tokens - pre_clear_baseline`. Post-clear: the
  rebuild's saving (typically negative).
* **`hook_latency_ms`** -- wall-clock from "PreCompact hook fires"
  to "rebuild block emitted", in milliseconds. Monotonic
  non-negative by construction (`time.monotonic()` floor at 0.0).

What the scaffolding does **NOT** do:

* **Continuation-fidelity scoring.** That's [#138][i138] (separate
  issue). The scaffolding only verifies the harness *runs*.
  Answer-match logic, LLM-judge integration, and the "did the
  agent continue correctly" headline number all land with #138.
* **Real tokenization.** The scaffolding uses the same 4-chars/token
  heuristic as `aelfrice.context_rebuilder._CHARS_PER_TOKEN`. A
  real tokenizer (tiktoken or model-specific) lands alongside the
  fidelity scorer.
* **Real rebuilder integration.** The scaffolding measures the
  *shape* of the latency channel; the synthetic clear injects a
  fixed-overhead rebuild block. Calling
  `aelfrice.context_rebuilder.rebuild()` on a real store happens
  in #138.

[i138]: https://github.com/robotrocketscience/aelfrice/issues/138

## How to run

End-to-end replay against the bundled synthetic fixture:

```bash
uv run python -m benchmarks.context_rebuilder.replay \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl
```

With a midpoint-clear injection at content-turn index 8:

```bash
uv run python -m benchmarks.context_rebuilder.replay \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl \
    --clear-at 8
```

Write JSON output to a file instead of stdout:

```bash
uv run python -m benchmarks.context_rebuilder.replay \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl \
    --clear-at 8 \
    --out /tmp/replay_out.json
```

The `python -m benchmarks.context_rebuilder` invocation is a synonym
for `python -m benchmarks.context_rebuilder.replay` -- both forms
hit the same argparse surface.

### Output schema

```json
{
  "fixture_path": "...",
  "n_turns": 16,
  "n_skipped_lines": 0,
  "clear_injected_at": 8,
  "full_replay_baseline_tokens": 657,
  "rebuild_block_tokens": 32,
  "turns": [
    {
      "turn_index": 0,
      "role": "user",
      "token_budget_delta": 0,
      "hook_latency_ms": 0.0,
      "cleared": false
    },
    ...
  ]
}
```

Every turn record carries `token_budget_delta` and `hook_latency_ms`
-- the two scaffolding metrics. `cleared` is `true` for exactly the
turn at which `--clear-at` fired (if any).

## Layout

```
benchmarks/
  context-rebuilder/                (this directory)
    README.md                       (this file)
    eval_harness.py                 (v1.2.0 skeleton)
    eval_corpus/                    (.gitignored, local-only)
    fixtures/
      README.md                     (fixture policy)
      synthetic/
        debugging_session_001.jsonl (~16-turn synthetic transcript)
    results/                        (per-run JSON; gitignored)
  context_rebuilder/                (v1.4.0 scaffolding package)
    __init__.py
    __main__.py                     (CLI surface)
    replay.py                       (transcript replay loader)
    inject.py                       (midpoint-clear injection)
    measure.py                      (token + latency primitives)
```

The hyphenated `context-rebuilder/` and underscored
`context_rebuilder/` siblings are deliberate: hyphens for the data
+ skeleton (matches the public-docs convention), underscores for
the importable Python package (Python identifiers can't contain
hyphens). Both share the same v1.4.0 milestone.

## Run modes (v1.2.0 skeleton; not yet wired in scaffolding)

- `--mode threshold-sweep` -- sweep PreCompact trigger thresholds
  per task type, emit the fidelity-vs-trigger curve. Default for
  calibration.
- `--mode dynamic` -- evaluate the dynamic-threshold heuristic
  against fixed thresholds.
- `--mode budget-sweep` -- sweep token budgets at a fixed trigger.
- `--mode regression` -- fixed config, pass/fail against the
  v1.0.0 baseline numbers.

These modes live on the v1.2.0 skeleton and are gated on the
fidelity-scorer landing (#138). The v1.4.0 scaffolding uses the
simpler `--clear-at` injection surface; it converges with the
skeleton when the integration points fill in.

## Fixture policy

Synthetic fixtures (this repo) and captured fixtures (lab repo)
follow the policy decided in
[`docs/eval_fixture_policy.md`](../../docs/eval_fixture_policy.md):

- **Public repo (`benchmarks/context-rebuilder/fixtures/`).**
  Synthetic, generator-built fixtures. Tracked in git. CI runs
  against this corpus. Headline number is computed here.
- **Lab repo (`~/projects/aelfrice-lab`).** Captured real-session
  `turns.jsonl` files. Never pushed to GitHub. Used for offline
  calibration only.
- **Local-only escape hatch (`eval_corpus/`).** Gitignored. Holds
  ad-hoc captured fixtures for individual developers' local runs.

See [`fixtures/README.md`](fixtures/README.md) for the synthetic
fixture contract (deterministic, bounded, schema-conformant).

## Open questions answered by the eval, not the spec

These remain open for the fidelity scorer (#138):

1. The default trigger threshold for v1.4.0.
2. The default token budget for v1.4.0.
3. Whether dynamic-mode improves over a fixed threshold.
4. Whether augment-mode loses fidelity vs. suppress-mode (matters
   for v2.x suppress-mode promotion decision).

## Status

* **#136 -- harness scaffolding:** SHIPPED in v1.4.0. Replay
  loader, midpoint-clear injection, token + latency measurement,
  CLI entry, synthetic fixture, 26 deterministic unit tests.
* **#138 -- continuation-fidelity scorer:** in progress. Adds
  answer-match logic on top of this scaffolding; produces the
  headline-fidelity number against the synthetic corpus.
* **#141 -- threshold-mode calibration:** waits on #138.
