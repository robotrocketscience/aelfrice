# Context-rebuilder eval-harness fixtures

This directory holds the **synthetic-only** fixture corpus that
backs the v1.4.0 context-rebuilder eval harness ([#136][i136]).

[i136]: https://github.com/robotrocketscience/aelfrice/issues/136

## Policy: synthetic only on the public repo

Per [`docs/eval_fixture_policy.md`][p] (decided at [#142][i142]):

[p]: ../../../docs/eval_fixture_policy.md
[i142]: https://github.com/robotrocketscience/aelfrice/issues/142

- **Public repo (this directory).** Synthetic, generator-built
  fixtures only. Tracked in git. CI runs against this corpus.
  The headline continuation-fidelity number documented in
  `docs/context_rebuilder.md § Headline metric` is computed on
  this corpus.
- **Lab repo (`~/projects/aelfrice-lab`).** Captured real-session
  `turns.jsonl` files held lab-side, never pushed to GitHub.
  Used for offline calibration only; never gates a release.
- **Local-only escape hatch.** `benchmarks/context-rebuilder/eval_corpus/`
  (gitignored, sibling of this directory) holds ad-hoc captured
  fixtures for individual developers' local calibration runs.

The boundary between "public" and "lab" is the directory and the
two-repo separation, not a filter. See `docs/eval_fixture_policy.md`
for the full reasoning.

## Layout

```
fixtures/
  README.md                          (this file)
  synthetic/                         (the synthetic corpus)
    debugging_session_001.jsonl      (~16-turn debugging task)
    ...
```

Each fixture is a `turns.jsonl` file matching the v1.2.0
[transcript-ingest schema][ts]. Per-line shape, lifted from
`aelfrice.transcript_logger._build_turn_line`:

[ts]: ../../../docs/transcript_ingest.md

```json
{
  "schema_version": 1,
  "ts":             "2026-04-27T15:40:10.000Z",
  "role":           "user" | "assistant",
  "text":           "...",
  "session_id":     "20260427T154010Z-3f8a",
  "turn_id":        "20260427T154010Z-0001",
  "context":        {"cwd": "..."}
}
```

## Adding a synthetic fixture

Synthetic fixtures must be:

1. **Self-contained.** No references to private project codenames,
   real user paths, or captured-session content. The
   `cwd` field on every line points at a synthetic path under
   `/Users/synthetic/...` -- a marker that this content was
   generator-built, not captured.
2. **Deterministic.** Same input seed -> identical output. The
   harness's reproducibility guarantee is downstream of this.
3. **Bounded.** ~10-30 turns is the sweet spot. Single-fixture
   replays must complete in well under the 5-second pytest
   timeout.
4. **Schema-conformant.** Each line is one JSON object carrying
   the fields above. Compaction-marker events (`{"event":
   "compaction_start"}`, `{"event": "compaction_complete"}`) are
   tolerated by the loader but discouraged in synthetic fixtures
   -- the scaffolding harness skips them and they add no signal.

## What this corpus is NOT for

- **Continuation-fidelity scoring.** That's [#138][i138] (separate
  issue, separate corpus design). The current scaffolding only
  measures `token_budget_delta` and `hook_latency_ms`; the
  fidelity scorer adds answer-match logic on top of this loader.
- **Latency benchmarking.** Hook-latency numbers from this corpus
  are scaffolding signals, not production benchmarks. Real
  latency calibration runs against captured corpus, lab-side.

[i138]: https://github.com/robotrocketscience/aelfrice/issues/138
