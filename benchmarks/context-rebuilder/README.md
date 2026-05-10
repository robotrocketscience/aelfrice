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

## What v1.4.0 measures

Three metrics travel together in the harness JSON output -- the
v1.4 ship-gate set:

* **`token_budget_delta`** (per turn) -- signed cumulative-token
  delta: `rebuilt_cumulative - full_cumulative`. Pre-clear: 0. At
  the clear turn: `rebuild_block_tokens - pre_clear_baseline`.
  Post-clear: the rebuild's saving (typically negative).
* **`hook_latency_ms`** (per turn) -- wall-clock from "PreCompact
  hook fires" to "rebuild block emitted", in milliseconds.
  Monotonic non-negative by construction (`time.monotonic()` floor
  at 0.0).
* **`continuation_fidelity`** (per replay) -- aggregate
  answer-match score in [0, 1]. Compares each post-clear
  assistant turn's answer against the original session's answer
  at the same turn (the ground truth). Aggregates per-turn binary
  match scores via mean. Landed in [#138][i138].

What the v1.4.0 harness still does **NOT** do:

* **Real tokenization.** The harness uses the same 4-chars/token
  heuristic as `aelfrice.context_rebuilder._CHARS_PER_TOKEN`. A
  real tokenizer (tiktoken or model-specific) is parked for v1.5.x.
* **Real rebuilder integration.** The synthetic clear still
  injects a fixed-overhead rebuild block. Calling
  `aelfrice.context_rebuilder.rebuild()` on a real store and
  feeding the agent's actual post-clear answers into the scorer
  is [#139][i139] -- the rebuilder hook itself.

[i138]: https://github.com/robotrocketscience/aelfrice/issues/138
[i139]: https://github.com/robotrocketscience/aelfrice/issues/139

## Continuation-fidelity scoring (#138)

### Method shipped at v1.4.0: `exact`

Per-turn comparison of normalized answer text. Normalization (in
order): NFC, lowercase, whitespace-collapse, strip. Per-turn match
is binary (1 or 0); aggregate is the mean over post-clear
assistant turns.

Why `exact` for v1.4.0:

* **Deterministic.** Same fixture + same answers -> identical
  score, byte for byte, on every machine.
* **Reproducible.** No model call, no embeddings, no clock.
* **No network.** Zero outbound calls; CI never blocks on
  network state.
* **Cheap.** O(n) on turn count.

### Known false-positive modes (`exact` says "match", true fidelity is lower)

* **Regression-shaped restatement.** Original = "v1.2.0 shipped
  the alpha rebuilder"; post-clear = same string -- but the agent
  silently re-derived the working state from prior turns and the
  spec's "working-state loss" failure mode is invisible.
* **Quoted-from-rebuild.** The `<aelfrice-rebuild>` block already
  quotes the original verbatim; the agent regurgitates rather than
  reasons. `exact` scores it 1.0; the failure goes uncaught.

### Known false-negative modes (`exact` says "miss", true fidelity is higher)

* **Paraphrase / re-ordering.** Semantic match, different surface
  form. `exact` rejects.
* **Trailing differences.** Post-clear answer adds a clarifying
  sentence; strict equality rejects.
* **Numeric formatting.** "32 tokens" vs "thirty-two tokens" --
  same fact, different surface form.

These modes are documented (not hidden) so v1.4.x calibration runs
can correlate the per-task-type fidelity numbers with the failure
modes the spec calls out
([`docs/context_rebuilder.md` § Failure modes][spec]).

[spec]: ../../docs/context_rebuilder.md

### Why LLM-judge is parked

LLM-judge introduces an eval-time outbound call -- acceptable per
spec § Open ("eval is not the runtime path") but not the v1.4.0
default. CI green would couple to network state. Embedding
similarity adds a heavyweight optional dep
(sentence-transformers or similar) and the LLM-judge path
subsumes its accuracy. Both are deferred to v1.5.x.

### Method toggle

Pick the method via `--score-method`:

```bash
uv run python -m benchmarks.context_rebuilder.replay \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl \
    --clear-at 8 \
    --score-method exact          # the v1.4.0 default
```

`--score-method embedding` and `--score-method llm-judge` exit 2
with a `NotImplementedError` pointing at #138 and the spec issue.
The literal-string surface (in `score.ScoreMethod`) accepts the
parked names so a future PR can wire them in without changing the
public signature.

### Vacuous cases

* **No clear injected.** `clear_injected_at is None`. There is no
  notion of "post-clear" without a clear; score = 1.0 with
  `n_post_clear_assistant_turns = 0`. This is the perfect-replay
  baseline.
* **Clear at last turn.** No post-clear answers exist; score =
  1.0 with `n = 0`. We pick 1.0 over `NaN` so dashboards stay
  numeric; `n` lets callers detect "no measurement happened".

### Reproducibility

Same fixture + same `post_clear_answers` -> identical score, byte
for byte, on every machine. Pinned by
`tests/test_continuation_fidelity_scorer.py::test_score_is_reproducible_across_runs`
and `::test_score_is_reproducible_with_explicit_answers`.

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
  ],
  "continuation_fidelity": {
    "score": 1.0,
    "method": "exact",
    "n_post_clear_assistant_turns": 4,
    "per_turn": [1, 1, 1, 1]
  }
}
```

Every turn record carries `token_budget_delta` and
`hook_latency_ms`. The top-level `continuation_fidelity` sub-object
carries the v1.4 answer-match score; together with
`rebuild_block_tokens` (token-cost surface) and per-turn
`hook_latency_ms`, the JSON ships all three v1.4 ship-gate
metrics. `cleared` is `true` for exactly the turn at which
`--clear-at` fired (if any).

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
  context_rebuilder/                (v1.4.0 harness package)
    __init__.py
    __main__.py                     (CLI surface)
    replay.py                       (transcript replay loader)
    inject.py                       (midpoint-clear injection)
    measure.py                      (token + latency primitives)
    score.py                        (continuation-fidelity scorer)
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

## LLM-judge stage (commit-3 of #592)

Open-ended replay rows that the deterministic substring scorer
cannot settle land in the run report tagged
`reason="needs_llm_judge"`. The judge stage ships as a separate,
opt-in pass that compares each candidate answer against the
reference using an off-band model call at the host CLI's anchor
tier.

The stage is **default-off** (`max_judge_calls=0`) so CI never
issues a model call. Operators opt in by passing a positive cap;
the cap binds before any request file is written, so a
misconfigured run cannot exceed its budget.

### Contamination boundary

The judge sees only `(turn_idx, expected, actual)`. Retrieval
context -- the rebuilt block and the user turn -- never reaches
the judge prompt, per [`docs/BENCHMARKS.md`][b] Pass 1 / Pass 2
separation. Letting the judge see the rebuilt block would allow
it to patch the candidate using context the candidate did not in
fact produce, inflating fidelity.

[b]: ../../docs/BENCHMARKS.md

### Operator flow

The harness (or an operator-driven helper) writes a per-run
`judge_requests.jsonl`; the host CLI's dispatcher issues one
off-band call per row at the anchor tier (`JUDGE_MODEL_TIER`)
and writes `judge_responses.jsonl` back to the same directory;
a follow-up call joins by `turn_idx` and folds the verdicts back
into the run report.

```
# 1. Replay produced rows tagged reason="needs_llm_judge". Write
#    the request file (replace 50 with your call budget):
uv run python - <<'PY'
from pathlib import Path
import json, sys
sys.path.insert(0, "benchmarks/context-rebuilder")
from judges import llm_judge

rows = [json.loads(l) for l in
        Path("run_dir/replay_results.jsonl").read_text().splitlines()
        if l.strip()]
n = llm_judge.write_judge_requests(rows, Path("run_dir"),
                                    max_judge_calls=50)
print(f"wrote {n} judge requests")
PY

# 2. Host-side dispatch (one off-band model call per row at the
#    anchor tier, prompt = llm_judge.JUDGE_PROMPT_TEMPLATE rendered
#    with the row's `expected` and `actual`). Write
#    run_dir/judge_responses.jsonl with rows of shape
#    {"turn_idx": <int>, "matched": <bool>, "rationale": "<str>"}.

# 3. Fold the verdicts back into the run report:
uv run python - <<'PY'
from pathlib import Path
import json, sys
sys.path.insert(0, "benchmarks/context-rebuilder")
from judges import llm_judge

rows = [json.loads(l) for l in
        Path("run_dir/replay_results.jsonl").read_text().splitlines()
        if l.strip()]
responses = llm_judge.read_judge_responses(Path("run_dir"))
folded = llm_judge.apply_judge_verdicts(rows, responses)
Path("run_dir/replay_results.jsonl").write_text(
    "\n".join(json.dumps(r) for r in folded) + "\n"
)
PY
```

Wiring this stage into the harness's main flow -- so steps 1 and
3 ride on the same `eval_harness.py` invocation -- is a follow-up
that joins onto the `--run-dir` plumbing landing in PR #601.

## Status

* **#136 -- harness scaffolding:** SHIPPED in v1.4.0. Replay
  loader, midpoint-clear injection, token + latency measurement,
  CLI entry, synthetic fixture, 26 deterministic unit tests.
* **#138 -- continuation-fidelity scorer:** SHIPPED in v1.4.0.
  `exact`-method scorer in `score.py`, wired into the harness
  JSON output, 20 deterministic acceptance tests in
  `tests/test_continuation_fidelity_scorer.py`. Produces the
  headline-fidelity number against the synthetic corpus.
  `embedding` / `llm-judge` methods parked for v1.5.x.
* **#139 -- rebuilder hook integration:** waits on real
  agent-output capture; will replace the synthetic
  fixed-overhead block with `aelfrice.context_rebuilder.rebuild()`
  and feed the agent's actual post-clear answers into the
  scorer's `post_clear_answers` parameter.
* **#141 -- threshold-mode calibration:** waits on #139.
