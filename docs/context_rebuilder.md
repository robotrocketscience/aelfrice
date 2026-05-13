# Context rebuilder

**Status:** v1.4.0 implementation landed (issue
[#139](https://github.com/robotrocketscience/aelfrice/issues/139));
suppress mode parked for v2.x.
**Target milestone:** v1.4.0 (post-v1.3.0 retrieval wave; the
entity-index L2.5 tier is what `retrieve()` returns under the
rebuilder hood).
**Dependencies (hard):**
- [transcript_ingest.md](transcript_ingest.md) (v1.2.0) — without a
  transcript log to query, this feature has nothing to read.
- v1.2.0 ingest enrichment — `session_id` must be a real field for
  session-scoped retrieval to work.
**Dependencies (soft, quality-gating):**
- v1.3.x partial Bayesian-weighted ranking (issue #146) — when it
  lands, posterior weighting flows through the rebuilder
  automatically because the rebuilder calls `retrieve()` as a
  black box. No follow-up code change in `context_rebuilder.py`
  required.

## What shipped at v1.4.0 vs. what's still parked

| Acceptance criterion | v1.4.0 |
|---|---|
| `aelf setup --rebuilder` installs the PreCompact hook idempotently | Shipped (v1.2.0a0; carried forward) |
| Hook contract: exit 0 on every failure, never block | Shipped |
| Empty transcript / missing store: silent exit 0 | Shipped |
| Reproducible: same inputs → byte-identical `additionalContext` | Shipped (regression test in `tests/test_context_rebuilder_hook.py`) |
| Median latency ≤ 200 ms on a 10k-belief store | Shipped (regression test); measured ~2 ms on a workstation |
| Locked + session-scoped + retrieve() hits in that order | Shipped via `rebuild_v14()` |
| `[rebuilder] turn_window_n` / `token_budget` in `.aelfrice.toml` | Shipped (defaults 50 / 4000) |
| Augment-mode coordination with the harness | Shipped |
| `additionalContext` JSON envelope on stdout | Shipped |
| Manual fire via `aelf rebuild` / `/aelf:rebuild` | Shipped (#141) |
| Trigger modes — manual + threshold | Shipped (#141; default `threshold` since #746) |
| Threshold default sourced from calibration data | Shipped (#141; `benchmarks/context-rebuilder/calibration_v1_4_0.json`) |
| Trigger mode — dynamic | **Parked for v1.5** (#141; see § Dynamic mode (parked v1.5) below) |
| Suppress-mode coordination with the harness | **Parked for v2.x** |
| Continuation-fidelity eval harness | Scaffolding shipped at v1.3 (#136); fidelity scoring is #138 |
- v1.2.0 triple-extraction port — better edge structure on the
  transcript ingest path produces better recall on the rebuild
  query.
**Risk:** high. Three sources of risk:
1. Quality. "Seamless from the user's perspective" is a strong
   claim. Needs an eval harness (see "Validation") to be
   measurable, not just intuited.
2. Latency. PreCompact is on the user-facing latency budget; the
   rebuild must complete fast enough that the user doesn't notice
   the swap.
3. Coordination with the harness. Claude Code's auto-compaction
   runs whether or not we hook PreCompact; we either suppress and
   replace it, or augment it. Both have failure modes.

## Summary

A `PreCompact` hook that, instead of letting Claude Code compact
the context window with its default summarization, queries
aelfrice for the highest-value beliefs from the current session
and emits them as the new session-start context block. The user
sees no `/clear`, no compaction summary, no resume prompt — the
agent simply continues with a leaner, retrieval-curated working
set.

```
context approaches threshold (configured: 50% / 80% / dynamic)
        ↓
PreCompact hook fires
        ↓
context_rebuilder.rebuild():
  1. Read last N turns of <project>/.git/aelfrice/transcripts/turns.jsonl
  2. Extract entities + intents from those turns (triple extractor)
  3. Query aelfrice with those entities:
       L0 locked beliefs (always in)
       L1 BM25 hits (today) → posterior-weighted (v1.3+)
       L2 session-scoped beliefs from same session_id
  4. Pack to budget B (default ~2000 tokens, tunable)
  5. Emit additionalContext block + a "continue" continuation cue
        ↓
Harness clears prior context; new session inherits the rebuild block
        ↓
User submits next prompt; agent continues without visible seam
```

## Motivation

Three failure modes the rebuilder is designed to fix:

1. **Recursive context cost.** A long-running session re-pays the
   full prior-context token cost on every turn until compaction
   fires. The rebuilder can fire earlier (at 50% rather than 80%+
   thresholds) when the eval shows continuation quality holds, which
   reduces the per-turn token cost across the steady-state work.

2. **Default compaction loses task state.** Claude Code's built-in
   compaction is a generic summarizer. It does not know which facts
   in the session are load-bearing for the user's current task. An
   aelfrice-driven rebuild can preferentially keep the user's
   active task's beliefs and decisions — closing a gap the earlier
   research line identified as the missing piece between cold-start
   and post-compaction state (the cold-start path injected too much
   general context; nothing closed the gap with a working-state
   delta).

3. **Manual `/clear` is a productivity tax.** Today the user is
   tracking when to clear, what to re-paste, and how to resume.
   The rebuilder makes that bookkeeping the system's job.

## Non-goals

- **A new ranker.** The rebuilder consumes whatever ranker
  aelfrice ships. v1.4.0 ships against v1.3.0's partial Bayesian
  ranker; v2.0.0 inherits the full one for free.
- **A summarizer.** The rebuild is *retrieval*, not summarization.
  No LLM is called inside the hook. Beliefs come back as their
  literal stored content.
- **A perfect zero-loss feature.** Set a regression band; do not
  promise lossless. See "Validation."
- **Codex parity at v1.4.0.** Codex has no equivalent of
  PreCompact. A Codex slice is future work — likely a manual
  `/aelf-resume` slash command + tool.

## Design

### Trigger policy

Three trigger modes are defined; two ship at v1.4.0, one is
investigated separately. Configured via `.aelfrice.toml`:

```toml
[rebuilder]
# Ship default since #746. The PreCompact hook fires when the
# harness compacts; opt out by setting this to "manual" if you
# only want the rebuilder to run on explicit `aelf rebuild` /
# `/aelf:rebuild` invocations.
trigger_mode = "threshold"

# Calibrated default from
# benchmarks/context-rebuilder/calibration_v1_4_0.json. Only
# consulted when `trigger_mode = "threshold"`. The actual gate is
# Claude Code's own PreCompact firing; this value documents the
# operating point and is the reproducible source-of-truth for the
# default.
threshold_fraction = 0.6
```

- **`manual`** *(opt-out since #746)*: PreCompact hook never fires
  the rebuild block. Only explicit invocations
  (`aelf rebuild` / `/aelf:rebuild`) emit a block. Use this if the
  augment-mode token-spend bump (~5-6 KB per compaction) is not
  worth the context-preservation win for your workflow.
- **`threshold`** *(default since #746; bench gates #592 + #687
  cleared 2026-05-13)*: PreCompact hook fires when called by Claude
  Code's harness. The harness's own threshold gating is the trigger
  signal; `threshold_fraction` documents the *calibrated operating
  point* and is the reproducible source-of-truth for the default
  value (re-derive via `python -m benchmarks.context_rebuilder.calibrate`).
- **`dynamic`** *(parked v1.5)*: heuristic-driven trigger (rate of
  context growth, entity-density delta). Investigated at v1.4 but
  did **not** clear the spec's "beats threshold by ≥ 5% absolute
  fidelity at same-or-lower token cost" gate — see § Dynamic mode
  (parked v1.5) below for the measurements. Setting
  `trigger_mode = "dynamic"` at v1.4 logs a "parked" trace and
  no-ops the hook.

Whatever the trigger, the hook contract is the same: exit 0 in
under 50ms even if ingest is in progress in the background.

### Threshold calibration

The `threshold_fraction` default is **sourced from the eval
harness, not picked by hand**. Re-run via:

```bash
python -m benchmarks.context_rebuilder.calibrate \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl \
    --out benchmarks/context-rebuilder/calibration_v1_4_0.json
```

The script sweeps thresholds 0.5 / 0.6 / 0.7 / 0.8 / 0.9 against
the bundled synthetic fixture, seeds an in-memory store with one
belief per pre-clear assistant turn, calls `rebuild_v14()` at a
compressed `token_budget` (200 — tighter than the production 4000
so the retrieved-beliefs section actually has to choose what to
surface), and scores each post-clear assistant turn by **content-
overlap** (fraction of `>=4-char` lowercase tokens from the
original answer that appear in the rebuild block's
`<retrieved-beliefs>` section).

The proxy is honest about what it measures: "what fraction of the
original answer's content tokens does the rebuild block surface?"
— the load-bearing precondition for the agent to be able to
reconstruct the answer. It is reproducible, deterministic, no LLM,
no network — same constraints as the v1.4.0 `exact` continuation-
fidelity scorer.

Sweep table (committed at
`benchmarks/context-rebuilder/calibration_v1_4_0.json`):

| `threshold_fraction` | `clear_at` | n post-clear | rebuild tokens | full-replay tokens | ratio | fidelity | efficiency |
|---|---|---|---|---|---|---|---|
| 0.5 | 8 | 4 | 641 | 657 | 0.976 | 0.051 | 0.052 |
| **0.6** | **9** | **3** | **668** | **657** | **1.017** | **0.068** | **0.067** |
| 0.7 | 11 | 2 | 756 | 657 | 1.151 | 0.069 | 0.060 |
| 0.8 | 12 | 2 | 851 | 657 | 1.295 | 0.069 | 0.054 |
| 0.9 | 14 | 1 | 938 | 657 | 1.428 | 0.095 | 0.067 |

`efficiency = continuation_fidelity / token_budget_ratio`.

**Choice rule** (deterministic, encoded in `_choose_threshold`):

1. Filter to points with `token_budget_ratio <= 1.5` (band).
2. Among those, take points with the highest efficiency, rounded
   to 3 decimals so cosmetically-tied points are treated as tied.
3. Tie-break by lowest `threshold_fraction` (earlier firing
   catches drift sooner).

At v1.4.0 the band-filter passes all five points; the highest
rounded efficiency is 0.067, tied between 0.6 and 0.9; lowest-
threshold tie-break picks **0.6**.

This value is fixture-bound. The synthetic fixture is small (16
turns) so absolute fidelity numbers are noisy — what matters is
the *ranking* across thresholds, which is stable. A v1.5.x re-
calibration on a captured corpus may move the chosen value.
Production users opting into `trigger_mode = "threshold"` should
re-run calibration on a representative session and override via
`[rebuilder] threshold_fraction = X` in `.aelfrice.toml`.

### Dynamic mode (parked v1.5)

Spec gate: "Dynamic mode ships only if its fidelity delta beats
the threshold default by a documented margin (≥ 5% absolute
fidelity at same-or-lower token cost). Otherwise: park to v1.5."

Two heuristic candidates were measured by
`benchmarks/context_rebuilder/dynamic_probe.py` against the same
synthetic fixture used for threshold calibration. Reproduce via:

```bash
python -m benchmarks.context_rebuilder.dynamic_probe \
    benchmarks/context-rebuilder/fixtures/synthetic/debugging_session_001.jsonl
```

**Reference point:** threshold-mode at `threshold_fraction = 0.6`
(the v1.4 calibrated default) — fires at content-turn 9, fidelity
**0.0677**, token ratio **1.017**.

**Candidate 1 — rate-of-context-growth.** Fire at the first turn
whose 4-turn rolling per-turn token average exceeds 1.5× the
fixture-wide median per-turn token count. Rationale: catches
moments of rapid working-state expansion. On the 16-turn fixture
the rule never trips (the fixture is balanced) and the trigger
falls back to the last turn (15). Resulting fidelity is vacuously
1.0 (zero post-clear assistant turns, by the documented vacuous-
case convention) but the token ratio (**1.461**) **exceeds** the
threshold-mode reference (**1.017**), so it fails the "same-or-
lower token cost" half of the gate.

**Candidate 2 — entity-density-delta.** Fire at the first turn
(after a 4-turn warmup) whose new-entity count drops below
`max(1, 0.5 × fixture-median new-entity count)`. Rationale: the
rebuilder's job is easier once the agent has stopped introducing
new state. On the same fixture the rule fires at content-turn 4 —
*earlier* than threshold-mode — at fidelity **0.0824** and a lower
token ratio (**0.682**). Fidelity delta vs. threshold-mode:
**+0.0147 absolute**, well below the **+0.05 spec gate**.

**Verdict — park.** Neither candidate clears the v1.4 ship-gate.
Per the spec's "either ships with evidence OR parks with
documented evidence" clause, dynamic mode is **parked for v1.5**.
Setting `trigger_mode = "dynamic"` at v1.4 logs a `parked v1.5`
trace to stderr and no-ops the hook (matching the manual-mode
no-op pathway). The implementation lives at
`benchmarks/context_rebuilder/dynamic_probe.py` so a v1.5.x
re-investigation can build on the same proxy metric and fixture
seeding code without re-deriving them.

The investigation is intentionally narrow. A captured-corpus
re-run with a richer entity-density signal, or a hybrid trigger
combining rate-of-growth with absolute context size, may clear
the gate at v1.5.x. The v1.4 measurement is recorded so that
re-investigation has a baseline to beat.

#### v1.5 captured-corpus revisit (issue #188) — outcome: re-park

The v1.5 revisit ran `dynamic_probe.py` in corpus mode
(`--corpus <lab-side> --corpus-label "<lab-side>"`) against the
lab-side captured-corpus fixture set. The lab corpus was empty at
the v1.5 ship-gate (no captured `turns.jsonl` files existed under
the corpus directory). Per issue #188 dependency clause — "The
issue does not ship until they do; if they don't exist by v1.5
ship-gate, defer to v1.5.x" — the probe returned zero fixtures
(`n_fixtures = 0`) and the aggregate verdict is **park**.

The synthetic-fixture re-run (included as `synthetic_reference` in
`benchmarks/context-rebuilder/calibration_v1_5_0_dynamic.json`)
confirmed the v1.4 measurements are unchanged:

| Candidate | fidelity | ratio | fidelity delta | passes gate? |
|---|---|---|---|---|
| rate-of-growth | 1.0 (vacuous) | 1.461 | +0.932 | No — ratio > 1.017 |
| entity-density-delta | 0.082 | 0.682 | +0.015 | No — delta < 0.05 |

**Outcome 2 (re-park).** Neither candidate clears the bar on
synthetic; no captured-corpus data was available to contradict
this. The `parked v1.5` log line and no-op hook behaviour are
preserved. A v1.6.x re-investigation should first populate the
lab-side corpus, then re-run:

```bash
python -m benchmarks.context_rebuilder.dynamic_probe \
    --corpus <lab-side-path> \
    --corpus-label "<lab-side>" \
    --out benchmarks/context-rebuilder/calibration_v1_6_0_dynamic.json
```

The `--corpus` flag (added in this PR) accepts a directory of
`*.jsonl` files and aggregates the verdict across all fixtures,
superseding the single-fixture positional-arg interface for
multi-file corpus runs. Single-fixture mode continues to work
unchanged for synthetic reproducibility.

### What the hook reads

The hook needs the most recent N turns of the conversation to
seed the rebuild query. Source: the project's
`<root>/.git/aelfrice/transcripts/turns.jsonl` (per
[transcript_ingest.md](transcript_ingest.md)). The hook reads the
tail of this file directly — fast, no network, no DB roundtrip
required for the seed.

`N` is a config key. Initial value: 10 (5 user, 5 assistant).
Eval-tunable.

### Rebuild algorithm (as shipped at v1.4.0)

```python
def rebuild_v14(
    recent_turns: list[RecentTurn],
    store: MemoryStore,
    *,
    token_budget: int = DEFAULT_REBUILDER_TOKEN_BUDGET,
) -> str:
    """v1.4 rebuild: L0 + session-scoped + L2.5/L1 via retrieve().

    Stage 1: build a query string from the recent turns.
        Entity extraction + triple extraction (no LLM) over the
        concatenated turn text. The downstream retrieve() path
        runs L2.5 entity lookup on this string and L1 BM25 over
        its tokens; both benefit from a high-signal query.

    Stage 2: pull the live session id off the most recent turn that
        carries one. Beliefs whose `session_id` matches are
        surfaced as a dedicated tier between L0 and L2.5/L1.

    Stage 3: call retrieve() once. retrieve() returns L0 + L2.5 +
        L1 in that order; we already have L0 from
        list_locked_beliefs() and rebuild it ourselves so the
        session-scoped tier slots in the right place.

    Stage 4: pack. Locked first (never trimmed), then
        session-scoped (capped at token_budget), then the L2.5/L1
        tail returned by retrieve() with budget honoured. Output
        is the formatted XML block.
    """
```

The `pre_compact()` hook in `aelfrice.hook` reads the JSON payload
from stdin, locates a transcript (canonical `turns.jsonl` first,
the harness's transcript_path as fallback), drives `rebuild_v14`,
and writes the result through `emit_pre_compact_envelope()` —
which wraps the block in
`{"hookSpecificOutput": {"hookEventName": "PreCompact",
"additionalContext": "<aelfrice-rebuild>...</aelfrice-rebuild>"}}`.

The new `aelfrice.context_rebuilder.main` is also wired as a
console-script-callable; either entry point produces identical
output for identical inputs.

### Output format

The hook emits a single XML-tag-delimited block, matching the v1.0
hook's existing `<aelfrice-memory>` envelope so the model's prompt
processing has one consistent signal:

```xml
<aelfrice-rebuild session_id="20260427T154010Z-3f8a">
  <recent-turns>
    <turn role="user">...</turn>
    <turn role="assistant">...</turn>
    ...
  </recent-turns>
  <retrieved-beliefs budget_used="1842/2000">
    <belief id="..." score="..." locked="true">...</belief>
    <belief id="..." score="...">...</belief>
    ...
  </retrieved-beliefs>
  <continue/>
</aelfrice-rebuild>
```

The `<continue/>` element is a marker prompt — a stable signal the
agent learns to interpret as "resume the prior task using the
above context, do not greet, do not summarize."

### Coordination with the harness

PreCompact hook fires *before* the harness compacts. Two modes:

1. **Augment.** Hook emits additionalContext; the harness still runs
   its own compaction afterwards. Result: both the harness summary
   and our rebuild are in the new context. Bigger token cost, lower
   risk.

2. **Suppress + replace.** Hook emits additionalContext; the harness
   skips compaction (Claude Code's `decision: "block"` semantics for
   PreCompact, if available). Result: only our rebuild is in the
   new context. Tighter token cost, higher risk if our rebuild is
   incomplete.

v1.4.0 ships **augment-mode only**. Suppress mode requires high
confidence in the eval harness scores. Suppress mode is a v2.x
candidate.

### What does NOT change

- The harness's transcript log under `~/.claude/projects/...` is
  untouched. We don't read it, don't write it, don't depend on it.
- The v1.0 `UserPromptSubmit` hook continues to inject memory
  retrieval into every prompt. The rebuilder is orthogonal — it
  fires only on PreCompact.
- The v1.0.1 hook→retrieval feedback loop continues to log every
  retrieval. The rebuilder's retrieval call goes through the same
  `retrieve()` codepath and writes the same `feedback_history`
  rows.

## Validation

This is the load-bearing section. Without a falsifiable eval, the
"seamless" claim is not a claim, it's a vibe.

### Eval harness

A new harness at `benchmarks/context-rebuilder/eval_harness.py`
(skeleton shipped alongside this spec) does:

1. Read a corpus of replayable transcripts from
   `benchmarks/context-rebuilder/eval_corpus/`. Each transcript
   is a captured `turns.jsonl` from a real working session, with
   PII scrubbed. Multiple transcripts; multiple task types
   (debugging, planning, code-review, exploratory).
2. For each transcript, fork at a midpoint turn `T`. Replay turns
   `0..T` to populate a fresh aelfrice store.
3. Force a clear at `T`. Run the rebuilder. Inject its output as
   the new context.
4. Replay turns `T+1..end` and let the agent continue. Compare:
   - **Continuation fidelity**: does the agent answer the same
     subsequent questions correctly? Scored binary by the agent's
     responses against the ground-truth original session.
   - **Token cost**: how many tokens did the rebuild emit vs. the
     full-replay baseline?
   - **Latency**: PreCompact hook wall-time from fire to
     additionalContext emit.

### Headline metric

**Continuation fidelity at fixed token budget.** A regression band:
v1.4.0 must hit ≥80% continuation fidelity at a token budget of
≤30% of the full-replay baseline. v1.5.x targets 90%; v2.0.0
targets 95%.

These numbers are placeholders until the eval harness produces
the v1.0.0 baseline. The actual targets get set after the first
calibration run, not from this spec.

### Per-trigger-mode tuning

The threshold (50% / 70% / 80%) is not a guess. The eval harness
sweeps trigger thresholds and produces a fidelity-vs-trigger curve
per task type. The default ships at the threshold where fidelity
is within the regression band on all task types.

The dynamic-mode metric (entropy-based or other) is gated on
showing it tracks fidelity better than a fixed threshold across
task types. If the eval doesn't show that, dynamic mode does not
ship.

### Failure modes the eval must catch

- **Working-state loss.** The agent had a partially-formed plan
  pre-clear; post-rebuild, the plan is gone and the agent
  re-derives from scratch (or worse, re-derives differently). The
  eval should mark these as fidelity failures even if the
  *eventual* answer is correct.
- **Hallucinated continuation.** The rebuild was incomplete; the
  agent confabulates plausible-sounding context. The eval scores
  these as failures and the eval corpus should specifically
  include questions the rebuild *cannot* recover (so the agent
  has the option to say "I don't have that context" rather than
  invent it).
- **Trigger storms.** Threshold misconfigured; rebuilder fires
  every 2 turns. Eval should detect via the per-session
  rebuild-event count and gate trigger configs accordingly.

## Acceptance criteria

1. `aelf setup --rebuilder` (or equivalent flag on existing
   `aelf setup`) installs the PreCompact hook idempotently.
2. The hook contract is honored: exit 0 in <50ms regardless of
   internal state.
3. The eval harness ships and produces a calibration JSON file
   showing fidelity-vs-trigger curves on at least 3 task types.
4. Default trigger threshold is set from the calibration, not
   hardcoded. The chosen threshold is documented in
   `benchmarks/context-rebuilder/calibration_v1.4.0.json` and
   referenced from the spec when v1.4.0 cuts.
5. Manual mode (`/aelf-rebuild`) works as an explicit testing
   surface and ships before threshold mode auto-triggers.
6. Round-trip test: a real 50-turn session, force a midpoint
   clear, observe the agent continues to answer correctly on
   ≥80% of subsequent questions about prior turns.

## Test plan

- `tests/test_rebuilder_unit.py` — unit tests on rebuild()
  determinism, token-budget packing, output format.
- `tests/test_rebuilder_hook.py` — PreCompact hook script,
  non-blocking contract, 50ms budget.
- `benchmarks/context-rebuilder/eval_harness.py` — the eval
  harness (skeleton shipped alongside this spec; corpus and
  scoring fill out through v1.2.0–v1.4.0).
- `tests/integration/test_rebuilder_roundtrip.py` — end-to-end
  round-trip with a fixture transcript.

Per project test policy: every test deterministic and short.
Eval-harness runs are explicitly slow and live in `benchmarks/`,
not `tests/` — they aren't run on every CI invocation.

## Roadmap

Listed in [ROADMAP.md](ROADMAP.md) as the v1.4.0 milestone. The
"Context rebuilder" entry there is the user-facing summary; this
document is the implementation contract.

## Open questions

These don't gate the spec but need answers before implementation:

1. **PreCompact `decision: "block"` semantics.** Does Claude Code
   currently honor it? If not, augment-mode is the only available
   path until it does.
2. **Trigger from where.** The harness's pre-compaction signal —
   is it exposed to hooks, or do we have to compute "approaching
   threshold" ourselves from token counts? If self-computed, that's
   another small spec.
3. **Slash command surface.** `/aelf-rebuild` for manual mode —
   does it live in the v1.0 slash-commands surface or as a CLI
   command piped through Bash?
4. **Per-project vs. user-global config.** Trigger threshold
   probably wants project-local override; the rebuilder behavior
   probably wants user-global default. Reuse `.aelfrice.toml`
   (v1.1.0) for the project layer.
