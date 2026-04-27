# Context rebuilder

**Status:** spec.
**Target milestone:** v1.4.0 (new milestone — slot is post-v1.3.0
retrieval wave so the partial Bayesian-weighted ranker is real
when the rebuilder ships).
**Dependencies (hard):**
- [transcript_ingest.md](transcript_ingest.md) (v1.2.0) — without a
  transcript log to query, this feature has nothing to read.
- v1.2.0 ingest enrichment — `session_id` must be a real field for
  session-scoped retrieval to work.
**Dependencies (soft, quality-gating):**
- v1.3.0 partial Bayesian-weighted ranking — without
  posterior-weighted ranking, the rebuilder is BM25-only and the
  "right" beliefs may not float to the top.
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

Three trigger modes, configured via `.aelfrice.toml` (the project
config file introduced in v1.1.0; the rebuilder adds new keys):

- **`threshold`** (default): fire when the harness's pre-compaction
  signal indicates context is at or above a configured fraction of
  the model's window. Default fraction is set from eval-harness
  calibration data, not hardcoded; an initial guess of 70% is used
  during early calibration runs.
- **`dynamic`**: fire based on a derived metric (e.g.,
  rolling-window entropy of new beliefs per turn — when the agent
  stops introducing new entities, working state is small enough to
  rebuild safely). Implementation gated on eval-harness evidence
  that the dynamic metric tracks continuation quality. May not
  ship in v1.4.0.
- **`manual`**: fire only on explicit `/aelf-rebuild` slash command
  invocation. Useful for users who don't trust auto-trigger and
  for the rebuilder's own QA.

Whatever the trigger, the hook contract is the same: exit 0 in
under 50ms even if ingest is in progress in the background.

### What the hook reads

The hook needs the most recent N turns of the conversation to
seed the rebuild query. Source: the project's
`<root>/.git/aelfrice/transcripts/turns.jsonl` (per
[transcript_ingest.md](transcript_ingest.md)). The hook reads the
tail of this file directly — fast, no network, no DB roundtrip
required for the seed.

`N` is a config key. Initial value: 10 (5 user, 5 assistant).
Eval-tunable.

### Rebuild algorithm

```python
def rebuild(
    transcript_path: Path,
    store: MemoryStore,
    *,
    n_recent_turns: int,
    token_budget: int,
    session_id: str | None,
) -> str:
    """Build a context block to inject as the new session start.

    Returns a string that goes into Claude Code's hook
    additionalContext. Must be deterministic given the same inputs
    so eval-harness runs are reproducible.
    """
    recent = read_tail(transcript_path, n=n_recent_turns)

    # Stage 1: seed the query. Triple-extract entities + intents
    # from the recent turns; build a query string from them.
    triples = extract_triples_batch(recent)
    query = triples_to_query(triples)

    # Stage 2: retrieve.
    hits = store.retrieve(
        query=query,
        token_budget=token_budget,
        # Session-scoped beliefs always pull. Cross-session
        # beliefs ranked by L1 BM25 / posterior.
        session_filter=session_id,
        include_locked=True,  # L0 always in
    )

    # Stage 3: pack. Locked first, then session-scoped, then
    # the open BM25 / posterior tail. Truncate at budget.
    return format_context_block(hits, recent_turns=recent)
```

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
