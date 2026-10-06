# Classifier-gated core admission

**Status:** implemented. The label cache, `aelf core-gate accept`, the gate
in core selection, the `aelf doctor core-gate` backlog drain, the
self-check, the re-run of a flagged batch, and the session-end batch are
implemented. Until a belief has a label, it follows today's rule.
Tracking issue: [#1638](https://github.com/robotrocketscience/aelfrice/issues/1638).

This spec proposes a content gate for core admission. Three consumers select
core independently: `aelf core` (`cli._qualifies_core`), the SessionStart
`<core>` lane (`hook._belief_qualifies_core`), and the core membership that
`aelf doctor` reports. None reads another's output, so the gate must apply in
all three. A belief qualifies through the
non-lock arms only if an offline classifier has judged it truth-apt. Locked
beliefs aren't affected.

## Problem

A belief that isn't locked qualifies for core through either of two arms:

- **Corroboration arm:** a corroboration count of at least 2, spread across at
  least two episodes. Sightings less than an hour apart form one episode
  (`CORROBORATION_EPISODE_GAP_SECONDS`, added in #1635).
- **Posterior arm:** α + β ≥ 4 and a posterior mean of at least 2/3.

Neither arm reads the belief's content. The measurements below come from two
real stores: 519 core items, each labeled by two independent blind raters,
with an inter-rater κ of about 0.86–0.93.

- 76–82% of core isn't truth-apt. These items are fragments, commands, and
  markup.
- `user_transcript` rows clear the posterior arm at birth, because their prior
  is (3, 1). Recurring fragments clear the corroboration arm.
- Real propositions enter core by the same routes and carry no more evidence
  than the junk does. They have zero feedback events apart from exposure, and
  few corroborations.

Each of the following was also measured:

- Tightening either arm on its own lowers the quality of core. For the
  recomputation, read the comments on #1638.
- Surface content filters reject at most about 40% of the junk and lose
  8–25% of the real content.
- No passive signal that the store records separates real items from junk.
  The best signal improves the truth-apt share by 1.29×.

All measurements were taken on `main` at e59df85, before #1635 shipped. They
come from private stores, so no script in this repository can reproduce them.
#1635 removed replay bursts from the corroboration arm. That change made the
arm stricter, but it didn't make the arm read content, so the problem above
still holds.

## Proposal

Add a content gate to core admission. A belief qualifies for core's non-lock
arms only if a classifier has labeled it as one of the following:

- **A:** a self-contained claim.
- **B:** a claim that needs context to read.
- **C:** not a claim. C items never qualify.

### Classifier

- A small language model labels each belief A, B, or C against a fixed
  rubric. The rubric is the same one that the reference labels used, and it
  ships with the implementation.
- The classifier runs offline and in batches, never on the hot path. It runs
  over new core candidates, for example at session end or from `aelf doctor`.
- Results are cached. The cache key is the hash of the belief's content
  together with a classifier version that identifies the rubric, the prompt,
  and the model. Changing any of the three bumps the version, so an old label
  is never applied under a new classifier. Given the cache, the gate is
  deterministic.
- The host-driven onboard classifier is the precedent for the call path. The
  CLI emits the candidates, the host dispatches its cheapest model, and the
  CLI accepts the labels. The aelfrice CLI makes no outbound call. For that
  flow, read [`design/llm_classifier.md`](design/llm_classifier.md).
- The precedent is the pattern, not the payload. The onboard handshake
  accepts `belief_type` and `persist` for the four belief types, and it
  rejects any other type. This gate needs its own accepted schema that
  carries the A, B, or C label and the classifier version.

#### Session-end batch

The Stop hook runs the session-end batch. The host's `SessionEnd` event
can't hand work to the model, so the hook continues the conversation from
a Stop instead. "Session end" means "after the session's beliefs are
ingested": the transcript logger folds a session's turns into beliefs only
when 12 turn lines (about six exchanges, by default;
`AELFRICE_INGEST_STOP_FLUSH_TURNS` sets it) have built up since the last
flush, or at a compaction. So the batch fires on the first Stop after each
ingest flush, not once per session (operator ruling on #1638,
2026-10-06):

- On every Stop, the hook runs one indexed query for this session's
  candidates. A candidate is an active, unlocked belief that this session
  created (its `beliefs.session_id`), that meets today's non-lock core
  rule, that has no label under the current classifier version, and whose
  content hash no open batch claims. A belief that an earlier session
  created and that reaches core during this one isn't a candidate; the
  backlog drain labels it (operator ruling on #1638, 2026-10-05).
- An open batch claims a hash when it is under the current classifier
  version and is either this session's own session-end batch or a doctor
  batch. A Stop therefore never asks about a belief that an earlier Stop
  already asked about, whether or not that batch was answered. An
  accepted batch claims nothing, because its labels already exclude its
  beliefs. A batch under an earlier classifier version can't be accepted,
  so it claims nothing either.
- With at least one candidate, the hook records one batch with origin
  `session_end` and the session's id, and writes
  `{"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext":
  ...}}` to stdout. The host continues the conversation with the context,
  which holds the classifier prompt and the `aelf core-gate accept
  <batch-id>` command to run with the reply on stdin. The host documents
  `additionalContext` on Stop as non-error feedback that continues the
  conversation; `decision: "block"` reads as an error the model must fix.
- The batch holds the newest candidates that fit, up to 50, within 9,500
  characters of context. The host inlines at most 10,000 characters of a
  hook's output and replaces a longer string with a file the model isn't
  asked to read. Snippets are never cut, because the label is cached under
  the hash of the whole content. A candidate that doesn't fit stays
  unclaimed for the next Stop or the backlog drain.
- The query, the batch record, and the context are one `BEGIN IMMEDIATE`
  transaction, so two Stops at once can't batch the same belief, and a
  failure leaves no batch behind.
- The hook doesn't continue the conversation while the host sets
  `stop_hook_active`, in a headless session (#1634), on the Codex host, or
  when you opt out with `AELFRICE_CORE_GATE_SESSION_END=0` or
  `[core_gate] session_end = false`. Codex documents `decision` for Stop
  but not `additionalContext`, so the hook writes nothing there.
- The context asks the host to run the prompt on its smallest model
  (`CLASSIFIER_MODEL_TIER`). That's a request, not a guarantee (operator
  ruling on #1638, 2026-10-05). If a larger model labels the batch, the
  labels are kept under the same classifier version, and nothing checks
  them automatically: the self-check runs on doctor batches only.
  `aelf doctor core-gate --rerun` refuses a session-end batch, so no
  command relabels one yet.
- An unanswered session-end batch stays open, and its beliefs stay
  unlabeled. The backlog drain doesn't read session-end batches, so
  `aelf doctor core-gate --emit` batches those beliefs again.

### Self-verification

Self-verification is required. In evaluation, one of four identical batch
runs failed silently. It agreed with the reference labels only 35% of the
time, while the other runs agreed 70–87% of the time. To detect such a
failure, shuffle the batches and check each run against its siblings:

1. Flag any batch whose share of C labels differs by more than 0.25 from the
   median of the other batches.
2. Re-run each flagged batch.

On the evaluation data, this check flagged the failed run and no healthy run.
The 0.25 threshold was chosen after the failure was seen, so validate it again
on new data.

#### Implementation

The backlog drain and the self-check work as follows:

- `aelf doctor core-gate --emit` batches the backlog: the active, unlocked
  beliefs that meet today's rule and have no label under the current
  classifier version. It splits the backlog evenly into as few batches of up
  to 50 beliefs as it can, with sizes that differ by at most one, so no small
  remainder batch skews the self-check. A batch of one snippet can only score
  a C share of 0 or 1. It records each batch, and prints the batch's prompt
  and the `aelf core-gate accept` command that takes the reply. All the
  batches of one emit share one creation time, which marks them as one emit
  run. The emit's reads and writes run under one write lock, so two emits at
  once can't batch the same beliefs.
- An emit doesn't batch a belief twice. If a batch from an earlier emit isn't
  accepted yet, and all of its beliefs are still unlabeled, unchanged, and
  in core, the next emit prints that batch again. Otherwise the emit sets the batch
  aside and puts its unlabeled beliefs in a new batch.
- A set-aside batch stays open rather than being closed, because its
  beliefs can return to the backlog (for example, when a lock is lifted)
  and a later emit then prints it again. Accepting it is harmless: each
  label is keyed to the content hash the model judged. It does replace a
  label that a newer batch wrote for the same belief, and that label row
  then counts toward the set-aside batch's emit run in the self-check.
- If `--out` can't write a prompt file, the batches stay recorded, and the
  next emit prints them again.
- When `aelf core-gate accept` accepts a doctor batch, it runs step 1 of the
  check over the accepted batches of the same emit run, and names each
  flagged batch on stderr. It doesn't check a session-end batch. It doesn't refuse the labels or change the exit
  code. The check runs only when the run has at least four accepted
  batches. With three, each batch's median of the others is the mean of two
  shares, so one failed batch also flags the healthy ones. Below four, the
  command says on stderr that it skipped the check. Each emit is its own
  run, so emitting with a small `--limit` and accepting between emits gives
  runs of fewer than four batches, and the check never runs for them. To
  keep the self-check, accept all batches of one emit before emitting
  again.
- Step 2, the re-run, is `aelf doctor core-gate --rerun <batch-id>`. In one
  transaction, it deletes the labels the batch still owns, and batches again
  the beliefs whose labels it deleted and that are still in core and
  unchanged. A label that a later batch wrote for the same content hash
  stays. The new batch joins the original emit run: it carries the
  original creation time, and the original batch, which now owns no
  labels, drops out of the comparison. A batch id is derived from the classifier version, the
  creation time, and the content hashes. So when every belief is kept, the
  new batch would have the original's id, and the command reopens the
  original batch instead of creating a second one. Until the new batch is
  accepted, the re-run beliefs have no label, so they follow today's rule,
  including the ones the batch had labeled C.
- A subset re-run, which holds fewer beliefs than the batch it re-ran, is
  left out of the self-check: it isn't checked, and it isn't a sibling
  for other batches. It isn't comparable with its full-size siblings, and
  the operator ruled it out on 2026-10-05. A whole-batch re-run
  keeps its size and is checked. No column marks a subset re-run: the
  batches one emit creates are disjoint, so a batch whose content hashes
  are a strict subset of another batch's in the same run can only be one.
- The emit forms batches in belief id order and doesn't shuffle them. Belief
  ids are hashes, so that order is unrelated to what a belief says or when
  it was created.
- Plain `aelf doctor` counts the core candidates with and without a label
  under the current version. The counts are informational. The count walks
  every belief. No read in the graph report can be reused: the auditor
  reads counts and the α and β pairs, not the lock level, content hash, or
  corroboration count that the core rule needs. On one machine, on a synthetic
  20,000-belief store, the walk took 0.24 seconds, and `aelf health` took
  0.51 seconds instead of 0.25.

### Expected effect

The following table shows the evaluation result, with the failed batch re-run.

| Metric | Today | Gated |
| --- | --- | --- |
| Truth-apt share of core, pooled | 0.206 | 0.476 (2.31×) |
| Share of real core items retained | — | 0.944 |

Caveats:

- Without the re-run, the registered result was 1.90×, just short of the 2×
  bar.
- Precision is modest. It's 0.59 on a store that user-typed text dominates,
  and 0.36 on a store that recurring fragments dominate. The gate keeps nearly
  all real items, but it also admits junk.
- The reference labels come from language-model raters, not from people.

## Precision options

The single-run gate admits junk. The following table compares three
configurations, in order of cost, on a development set and on a fresh
holdout set.

| Configuration | Classifier calls | Truth-apt share (dev / holdout) | Real items retained (dev / holdout) |
| --- | --- | --- | --- |
| Single run: admit A, or admit B with fewer than 2 corroborations | 1× | 0.56 / 0.80 | 0.89 / 0.76 |
| Majority of 3 runs | 3× | 0.52 / 0.64 | 0.92 / 0.85 |
| Unanimous of 3 runs | 3× | 0.63 / 0.82 | 0.84 / 0.75 |

For comparison, today's truth-apt share of core is 0.21 / 0.25, and the
single-run gate that admits A or B scores 0.48 / 0.77.

- The ordering of the configurations is stable, but the absolute values move
  by 0.1–0.3 between samples and between label sets. Quote them as ranges.
- The recommended default is the 1× configuration: a single run, with the
  corroboration condition on B. Where core precision matters more than cost,
  use unanimous of 3.
- A stricter wording of the rubric was also tested. It's dominated by the
  configurations in the table.

The corroboration counts in this table were measured before #1635. The B
condition was re-measured against the episode rule in #1638 and gave the same
result with either count.

## Optional layer: explicit confirmation

A per-session prompt that asks the user whether to keep the gate's positives
raises precision toward 1.0. The prompt budget limits how many items the
prompt can retain. With a carry-over queue and prompts ranked by the gate, a
retention of 50% needs about six prompts per session. Three prompts per
session retain 28%. The prompt is therefore useful only as an opt-in layer,
not as the admission route.

## Not proposed

- **Lowering the `user_transcript` prior on its own.** On a store that is
  mostly text, this change empties core, and it lowers the pooled quality.
- **Admission based on use in reply text.** About 10% of core is ever
  injected, and detecting use from lexical overlap is too crude. Revisit this
  route only with instrumentation, such as the model citing belief IDs.

## Decisions

These questions were open in earlier versions of this spec. They're ruled in
#1638.

1. **Where the batch runs:** at session end, over that session's new core
   candidates. `aelf doctor` drains the backlog of older candidates.
2. **Beliefs that aren't classified yet:** they follow today's rule until
   they're classified, so nothing leaves core before its first label.
3. **Cost ceiling:** the 1× configuration, a single classifier run that admits
   A, or B with fewer than two corroborations.
