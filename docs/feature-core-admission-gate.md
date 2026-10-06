# Classifier-gated core admission

**Status:** partly implemented. The label cache, `aelf core-gate accept`, and
the gate in core selection are implemented. Nothing emits a classifier batch
yet, so every belief is unlabeled and core follows today's rule.
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
