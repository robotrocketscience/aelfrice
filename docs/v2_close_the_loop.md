# Close the loop — measuring injection relevance against Bayesian posterior

**Issue:** [#317](https://github.com/robotrocketscience/aelfrice/issues/317)
**Status:** spec, no implementation. Implementation lands as a follow-up
PR against this spec's acceptance list.
**Target:** v2.x — deferred per V2_REENTRY_QUEUE; original v2.0.0 anchor
slipped past the v2.0/v2.1 cuts without a `close_the_loop` module landing.

---

## Problem

Storage, retrieval, provenance, and posterior updating are all in tree.
The remaining loop closure is empirical: when the hook injects
`<aelfrice-memory>`, are the injected beliefs **actually relevant** to
the user's prompt, and is the Bayesian ranking signal pushing useful
beliefs up over time?

Today we have:

- A per-turn audit JSONL (`hook_audit.jsonl`) capturing exactly which
  beliefs were injected, with `posterior_mean`, `α/β`, lane, and the
  full rendered block — `aelfrice/hook.py:_write_hook_audit_record`.
- A deferred-feedback sweeper (`aelfrice/deferred_feedback.py`) that
  credits a small `+ε` to retrieved beliefs whose grace window passed
  with no contradicting explicit feedback. This is the *implicit*
  half of the loop and is shipped.
- Posterior-weighted L1 ranking (`docs/bayesian_ranking.md`,
  `docs/v2_posterior_ranking_residual.md`) consumes the posterior in
  retrieval scoring with a fixed log-additive weight `λ`.

What we **don't** have is a measurement that answers two questions
end-to-end against live session data:

1. Is the implicit feedback signal pointing in the right direction —
   i.e., do beliefs the deferred sweeper credits as "useful" actually
   correlate with downstream user-visible value (explicit positive
   feedback, repeat retrieval, lock promotion)?
2. Is the posterior-weighted ranking firing correctly — i.e., are
   higher-posterior beliefs *more* relevant to the prompts that
   triggered their injection, or is `λ` mis-calibrated and burying
   useful cold beliefs / boosting confirmation-biased ones?

#317's body posits the answer is "analyze session logs and the
corresponding Bayesian score for relevance." This memo turns that
into a benchmark contract.

## Out of scope

- Live online tuning of `λ`. v2.0 ships the measurement; weight
  retuning lives downstream of the numbers this benchmark produces.
- Replacing the deferred-feedback sweeper. Sweeper stays; this memo
  measures it.
- Embeddings-based relevance scoring. Aelfrice is no-embeddings by
  design (per `docs/no_embeddings_first.md`); the benchmark uses
  lexical and structural signals only.
- Cross-domain contamination diagnosis (already covered by the
  hook-injection audit, `docs/hook-injection-audit.md`).
- Lab-side wonder evaluation (#228) — different campaign.

## Inputs

The benchmark joins three on-disk artifacts already produced by live
operation. No new logging is required for the v2.0 slice.

| Artifact | Path | Producer |
|---|---|---|
| Hook audit | `<git-common-dir>/aelfrice/hook_audit.jsonl` | `aelfrice.hook:_write_hook_audit_record` |
| Feedback events | `feedback_history` table | `MemoryStore.append_feedback_history` |
| Belief snapshot | `beliefs` table (current `(α, β)`) | live store |

The hook audit gives a per-turn record of `(prompt_prefix,
injected_beliefs[], posterior_mean_at_injection_time)`. Feedback
events give the downstream signal. Belief snapshots give the current
posterior for drift comparisons.

## Ground truth — what counts as "relevant"

A belief `b` injected on turn `t` for prompt `p` is **relevant** if
any of the following holds within the next `K` turns of the same
session (`K = 5` default; sweep `{1, 3, 5, 10}`):

1. **Explicit positive feedback.** A `feedback_history` row exists
   with `belief_id = b.id`, `valence > 0`, `source != "retrieval"`
   (i.e., not the deferred-sweeper's auto-credit), `created_at` in
   `[t, t+K]`.
2. **Lock promotion.** `b.lock_level` advanced to `"user"` within the
   window — strongest possible signal of usefulness.
3. **Re-retrieval against a downstream prompt.** `b` appears in a
   later hook-audit record within the window. Re-retrieval is a weak
   positive (operator may have continued in the same neighborhood).

A belief is **not** relevant by default. Sweeper-credited
`+ε` rows from `deferred_feedback` are deliberately excluded — they
are the signal under measurement, not ground truth.

This is a proxy. Failure modes (in order of severity):

- A belief was correctly injected but the user redirected — counted
  as not-relevant. Acceptable: the score is a lower bound on
  precision.
- A belief was irrelevant but user noise produced a positive
  feedback (e.g., user lock-promotes for an unrelated reason).
  Acceptable noise floor; the calibration metric (ECE) is more
  robust to it than the ranking metric (MRR).

The benchmark reports both metrics so the failure modes are visible.

## Metrics

For each turn `t` with audit record `r`:

- **MRR@K of injected ranking.** Compute the rank within `r.beliefs`
  of the *first* relevant belief; reciprocal rank. Average over all
  turns with at least one injection. Reported at `K = 5` and `K = 10`.
- **MRR uplift vs. pure-BM25 ranking.** Re-rank the same
  `r.beliefs` set ignoring the posterior term (`λ = 0`) and compute
  MRR. Uplift = `MRR(λ=spec) - MRR(λ=0)`. Sign-and-magnitude is the
  headline number for "is the posterior helping ranking?".
- **ECE of posterior_mean.** Bucket injected beliefs by
  `posterior_mean` into 10 bins. Per bin: empirical relevance rate
  vs. mean predicted posterior. ECE is the bucket-weight-averaged
  absolute residual. Lower is better; calibrated posterior should
  hit ≤ 0.10.
- **Cold-belief lift.** Among beliefs with `α + β ≤ 2` (effectively
  cold), what fraction were injected and relevant? Compared to the
  baseline relevance rate of all injections. A cold-belief
  suppression bug would show this number sliding to zero over time.
- **Sweeper-credit alignment.** For each belief the deferred-sweeper
  credited in the window, did it also satisfy the relevance
  definition (explicit feedback / promotion / re-retrieval)?
  Reported as precision/recall of the sweeper signal against
  ground truth. This answers the "is the implicit feedback signal
  pointing in the right direction" question directly.

## CLI surface

One new subcommand under the existing `aelf bench` group:

```
aelf bench close-loop \
    [--audit-path PATH]    # default: db_path()/../hook_audit.jsonl
    [--db PATH]            # default: AELFRICE_DB / db_path()
    [--window 5]           # K turns
    [--report json|md]     # default: md
    [--since ISO8601]      # default: all-time
    [--until ISO8601]
```

Pure read; no DB writes. Exits 0 with the report on stdout.

A second helper, `aelf bench close-loop --replay`, re-ranks the same
beliefs at `λ = 0` and prints the MRR uplift number standalone for
quick sanity checks.

## Acceptance criteria

To merge the implementation PR closing #317:

1. `aelf bench close-loop` produces a Markdown or JSON report against
   any DB + audit JSONL pair, with all five metrics defined above.
2. A regression test seeds an audit JSONL + a feedback-history table
   with three known scenarios — (a) high-posterior-relevant, (b)
   high-posterior-irrelevant, (c) cold-belief-relevant — and asserts
   each metric matches the hand-computed expected value.
3. A live-fire dry run on the maintainer's primary DB
   (`~/projects/aelfrice/.git/aelfrice/memory.db`) produces a report
   committed under `docs/close_the_loop_baseline_<DATE>.md`. This is
   the calibration anchor; subsequent ranking-tuning issues are
   measured against it.
4. The report explicitly states whether each metric meets the v2.0
   pass thresholds:
   - **MRR@5 ≥ 0.30** (random baseline depends on injection
     budget; with token-budget-15 a non-degenerate ranker beats
     0.20 trivially).
   - **MRR uplift > 0** (posterior must move ranking the right way).
   - **ECE ≤ 0.15** (calibrated within ±15 percentage points; v2.0
     bar; v2.x can tighten).
   - **Sweeper precision ≥ 0.50** (at least half the beliefs the
     sweeper credits also have an independent positive signal).
5. Failure of any threshold opens a follow-up tuning issue with the
   metric values and the suspected lever (`λ`, `ε`, grace window,
   K). Failure does **not** block #317 closure — the deliverable is
   the measurement and the baseline, not a passing grade.

## Implementation sketch

Three files, ~400 LOC plus tests:

- `src/aelfrice/close_loop.py` — pure functions:
  - `load_audit_window(path, since, until) -> list[AuditRecord]`
  - `relevant_beliefs_for_record(record, store, window_turns) -> set[str]`
  - `mrr_at_k(records, k, *, lambda_override=None) -> float`
  - `ece_posterior(records, n_bins=10) -> float`
  - `cold_belief_lift(records) -> tuple[float, float]`
  - `sweeper_alignment(records, store) -> tuple[float, float]`
- `src/aelfrice/cli.py` — new subcommand `bench close-loop` wiring the
  above; report formatter (md + json).
- `tests/test_close_loop.py` — three-scenario regression fixture.

The re-rank-at-`λ=0` path reuses the existing scoring contract from
`docs/bayesian_ranking.md` so both numbers are produced from one
algorithm with one parameter changed, not two parallel implementations.

## Risks and judgment calls

- **Audit JSONL gaps.** The audit is best-effort fail-soft; a session
  with the writer disabled produces no rows. The benchmark must
  surface "n_records = 0 for some sessions" honestly rather than
  silently averaging over a smaller denominator.
- **Privacy.** Audit records contain prompt prefixes and rendered
  blocks. The benchmark's report must not echo prompts; only IDs,
  posteriors, and counts. The unit test asserts the rendered report
  contains no substring of the input prompts.
- **Sample size.** ECE with < 200 injected events per bin is unstable.
  Report sample sizes per bin; flag bins with fewer than 50 as
  low-confidence.
- **Selection bias.** Only beliefs that *passed retrieval* show up in
  the audit. The benchmark cannot say anything about beliefs the
  ranker buried. That's a separate counterfactual study; explicitly
  out of scope.

## Follow-ups (deliberately deferred)

- Counterfactual ranking study (what *would* have been injected at
  λ=0?). Requires re-running retrieval at audit time, not a
  post-hoc re-rank.
- Per-corpus `λ` tuning. v2.x.
- Online dashboard / `aelf health close-loop`. v2.x; one-shot CLI is
  enough for the v2.0 measurement.
- Window-K ablation as a default report column. Implementation can
  expose it but the spec only requires K=5.
