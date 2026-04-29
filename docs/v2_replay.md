# v2.x spec: `replay_full_equality` — flip-readiness probe

Spec for issue [#262](https://github.com/robotrocketscience/aelfrice/issues/262). Cascade addendum to [`design/write-log-as-truth.md`](design/write-log-as-truth.md). Gates the view-flip in [`v2_view_flip.md`](v2_view_flip.md).

Status: spec, no implementation. Recommendation included; decision is the user's.

## What's being decided

Whether `replay_full_equality()` — the validation harness the design memo calls for — ships under a *shape-equality at ingest time* contract or a *bit-equality across all fields* contract, and how the report distinguishes drift caused by the rule set from drift caused by post-ingest mutation (feedback, decay).

The function re-runs the deterministic derivation function (#261) over every `ingest_log` row written since v2.0 and compares synthesized beliefs to canonical `beliefs`. Output is a `FullEqualityReport` with row counts in five buckets (match / mismatch / derived_orphan / canonical_orphan / drift_examples). It is the gate the chain has to clear before view authority can flip.

## Substrate / chain dependency

- **Depends on:** #261 (pure derivation function). The probe has nothing to invoke until that lands.
- **Depends on:** #263 (`legacy_unknown` migration). Pre-v2.0 stores have beliefs without a corresponding ingest_log row; without the migration synthesizing rows, every legacy belief reports as `canonical_orphan` and the probe is useless on existing user stores.
- **Excludes from comparison:** rows whose `source_kind = legacy_unknown`. They have no raw_text the derivation function can consume; the migration backfills shape-only metadata.

## Recommendation

**Ship at v2.x with shape-equality at ingest time, not bit-equality.**

Three reasons:

1. **alpha/beta evolve via feedback.** A belief ingested at α=1, β=1 today and given two upvotes is α=3, β=1 tomorrow. Re-deriving from the log produces α=1, β=1 — the prior. Bit-equality flags every fed-back belief as drift, which is the wrong signal. The right equality is *what would derivation produce on a fresh store ingesting the same log row?* Posterior drift is **not drift** — it is feedback doing its job.
2. **Origin can legitimately rewrite.** #224 added origin propagation; pre-#224 beliefs have `origin=NULL` while a re-derivation today produces `origin=ingest_turn` (or whichever entry point). This is a known cohort, not a bug. The report should bucket these under `legacy_origin_backfill` rather than `mismatch`.
3. **Edge equality is partial.** `triple_extractor` edges are deterministic and replay-equal; `propagate_valence` edges are feedback-driven and won't replay. Scope edge equality to the deterministic set; report the feedback-driven subset under a separate `feedback_derived_edges` count, never as drift.

### Equality contract (concrete)

A canonical belief and a derived belief are **shape-equal** iff:
- `content_hash` matches, AND
- `type` matches, AND
- `origin` matches OR canonical `origin IS NULL` (legacy backfill cohort), AND
- the deterministic edge set (FROM `triple_extractor`) matches.

α/β/last_retrieved_at/feedback-driven edges are explicitly out of scope for the equality check. They are tracked in separate counters for human inspection but never trigger the drift alarm.

## Decision asks

- [ ] **Confirm shape-equality contract.** If the user wants strict bit-equality (e.g., to catch posterior-write bugs), the report must bucket every fed-back belief separately. Default recommendation: ship shape-equality.
- [ ] **Drift threshold for `aelf doctor --replay` exit code.** Recommendation: exit 0 if `mismatch + derived_orphan == 0` (canonical_orphan and legacy_origin_backfill are not drift); exit 1 otherwise. Configurable via `--max-drift N`.
- [ ] **Scope of the legacy backfill cohort.** When `replay_full_equality` runs on a store with `legacy_unknown` rows, those rows are excluded from `total_log_rows`. Beliefs whose only log row is `legacy_unknown` are excluded from `canonical_orphan`. This is the right call but should be ratified — it does mean the probe is silent about pre-v2.0 ingest correctness.
- [ ] **Drift example sample size.** Recommendation: up to 10 representative cases per drift bucket, raw_text truncated to 200 chars. Configurable.

## Why this is judgment-scope

The decisions above are the design work. Once equality is contracted, implementation is mechanical (~150 LOC code + ~250 LOC tests).

## Downstream impact

- `aelf doctor --replay` becomes the gate for #265 (view-flip). Required to exit 0 on a clean store before the flag flips default-on.
- `LIMITATIONS.md`: section on "what re-onboarding can and can't reproduce" gets to point at the probe instead of describing the limitation in prose.
- New CLI flags: `--max-drift N`, `--drift-examples N`, `--scope <all|since-v2>`.

## Out of scope (deferred to follow-up)

- **Time-travel replay** (`at_timestamp` parameter). Requires per-version derivation library. Tracked separately.
- **Multi-rule-set replay** (replay against version N of the rule set). Same blocker.
- **Auto-fix.** The probe reports drift; #265's view-flip is what corrects it.

## Provenance

- Source-of-truth: [`docs/design/write-log-as-truth.md`](design/write-log-as-truth.md) §§ "What changes under the proposed contract", "Smallest first step".
- Substrate ratification: [`substrate_decision.md`](substrate_decision.md) (#196 Option B). Beta-Bernoulli's posterior drift behavior is what motivates shape-equality over bit-equality here.
- Upstream chain: #205 (parallel-write phase, merged) → #261 (derivation function) → **#262 this issue** → #264 (worker) → #265 (view-flip).
