# v2.x spec: view-flip — `ingest_log` becomes canonical

Spec for issue [#265](https://github.com/robotrocketscience/aelfrice/issues/265). Terminal step of [`design/write-log-as-truth.md`](design/write-log-as-truth.md). Highest-risk issue in the chain.

Status: spec, no implementation. Recommendation included; decision is the user's.

## What's being decided

Three calls left open by the issue body:

1. **How long the feature flag soaks** before defaulting on (and how long before the rollback path is removed).
2. **What `aelf rebuild` actually drops and reconstructs** — beliefs only, beliefs + edges, beliefs + edges + α/β reset, or beliefs + edges + α/β reset + feedback_history replay.
3. **What `aelf doctor --explain <belief_id>` shows** when the belief is fed-back-driven (no single log row produced it).

After this issue ships, the contract change the chain has been preparing is real: `ingest_log` is the source of truth; `beliefs`/`edges` are computed views; classifier-fix-then-rebuild becomes a supported user workflow; the federation atom for v3 exists.

## Recommendation

**Ship at v2.x behind `WRITE_LOG_AUTHORITATIVE` flag (default off), soak one full minor release with no drift reports before flipping the default, keep the rollback path through the release after that.**

### Soak schedule

| Release | Default | Rollback path | Direct `insert_belief()` |
|---|---|---|---|
| v2.x (this issue) | off | works | warns under flag, allowed |
| v2.x+1 | off | works | warns regardless |
| v2.x+2 | **on** | works | raises under flag (default-on) |
| v2.x+3 | on | **removed** | raises always |

Two-release soak before flipping the default is the conservative call. The risk profile is "silent canonical drift" — the kind of bug that only manifests over weeks of real use, not under unit tests. One release is not enough; three is overkill. Two with explicit drift-alarm telemetry is the recommendation.

### `aelf rebuild` scope

**Default behavior:** drop and recompute beliefs + deterministic edges. Preserve α/β by reading the prior posteriors from existing canonical state, then re-applying `feedback_history` against the re-derived beliefs (matched by content_hash). Feedback-driven edges are dropped and re-propagated by the existing `propagate_valence` path.

**Why:** the only way a user's existing posterior survives a rebuild is if rebuild explicitly preserves it. The user's intuition is "fix the classifier, keep my feedback" — anything else is a footgun. The cost is one extra pass over `feedback_history` after the worker pass; trivial.

**Flags:**
- `--reset-feedback`: drop feedback_history too. Equivalent to a fresh ingest. For "the classifier change invalidated my feedback judgments" cases.
- `--rule-set <hash>`: rebuild against a specific rule-set version (requires the per-version derivation library, deferred to follow-up; in v2.x this flag accepts only the current hash and errors on others).
- `--as-of <timestamp>`: rebuild against log rows ≤ timestamp. Same per-version-rule-set blocker; in v2.x this flag accepts only a timestamp ≥ the last rule-set bump and errors on older.

### `aelf doctor --explain <belief_id>` output

For a belief with one or more `ingest_log` rows: print the rows in chronological order with raw_text, source_kind, source_path, classifier_version, rule_set_hash, ts.

For a belief with zero `ingest_log` rows (legacy backfill cohort): print "no ingest record; created pre-v2.0 at <belief.created_at> via path now indeterminate." Don't fabricate a synthetic explanation.

For a belief in canonical with `legacy_unknown` source rows only: print the legacy_unknown row with a banner explaining it's a migration-synthesized stub.

For a belief that has α/β diverging from prior because of feedback: append `feedback_history` rows to the explain output, separated visually. The "explain" surface is the user's way to answer "why does the system think this?" — feedback is part of the why.

## Decision asks

- [ ] **Confirm two-release soak** before default-on. Alternative: one-release (faster to the user-visible payoff, more drift risk) or three-release (slower, lower risk). Recommendation: two.
- [ ] **Confirm rebuild preserves α/β + replays feedback by default.** The alternative — drop everything and let the user re-feedback — is cleaner architecturally but a worse user experience. Recommendation: preserve by default, add `--reset-feedback` for the clean-slate case.
- [ ] **Confirm `--rule-set <hash>` and `--as-of <timestamp>` ship as stubs in v2.x** (accepting only current values), with full multi-version support deferred to follow-up. Alternative: ship only `aelf rebuild` (current rules, now) and add the flags later. Recommendation: stub the flags so the surface is final and the follow-up work is purely internal.
- [ ] **Confirm explain semantics for the legacy cohort** — print the stub honestly rather than fabricate. Recommendation: yes; the alternative ("hide the legacy rows from explain output") would mask the migration boundary.
- [ ] **`WRITE_LOG_AUTHORITATIVE` config name and surface.** Env var is fine for ops; should it also be a `.aelfrice.toml` key for project-level pinning? Recommendation: both — env var wins if both are set.

## Why this is judgment-scope

The risk model for this issue is what makes it judgment-heavy. Implementation is bounded (~250 LOC + ~250 LOC tests); the calls above are the bet — soak length, rebuild semantics, and the flag's surface are user-visible commitments that will be hard to revise once shipped.

## Downstream impact

- Federation atom for v3: log rows become the unit of cross-scope replication. This issue defines what "a complete log row" means; v3 ships rows over the wire.
- README claim alignment: the README's "fix the classifier and re-derive" pitch becomes true (it is currently aspirational).
- `LIMITATIONS.md`: drops the "re-onboard required after rule changes" caveat. Adds a new section on the soak path and the rollback flag — for honesty about what default-off means in v2.x and v2.x+1.
- `CHANGELOG.md`: this is the headline v2.x change. The flag, the rollback path, the rebuild CLI, the explain CLI, and the soak schedule all need user-facing copy.
- New CLI surfaces: `aelf rebuild [--reset-feedback] [--rule-set H] [--as-of T]`, `aelf doctor --explain <belief_id>`.

## Risk and mitigations

The chain's correctness is verified up to this point against canonical-as-authority. Flipping authority means canonical is now a function of the log; if the worker has a bug missed by #262/#264, canonical drifts silently.

- **Drift alarm in `aelf doctor`.** Continuous self-check on every doctor invocation: if `replay_full_equality` (#262) reports drift on a flag-enabled store, surface as P0 with a clear remediation message ("rollback by setting WRITE_LOG_AUTHORITATIVE=False; report the drift example to <issue tracker>").
- **CI regression on the rollback path.** New test: enable flag, ingest, disable flag, re-open store, verify canonical is regenerated bit-for-bit (modulo feedback posteriors, per the equality contract in #262).
- **Telemetry opt-in.** Anonymized drift-event counter, opt-in via `.aelfrice.toml`, default off. Helps catch "wild" drift the user wouldn't otherwise notice. (Out of scope for this issue if telemetry hasn't shipped yet — flag for follow-up.)

## Out of scope (deferred)

- **Async / cross-process derivation.** v3.
- **Federation log shipping.** v3.
- **Removing the rollback flag.** v2.x+3 follow-up after the soak completes cleanly.
- **Per-rule-set-version derivation library.** Required for full multi-rule-set rebuild; the flags ship as stubs in v2.x with implementation deferred.

## Provenance

- Source-of-truth: [`docs/design/write-log-as-truth.md`](design/write-log-as-truth.md) §§ "The contract", "What changes under the proposed contract".
- Federation cross-link: [`docs/design/federation-primitives.md`](design/federation-primitives.md).
- Upstream chain: #205 → #261 → #262 → #264 → **#265 this issue**.
- Equality contract this issue's `aelf doctor --replay` invokes: [`v2_replay.md`](v2_replay.md).
- Worker contract this issue assumes: [`v2_derivation_worker.md`](v2_derivation_worker.md).
- Successor: #254 (T1 corroboration recorder trigger gap) likely becomes trivially correct once derivation is canonical; revisit after the flip.
