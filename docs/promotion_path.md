# `agent_inferred → user_validated` promotion path

Design memo for the v1.1.0 gate (issue #95). Implementation lands in
v1.2.0 per [ROADMAP.md § v1.2.0](ROADMAP.md#v120--auto-capture-and-triple-extraction);
the v1.1.0 deliverable is the schema field plus this memo. Reviewer
reads this to decide whether the proposed surface is the one v1.2.0
should ship.

Cross-references: [LIMITATIONS.md § Onboarding](LIMITATIONS.md#onboarding--v101--v110)
(the gap), [`src/aelfrice/contradiction.py`](../src/aelfrice/contradiction.py)
(the precedence ordering this slots into), [`src/aelfrice/feedback.py`](../src/aelfrice/feedback.py)
(the audit-row format and `apply_feedback` semantics).

Status: design only. No code changes implied by merging this file.

## 1. What is being promoted

A belief whose provenance is the onboarding scanner —
[`src/aelfrice/scanner.py:146-158`](../src/aelfrice/scanner.py)
inserts every onboard hit with `lock_level=LOCK_NONE` and no
origin signal — graduating to a state that is recognisably
"user has acknowledged this is correct" without being upgraded
to a full lock.

### Schema as it exists today (v1.0.x)

[`src/aelfrice/models.py:65-83`](../src/aelfrice/models.py) defines `Belief`
with these fields relevant to provenance:

| Field | Type | Notes |
|---|---|---|
| `type` | `str` | One of `factual`, `correction`, `preference`, `requirement`. Not a provenance signal. |
| `lock_level` | `str` | `none` or `user`. Two-valued. |
| `locked_at` | `str \| None` | Set when `lock_level == LOCK_USER`. |
| `demotion_pressure` | `int` | Counter on locks, irrelevant to non-locked. |

There is **no `origin` field**. The contradiction tie-breaker
([`src/aelfrice/contradiction.py:30-42`](../src/aelfrice/contradiction.py))
calls this out explicitly:

> The fourth — `agent_inferred` — needs a `Belief.origin` field that
> the v1.0 schema does not have; in practice no v1.0 code path
> produces beliefs that would map to agent_inferred (every insert
> path is one of: user lock, MCP remember, scan_repo, or correction
> detection). [...] v1.1.0 will add it alongside the project-identity
> work.

### Schema change required for v1.1.0

Add one column to `beliefs`. Recommended: `origin TEXT NOT NULL DEFAULT 'unknown'`.

| Origin value | Set by | Meaning |
|---|---|---|
| `agent_inferred` | `scan_repo` (all three extractors) | Onboard-derived; no human has seen it. |
| `user_validated` | the new promotion command (this memo) | Human has explicitly acknowledged. |
| `user_stated` | `aelf lock` / `aelf:lock` | Human asserted as ground truth. Mirrors `lock_level=user`. |
| `agent_remembered` | MCP `aelf:remember` (when it lands) | Agent decided this was worth keeping mid-session. |
| `unknown` | default for migration | v1.0.x rows whose origin can't be reconstructed. |

Forward compatibility: per the v1.x compatibility commitment
([ROADMAP.md § Compatibility commitment](ROADMAP.md#compatibility-commitment)),
v1.1.0 adds the column with a default; v1.0.x rows read forward as
`origin='unknown'` until backfilled.

**Backfill on first v1.1.0 startup:** rows where `lock_level=LOCK_USER`
become `origin='user_stated'`. Rows where `type=BELIEF_CORRECTION`
become `origin='user_corrected'` (this is where that origin name
comes from — the existing `BELIEF_CORRECTION` type). Everything else
stays `unknown` rather than being misclassified. `agent_inferred`
is **not** assigned by backfill; it is forward-only because the
v1.0.x scanner did not commit to that label and we don't want to
retroactively claim it. Beliefs that should be `agent_inferred`
will be re-marked when v1.2.0 reruns onboard against v1.1.0+ stores.

TBD: whether `unknown` rows surface in `aelf locked --pressured`-style
listings as a separate category. Not load-bearing for promotion.

### What promotion does not do

- It does **not** change `lock_level`. A `user_validated` belief
  remains `lock_level=LOCK_NONE`. This is the central design point:
  validation is weaker than locking. The user is saying "yes, this
  is correct" not "this is non-negotiable."
- It does **not** change `type`. A `factual` belief stays `factual`;
  promotion is orthogonal to the four `BELIEF_TYPES`
  ([`src/aelfrice/models.py:18-23`](../src/aelfrice/models.py)).
- It does **not** insert any edges. Compare `aelf lock` which
  doesn't either ([`src/aelfrice/cli.py:151-180`](../src/aelfrice/cli.py)).

## 2. Trigger

Explicit user action only. No implicit promotion path.

### Why no implicit promotion

Three implicit paths suggest themselves and all three are rejected:

1. **Promote on N positive feedback events.** Conflates two signals.
   `apply_feedback` ([`src/aelfrice/feedback.py:106`](../src/aelfrice/feedback.py))
   already moves posteriors on positive valence. Adding a side effect
   that flips `origin` introduces a hidden state-machine edge that
   isn't visible from the feedback row — and the threshold (3? 5?
   posterior > 0.8?) would be tuning-required without a benchmark.
2. **Promote on retrieval count.** Hook retrievals already write
   audit rows tagged `source='hook'` with valence 0.1 per
   [LIMITATIONS.md § Hook layer](LIMITATIONS.md#hook-layer--v101).
   Re-using that as a promotion trigger means a belief promotes
   purely because the user happens to ask similar questions. That's
   selection bias, not validation.
3. **Promote on user_corrected adjacency.** A belief with a
   SUPPORTS edge from a correction has stronger evidence, but
   "the agent corrected something nearby" isn't the user
   acknowledging the original.

The unifying argument: validation is a UI act, not a math act.
The user looked at the belief and said "yes." There is no signal
the system can derive on its own that substitutes for that.

### Recommended surface: new CLI command

```
aelf validate <belief_id> [--source <label>]
```

And the matching MCP tool:

```
aelf:validate { belief_id: str, source: str = "user" }
```

Naming rationale, against the existing surface:

| Name | Existing usage | Why not |
|---|---|---|
| `lock` | [`cli.py:151`](../src/aelfrice/cli.py) — α=9.0, β=0.5, lock_level=user | Wrong tier. Lock is non-negotiable; validate is "yes, correct." |
| `confirm` | reserved for v2.0.0 ([ROADMAP.md § v2.0.0](ROADMAP.md#v200--feature-parity-and-reproducibility)) | Already claimed by the wonder/reason/core/unlock/delete/confirm tranche. |
| `accept` | unused | Acceptable alternative. Lukewarm — it reads like one-shot triage rather than a state change. TBD. |
| `validate` | unused | Verb matches `user_validated` literal. Recommended. |
| `endorse` | unused | Reads odd in CLI. Rejected. |

The inverse (`aelf demote` at [`cli.py:203`](../src/aelfrice/cli.py))
already takes a single `belief_id` and is an explicit user act with
no implicit state change. `aelf validate` mirrors that surface shape.

### Args

- `belief_id` (positional, required): the belief to promote.
- `--source` (optional, default `"user"`): audit-row source label,
  same semantics as `aelf feedback --source`
  ([`cli.py:874-875`](../src/aelfrice/cli.py)). Lets a UI client
  tag promotions distinctly (`"mcp:validate"`, `"slash:aelf:validate"`).

No bulk operation in v1.2.0 — bulk validate is parked under the
"No bulk operations" sharp-edge in
[LIMITATIONS.md § Surface limits at v1.0](LIMITATIONS.md#surface-limits-at-v10).
Lift if usage justifies.

## 3. Effect on confidence (α/β)

Three options were considered.

| Option | What it does | Cost |
|---|---|---|
| **A. Lock-style prior bump** | Set α=9.0, β=0.5 (matches `aelf lock` at [`cli.py:163-164`](../src/aelfrice/cli.py)) | Promotion would carry the same confidence floor as locks. Defeats the tier distinction — a promoted onboard belief would outrank a high-feedback regular belief on retrieval. |
| **B. Flag-only flip** | Change `origin`, do not touch `alpha` / `beta` | Posterior is preserved as-is. Cleanest: validation is provenance, not evidence. |
| **C. Positive `apply_feedback` event** | Call `apply_feedback(belief_id, valence=+1.0, source='validation')` | One row in `feedback_history`, one Beta-Bernoulli increment. Reversible by replaying with negative valence. |

### Recommended: B (flag-only flip), with audit row

Promotion is a provenance change. The math at retrieval time should
not move because the user clicked a button. If the belief had
posterior 0.5 (Jeffreys floor — see
[LIMITATIONS.md § Sharp edges](LIMITATIONS.md#sharp-edges)) before
promotion, it should have posterior 0.5 after. The user has not
provided new evidence; they have provided a label.

### Why not C (apply_feedback)

`apply_feedback` rejects zero valence
([`feedback.py:135-136`](../src/aelfrice/feedback.py)) and triggers
the demotion-pressure walk on positive valence
([`feedback.py:160-165`](../src/aelfrice/feedback.py)). Using it
as the promotion path would mean: validating a belief that
contradicts a user-locked belief would pressure-walk the lock.
That couples promotion semantics to the contradiction graph, which
is precisely what we don't want — promotion is about the belief
itself, not its neighbourhood.

C is also irreversible-in-place. To "undo" you'd `apply_feedback`
with negative valence, but the audit log then shows two real
feedback events for what was a UI accident. Replay of the audit
trail (deferred but real, see the `feedback_history` design at
[`feedback.py:7-9`](../src/aelfrice/feedback.py)) would treat
both as evidence rather than as one provenance flip and one undo.

### Why not A

A makes promotion a **stronger** floor than is actually warranted.
A promoted belief that turns out wrong takes 5 contradictions to
auto-demote ([`feedback.py:26`](../src/aelfrice/feedback.py)) — the
same as a real lock. The point of having a tier between
`agent_inferred` and `user_stated` is that the floor is **softer**.

### Audit row regardless

Even with option B, the promotion writes one `feedback_history`
row (zero valence, like the contradiction-tiebreaker pattern at
[`contradiction.py:55-66`](../src/aelfrice/contradiction.py)).
Replay logic ignores zero-valence rows; the row exists for
"who promoted what when" reconstruction. See § 6.

## 4. Effect on contradiction tie-breaker

v1.0.1 precedence ([`contradiction.py:9-22`](../src/aelfrice/contradiction.py)):

```
user_stated > user_corrected > document_recent
            (agent_inferred collapsed into document_recent)
```

v1.1.0 expands to four classes once `origin` exists; v1.2.0 adds
`user_validated` and the order becomes:

```
user_stated > user_corrected > user_validated > document_recent > agent_inferred
```

### Where `user_validated` slots in

**Below `user_corrected`, above `document_recent`.**

Worked scenarios:

**(i) `user_validated` vs `document_recent`.** A belief the user
acknowledged from onboard contradicts an unmarked belief from a
later doc edit. The acknowledged belief should win — the user's
one positive signal, however weak, is stronger than no signal
at all. Recency does not flip this; the unmarked belief gets the
later-recency tiebreak only within its own class.

**(ii) `user_validated` vs `user_corrected`.** A belief the user
acknowledged from onboard contradicts a `BELIEF_CORRECTION`. The
correction wins. A correction is the user actively saying "the
prior thing was wrong, here is the right one"; validation is the
user passively saying "this looks right." Active beats passive.

**(iii) `user_validated` vs `user_stated`.** Lock wins. A lock is
non-negotiable; validation is acknowledgement. The user can
explicitly upgrade a `user_validated` belief to a lock via
`aelf lock` if they want this contest to flip — that's a
deliberate act, not a tie-breaker concern.

**(iv) `user_validated` vs `agent_inferred`.** Validation wins.
This is the whole point of having a tier — validated beliefs
should outrank unvalidated peers under contradiction.

**(v) Two `user_validated` beliefs contradict.** Within-class:
fall through to recency, then id, exactly as
[`_pick_winner` at `contradiction.py:155-200`](../src/aelfrice/contradiction.py)
already does. No new tiebreak axis.

### Implementation impact on contradiction.py

v1.2.0 adds `PRECEDENCE_USER_VALIDATED` between
`PRECEDENCE_USER_CORRECTED` (2) and `PRECEDENCE_DOCUMENT_RECENT` (1).
The numeric mapping shifts:

```
PRECEDENCE_USER_STATED      = 5  (was 3)
PRECEDENCE_USER_CORRECTED   = 4  (was 2)
PRECEDENCE_USER_VALIDATED   = 3  (new)
PRECEDENCE_DOCUMENT_RECENT  = 2  (was 1)
PRECEDENCE_AGENT_INFERRED   = 1  (new)
```

`CLASS_NAMES` gains the two new entries. `precedence_class` reads
`belief.origin` as its primary axis once that field exists, with
`lock_level == LOCK_USER` continuing to short-circuit to
`user_stated` (preserves the lock-takes-priority test at
[`tests/test_contradiction.py:86-92`](../tests/test_contradiction.py)).

### `aelf resolve` behaviour

`aelf resolve` ([`cli.py:223-252`](../src/aelfrice/cli.py))
operates on unresolved CONTRADICTS edges and writes one
`feedback_history` row per resolution. With `user_validated` in the
order, a `user_validated`-vs-`document_recent` contradiction
resolves with rule string
`user_validated_beats_document_recent`, the loser gets
SUPERSEDES'd, and the audit row reads
`source='contradiction_tiebreaker:user_validated_beats_document_recent'`.

No new CLI flag. `aelf resolve` continues to resolve everything.

## 5. Reversibility

**Yes. Promotion is reversible.**

Justification: validation is a UI act and UIs make mistakes. A user
clicking validate on the wrong belief, or validating a belief that
later turns out wrong, must be able to walk back without
schema-level surgery.

### Recommended mechanism: extend `aelf demote`

`aelf demote <id>` ([`cli.py:203-220`](../src/aelfrice/cli.py))
currently flips `lock_level=LOCK_USER` → `LOCK_NONE`. Extend it to
also flip `origin=user_validated` → `agent_inferred` when
`lock_level` is already `none`.

| Belief state | `aelf demote` effect today | `aelf demote` effect v1.2.0 |
|---|---|---|
| `lock_level=user` | → `lock_level=none`, prints "demoted" | unchanged |
| `lock_level=none, origin=user_validated` | prints "belief is not locked", no-op | → `origin=agent_inferred`, prints "devalidated" |
| `lock_level=none, origin=agent_inferred` | prints "belief is not locked", no-op | unchanged |
| `lock_level=user, origin=user_validated` | → `lock_level=none`, `origin` untouched | → `lock_level=none`, `origin` stays `user_validated` (one tier per `demote` call) |

One tier per `demote` call. Same as how
[`feedback._pressure_and_maybe_demote`](../src/aelfrice/feedback.py)
demotes one tier at a time even if pressure thresholds were already
crossed.

### Alternative: separate `aelf devalidate` command

Cleaner separation but adds CLI surface area. Rejected: the cost
of an extra command exceeds the cost of `demote` having two cases.
Re-evaluate at v1.3 if `demote`'s help text becomes confusing.

TBD: command name when devalidate becomes the user-facing verb in
help output (e.g. `aelf demote --help` says "demote a lock or
devalidate a validated belief").

### Audit row on devalidate

Same shape as the promotion row, with a different rule string.
See § 6.

## 6. Audit row

Every promotion writes one row to `feedback_history`. Format
matches the contradiction-tiebreaker pattern
([`contradiction.py:243-251`](../src/aelfrice/contradiction.py)).

### Promotion row

| Column | Value | Notes |
|---|---|---|
| `id` | autoincrement | row id |
| `belief_id` | the promoted belief's id | the subject of promotion |
| `valence` | `0.0` | bookkeeping; replay logic ignores zero-valence rows. Same as the tie-breaker convention. |
| `source` | `"promotion:user_validated"` | wire-format string. Source prefix is `"promotion"` so all promotion-related events filter together. |
| `created_at` | ISO-8601 UTC, `Z`-suffixed | as `_utc_now_iso()` at [`feedback.py:48-50`](../src/aelfrice/feedback.py) |

The `source` field carries the new origin as the suffix so a future
`origin=agent_validated` (if such a tier ever lands) doesn't
collide on the source string.

### Devalidation row

| Column | Value |
|---|---|
| `belief_id` | the devalidated belief's id |
| `valence` | `0.0` |
| `source` | `"promotion:revert_to_agent_inferred"` |
| `created_at` | ISO-8601 UTC |

### Reading the audit log

`source LIKE 'promotion:%'` filters to all promotion-related events.
`source = 'promotion:user_validated'` counts how many beliefs were
ever promoted. Pairing each promotion with its eventual devalidation
(if any) is a join on `belief_id` ordered by `created_at`.

## 7. MCP / CLI surface

### `aelf validate <belief_id>`

```
$ aelf validate 8f3a2b1c4d5e6f78
validated: 8f3a2b1c4d5e6f78 (origin: agent_inferred -> user_validated)
```

Exit codes:

| Code | When |
|---|---|
| 0 | promotion succeeded, including the idempotent-already-validated case |
| 1 | belief not found, belief is locked (cannot promote a lock further), origin already at a higher tier than `user_validated` |
| 2 | argparse usage error |

### Worked examples

**Success case.**

```
$ aelf validate 8f3a2b1c4d5e6f78
validated: 8f3a2b1c4d5e6f78 (origin: agent_inferred -> user_validated)
```

**Already validated (idempotent).** Does not re-write an audit row.
Same convention as `aelf setup` already-installed
([`cli.py:444-449`](../src/aelfrice/cli.py)).

```
$ aelf validate 8f3a2b1c4d5e6f78
already validated: 8f3a2b1c4d5e6f78
```

Exit 0.

**Belief not found.** stderr.

```
$ aelf validate ghost
belief not found: ghost
```

Exit 1.

**Belief is locked.** stderr.

```
$ aelf validate ab12cd34ef56gh78
cannot validate locked belief: ab12cd34ef56gh78
(locks already exceed user_validated; use 'aelf demote' to drop the lock first)
```

Exit 1.

Justification for refusing rather than no-oping: a lock is a
strictly stronger statement than validation. Silently no-oping
hides the fact that the user's intent (probably "I want this to
be in user_validated tier") doesn't match the current state.
Demote-then-validate is the explicit two-step.

**Belief is `user_stated` origin without lock_level=user.** Should
not occur if backfill is correct (the two are coupled at insert
time). If it does occur (hand-edited DB), refuse with a clear
message.

```
$ aelf validate cd56ef78ab90gh12
cannot validate user_stated belief: cd56ef78ab90gh12
(origin already at higher tier; use 'aelf demote' to drop the lock first)
```

Exit 1. TBD whether to make this a 2 (data inconsistency) or a 1
(refuse like the locked case). Lean toward 1 for surface
consistency.

### MCP `aelf:validate`

```python
{
    "name": "aelf:validate",
    "description": "Promote an agent_inferred belief to user_validated.",
    "input": {
        "belief_id": "8f3a2b1c4d5e6f78",
        "source": "mcp:validate"  # optional
    },
    "output": {
        "belief_id": "8f3a2b1c4d5e6f78",
        "prior_origin": "agent_inferred",
        "new_origin": "user_validated",
        "audit_event_id": 1234
    }
}
```

Mirrors the `FeedbackResult` return shape from
[`feedback.py:32-45`](../src/aelfrice/feedback.py) — caller can read
prior+new state and the audit-row id without re-reading the store.

Errors as MCP tool errors with the same exit-code semantics as the
CLI: `belief not found`, `cannot validate locked belief`,
`cannot validate user_stated belief`.

## 8. Test plan

Atomic property tests in `tests/test_promotion.py`. Style matches
[`tests/test_contradiction.py`](../tests/test_contradiction.py):
in-memory store via `_seed`, one property per test, no fixtures
beyond a builder helper.

### Provenance flip

- `test_validate_changes_origin_to_user_validated` — happy path.
- `test_validate_does_not_change_lock_level` — `lock_level=none` stays `none`.
- `test_validate_does_not_change_alpha_beta` — posteriors preserved (option B from § 3).
- `test_validate_does_not_change_type` — `type` preserved.
- `test_validate_idempotent_no_double_audit_row` — second call writes no row.

### Refusal cases

- `test_validate_raises_on_missing_belief` — ValueError, parallel to `test_resolve_raises_on_missing_a` at [`tests/test_contradiction.py:262-265`](../tests/test_contradiction.py).
- `test_validate_refuses_locked_belief` — exit 1, error message includes "locked".
- `test_validate_refuses_user_stated_origin` — exit 1.

### Audit row shape

- `test_validate_writes_audit_row_with_source_prefix` — `source.startswith("promotion:")`.
- `test_validate_audit_row_has_zero_valence` — replay-safe (mirrors `test_resolve_audit_row_zero_valence_does_not_affect_replay` at [`tests/test_contradiction.py:246-256`](../tests/test_contradiction.py)).
- `test_validate_audit_row_carries_now_kwarg` — clock injection, parallel to [`tests/test_contradiction.py:236-243`](../tests/test_contradiction.py).
- `test_validate_audit_row_belief_id_is_subject` — the row's `belief_id` is the promoted belief, not anything else.

### Tie-breaker integration

- `test_user_validated_beats_document_recent` — new precedence rule fires.
- `test_user_corrected_beats_user_validated` — order respected upward.
- `test_user_stated_beats_user_validated` — lock still wins.
- `test_user_validated_beats_agent_inferred` — full four-class split now works.
- `test_two_user_validated_break_by_recency` — within-class falls through.

### Reversibility

- `test_demote_devalidates_user_validated_belief` — `demote` on
  `lock_level=none, origin=user_validated` flips origin back, writes a
  `promotion:revert_to_agent_inferred` audit row.
- `test_demote_one_tier_at_a_time` — demoting a
  `lock_level=user, origin=user_validated` belief drops the lock first,
  leaves `origin=user_validated` for the next call.
- `test_devalidate_then_revalidate_preserves_alpha_beta` — round-trip
  is provenance-only, posterior never moves.

### CLI smoke

- `test_cli_validate_prints_origin_transition` — stdout matches the worked example.
- `test_cli_validate_exit_1_on_missing` — exit code.
- `test_cli_validate_exit_1_on_locked` — exit code.

## 9. Open questions

Marked `TBD` for items that need implementation feedback to settle.

- **TBD: column name.** `origin` is the recommended name. Alternatives
  considered: `provenance`, `source_type`, `tier`. `origin` is short
  and matches the prose in
  [`contradiction.py:30-38`](../src/aelfrice/contradiction.py).
  Locked unless an SQLite reserved word collision surfaces.
- **TBD: backfill of pre-v1.1.0 rows.** Recommendation in § 1 is to
  leave non-locked, non-correction rows as `unknown` rather than
  retroactively label as `agent_inferred`. Open to flipping this if
  the cost of a "ghost" `unknown` tier is worse than the cost of one
  fuzzy retroactive label.
- **TBD: should `aelf:remember` (MCP) write `origin='agent_remembered'`
  or `origin='agent_inferred'`?** They are semantically distinct
  (the agent decided mid-session vs. the scanner extracted at onboard).
  Implementation-side decision; this memo doesn't constrain it. Affects
  whether `aelf validate` accepts `agent_remembered` beliefs as input.
- **TBD: bulk validate.** Out of scope at v1.2.0. Open to revisit if
  usage shows users wanting to mass-validate after an `aelf onboard`
  run. Would need its own UX (interactive review? --pattern?).
- **TBD: separate `aelf devalidate` command.** Recommended path is to
  extend `aelf demote`. If `demote`'s help text becomes confusing,
  split. Decision deferred to v1.2.0 implementation.
- **TBD: the `unknown` origin in the contradiction tie-breaker.** Where
  does it slot? Recommendation: same class as `document_recent` (it's
  the same pre-v1.1.0 absorption that the tie-breaker already does).
  Confirm during v1.2.0 implementation.
- **TBD: exit code for the data-inconsistency case** (origin already
  `user_stated` without lock). Lean toward 1; arguable for 2.
- **TBD: validation gives access to a softer auto-demote threshold?**
  The locked-belief auto-demote at `DEMOTION_THRESHOLD=5`
  ([`feedback.py:26`](../src/aelfrice/feedback.py)) is calibrated for
  locks. `user_validated` beliefs aren't locked, so the threshold
  doesn't apply today. Future question: should `user_validated`
  beliefs accumulate their own (lower) demotion pressure? Out of
  scope for v1.2.0; flagged for v1.3+ once feedback-into-ranking
  data exists.
- **TBD: interaction with [LIMITATIONS.md § harness conflict](LIMITATIONS.md#harness-conflict--claude-code-auto-memory-write-path).**
  At v1.2.0 the MCP write path is documented but not canonical. Does
  `aelf:validate` offer the user a path to graduate harness-side
  beliefs into the MCP store? Probably no — the harness store and
  the MCP store are different stores; cross-store promotion is a
  separate feature. Confirm with implementation.
