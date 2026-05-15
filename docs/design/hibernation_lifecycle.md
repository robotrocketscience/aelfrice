# Hibernation lifecycle (v2.0 #196 behavior half)

## Status

**Design memo — UNIMPLEMENTED on `github/main` as of v3.1.** Storage
half landed in PR #282 (`hibernation_score REAL` +
`activation_condition TEXT`, both nullable on `beliefs`); both columns
sit unused. `git grep 'list_hibernated\|hibernate_eligible\|def.*hibernat' src/`
returns nothing; `git grep 'hibernation_score\s*=' src/` shows only the
row→model load and the `INSERT/UPDATE` statements that write `NULL` —
no scorer populates the column.

Treat the rest of this file as design-stage notes for a future
implementation pass, not as documentation of current behavior.
The trigger / grammar / sweeper sections below describe a proposed
shape; nothing here ships in v3.x. Re-open this memo for ratification
when an eligibility surface lands.

The original spec also keyed the trigger off `demotion_pressure`;
that column was removed by #814 / PR #820, so any future
implementation will need to choose new signal columns. The §
"Detailed proposal" SQL examples below have been pruned of
`demotion_pressure` references already (see #822 / PR #824), but the
rest of the trigger logic is unchanged from the original draft and
remains hypothetical.

## Problem

The substrate decision (`docs/design/substrate_decision.md`) ratified
hibernation as a separable lifecycle feature on the scalar
Beta-Bernoulli substrate. The columns are now in place but unused.
Until behavior lands, `hibernation_score` is always `NULL`, no belief
hibernates, and `activation_condition` is dead schema.

Three open questions:

1. **Trigger.** Under what condition does a belief's
   `hibernation_score` transition `NULL → float`?
2. **Predicate grammar.** What does `activation_condition` actually
   contain? "JSON" was ratified; the schema inside the JSON is not.
3. **Sweeper.** What process *applies* hibernation? When? Where?

## Recommendation summary

- **Trigger:** scheduled doctor-pass sweep, not online. Score is
  `1.0 - posterior_mean(α, β)` clamped to `[0, 1]`, written only when
  `last_retrieved_at` is older than 30 days *and*
  `lock_level == LOCK_NONE`. NULL otherwise.
- **Activation condition grammar:** narrow, declarative JSON object
  with three optional clauses — `keywords_any`, `source_kind`, and
  `after_ts`. All clauses combine with implicit AND. No nesting, no
  arbitrary boolean trees, no expressions.
- **Sweeper:** lives in `aelfrice.hibernation`. Two entry points —
  `score_pass()` writes hibernation_score, `wake_pass(query)` clears
  hibernation_score for beliefs whose activation_condition matches an
  incoming retrieval query. Both invoked from `aelf doctor`; the
  retrieval pipeline calls `wake_pass` opportunistically on prompt
  submit.

## Detailed proposal

### 1. Hibernation trigger — when does the score get written

Three trigger candidates, ordered by how much I trust them:

#### Candidate A — passive decay against retrieval

The original spec proposed a passive-decay trigger keyed off
`last_retrieved_at`, `lock_level`, posterior mean, and (originally)
`demotion_pressure`. The latter column was removed by #814 / PR #820;
no replacement signal has been chosen. As of v3.1, no SQL trigger of
this shape exists in the codebase — `hibernation_score` is always
written as `NULL`. The 30-day window referenced below would
notionally match `feedback_history` retention semantics.

This section, like the rest of this memo, is a design sketch for a
future implementation pass. Re-open it with concrete SQL when an
implementation lands.

**Why this and not the others:**

- *Pure age* hibernates beliefs that were correct-and-quiet
  (the user just hasn't talked about them).
- *Pure negative-feedback* hibernates beliefs the user actively
  marked wrong via `harmful` — but those should be *deleted* via the
  existing feedback path, not soft-suspended.
- *Pure low-posterior* hibernates beliefs the system never had
  evidence for — but new beliefs start at α=β=1.0 (posterior 0.5),
  so this would hibernate brand-new beliefs.

The conjunction is what makes the rule safe: a belief has to be
*old, contested, and unsupported* before it hibernates. Beliefs that
are old-and-quiet (probably true, just not the topic du jour) keep
hibernation_score NULL and stay first-class.

#### Candidate B — corroboration-aware trigger

Same as A, but multiply `posterior_mean` against
`corroboration_count` weight before testing. Belief with many
corroborations is harder to hibernate.

Defer until #190 corroboration-rank-influence work lands and we have
a calibrated weight. Using corroboration_count raw would over-protect
beliefs from chatty sources.

#### Candidate C — wake by retrieval, never trigger automatically

Hibernation_score is NULL forever; `activation_condition` is the
only mechanism. Score column becomes vestigial.

Rejected: the substrate decision specifically ratified the score as
a soft-suspension mechanism. Without trigger, score is unused.

**Decision ask (deferred):** confirm a Candidate-A-style trigger
shape. The original draft listed `≥1 demotion_pressure` as one
threshold; that column no longer exists, so any future ratification
needs to choose a replacement signal first. Constants (30 days,
posterior cutoff) are individually defensible but not benchmarked.
Ratification deferred until an implementation pass is scheduled.

### 2. Activation-condition predicate grammar

`activation_condition` stores a JSON object. **Three optional clauses,
implicit AND, no nesting:**

```json
{
  "keywords_any": ["term_a", "term_b"],
  "source_kind": "git",
  "after_ts": "2026-05-01T00:00:00Z"
}
```

| Clause | Type | Semantics |
|---|---|---|
| `keywords_any` | `list[str]` | match if any keyword (case-insensitive substring) appears in incoming query/prompt |
| `source_kind` | `str` | match if incoming retrieval call's `source_kind` equals this value (one of `INGEST_SOURCE_*`) |
| `after_ts` | ISO-8601 string | match only if current wall-clock time is past this timestamp |

A clause that is absent does not constrain. An object with no clauses
matches always (use sparingly — semantically equivalent to "wake on
any retrieval"). Unknown top-level keys raise `ValueError` at write
time. There is no `all`, `any`, `not`, or arbitrary expression — the
grammar is closed.

**Why the closed grammar:** an open grammar (Python-style booleans,
JSONLogic, jq) is one DSL the user has to learn, one parser to
maintain, and one eval-injection surface (`activation_condition` is
written from places the user does not always inspect). The closed
shape covers the three use cases the substrate decision named —
keyword-driven wake, source-scoped wake, time-deferred wake — and
nothing else. Widening is purely additive when a fourth use case
earns its keep; narrowing later breaks beliefs in the wild.

### 3. Sweeper

New module `src/aelfrice/hibernation.py`, two pure entry points:

```python
def score_pass(store: MemoryStore, *, now: datetime) -> int:
    """Walk eligible beliefs and write hibernation_score. Returns
    the count of newly-hibernated beliefs. Idempotent — beliefs
    already hibernated keep their score (re-scoring is a separate
    decision that should not collide with this pass)."""

def wake_pass(
    store: MemoryStore,
    query: str,
    *,
    source_kind: str | None = None,
    now: datetime,
) -> int:
    """Walk hibernated beliefs, evaluate activation_condition against
    (query, source_kind, now), clear hibernation_score on hits.
    Returns the count of newly-woken beliefs. Pure with respect to
    its inputs — same store + same query → same wakes."""
```

**Where these run:**

- `score_pass` — invoked from `aelf doctor --hibernate` (new flag).
  Not run automatically; opting into hibernation is an explicit user
  step in v2.0. v2.1 may make it default-on once the trigger
  thresholds have been benchmarked.
- `wake_pass` — invoked from the `UserPromptSubmit` retrieval path
  *before* the L0/L1/L2.5 search. A belief that wakes on this prompt
  is eligible for retrieval on this same prompt; the wake clears
  `hibernation_score` so the FTS5/BM25 path returns it normally.
  Cost: one extra SELECT against `WHERE hibernation_score IS NOT NULL`,
  cheap because most beliefs aren't hibernated.

**Retrieval interaction:** beliefs with `hibernation_score IS NOT
NULL` are filtered *out* of L1/L2.5 retrieval (lock-tier L0 is
unaffected — locked beliefs cannot hibernate per the trigger rule).
The filter lives in `retrieval.py` next to the existing
`lock_level` filter; `wake_pass` running before retrieval means a
belief that should be visible this turn is already woken by the time
the filter runs.

### 4. Out of scope

- **Re-hibernation cadence.** A woken belief stays awake; whether to
  re-hibernate after a follow-up cold period is a v2.x decision.
- **Per-aspect hibernation.** Substrate decision rejected
  multi-axis; a hibernated-on-aspect-X belief does not exist.
- **User-visible CLI for hibernation_score.** No `aelf hibernate`
  command, no list-hibernated subcommand. Doctor pass is the surface.
  Add later if users ask.
- **Migration of existing rows.** All beliefs ship with
  hibernation_score NULL today (PR #282). The first `score_pass` run
  is what populates it.

## Decision asks

- [ ] **Trigger rule (Candidate A).** Confirm conjunction of
  *unlocked, retrieved-and-cold, demoted, low-posterior*. Constants
  (30 days, ≥1 demotion, <0.6 posterior) are tunable in v2.x; only
  the shape needs ratification now.
- [ ] **Predicate grammar.** Confirm closed three-clause grammar
  (`keywords_any`, `source_kind`, `after_ts`) with implicit AND and
  no nesting. Reject if an open grammar is preferred — note which
  use case isn't covered.
- [ ] **Sweeper placement.** Confirm `score_pass` opt-in via `aelf
  doctor --hibernate` and `wake_pass` automatic on
  `UserPromptSubmit`. Reject if either should be the other.
- [ ] **Retrieval filter.** Confirm hibernated beliefs are filtered
  from L1/L2.5 (and L0 cannot hibernate). Reject if hibernation
  should affect ranking only (down-weight, not exclude).

## Implementation tracker (post-ratification)

Once this memo is ratified, the implementation is roughly three PRs:

1. `aelfrice.hibernation` module — pure functions for scoring,
   predicate evaluation, and the two sweeper entry points. Pytest
   coverage of the trigger conjunction, the grammar (round-trip,
   unknown-key rejection, every clause matched/missed), the wake
   path. Implementation needs design judgment; tests are mechanical.
2. `aelf doctor --hibernate` flag wiring + retrieval-pipeline filter
   + `UserPromptSubmit` `wake_pass` invocation. Touches the existing
   doctor CLI and `hook.py`.
3. `LIMITATIONS.md`, `PHILOSOPHY.md`, and the user-facing README
   `## Beliefs hibernate` section. Doc-only.

## Provenance

- Storage half: PR #282 (squash-merged 2026-04-29).
- Substrate ratification: `docs/design/substrate_decision.md` § Decision
  asks #3 + #4.
- Issue #196 stays open as the v2.0 substrate tracking issue until
  the three implementation PRs above ship.
