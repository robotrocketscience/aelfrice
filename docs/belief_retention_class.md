# Belief retention class + aging policy (#290)

## Status

**Spec — proposing for ratification.** Doc-only. Implementation
blocks on this memo + #288 logs (for retention-class distribution
audit) before any aging coefficient is calibrated.

## Problem

The existing belief-type axis (`BELIEF_FACTUAL`,
`BELIEF_CORRECTION`, `BELIEF_PREFERENCE`, `BELIEF_REQUIREMENT` —
`models.py:13–22`) describes the *form* of a claim, not its
*expected lifetime*. Three categorically different retention
profiles share the type axis today and are ranked identically:

1. **Stable facts about the codebase / project.** "X uses SQLite."
   Should age slowly. Wrong if the code changes and the belief
   doesn't.
2. **Snapshot-of-thought / research notes.** "I think we should
   consider Y." True at the moment it was written. Should age out
   fast unless promoted.
3. **Transient debugging state.** "Fix applied." True for ~one PR.
   Useless after merge.

`1bc8ab45a40351d9` ("store.insert_belief() does content-hash
dedup" — #281) is a class-3 belief that scored identically to a
class-1 belief and was retrieved as authoritative. The form is
"factual"; the retention profile should have been "transient." No
existing axis carries that signal.

## Recommendation summary

- **New orthogonal axis: `retention_class`.** Three values:
  `fact`, `snapshot`, `transient`. Stored as a column on `beliefs`,
  defaulting to `unknown` for migrated rows.
- **Default per ingest path.** Filesystem / git ingest →
  `fact`. Transcript ingest → `snapshot`. Hook ingest →
  `snapshot`. CLI remember / MCP remember → operator-supplied;
  default `fact` (operator typed it deliberately).
  Consolidation-migration synthetic rows inherit the canonical's
  retention class.
- **Soft down-weight in ranking, not hard expiry.** Beliefs don't
  *expire*; their `final_score` gets a class-and-age multiplier
  before the floor (#289) is applied. This means a sufficiently
  high BM25 + posterior + corroboration signal can still surface a
  6-month-old transient belief, but it has to *earn* it.
- **Promotion: snapshot → fact when corroborated N=3 times across
  M=2 sessions.** Reuses the corroboration recorder (#190) — no new
  signal, no new table.
- **Locks override retention.** A locked belief is operator-asserted
  ground truth and skips the age multiplier. Hibernation (#196)
  already excludes locks from its trigger; this rule mirrors that.
- **Migration: bulk-mark existing 20K+ beliefs as `unknown`,
  classify by `source_kind` heuristic in a one-shot pass after
  ratification.**

## Detailed proposal

### 1. Schema axis

New column on `beliefs`:

```sql
retention_class TEXT NOT NULL DEFAULT 'unknown'
    CHECK (retention_class IN ('fact', 'snapshot', 'transient', 'unknown'))
```

`unknown` exists only for the migration window. New writes always
land with one of the three live values.

`models.py` constants:

```python
RETENTION_FACT: Final[str] = "fact"
RETENTION_SNAPSHOT: Final[str] = "snapshot"
RETENTION_TRANSIENT: Final[str] = "transient"
RETENTION_UNKNOWN: Final[str] = "unknown"  # migration-only

RETENTION_CLASSES: Final[frozenset[str]] = frozenset({
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_TRANSIENT,
    RETENTION_UNKNOWN,
})
```

**Decision asks for the axis:**

- [ ] **Three live values.** Confirm `fact`, `snapshot`,
  `transient`. Reject if a fourth class is needed (and name what
  it covers that the three above don't).
- [ ] **Orthogonal to type.** Confirm `retention_class` is its own
  column, not a refinement of the existing `type` enum. Reject if
  the type enum should be extended instead (and accept that
  `BELIEF_FACTUAL` would split into `BELIEF_FACTUAL_FACT` /
  `_SNAPSHOT` / `_TRANSIENT`, doubling the type count).

### 2. Defaults per ingest path

| Ingest path | Default `retention_class` | Rationale |
|---|---|---|
| `filesystem_ingest` (scanner, onboard) | `fact` | File contents are observed, not asserted; should age slowly |
| `git_ingest` (commit ingest hook) | `fact` | Commit messages are durable history |
| `python_ast` (#205 lane) | `fact` | Code structure facts |
| `transcript_ingest` (turns log) | `snapshot` | Conversation content; opinions and proposals dominate |
| `hook_ingest` (UserPromptSubmit feedback rows) | `snapshot` | Operator-typed prose |
| `cli_remember` (`aelf lock` / `aelf remember`) | `fact` | Operator typed it deliberately, intent is preservation |
| `mcp_remember` (MCP write tool) | `fact` | Same as cli_remember |
| `feedback_loop_synthesis` | `snapshot` | Generated, not asserted |
| `consolidation_migration` (synthetic) | inherits canonical | Migration row, not a new claim |
| `legacy_unknown` (pre-v2.0 backfill) | `unknown` | Migration window |

`transient` is **never assigned by default**. It exists only for
operator-supplied "this is debugging state" markers via a new
`aelf remember --transient` flag (out-of-scope for this spec, lands
when the first transient use case surfaces).

**Decision ask:**

- [ ] **Defaults per ingest path.** Confirm the table above.
  Reject specific rows if a different default is preferred (note
  which row + reason).
- [ ] **No automatic `transient` assignment.** Confirm `transient`
  requires explicit operator opt-in. Reject if a heuristic
  (e.g. "fix applied" string match) should auto-classify.

### 3. Aging multiplier

The composite score from #289 becomes:

```
final_score = bm25_normalized
            * (0.5 + 0.5 * posterior_mean)
            * retention_age_multiplier(retention_class, age_days)
```

`retention_age_multiplier` is monotonically non-increasing in age:

| Class | Multiplier formula | Half-life |
|---|---|---|
| `fact` | `max(0.7, 1.0 - 0.001 * age_days)` | ~300 days to floor |
| `snapshot` | `max(0.4, 0.95 ** (age_days / 14))` | 14-day half-life, floors at 0.4 |
| `transient` | `max(0.1, 0.5 ** (age_days / 1))` | 1-day half-life, floors at 0.1 |
| `unknown` | `0.7` | constant — migration discount |

The floors prevent the multiplier from ever zeroing; combined with
#289's hard floor, a sufficiently low score still drops out, but
strong other signals can rescue an old belief.

**Lock override.** If `lock_level == LOCK_USER`, the multiplier is
forced to `1.0` regardless of class or age. Locks are operator
ground truth; aging would silently demote them.

**Calibration.** The half-life and floor constants are placeholders.
#288 logs let an operator audit the actual age distribution per
class and tune. The *shape* (exponential decay with floor) is
ratified here; the constants are tunable via `[retention_aging]`
config.

**Decision asks for aging:**

- [ ] **Multiplier shape — exponential decay with floor.** Confirm
  per-class formulas with min-floors. Reject if hard expiry is
  preferred (and accept that strong-signal old beliefs disappear).
- [ ] **Lock override forces multiplier = 1.0.** Confirm locks
  bypass aging. Reject if locks should still age (note what locks
  *would* age toward).
- [ ] **Ship with placeholder constants; calibrate from #288 logs**
  in a follow-up PR.

### 4. Promotion: snapshot → fact

A snapshot belief is promoted to `fact` when:

```
corroboration_count >= 3
AND distinct_sessions(corroborations) >= 2
AND no contradiction edge (CONTRADICTS) targets this belief
```

Reuses the existing `belief_corroborations` table (#190) — no new
signal infrastructure needed. Distinct-sessions is enforced via
the corroboration row's `session_id` (already populated by #192
T3 once it lands; until then, the rule degrades to
`corroboration_count >= 3` only).

Promotion happens during the doctor pass (`aelf doctor
--promote-retention` — new flag, opt-in for v1.x same as
`--classify-orphans`). On promotion, write a synthetic
`feedback_history` row with
`source = 'retention_promotion'` and the multiplier resets to the
`fact` curve from `created_at = now`.

**No demotion.** A `fact` belief never becomes a `snapshot` again.
If it's wrong, the contradiction / correction lane handles it.
Asymmetry is intentional: promotion is cheap (we found more
evidence); demotion would be a substantively new operator
intervention.

**Decision asks for promotion:**

- [ ] **N=3 corroborations, M=2 distinct sessions, no
  contradiction.** Confirm thresholds. Reject if different
  N / M is preferred.
- [ ] **Doctor-pass opt-in for v1.x.** Confirm
  `aelf doctor --promote-retention` is opt-in (mirrors
  `--classify-orphans` from #253). Reject if automatic on every
  ingest (and accept the cost).
- [ ] **No demotion.** Confirm `fact → snapshot` does not exist;
  contradictions handle correctness. Reject if demotion is wanted
  (note the trigger).

### 5. Migration

One-shot pass on first open after upgrade,
`_maybe_classify_retention_class()`, schema-meta gated. Pattern
matches `_maybe_consolidate_content_hash_duplicates` from #283 /
#219.

Classification heuristic by `source_kind`:

```python
def _classify_retention_from_source(source_kind: str) -> str:
    if source_kind in (INGEST_SOURCE_FILESYSTEM, INGEST_SOURCE_GIT,
                       INGEST_SOURCE_PYTHON_AST,
                       INGEST_SOURCE_CLI_REMEMBER,
                       INGEST_SOURCE_MCP_REMEMBER):
        return RETENTION_FACT
    if source_kind in ("transcript_ingest", "hook_ingest",
                       INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS):
        return RETENTION_SNAPSHOT
    return RETENTION_UNKNOWN  # legacy_unknown stays unknown
```

`legacy_unknown` rows from #271 keep `retention_class = 'unknown'`
and pick up the `0.7` constant multiplier. Operators can run
`aelf doctor --classify-retention` (separate from the promote
flag) to triage these manually if they care; default is to leave
them at `unknown` and let the multiplier handle it.

**Idempotence.** Schema-meta marker
`SCHEMA_META_RETENTION_CLASS_CLASSIFIED = "retention_class_classified"`
gates re-runs. Mirrors the existing schema-meta pattern.

**Decision ask:**

- [ ] **Heuristic-based migration.** Confirm bulk classify by
  `source_kind` with `unknown` as the fallback. Reject if an
  operator-driven manual triage is preferred for the migration
  pass (accept the operator-time cost).

### Out of scope

- **Phantom / wonder beliefs (#228).** Already a separate type axis
  and a separate generation path. Phantom beliefs get
  `retention_class = 'snapshot'` by default but the wonder lane's
  promotion trigger (#229) is independent of this memo's
  promotion rule.
- **Multi-axis posterior (#196).** Orthogonal. The
  `posterior_mean` term in the score formula is whatever the
  substrate ratifies.
- **The aging coefficients themselves.** Calibrated from #288
  data. Spec ratifies the shape; tuning lands in a follow-up PR.
- **`aelf remember --transient` operator flag.** Out of scope
  until a concrete transient-marking use case surfaces. The
  `transient` class exists in the schema and the multiplier table
  so the lane is ready when called for.
- **Per-project retention overrides.** Some projects may want
  longer snapshot half-lives (research repos) or shorter
  (rapid-iteration codebases). Defer.

## Decision asks (consolidated)

- [ ] **Schema** — three live values + `unknown`, orthogonal column.
- [ ] **Defaults per ingest path** — table in §2.
- [ ] **`transient` requires explicit opt-in.**
- [ ] **Multiplier shape** — per-class exponential decay with floor.
- [ ] **Lock override** forces multiplier = 1.0.
- [ ] **Ship with placeholder constants; calibrate from #288.**
- [ ] **Promotion** — N=3, M=2, no-contradiction, doctor-pass opt-in.
- [ ] **No demotion** — contradictions handle correctness.
- [ ] **Heuristic-based migration** with `unknown` fallback.

## Implementation tracker (post-ratification)

Roughly three PRs.

1. **Schema + constants + score-formula change.** Migration
   helper (`_maybe_classify_retention_class`), `models.py`
   constants, `[retention_aging]` config, score-multiplier hook
   into `rebuild_v14`. Tests for migration idempotence,
   per-class multiplier formula, lock override, score-formula
   continuity at retention transitions. ~500 lines net.
2. **Per-ingest defaults.** Wire each ingest path to set its
   default. Touches `scanner.py`, `ingest.py`,
   `hook_commit_ingest.py`, `triple_extractor.py`. Tests covering
   each path's default. ~250 lines net.
3. **Promotion lane.** `aelf doctor --promote-retention` flag
   wiring + promotion query + synthetic-feedback row write +
   tests. ~300 lines net.

## Provenance

- Parent: #286.
- Symptom evidence: #281 (`1bc8ab45a40351d9` example).
- Adjacent: #196 (substrate decision — orthogonal axis), #229
  (promotion trigger for wonder lane — distinct from this memo's
  promotion rule but uses the same corroboration table), #190
  (corroboration recorder — provides the N≥3 signal), #253
  (`aelf doctor --classify-orphans` precedent for opt-in flags),
  #283 / #219 (`_maybe_consolidate_content_hash_duplicates`
  precedent for one-shot migrations).
- Code touchpoints:
  - `src/aelfrice/models.py` (axis constants).
  - `src/aelfrice/store.py` (schema column + migration helper).
  - `src/aelfrice/context_rebuilder.py` (score formula extension).
  - `src/aelfrice/scanner.py`, `ingest.py`,
    `hook_commit_ingest.py`, etc. (per-path defaults).
  - `src/aelfrice/doctor.py` (promote-retention flag).
