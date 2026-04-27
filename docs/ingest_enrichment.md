# Ingest enrichment

**Status:** spec.
**Target milestone:** v1.2.0 (alongside the commit-ingest hook,
transcript-ingest, and triple-extraction port already on the public
roadmap).
**Dependencies:** stdlib only.
**Risk:** medium. Three small schema additions plus a contract for
what new ingest paths must populate. Backward-compatible — existing
v1.0 stores read forward without alteration.

## Summary

Three coupled schema additions to the ingest path that, together,
unblock several downstream retrieval techniques whose synthetic
results cannot fire on real production data because the upstream
ingest does not currently write the fields they read:

1. **Sessions on beliefs.** A new ingest call carries a `session_id`;
   the store records it on every belief inserted in that call.
2. **Anchor text on edges.** A new `anchor_text` column on `edges`,
   populated by ingest paths that have access to the citing belief's
   own phrasing of the relationship.
3. **`DERIVED_FROM` re-added** to the edge-type vocabulary, with an
   ingest path positioned to emit it.

These changes are bundled in one spec because they share an
implementation surface (the ingest module + the store mutators), have
the same migration shape (nullable column / opt-in edge type), and
the same correctness story (fields populated only on the new write
path; existing rows unchanged).

## Motivation

A retrieval technique that reads a field the ingest pipeline never
writes lands as dead code. Several downstream v1.3+ retrieval
techniques on the roadmap depend on data the v1.0 ingest paths do not
produce:

- Posterior-weighted ranking reads `(α, β)` updated by
  `apply_feedback`. Without dense feedback events, the signal stays
  flat against the prior — the technique would not move ranking on
  real corpora.
- Augmented BM25F reads `Edge.anchor_text`. v1.0 edges have no
  anchor text — the technique would index empty strings.
- Session-coherent supersession reads `Belief.session_id` plus
  `RELATES_TO` / `DERIVED_FROM` edges. v1.0 stores have neither
  populated densely.

This spec lands the schema. The producers that populate it densely
are the v1.2.0 transcript-ingest hooks, commit-ingest hook, and
triple-extraction port (each documented separately).

## Design

### 1. Sessions on beliefs

Add a nullable `session_id` column to `beliefs`:

```sql
ALTER TABLE beliefs ADD COLUMN session_id TEXT;
```

A new ingest entry point `start_ingest_session(model: str | None,
project_context: str | None) -> str` returns a fresh session id and
records the open session. Subsequent ingest calls take an optional
`session_id` parameter; when present, the store writes it on every
belief inserted in the call. Caller is responsible for calling
`complete_ingest_session(session_id)` at the end of a logical group.

Bare ingest (no session) is still allowed for backward compatibility
— `session_id` stays NULL on those beliefs, and any downstream
technique that requires session-coherent grouping simply does not
fire on legacy rows. Conservative-failure behavior, no false
positives on legacy data.

The existing `MemoryStore.create_session` / `complete_session`
methods (which today are no-ops) become the natural entry points.

### 2. Anchor text on edges

Add a nullable `anchor_text` column to `edges`:

```sql
ALTER TABLE edges ADD COLUMN anchor_text TEXT;
```

`anchor_text` is a short free-text label the citing belief's author
wrote describing the relationship. Example: belief A `CITES` belief
B with `anchor_text = "the WAL discussion"`.

The `Edge` dataclass gains a corresponding optional field:

```python
@dataclass
class Edge:
    src: str
    dst: str
    type: str
    weight: float
    anchor_text: str | None = None
```

Ingest contract: any path that has access to the citing prose at
edge-creation time MUST populate `anchor_text`. This includes the
v1.2.0 triple-extraction port (which reads sentences and identifies
relationships, so the surrounding prose is in scope) and the
v1.2.0 commit-ingest hook (which reads commit messages, where
"because of X" / "supersedes Y" framing is common).

Ingest paths that lack the source prose (programmatic edge creation
from internal logic; tests; bulk imports) leave `anchor_text` NULL.
Downstream augmented-BM25F retrieval ignores NULL anchors, so legacy
edges are neutral — neither helping nor hurting retrieval.

### 3. `DERIVED_FROM` edge type

Re-add `DERIVED_FROM` to `EDGE_TYPES` and `EDGE_VALENCE`:

```python
EDGE_DERIVED_FROM: Final[str] = "DERIVED_FROM"

EDGE_VALENCE[EDGE_DERIVED_FROM] = 0.5   # mirrors CITES
```

The valence weight matches `CITES` because the relationships are
structurally similar: B `CITES` A and B `DERIVED_FROM` A both
indicate B's content depends on A's existence. Distinct types because
`DERIVED_FROM` carries a stronger contextual coupling (the sibling
becomes stale if A is superseded; a citer does not).

The producer is the v1.2.0 triple-extraction port. Triples of the
form `(B, derived-from, A)` map directly. Programmatic
`store.insert_edge` calls can also emit `DERIVED_FROM` when the
caller knows the relationship is derivational rather than citational.

### Interlock with `apply_feedback`

This spec does NOT add a new feedback path. The v1.0.1 hook→retrieval
wiring already commits to writing one `feedback_history` row per
hook-time retrieval; that fix is the natural producer of feedback
events on the read side. Once v1.0.1 is exercised in agent loops,
feedback density should climb to a value sufficient to drive
posterior-weighted ranking.

If a re-survey on representative production data shows density still
lagging, this spec needs a follow-up patch with explicit feedback-
emission rules (e.g., one feedback row per `aelf onboard` call; one
per `aelf remember`). For now, leave that to v1.0.1's loop and
re-measure.

### Backward compatibility

All three changes are forward-compatible:

- `ALTER TABLE` adds nullable columns; existing rows remain readable.
- `EDGE_TYPES` is a frozenset; adding to it does not break enumeration
  callers.
- The `Edge` dataclass field has a default of `None`; existing
  callers that construct without `anchor_text` keep working.
- The new `start_ingest_session` entry point is additive; existing
  ingest paths that don't use it produce beliefs with `session_id =
  NULL`, which is the same as today.

A v1.x store opened by a v1.0 reader sees the new columns ignored
(SQLite is permissive about unknown columns on SELECT *). A v1.0
store opened by a v1.x reader sees the new columns NULL. No
schema-version table needed for this round.

## Acceptance criteria

### Sessions

1. `start_ingest_session(model="x")` returns a fresh non-empty id and
   records it in the `sessions` table (or equivalent).
2. Ingest calls passing `session_id=X` write `X` to `beliefs.session_id`
   on every inserted belief.
3. `complete_ingest_session(X)` is idempotent and does not fail when
   the session is already completed.
4. Ingest calls without `session_id` produce beliefs with
   `session_id = NULL`. Existing v1.0 ingest tests continue to pass.

### Anchor text

5. `Edge` constructed without `anchor_text` has the field set to None.
6. `MemoryStore.insert_edge(e)` persists `e.anchor_text` exactly,
   including None.
7. The store's `Edge` round-trip preserves `anchor_text` through
   `insert_edge` → `get_edge`.

### `DERIVED_FROM`

8. `EDGE_TYPES` contains `DERIVED_FROM` and `EDGE_VALENCE` returns
   `0.5` for it.
9. `MemoryStore.insert_edge` accepts and persists `DERIVED_FROM`-typed
   edges.
10. `propagate_valence` walks `DERIVED_FROM` edges with the same
    semantics as `CITES` (positive propagation, attenuated by broker
    confidence).

### Migration

11. A v1.0 SQLite fixture file opens cleanly with the v1.x store and
    all CRUD round-trips succeed on existing rows.
12. The `ALTER TABLE` migrations are idempotent (running twice does
    not error).

## Test plan

- `tests/test_ingest_session.py` — sessions sub-feature (criteria 1–4).
- `tests/test_edge_anchor_text.py` — anchor sub-feature (criteria 5–7).
- `tests/test_edge_type_derived_from.py` — `DERIVED_FROM`
  sub-feature (criteria 8–10).
- `tests/test_v1_to_v1x_migration.py` — backward compat (criteria
  11–12). Requires a small fixture v1.0 SQLite file under
  `tests/fixtures/`.
- All deterministic, in-memory `:memory:` store, < 200 ms per test
  except the migration test (< 1 s).

## Out of scope

- Implementing the downstream techniques that consume these fields
  (augmented BM25F, posterior-weighted ranking, session-coherent
  supersession). Each lands in v1.3.0+ with its own spec.
- Backfilling `anchor_text` or `session_id` on existing beliefs.
  Legacy rows stay NULL; downstream techniques degrade gracefully on
  NULL inputs.
- LLM-generated anchor text. The spec is purely mechanical — anchor
  text comes from the ingest path's source prose, not from a model.
- Cross-session belief sharing. A belief belongs to exactly one
  session (or none). Multi-session membership is a future v2.x
  consideration if a use case emerges.
- Session expiration / garbage collection. Sessions live until
  explicitly completed; orphan-session cleanup is a follow-up.

## Open questions

- Should `start_ingest_session` write a `sessions` row immediately,
  or only on first belief insert? Immediate is simpler;
  lazy avoids empty-session rows. Recommendation: immediate
  (simpler; orphan cleanup handles the rare empty-session case).
- Should `anchor_text` have a max length cap? Real anchor text is
  short prose (~20–200 chars). A max of 1000 protects against
  pathological writes without constraining real use. Recommendation:
  cap at 1000 chars at the dataclass level; truncate with a warning
  rather than reject.
- Should `DERIVED_FROM` participate in the supersession cascade
  predicate? Yes — that's the entire point of re-adding it.
  Confirmed by the future v1.3+/v2.0 supersession-cascade work that
  reads it.

## Producers (the actual ingest paths that populate the new fields)

The schema this spec adds is necessary but not sufficient — fields
must be populated by some ingest path. The v1.2.0 milestone names
three producers, each documented separately:

- [`triple_extractor.md`](triple_extractor.md) — pure function
  reading prose and emitting `(subject, relation, object)` triples
  with `anchor_text`. Reusable by every ingest caller.
- [`commit_ingest_hook.md`](commit_ingest_hook.md) — Claude Code
  PostToolUse hook that fires on `git commit`, runs the triple
  extractor on the commit message, and inserts under a session
  derived from git context.
- [`transcript_ingest.md`](transcript_ingest.md) — per-turn
  conversation logger plus `PreCompact` rotation that ingests entire
  conversations under a per-session id.

Together these three producers populate `session_id`, `anchor_text`,
and `DERIVED_FROM` densely on real corpora during normal Claude Code
sessions.
