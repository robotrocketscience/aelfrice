# Content-hash deduplication contract (#219)

## Problem

`belief_id` is `sha256(source + NUL + sentence)[:16]`. The same sentence
arriving from two different sources (e.g. a git commit message and a
filesystem scan of the same file) produces **different** `belief_id` values
but **identical** `content_hash` values. Without a UNIQUE constraint on
`content_hash`, repeated ingest inflates the table — a 5.3x blow-up was
observed on real stores.

## Fix (v1.x, ships in the release that closes #219)

Two layers of protection, both in `MemoryStore`:

### 1. `insert_or_corroborate()` — live dedup at ingest time

```python
def insert_or_corroborate(
    self, b: Belief, *, source_type: str,
    session_id: str | None = None,
    source_path_hash: str | None = None,
) -> tuple[str, bool]:
```

All ingest call sites use this instead of `insert_belief()` directly.

- If a belief with the same `content_hash` already exists: records a
  corroboration row on the existing belief and returns `(existing_id, False)`.
- Otherwise: inserts the new belief and returns `(b.id, True)`.

The returned `bool` tells the caller whether a new row was created so it can
maintain accurate `beliefs_inserted` counts.

### 2. One-shot migrations on store open

Both migrations are idempotent via `schema_meta` markers. They run on every
store open until their respective markers are set, then become a no-op
single-row read.

#### `_maybe_consolidate_content_hash_duplicates()`

Marker: `SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE = "content_hash_dedup_complete"`

Runs **before** the UNIQUE migration. For each duplicate group (beliefs
sharing a `content_hash`) it:

- Picks the **canonical** row: `ORDER BY created_at ASC, id ASC` (oldest first).
- Sums `alpha` and `beta` across the group (each row carries independent
  Bayesian evidence accumulated from a distinct source).
- Propagates `lock_level = 'user'` if any member holds it.
- Propagates the highest-precedence `origin` (`user_stated` > `user_corrected`
  > `user_validated` > `agent_remembered` > `agent_inferred` > `document_recent`
  > `unknown`).
- Takes `MAX(last_retrieved_at)` across the group.
- Rewrites FK references: `feedback_history`, `belief_corroborations`, `edges`
  (both `src` and `dst`).
- Drops `belief_entities` and `belief_versions` rows for duplicates (same
  content → same entities; the canonical row already has them).
- Inserts one `belief_corroborations` row per duplicate consumed with
  `source_type = 'consolidation_migration'` to preserve the count signal.
- Deletes all duplicate rows from `beliefs` and `beliefs_fts` in a single
  bulk `DELETE ... WHERE id IN (...)`.

All writes are inside one transaction. On a 20 K-belief store with ~2 K
duplicate groups the pass completes in under 2 seconds.

#### `_maybe_apply_content_hash_unique()`

Marker: `SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED = "content_hash_unique_applied"`

Adds `UNIQUE(content_hash)` to the `beliefs` table via a SQLite table-swap
(SQLite does not support `ALTER TABLE ADD CONSTRAINT`):

1. Read column definitions via `PRAGMA table_info(beliefs)` — preserves any
   columns added by prior `ALTER TABLE` migrations (e.g. `hibernation_score`,
   `activation_condition`).
2. `DROP TABLE IF EXISTS beliefs_new` — clears any partial state from a prior
   failed attempt.
3. `CREATE TABLE beliefs_new` with `UNIQUE` added to `content_hash`.
4. `INSERT INTO beliefs_new SELECT ... FROM beliefs`.
5. `DROP TABLE beliefs`.
6. `ALTER TABLE beliefs_new RENAME TO beliefs`.
7. Recreate `idx_beliefs_session` and `idx_beliefs_origin`.

Fresh stores (created with the new `_SCHEMA`) already have the constraint in
DDL; they skip the swap and only stamp the marker.

## Corroboration source types added

| Constant | Value | Used by |
|---|---|---|
| `CORROBORATION_SOURCE_FILESYSTEM_INGEST` | `"filesystem_ingest"` | `scanner.py`, `classification.py` |
| `CORROBORATION_SOURCE_CLI_REMEMBER` | `"cli_remember"` | `cli.py` |
| `CORROBORATION_SOURCE_CONSOLIDATION_MIGRATION` | `"consolidation_migration"` | migration pass |

## Ingest call sites migrated

| Module | Old call | New call |
|---|---|---|
| `ingest.py` | `store.insert_belief(out.belief)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST)` |
| `scanner.py` (LLM + regex paths) | `store.insert_belief(b)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_FILESYSTEM_INGEST)` |
| `classification.py` | `store.insert_belief(b)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_FILESYSTEM_INGEST)` |
| `cli.py` (`_cmd_lock`) | `store.insert_belief(b)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_CLI_REMEMBER)` |
| `triple_extractor.py` | `store.insert_belief(b)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_COMMIT_INGEST)` |
| `mcp_server.py` (`tool_lock`) | `store.insert_belief(b)` | `store.insert_or_corroborate(..., source_type=CORROBORATION_SOURCE_MCP_REMEMBER)` |

## Invariants

- `beliefs.content_hash` is UNIQUE. Any attempt to INSERT a duplicate via raw
  SQL raises `sqlite3.IntegrityError`.
- `insert_or_corroborate()` is the only sanctioned way to insert a belief from
  ingest paths. `insert_belief()` remains for test fixtures and the migration
  pass itself.
- The consolidation migration always runs before the UNIQUE migration on the
  same store open, guaranteeing no UNIQUE violation can occur during the swap.
