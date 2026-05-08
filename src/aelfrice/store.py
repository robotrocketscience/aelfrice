"""SQLite-backed store for Beliefs and Edges, with FTS5 full-text search.

Stdlib-only (sqlite3). WAL journal mode for concurrent reads. FTS5 virtual
table mirrors `beliefs.content` for keyword retrieval.

`propagate_valence` lives here so v0.1.0 stays a small module set.
Broker-confidence attenuation: each hop through an intermediate belief is
dampened by that belief's alpha/(alpha+beta), so low-confidence brokers
absorb propagation rather than amplify it.

`demotion_pressure` is both written and read end-to-end here; the test
suite locks that behaviour in from day one
(see tests/test_demotion_pressure.py).
"""
from __future__ import annotations

import inspect
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Final, Iterable, Iterator

from aelfrice.models import (
    CORROBORATION_SOURCE_CONSOLIDATION_MIGRATION,
    CORROBORATION_SOURCE_TYPES,
    EDGE_VALENCE,
    INGEST_SOURCE_KINDS,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    ONBOARD_STATE_COMPLETED,
    ONBOARD_STATE_PENDING,
    RETENTION_CLASSES,
    RETENTION_UNKNOWN,
    Belief,
    Edge,
    FeedbackEvent,
    OnboardSession,
    Session,
)
from aelfrice.ulid import ulid

# --- v2.x view-flip gate --------------------------------------------------
#
# When AELFRICE_WRITE_LOG_AUTHORITATIVE is on, `beliefs` becomes a
# materialized view of `ingest_log` and direct `insert_belief()` calls
# from outside the allowlist raise. The flag itself was wired in
# derivation_worker.is_write_log_authoritative (#265 PR-A); store reads
# os.environ inline to avoid a circular import (worker depends on store).
#
# Allowlist matches the ratification on PR #478:
#   - aelfrice.derivation_worker  — the single canonical writer
#   - aelfrice.wonder.simulator   — synthetic corpus seeder (test fixture)
#   - aelfrice.benchmark          — benchmarking fixtures
#   - aelfrice.migrate            — v1→v2 store migration; legacy_unknown
#                                    rows are emitted by the migration tool
#                                    itself, not through ingest_log
#
# Anything outside the allowlist must go through
# `record_ingest` + `run_worker` so the write trail starts on the log.

_ENV_WRITE_LOG_AUTHORITATIVE: Final[str] = "AELFRICE_WRITE_LOG_AUTHORITATIVE"
_WLAUTH_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

INSERT_BELIEF_ALLOWLIST: Final[frozenset[str]] = frozenset({
    "aelfrice.derivation_worker",
    "aelfrice.wonder.simulator",
    "aelfrice.benchmark",
    "aelfrice.migrate",
})


class WriteLogAuthorityViolation(RuntimeError):
    """Raised when `MemoryStore.insert_belief` is called from outside
    `INSERT_BELIEF_ALLOWLIST` while `AELFRICE_WRITE_LOG_AUTHORITATIVE`
    is on. Indicates a code path that would write to `beliefs` without
    a corresponding `ingest_log` row — the write trail invariant the
    v2.x view-flip is designed to enforce.

    Resolution: route the caller through `record_ingest` + `run_worker`
    instead of calling `insert_belief` directly. Or, if the caller is a
    legitimate fixture/migration site, add it to `INSERT_BELIEF_ALLOWLIST`
    after design review.
    """


def _is_write_log_authoritative_inline() -> bool:
    """Inline reader to avoid the store→derivation_worker→store cycle."""
    raw = os.environ.get(_ENV_WRITE_LOG_AUTHORITATIVE)
    if raw is None:
        return False
    return raw.strip().lower() in _WLAUTH_TRUTHY


def _check_insert_belief_authority() -> None:
    """Stack-walk gate for `insert_belief`.

    No-op when the flag is off (production default at this commit).
    When on, walks frames until it finds one outside `aelfrice.store`,
    then checks that frame's module against `INSERT_BELIEF_ALLOWLIST`.
    Raises `WriteLogAuthorityViolation` on a non-allowlisted caller.

    `insert_or_corroborate` calls `insert_belief` internally; the walk
    skips over `aelfrice.store` frames so the *true* caller (the worker
    or another module) is what gets checked.
    """
    if not _is_write_log_authoritative_inline():
        return
    frame = inspect.currentframe()
    if frame is None:  # platforms without frame inspection
        return
    # Skip our own frame plus any aelfrice.store frames (insert_belief,
    # insert_or_corroborate, …). The first non-store frame is the
    # true caller.
    caller = frame.f_back
    while caller is not None:
        mod = caller.f_globals.get("__name__", "")
        if not mod.startswith("aelfrice.store"):
            if mod in INSERT_BELIEF_ALLOWLIST:
                return
            raise WriteLogAuthorityViolation(
                f"insert_belief() called from {mod!r}; not in "
                f"INSERT_BELIEF_ALLOWLIST {sorted(INSERT_BELIEF_ALLOWLIST)}. "
                f"With AELFRICE_WRITE_LOG_AUTHORITATIVE on, only the "
                f"derivation worker and three allowlisted fixture/"
                f"migration modules may write to `beliefs` directly; "
                f"all other callers must go through "
                f"`record_ingest` + `run_worker`."
            )
        caller = caller.f_back
    # No non-store frame found (shouldn't happen in practice). Fail
    # closed — the caller is opaque, treat it as a violation.
    raise WriteLogAuthorityViolation(
        "insert_belief() called from an opaque caller; could not "
        "identify the calling module while the write-log gate is on."
    )


# --- Schema ---------------------------------------------------------------

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS beliefs (
        id                  TEXT PRIMARY KEY,
        content             TEXT NOT NULL,
        content_hash        TEXT NOT NULL UNIQUE,
        alpha               REAL NOT NULL,
        beta                REAL NOT NULL,
        type                TEXT NOT NULL,
        lock_level          TEXT NOT NULL,
        locked_at           TEXT,
        demotion_pressure   INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL,
        last_retrieved_at   TEXT,
        session_id          TEXT,
        origin              TEXT NOT NULL DEFAULT 'unknown',
        hibernation_score   REAL,
        activation_condition TEXT,
        -- #290: orthogonal-to-type retention axis. CHECK constraint
        -- enforces the four-value enum on fresh stores; migrated
        -- stores rely on the python-side RETENTION_CLASSES frozenset
        -- for validation (ALTER TABLE ADD COLUMN can't carry the
        -- CHECK across SQLite versions reliably). 'unknown' is the
        -- migration default; new writes pick one of the three live
        -- values per docs/belief_retention_class.md § 2 defaults.
        retention_class     TEXT NOT NULL DEFAULT 'unknown'
            CHECK (retention_class IN ('fact', 'snapshot', 'transient', 'unknown'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src         TEXT NOT NULL,
        dst         TEXT NOT NULL,
        type        TEXT NOT NULL,
        weight      REAL NOT NULL,
        anchor_text TEXT,
        PRIMARY KEY (src, dst, type)
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts
    USING fts5(id UNINDEXED, content, tokenize='porter unicode61')
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        belief_id   TEXT NOT NULL,
        valence     REAL NOT NULL,
        source      TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS onboard_sessions (
        session_id      TEXT PRIMARY KEY,
        repo_path       TEXT NOT NULL,
        state           TEXT NOT NULL,
        candidates_json TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        completed_at    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id              TEXT PRIMARY KEY,
        started_at      TEXT NOT NULL,
        completed_at    TEXT,
        model           TEXT,
        project_context TEXT
    )
    """,
    # v1.3.0 entity-index (L2.5 retrieval). One row per (belief, entity
    # span). Composite PK permits the same entity_lower to appear in
    # one belief at distinct span_starts (e.g. a file path mentioned
    # twice). Indexes cover the three lookup patterns: entity → beliefs
    # (hot path), belief → entities (refresh / debug), and kind filters
    # (future kind-weighted ranker). Additive: forward-compatible with
    # v1.0/v1.1/v1.2 stores per docs/entity_index.md § Migration story.
    """
    CREATE TABLE IF NOT EXISTS belief_entities (
        belief_id    TEXT NOT NULL,
        entity_lower TEXT NOT NULL,
        entity_raw   TEXT NOT NULL,
        kind         TEXT NOT NULL,
        span_start   INTEGER NOT NULL,
        span_end     INTEGER NOT NULL,
        PRIMARY KEY (belief_id, entity_lower, span_start)
    )
    """,
    # v1.3.0 schema_meta. Single-row registry of one-shot migrations
    # that need to be tracked across opens (currently: the entity-index
    # backfill flag). Key/value to keep the schema generic. Values are
    # ISO timestamps (or empty strings) — no booleans across SQLite.
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # v1.5.0 belief_corroborations (#190). Records each re-ingest of
    # content whose content_hash already exists in `beliefs`. The
    # canonical belief row is unchanged; this table makes re-assertions
    # observable as a first-class signal without disturbing the
    # existing dedup contract. ON DELETE CASCADE removes corroboration
    # rows when a belief is deleted (rare; covered for hygiene).
    # NOTE: belief_id is TEXT NOT NULL (not INTEGER) because beliefs.id
    # is TEXT PRIMARY KEY in this codebase; the spec sketch said INTEGER
    # but that was adapted to match the actual schema.
    """
    CREATE TABLE IF NOT EXISTS belief_corroborations (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        belief_id         TEXT    NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
        ingested_at       TEXT    NOT NULL,
        source_type       TEXT    NOT NULL,
        session_id        TEXT,
        source_path_hash  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_belief_corroborations_belief_id "
    "ON belief_corroborations(belief_id)",
    # v1.6.0 deferred_feedback_queue (#191). One row per surfaced
    # belief from `retrieve()`; processed by the `aelf sweep-feedback`
    # CLI subcommand after T_grace elapses. status transitions:
    # 'enqueued' -> 'applied' (no contradiction in grace window) or
    # 'cancelled' (explicit feedback / contradiction within grace).
    # event_type is open-ended so future signals (search exposure,
    # MCP tool reads) can ride the same sweeper. ON DELETE CASCADE
    # keeps the queue consistent if a belief is deleted.
    """
    CREATE TABLE IF NOT EXISTS deferred_feedback_queue (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        belief_id    TEXT    NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
        enqueued_at  TEXT    NOT NULL,
        event_type   TEXT    NOT NULL,
        applied_at   TEXT,
        status       TEXT    NOT NULL DEFAULT 'enqueued'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dfq_status_enq "
    "ON deferred_feedback_queue(status, enqueued_at)",
    "CREATE INDEX IF NOT EXISTS idx_dfq_belief "
    "ON deferred_feedback_queue(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_belief ON feedback_history(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_onboard_state ON onboard_sessions(state)",
    "CREATE INDEX IF NOT EXISTS idx_belief_entities_lower "
    "ON belief_entities(entity_lower)",
    "CREATE INDEX IF NOT EXISTS idx_belief_entities_belief "
    "ON belief_entities(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_belief_entities_kind "
    "ON belief_entities(kind)",
    # v1.5.0 #204 forward-compat for v3 federation. One row per
    # (belief, scope) so each belief carries a version vector
    # `{scope_id: counter}`. SQLite-idiomatic sidecar table beats
    # the JSON-column alternative because we never need to filter
    # in WHERE on individual scope counters; the rows are
    # accumulated locally and consumed by federation reconcile
    # (which lands at v3, not now).
    """
    CREATE TABLE IF NOT EXISTS belief_versions (
        belief_id TEXT NOT NULL,
        scope_id  TEXT NOT NULL,
        counter   INTEGER NOT NULL,
        PRIMARY KEY (belief_id, scope_id)
    )
    """,
    # Edge version vectors. Edges have a composite key
    # `(src, dst, type)` per the v1.0 schema; the version sidecar
    # mirrors that.
    """
    CREATE TABLE IF NOT EXISTS edge_versions (
        src       TEXT NOT NULL,
        dst       TEXT NOT NULL,
        type      TEXT NOT NULL,
        scope_id  TEXT NOT NULL,
        counter   INTEGER NOT NULL,
        PRIMARY KEY (src, dst, type, scope_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_belief_versions_belief "
    "ON belief_versions(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_edge_versions_edge "
    "ON edge_versions(src, dst, type)",
    # v2.0 #205 ingest_log. Append-only record of every raw input
    # that produced a belief or edge. Parallel-write for v2.0 first
    # slice; not yet authoritative. Spec: docs/design/write-log-as-truth.md.
    # `id` is a Crockford base32 ULID (26 chars) — lexicographic sort
    # equals time sort. `raw_meta`, `derived_belief_ids`, and
    # `derived_edge_ids` are JSON-encoded strings (TEXT) because
    # SQLite has no native JSON type and we never filter in WHERE
    # on their interior; access is read-then-deserialize. The
    # `(source_kind, source_path)` index covers the spec's required
    # O(log n) (source_path, raw_text) lookup.
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        id                 TEXT PRIMARY KEY,
        ts                 TEXT NOT NULL,
        source_kind        TEXT NOT NULL,
        source_path        TEXT,
        raw_text           TEXT NOT NULL,
        raw_meta           TEXT,
        derived_belief_ids TEXT,
        derived_edge_ids   TEXT,
        classifier_version TEXT,
        rule_set_hash      TEXT,
        session_id         TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingest_log_source "
    "ON ingest_log(source_kind, source_path)",
    "CREATE INDEX IF NOT EXISTS idx_ingest_log_session "
    "ON ingest_log(session_id)",
    # v2.0 #205 ingest_log version vectors. Mirrors the #204 pattern
    # so federation reconcile (v3) treats log rows as first-class
    # replication units. Local-write rule applies: vv[local_scope] +=
    # 1 on every record_ingest. Backfill stamps {local_scope: 1} on
    # every pre-existing log row at first v2.0 open.
    """
    CREATE TABLE IF NOT EXISTS log_versions (
        log_id   TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        counter  INTEGER NOT NULL,
        PRIMARY KEY (log_id, scope_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_log_versions_log "
    "ON log_versions(log_id)",
    # v2.0 #435 doc linker. One row per (belief, doc_uri). PK gives
    # idempotency on re-ingest of the same belief from the same source.
    # ON DELETE CASCADE removes rows when a belief is hard-deleted (#440)
    # — anchors are a derived projection of belief origin, not an audit
    # trail (belief_corroborations #190 is the audit-trail sibling).
    # `created_at` is REAL (unix seconds) — the linker is hot enough that
    # we want numeric ordering rather than ISO-string comparison.
    """
    CREATE TABLE IF NOT EXISTS belief_documents (
        belief_id     TEXT NOT NULL,
        doc_uri       TEXT NOT NULL,
        anchor_type   TEXT NOT NULL CHECK (anchor_type IN ('ingest', 'manual', 'derived')),
        position_hint TEXT,
        created_at    REAL NOT NULL,
        PRIMARY KEY (belief_id, doc_uri),
        FOREIGN KEY (belief_id) REFERENCES beliefs(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_belief_documents_belief_id "
    "ON belief_documents(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_belief_documents_doc_uri "
    "ON belief_documents(doc_uri)",
)

# Marker key for the entity-index one-shot backfill. Empty value =
# not yet run. ISO timestamp = completed at that time. Tracked in
# `schema_meta` so a v1.3+ binary opening a pre-v1.3 store knows to
# re-extract entities for every existing belief on first open.
SCHEMA_META_ENTITY_BACKFILL: Final[str] = "entity_backfill_complete"

# v1.5.0 #204 federation forward-compat. Stable per-DB scope id,
# generated on first v1.5+ open and persisted in `schema_meta`.
# Today aelfrice is single-scope per DB; the local scope id is
# just a UUID. When federation ships at v3, peer scopes appear
# alongside this one in the version-vector dicts.
SCHEMA_META_LOCAL_SCOPE_ID: Final[str] = "local_scope_id"
# Marker for the v1.5.0 backfill that stamps `{local_scope: 1}`
# on every pre-#204 belief and edge row. ISO timestamp on
# completion; absence triggers the backfill on next open.
SCHEMA_META_VERSION_VECTOR_BACKFILL: Final[str] = (
    "version_vector_backfill_complete"
)
# v2.0 #205. Marker for the one-shot backfill that stamps
# `{local_scope: 1}` on every pre-existing ingest_log row when a v2.0
# binary first opens a store with the new ingest_log table populated.
SCHEMA_META_LOG_VERSION_VECTOR_BACKFILL: Final[str] = (
    "log_version_vector_backfill_complete"
)
# v2.x #263. Marker for the one-shot migration that synthesizes a
# `source_kind=legacy_unknown` ingest_log row for every pre-v2.0 belief
# (i.e. beliefs with no log row pointing at them when the migration
# runs). ISO timestamp on completion; absence triggers the migration on
# next open.
SCHEMA_META_LEGACY_LOG_SYNTH: Final[str] = "legacy_log_synth_complete"
# #219 content_hash dedup. One-shot pass that collapses duplicate
# content_hash rows onto the canonical (oldest created_at, then id ASC)
# belief before the UNIQUE constraint is applied. ISO timestamp on
# completion.
SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE: Final[str] = "content_hash_dedup_complete"
# #219 content_hash UNIQUE. Set after the table-swap that adds
# UNIQUE(content_hash) to the beliefs table. Fresh stores skip the
# swap because their _SCHEMA already includes the constraint.
SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED: Final[str] = "content_hash_unique_applied"

# v1.0 -> v1.2 column additions. Each ALTER runs after _SCHEMA. ALTERs
# are idempotent: a duplicate-column OperationalError on a v1.2-fresh
# DB is caught and ignored. The CREATE INDEX after the ALTERs
# references a column that exists only post-migration; placing it here
# (rather than in _SCHEMA) is what lets a v1.0 store open at all.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
    "ALTER TABLE edges ADD COLUMN anchor_text TEXT",
    "ALTER TABLE beliefs ADD COLUMN origin TEXT NOT NULL DEFAULT 'unknown'",
    # v2.0 #196 hibernation lifecycle columns. Both nullable; behavior
    # is deferred to a follow-up issue. Storage round-trip only at this
    # commit. activation_condition is JSON-encoded TEXT (predicate
    # language ratified at substrate_decision.md § Decision asks #4).
    "ALTER TABLE beliefs ADD COLUMN hibernation_score REAL",
    "ALTER TABLE beliefs ADD COLUMN activation_condition TEXT",
    # #290 retention class. CHECK constraint omitted — ALTER TABLE
    # ADD COLUMN with a CHECK is brittle across SQLite versions.
    # Python-side RETENTION_CLASSES validates inserts.
    "ALTER TABLE beliefs ADD COLUMN retention_class TEXT NOT NULL DEFAULT 'unknown'",
)

# Indexes that depend on migrated columns. Run after _MIGRATIONS so
# they see the post-ALTER schema.
_POST_MIGRATION_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_beliefs_session ON beliefs(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_beliefs_origin ON beliefs(origin)",
)

# One-shot backfill for v1.0/v1.1 stores opening on v1.2+. Each row
# only flips if it is still at the default 'unknown'. Idempotent —
# reruns find no candidates after the first pass. See promotion_path.md
# § 1: locked beliefs become user_stated; correction beliefs become
# user_corrected; everything else stays unknown to avoid retroactively
# claiming agent_inferred for content the v1.0 scanner didn't commit
# to that label.
_BACKFILL_STATEMENTS: tuple[str, ...] = (
    "UPDATE beliefs SET origin = 'user_stated' "
    "WHERE origin = 'unknown' AND lock_level = 'user'",
    "UPDATE beliefs SET origin = 'user_corrected' "
    "WHERE origin = 'unknown' AND type = 'correction'",
)


def _escape_fts5_query(query: str) -> str:
    """Convert free-form user input into a safe FTS5 MATCH expression.

    Strategy: tokenize on whitespace, drop tokens that are empty after
    stripping, double-quote-wrap each remaining token (with embedded
    `"` doubled per FTS5 syntax), join with spaces. The result is a
    valid FTS5 query that ANDs the tokens implicitly.

    Empty / whitespace-only input returns the empty string; callers
    treat that as "no match" without hitting the FTS5 engine.
    """
    tokens = query.split()
    if not tokens:
        return ""
    escaped: list[str] = []
    for t in tokens:
        if not t:
            continue
        inner = t.replace('"', '""')
        escaped.append(f'"{inner}"')
    return " ".join(escaped)


def _row_to_belief(row: sqlite3.Row) -> Belief:
    keys = row.keys()
    corroboration_count = int(row["corroboration_count"]) if "corroboration_count" in keys else 0
    # retention_class column added in v1.6 (#290). Pre-migration rows
    # don't have it — fall back to RETENTION_UNKNOWN. The store __init__
    # runs the ALTER on open, so this branch is mainly for queries that
    # project a custom column list excluding the new column.
    retention_class = (
        row["retention_class"] if "retention_class" in keys
        else RETENTION_UNKNOWN
    )
    return Belief(
        id=row["id"],
        content=row["content"],
        content_hash=row["content_hash"],
        alpha=row["alpha"],
        beta=row["beta"],
        type=row["type"],
        lock_level=row["lock_level"],
        locked_at=row["locked_at"],
        demotion_pressure=row["demotion_pressure"],
        created_at=row["created_at"],
        last_retrieved_at=row["last_retrieved_at"],
        session_id=row["session_id"],
        origin=row["origin"],
        corroboration_count=corroboration_count,
        hibernation_score=row["hibernation_score"],
        activation_condition=row["activation_condition"],
        retention_class=retention_class,
    )


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        model=row["model"],
        project_context=row["project_context"],
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    # row["anchor_text"] safe: v1.2 migration guarantees the column on
    # any store this code instantiates. Legacy rows return None.
    return Edge(
        src=row["src"],
        dst=row["dst"],
        type=row["type"],
        weight=row["weight"],
        anchor_text=row["anchor_text"],
    )


def _row_to_feedback(row: sqlite3.Row) -> FeedbackEvent:
    return FeedbackEvent(
        id=row["id"],
        belief_id=row["belief_id"],
        valence=row["valence"],
        source=row["source"],
        created_at=row["created_at"],
    )


def _drop_stale_ingest_log(conn: sqlite3.Connection) -> None:
    """Drop a pre-#205 experimental `ingest_log` table if present.

    Some local stores carry a stale `ingest_log` from prior off-branch
    experimentation (id INTEGER PK, `raw_meta_json` column, no
    `session_id`). The schema never landed on main, never persisted
    data, and is incompatible with the v2.0 #205 contract. We drop it
    on open if (a) it exists, AND (b) its column set differs from the
    canonical v2.0 schema, AND (c) it holds zero rows.

    Idempotent: a table that already matches the canonical schema is
    left alone. A non-empty stale table is left alone (operator must
    intervene; we will not silently destroy data).
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_log'"
    )
    if cur.fetchone() is None:
        return
    cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(ingest_log)").fetchall()
    }
    canonical = {
        "id", "ts", "source_kind", "source_path", "raw_text", "raw_meta",
        "derived_belief_ids", "derived_edge_ids", "classifier_version",
        "rule_set_hash", "session_id",
    }
    if cols == canonical:
        return
    n = conn.execute("SELECT COUNT(*) AS n FROM ingest_log").fetchone()["n"]
    if n != 0:
        # Operator must inspect: leaving the table in place will surface
        # as a CREATE INDEX failure later, which is the right signal.
        return
    conn.execute("DROP TABLE ingest_log")


def _ingest_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    """Decode an `ingest_log` sqlite row into a Python dict.

    JSON-encoded TEXT columns (raw_meta, derived_belief_ids,
    derived_edge_ids) are deserialized; absent values become None.
    Used by `MemoryStore.get_ingest_log_entry` and the validation
    harness.
    """
    def _maybe_json(v: object) -> object:
        if v is None:
            return None
        return json.loads(str(v))

    return {
        "id": str(row["id"]),
        "ts": str(row["ts"]),
        "source_kind": str(row["source_kind"]),
        "source_path": row["source_path"],
        "raw_text": str(row["raw_text"]),
        "raw_meta": _maybe_json(row["raw_meta"]),
        "derived_belief_ids": _maybe_json(row["derived_belief_ids"]),
        "derived_edge_ids": _maybe_json(row["derived_edge_ids"]),
        "classifier_version": row["classifier_version"],
        "rule_set_hash": row["rule_set_hash"],
        "session_id": row["session_id"],
    }


def _row_to_onboard_session(row: sqlite3.Row) -> OnboardSession:
    return OnboardSession(
        session_id=row["session_id"],
        repo_path=row["repo_path"],
        state=row["state"],
        candidates_json=row["candidates_json"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


class MemoryStore:
    """SQLite store. Pass `:memory:` for tests, a path otherwise."""

    def __init__(self, path: str) -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        # WAL only meaningful on-disk; harmless on :memory:.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        # Block up to 5s waiting for a write lock instead of failing
        # with `database is locked` immediately. Required for safe
        # multi-worktree access where two processes share one .git/
        # aelfrice/memory.db. Per the v1.1.0 #89 concurrency tests.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        _drop_stale_ingest_log(self._conn)
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        for stmt in _MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError as e:
                # "duplicate column name: X" — column already present
                # (either from CREATE TABLE on a fresh v1.2 DB or from
                # a previous migration pass). Anything else re-raises.
                if "duplicate column name" not in str(e):
                    raise
        for stmt in _POST_MIGRATION_INDEXES:
            self._conn.execute(stmt)
        for stmt in _BACKFILL_STATEMENTS:
            self._conn.execute(stmt)
        self._conn.commit()
        self._invalidation_callbacks: list[Callable[[], None]] = []
        # v1.5.0 #204 federation forward-compat. Resolve (or
        # generate) the local scope id BEFORE any belief/edge
        # write path runs — write hooks consume `_local_scope_id`
        # to bump the version-vector counter.
        self._local_scope_id: str = self._resolve_local_scope_id()
        # v1.3 entity-index one-shot backfill. Skipped on a v1.3-fresh
        # store (no rows in beliefs yet) — the marker just stamps to
        # the current time so the next open is a no-op. Idempotent.
        self._maybe_backfill_entity_index()
        # v1.5.0 #204 version-vector backfill. Stamps
        # `{local_scope: 1}` on every pre-existing belief and edge
        # the first time a v1.5+ binary opens this DB. Idempotent.
        self._maybe_backfill_version_vectors()
        # v2.x #263 legacy log synthesis. Must run BEFORE the log
        # version-vector backfill so synthesized rows are included in
        # the backfill's INSERT OR IGNORE pass on the same open.
        self._maybe_synthesize_legacy_log_rows()
        # v2.0 #205 ingest_log version-vector backfill. Same shape
        # as #204 but for the parallel-write log table. Idempotent.
        self._maybe_backfill_log_version_vectors()
        # #219 content_hash dedup. Must run BEFORE the table-swap that
        # adds UNIQUE(content_hash) so we eliminate all duplicates first.
        self._maybe_consolidate_content_hash_duplicates()
        # #219 UNIQUE(content_hash). Table-swap migration; runs once
        # after the dedup pass guarantees no duplicates remain.
        self._maybe_apply_content_hash_unique()

    def close(self) -> None:
        self._conn.close()

    # --- Invalidation callbacks ------------------------------------------
    #
    # External components (e.g. RetrievalCache) register a zero-arg callback
    # that is fired on every belief/edge mutation. Used to wipe derived
    # state — query result caches, materialized views — that depends on the
    # store's contents. Callback order is registration order; exceptions
    # raised by a callback propagate (better to fail loudly than silently
    # serve stale data).

    def add_invalidation_callback(self, fn: Callable[[], None]) -> None:
        """Register a zero-arg callback fired after every store mutation."""
        self._invalidation_callbacks.append(fn)

    def _fire_invalidation(self) -> None:
        for fn in self._invalidation_callbacks:
            fn()

    # --- Schema meta (v1.3+ migration tracker) ---------------------------

    def get_schema_meta(self, key: str) -> str | None:
        """Return the value stored under `key` in schema_meta, or None.

        Used by v1.3+ migrations (entity-index backfill) to decide
        whether a one-shot pass has already run on this DB.
        """
        cur = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else None

    def set_schema_meta(self, key: str, value: str) -> None:
        """Insert or replace one schema_meta row. Idempotent."""
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # --- v1.5.0 #204 version-vector helpers ------------------------------

    @property
    def local_scope_id(self) -> str:
        """The stable per-DB scope id used for federation forward-compat.

        Generated on first open of a v1.5+ binary against this DB and
        persisted in `schema_meta`. Stays stable for the life of the
        DB; never rotated.
        """
        return self._local_scope_id

    def _resolve_local_scope_id(self) -> str:
        """Read the persisted scope id, generating one on first open."""
        existing = self.get_schema_meta(SCHEMA_META_LOCAL_SCOPE_ID)
        if existing:
            return existing
        new_scope = secrets.token_hex(16)
        self.set_schema_meta(SCHEMA_META_LOCAL_SCOPE_ID, new_scope)
        return new_scope

    def _bump_belief_version(self, belief_id: str) -> None:
        """Increment `belief_versions[belief_id, local_scope_id]` by 1.

        Called from every belief insert / update path. Local-write
        rule per the #204 spec: `vv[local_scope] += 1` on every
        write. Idempotent INSERT OR REPLACE on the composite PK keeps
        the row count bounded by `n_beliefs * n_scopes`.
        """
        self._conn.execute(
            "INSERT INTO belief_versions (belief_id, scope_id, counter) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(belief_id, scope_id) "
            "DO UPDATE SET counter = counter + 1",
            (belief_id, self._local_scope_id),
        )

    def _bump_log_version(self, log_id: str) -> None:
        """Increment `log_versions[log_id, local_scope_id]` by 1.

        v2.0 #205. Mirrors `_bump_belief_version` so ingest_log rows
        carry the same federation-replication primitive as beliefs and
        edges.
        """
        self._conn.execute(
            "INSERT INTO log_versions (log_id, scope_id, counter) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(log_id, scope_id) "
            "DO UPDATE SET counter = counter + 1",
            (log_id, self._local_scope_id),
        )

    def get_log_version_vector(self, log_id: str) -> dict[str, int]:
        """Return `{scope_id: counter}` for one ingest_log row.

        Empty dict for rows that pre-date the v2.0 backfill (until the
        next open triggers it). v2.0 #205.
        """
        cur = self._conn.execute(
            "SELECT scope_id, counter FROM log_versions WHERE log_id = ?",
            (log_id,),
        )
        return {str(r["scope_id"]): int(r["counter"]) for r in cur.fetchall()}

    def _bump_edge_version(self, src: str, dst: str, type_: str) -> None:
        """Increment `edge_versions[(src, dst, type), local_scope_id]`."""
        self._conn.execute(
            "INSERT INTO edge_versions (src, dst, type, scope_id, counter) "
            "VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT(src, dst, type, scope_id) "
            "DO UPDATE SET counter = counter + 1",
            (src, dst, type_, self._local_scope_id),
        )

    def get_belief_version_vector(self, belief_id: str) -> dict[str, int]:
        """Return `{scope_id: counter}` for `belief_id`.

        Empty dict on a belief that has no version rows yet (e.g., a
        store row written by a pre-v1.5 binary that has not yet been
        backfilled). Federation reconcile (v3) consumes this map.
        """
        cur = self._conn.execute(
            "SELECT scope_id, counter FROM belief_versions "
            "WHERE belief_id = ?",
            (belief_id,),
        )
        return {str(r["scope_id"]): int(r["counter"]) for r in cur.fetchall()}

    def get_edge_version_vector(
        self, src: str, dst: str, type_: str,
    ) -> dict[str, int]:
        """Return `{scope_id: counter}` for the edge `(src, dst, type)`."""
        cur = self._conn.execute(
            "SELECT scope_id, counter FROM edge_versions "
            "WHERE src = ? AND dst = ? AND type = ?",
            (src, dst, type_),
        )
        return {str(r["scope_id"]): int(r["counter"]) for r in cur.fetchall()}

    def _maybe_backfill_version_vectors(self) -> int:
        """One-shot backfill stamping `{local_scope: 1}` on every
        pre-existing belief and edge that lacks a version row.

        Idempotent: stamped marker short-circuits subsequent opens.
        Re-running by dropping the marker also no-ops because of
        `INSERT OR IGNORE` against the composite PK.

        Returns the count of rows inserted across both tables.
        """
        if self.get_schema_meta(SCHEMA_META_VERSION_VECTOR_BACKFILL):
            return 0
        scope = self._local_scope_id
        belief_cur = self._conn.execute(
            "INSERT OR IGNORE INTO belief_versions "
            "(belief_id, scope_id, counter) "
            "SELECT id, ?, 1 FROM beliefs",
            (scope,),
        )
        belief_inserted = belief_cur.rowcount or 0
        edge_cur = self._conn.execute(
            "INSERT OR IGNORE INTO edge_versions "
            "(src, dst, type, scope_id, counter) "
            "SELECT src, dst, type, ?, 1 FROM edges",
            (scope,),
        )
        edge_inserted = edge_cur.rowcount or 0
        self._conn.commit()
        self.set_schema_meta(
            SCHEMA_META_VERSION_VECTOR_BACKFILL,
            datetime.now(timezone.utc).isoformat(),
        )
        return belief_inserted + edge_inserted

    def _maybe_backfill_log_version_vectors(self) -> int:
        """Stamp `{local_scope: 1}` on every pre-existing ingest_log row.

        v2.0 #205. Same shape as `_maybe_backfill_version_vectors`:
        idempotent via the schema-meta marker. Runs once when a v2.0
        binary first opens a store that already has ingest_log rows
        but no log_versions entries (e.g. after the parallel-write
        phase ships and stores accumulate log rows before federation).
        """
        if self.get_schema_meta(SCHEMA_META_LOG_VERSION_VECTOR_BACKFILL):
            return 0
        scope = self._local_scope_id
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO log_versions (log_id, scope_id, counter) "
            "SELECT id, ?, 1 FROM ingest_log",
            (scope,),
        )
        inserted = cur.rowcount or 0
        self._conn.commit()
        self.set_schema_meta(
            SCHEMA_META_LOG_VERSION_VECTOR_BACKFILL,
            datetime.now(timezone.utc).isoformat(),
        )
        return inserted

    def _maybe_synthesize_legacy_log_rows(self) -> int:
        """One-shot migration: synthesize a `source_kind=legacy_unknown`
        ingest_log row for every belief that has no log row pointing at it.

        v2.x #263. Pre-v2.0 stores accumulate beliefs through ingest
        paths that pre-date the parallel-write log; those beliefs have
        no `derived_belief_ids` coverage and are flagged as orphans by
        the reachability check. This migration backfills exactly one
        synthesized row per orphan belief, using:
          - ts          = belief.created_at
          - raw_text    = belief.content
          - session_id  = belief.session_id (NULL when absent)
          - source_path = NULL
          - derived_belief_ids = [belief.id]
          - all other nullable columns = NULL

        Idempotent: the `SCHEMA_META_LEGACY_LOG_SYNTH` marker is stamped
        on completion; subsequent opens short-circuit on the marker and
        return 0 without touching any rows.

        Must run BEFORE `_maybe_backfill_log_version_vectors` in `__init__`
        so the synthesized rows get their version-vector stamp on the same
        open.

        Returns the number of synthesized rows inserted.
        """
        if self.get_schema_meta(SCHEMA_META_LEGACY_LOG_SYNTH):
            return 0
        # Find beliefs whose id does not appear in any
        # derived_belief_ids JSON array. json_each is available in
        # SQLite ≥ 3.38 (standard in Python 3.12+). The subquery
        # builds the set of covered belief ids from all log rows.
        cur = self._conn.execute(
            """
            SELECT b.id, b.content, b.created_at, b.session_id
            FROM beliefs b
            WHERE b.id NOT IN (
                SELECT je.value
                FROM ingest_log il, json_each(il.derived_belief_ids) je
                WHERE il.derived_belief_ids IS NOT NULL
            )
            """
        )
        orphan_rows = cur.fetchall()
        inserted = 0
        for row in orphan_rows:
            belief_id = str(row["id"])
            log_id = ulid()
            ts = str(row["created_at"])
            raw_text = str(row["content"])
            session_id = row["session_id"]
            self._conn.execute(
                """
                INSERT INTO ingest_log (
                    id, ts, source_kind, source_path, raw_text, raw_meta,
                    derived_belief_ids, derived_edge_ids,
                    classifier_version, rule_set_hash, session_id
                )
                VALUES (?, ?, ?, NULL, ?, NULL, ?, NULL, NULL, NULL, ?)
                """,
                (
                    log_id,
                    ts,
                    INGEST_SOURCE_LEGACY_UNKNOWN,
                    raw_text,
                    json.dumps([belief_id]),
                    session_id,
                ),
            )
            self._bump_log_version(log_id)
            inserted += 1
        self._conn.commit()
        self.set_schema_meta(
            SCHEMA_META_LEGACY_LOG_SYNTH,
            datetime.now(timezone.utc).isoformat(),
        )
        return inserted

    def _maybe_consolidate_content_hash_duplicates(self) -> int:
        """One-shot migration: collapse duplicate content_hash rows.

        #219. Before UNIQUE(content_hash) can be applied the DB must
        have at most one belief row per content_hash. This pass finds
        all duplicate groups and merges them:

        - Canonical row: oldest created_at, tie-break id ASC.
        - alpha / beta: sum across the group (each carries Bayesian
          evidence).
        - lock_level: MAX (user > none).
        - origin: highest-precedence value across the group (user_*
          beats agent_* beats everything else).
        - last_retrieved_at: MAX (most recent retrieval).
        - FK rewrites: feedback_history, belief_corroborations, edges
          (src and dst). belief_entities and belief_versions are
          dropped for duplicates (same content → same entities; the
          canonical row already has them).
        - One synthetic belief_corroborations row (source_type=
          'consolidation_migration') per duplicate consumed preserves
          the count signal.
        - Duplicate belief rows deleted from beliefs and beliefs_fts.

        Implementation uses bulk SQL (one SELECT + executemany) to
        stay well under 5 seconds on stores with tens of thousands of
        beliefs. The full pass on a 20K-belief store with ~2K duplicate
        groups takes < 2 seconds.

        Idempotent: the SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE marker
        short-circuits on subsequent opens. All work is inside a single
        transaction so a crash mid-migration leaves no partial state.

        Returns the count of duplicate rows eliminated.
        """
        if self.get_schema_meta(SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE):
            return 0

        # Fetch all beliefs in duplicate groups in one query, ordered
        # so the canonical (oldest created_at, then id ASC) is first.
        rows = self._conn.execute(
            """
            SELECT id, content_hash, alpha, beta, lock_level,
                   origin, last_retrieved_at
            FROM beliefs
            WHERE content_hash IN (
                SELECT content_hash FROM beliefs
                GROUP BY content_hash HAVING COUNT(*) > 1
            )
            ORDER BY content_hash ASC, created_at ASC, id ASC
            """
        ).fetchall()

        if not rows:
            self.set_schema_meta(
                SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
                datetime.now(timezone.utc).isoformat(),
            )
            return 0

        # Origin precedence: higher number = higher priority.
        _ORIGIN_RANK: dict[str, int] = {
            "user_stated": 10,
            "user_corrected": 9,
            "user_validated": 8,
            "agent_remembered": 5,
            "agent_inferred": 4,
            "document_recent": 3,
            "unknown": 0,
        }

        # Group in Python — O(n) pass.
        from collections import defaultdict as _defaultdict
        groups: dict[str, list[sqlite3.Row]] = _defaultdict(list)
        for row in rows:
            groups[str(row["content_hash"])].append(row)

        all_dupe_ids: list[str] = []
        canonical_updates: list[tuple[object, ...]] = []
        dupe_to_canon: dict[str, str] = {}
        corroboration_rows: list[tuple[str, str]] = []

        ts_now = datetime.now(timezone.utc).isoformat()

        for _ch, group in groups.items():
            if len(group) < 2:
                continue
            canonical_id = str(group[0]["id"])
            dupe_ids = [str(r["id"]) for r in group[1:]]
            all_dupe_ids.extend(dupe_ids)

            total_alpha = sum(float(r["alpha"]) for r in group)
            total_beta = sum(float(r["beta"]) for r in group)
            lock_level = (
                "user"
                if any(str(r["lock_level"]) == "user" for r in group)
                else "none"
            )
            origin = max(
                [str(r["origin"]) for r in group],
                key=lambda o: _ORIGIN_RANK.get(o, 0),
            )
            retrieved_vals = [
                str(r["last_retrieved_at"])
                for r in group
                if r["last_retrieved_at"] is not None
            ]
            last_retrieved_at = max(retrieved_vals) if retrieved_vals else None

            canonical_updates.append(
                (total_alpha, total_beta, lock_level, origin,
                 last_retrieved_at, canonical_id)
            )
            for did in dupe_ids:
                dupe_to_canon[did] = canonical_id
                corroboration_rows.append((canonical_id, did))

        if not all_dupe_ids:
            self.set_schema_meta(
                SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
                datetime.now(timezone.utc).isoformat(),
            )
            return 0

        # All writes in one transaction.
        with self._conn:
            # Update canonical belief rows (alpha/beta/lock/origin/lra).
            self._conn.executemany(
                """
                UPDATE beliefs SET
                    alpha = ?, beta = ?, lock_level = ?,
                    origin = ?, last_retrieved_at = ?
                WHERE id = ?
                """,
                canonical_updates,
            )

            # Rewrite FK references: feedback_history,
            # belief_corroborations, edges (src and dst).
            self._conn.executemany(
                "UPDATE feedback_history SET belief_id = ? "
                "WHERE belief_id = ?",
                [(canon, dupe) for dupe, canon in dupe_to_canon.items()],
            )
            self._conn.executemany(
                "UPDATE belief_corroborations SET belief_id = ? "
                "WHERE belief_id = ?",
                [(canon, dupe) for dupe, canon in dupe_to_canon.items()],
            )
            self._conn.executemany(
                "UPDATE edges SET src = ? WHERE src = ?",
                [(canon, dupe) for dupe, canon in dupe_to_canon.items()],
            )
            self._conn.executemany(
                "UPDATE edges SET dst = ? WHERE dst = ?",
                [(canon, dupe) for dupe, canon in dupe_to_canon.items()],
            )

            # belief_entities: same content → same entities extracted;
            # the canonical rows already exist. Drop duplicates to
            # avoid PK conflict.
            # belief_versions: drop duplicate rows; canonical
            # accumulates version bumps going forward.
            ph = ",".join("?" * len(all_dupe_ids))
            self._conn.execute(
                f"DELETE FROM belief_entities WHERE belief_id IN ({ph})",
                all_dupe_ids,
            )
            self._conn.execute(
                f"DELETE FROM belief_versions WHERE belief_id IN ({ph})",
                all_dupe_ids,
            )

            # Synthetic corroboration rows: one per duplicate consumed.
            # Best-effort: log and skip on IntegrityError rather than
            # aborting the dedup work, but do not swallow silently —
            # #336 (silent count-signal loss) was caused by an earlier
            # blanket `except: pass` here combined with a CASCADE-on-
            # DROP wipe in the unique-applying migration. The cascade
            # bug is fixed in `_maybe_apply_content_hash_unique`; this
            # except is retained only as a guard against unforeseen
            # legacy-schema integrity failures.
            try:
                self._conn.executemany(
                    """
                    INSERT INTO belief_corroborations
                        (belief_id, ingested_at, source_type,
                         session_id, source_path_hash)
                    VALUES (?, ?, ?, NULL, NULL)
                    """,
                    [
                        (canon_id, ts_now,
                         CORROBORATION_SOURCE_CONSOLIDATION_MIGRATION)
                        for canon_id, _dupe_id in corroboration_rows
                    ],
                )
            except sqlite3.IntegrityError as err:
                import logging
                logging.getLogger(__name__).warning(
                    "consolidation-migration: synthetic corroboration "
                    "insert failed (%d rows skipped): %s",
                    len(corroboration_rows),
                    err,
                )

            # Delete duplicate rows from beliefs and FTS.
            self._conn.execute(
                f"DELETE FROM beliefs WHERE id IN ({ph})",
                all_dupe_ids,
            )
            self._conn.execute(
                f"DELETE FROM beliefs_fts WHERE id IN ({ph})",
                all_dupe_ids,
            )

        self.set_schema_meta(
            SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
            datetime.now(timezone.utc).isoformat(),
        )
        return len(all_dupe_ids)

    def _maybe_apply_content_hash_unique(self) -> bool:
        """One-shot table-swap that adds UNIQUE(content_hash) to beliefs.

        #219. SQLite does not support ALTER TABLE ADD CONSTRAINT, so we
        use the standard rename/create/copy/drop pattern:

          1. CREATE TABLE beliefs_new (... UNIQUE(content_hash)).
          2. INSERT INTO beliefs_new SELECT * FROM beliefs.
          3. DROP TABLE beliefs.
          4. ALTER TABLE beliefs_new RENAME TO beliefs.
          5. Recreate all indexes that were on beliefs.

        Precondition: _maybe_consolidate_content_hash_duplicates must
        have already run (no duplicate content_hash rows exist). If a
        UNIQUE violation occurs the transaction rolls back and the
        error propagates — it means the consolidation pass was skipped
        or did not complete, which is a programming error.

        Fresh stores skip this migration because _SCHEMA already
        includes UNIQUE(content_hash). Idempotent via
        SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED marker.

        Returns True if the swap ran, False if it was already applied.
        """
        if self.get_schema_meta(SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED):
            return False

        # Check whether the constraint already exists (e.g. fresh store
        # created with the new _SCHEMA that includes UNIQUE).
        idx_cur = self._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='beliefs'"
        )
        row = idx_cur.fetchone()
        if row and "UNIQUE" in str(row["sql"]):
            # Fresh store — constraint already present; just stamp marker.
            self.set_schema_meta(
                SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,
                datetime.now(timezone.utc).isoformat(),
            )
            return False

        # Build CREATE TABLE DDL from the existing beliefs table.
        # Using PRAGMA table_info avoids parsing the sqlite_master DDL
        # string and correctly handles extra columns added by ALTER TABLE
        # on legacy stores (e.g. hibernation_score, activation_condition).
        col_info = self._conn.execute(
            "PRAGMA table_info(beliefs)"
        ).fetchall()
        col_defs: list[str] = []
        col_names: list[str] = []
        for col in col_info:
            cname = str(col["name"])
            ctype = str(col["type"])
            notnull = " NOT NULL" if col["notnull"] else ""
            dflt = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] is not None else ""
            pk = " PRIMARY KEY" if col["pk"] else ""
            unique = " UNIQUE" if cname == "content_hash" else ""
            col_defs.append(f"    {cname} {ctype}{pk}{unique}{notnull}{dflt}")
            col_names.append(cname)

        create_sql = (
            "CREATE TABLE beliefs_new (\n"
            + ",\n".join(col_defs)
            + "\n)"
        )

        col_list = ", ".join(col_names)

        # Disable FK enforcement around the table swap. Without this,
        # `DROP TABLE beliefs` fires `ON DELETE CASCADE` on
        # belief_corroborations (and any other table referencing
        # beliefs.id), wiping rows that the dedup migration just
        # inserted. This is the SQLite-recommended pattern for
        # schema-mutation migrations — see
        # https://www.sqlite.org/lang_altertable.html#otheralter
        # ("disable foreign key constraints with PRAGMA foreign_keys=OFF").
        # Toggling foreign_keys must happen OUTSIDE any transaction,
        # so the swap runs in its own explicit BEGIN/COMMIT.
        self._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self._conn.execute("BEGIN")
            try:
                # Drop any leftover temp table from a failed previous attempt.
                self._conn.execute("DROP TABLE IF EXISTS beliefs_new")
                self._conn.execute(create_sql)
                self._conn.execute(
                    f"INSERT INTO beliefs_new ({col_list}) "
                    f"SELECT {col_list} FROM beliefs"
                )
                self._conn.execute("DROP TABLE beliefs")
                self._conn.execute(
                    "ALTER TABLE beliefs_new RENAME TO beliefs"
                )
                # Recreate indexes that were on the original beliefs table.
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_beliefs_session "
                    "ON beliefs(session_id)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_beliefs_origin "
                    "ON beliefs(origin)"
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON")

        self.set_schema_meta(
            SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,
            datetime.now(timezone.utc).isoformat(),
        )
        return True

    def list_belief_ids(self) -> list[str]:
        """All belief ids in insertion-time order. Used by the v1.3
        entity-index backfill to walk every existing belief once."""
        cur = self._conn.execute("SELECT id FROM beliefs ORDER BY id ASC")
        return [str(r["id"]) for r in cur.fetchall()]

    # --- Belief CRUD ------------------------------------------------------

    def insert_belief(self, b: Belief) -> None:
        _check_insert_belief_authority()
        if b.retention_class not in RETENTION_CLASSES:
            raise ValueError(
                f"invalid retention_class {b.retention_class!r}; "
                f"must be one of {sorted(RETENTION_CLASSES)}"
            )
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, demotion_pressure,
                created_at, last_retrieved_at, session_id, origin,
                hibernation_score, activation_condition,
                retention_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                b.id, b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.session_id, b.origin,
                b.hibernation_score, b.activation_condition,
                b.retention_class,
            ),
        )
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        self._write_belief_entities(b.id, b.content)
        self._bump_belief_version(b.id)
        self._conn.commit()
        self._fire_invalidation()

    def get_belief_by_content_hash(self, content_hash: str) -> Belief | None:
        """Look up a belief by its content_hash. Returns None if not found.

        Used by ingest paths to detect re-ingest of identical content
        across different (source, text) pairs — e.g. the same sentence
        ingested once via transcript-ingest and once via commit-ingest.
        When found, the caller records a belief_corroborations row
        instead of silently dropping the duplicate.
        """
        cur = self._conn.execute(
            """
            SELECT b.*,
                   (SELECT COUNT(*) FROM belief_corroborations bc
                    WHERE bc.belief_id = b.id) AS corroboration_count
            FROM beliefs b
            WHERE b.content_hash = ?
            LIMIT 1
            """,
            (content_hash,),
        )
        row = cur.fetchone()
        return _row_to_belief(row) if row else None

    def get_belief(self, belief_id: str) -> Belief | None:
        cur = self._conn.execute(
            """
            SELECT b.*,
                   (SELECT COUNT(*) FROM belief_corroborations bc
                    WHERE bc.belief_id = b.id) AS corroboration_count
            FROM beliefs b
            WHERE b.id = ?
            """,
            (belief_id,),
        )
        row = cur.fetchone()
        return _row_to_belief(row) if row else None

    def update_belief(self, b: Belief) -> None:
        """Full-row update; demotion_pressure included."""
        self._conn.execute(
            """
            UPDATE beliefs SET
                content = ?,
                content_hash = ?,
                alpha = ?,
                beta = ?,
                type = ?,
                lock_level = ?,
                locked_at = ?,
                demotion_pressure = ?,
                created_at = ?,
                last_retrieved_at = ?,
                session_id = ?,
                origin = ?
            WHERE id = ?
            """,
            (
                b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.session_id,
                b.origin, b.id,
            ),
        )
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (b.id,))
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        # Rewrite entity rows for this belief: drop and re-extract.
        # Single-transaction with the FTS5 rewrite so we never expose
        # a half-updated index to readers between commits.
        self._conn.execute(
            "DELETE FROM belief_entities WHERE belief_id = ?", (b.id,)
        )
        self._write_belief_entities(b.id, b.content)
        self._bump_belief_version(b.id)
        self._conn.commit()
        self._fire_invalidation()

    def delete_belief(self, belief_id: str) -> None:
        self._conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (belief_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE src = ? OR dst = ?",
            (belief_id, belief_id),
        )
        self._conn.execute(
            "DELETE FROM belief_entities WHERE belief_id = ?", (belief_id,)
        )
        self._conn.commit()
        self._fire_invalidation()

    # --- Entity index (v1.3 L2.5 retrieval) ------------------------------

    def _write_belief_entities(self, belief_id: str, content: str) -> None:
        """Extract entities from `content` and insert one row per match.

        Called inside `insert_belief` and `update_belief` BEFORE the
        outer commit. Failure here aborts the parent transaction —
        the index ↔ store invariant is preserved (either both rows
        commit or neither does). The extractor is pure regex; it does
        not raise under normal input.

        Lazy import of `aelfrice.entity_extractor` to keep the import
        graph acyclic (entity_extractor only imports triple_extractor,
        not store) and so the extra cost of regex compilation is paid
        once on first insert rather than at module load.
        """
        from aelfrice.entity_extractor import extract_entities

        entities = extract_entities(content)
        if not entities:
            return
        rows = [
            (belief_id, e.lower, e.raw, e.kind, e.span_start, e.span_end)
            for e in entities
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO belief_entities "
            "(belief_id, entity_lower, entity_raw, kind, span_start, span_end) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def lookup_entities(
        self,
        entity_lowers: Iterable[str],
        *,
        limit: int,
    ) -> list[tuple[str, int]]:
        """Return [(belief_id, overlap_count)] sorted overlap DESC, id ASC.

        `entity_lowers` is the lowercased entity-text key per row in
        `belief_entities`. Distinct keys are GROUP BY'd so the same
        entity occurring twice in one belief still counts as one
        overlap (the spec ranks by entity overlap COUNT, not by
        match count).

        Empty input returns [] without hitting SQLite. The L2.5
        retrieval tier consumes this via `lookup_entities` directly —
        there is no separate index object to keep in sync with the
        underlying table.
        """
        keys = [k for k in entity_lowers if k]
        if not keys or limit <= 0:
            return []
        # SQL placeholder expansion: avoid building a query with
        # unbounded `?` count by using a temp table-style IN clause.
        # SQLite's parameter cap is 999 by default; we trim above
        # that defensively (the L2.5 query-side extraction caps at
        # 16 entities per call so this is generous).
        keys = list(dict.fromkeys(keys))[:512]
        placeholders = ",".join("?" * len(keys))
        sql = (
            "SELECT belief_id, COUNT(DISTINCT entity_lower) AS overlap "
            "FROM belief_entities "
            f"WHERE entity_lower IN ({placeholders}) "
            "GROUP BY belief_id "
            "ORDER BY overlap DESC, belief_id ASC "
            "LIMIT ?"
        )
        cur = self._conn.execute(sql, (*keys, limit))
        return [(str(r["belief_id"]), int(r["overlap"])) for r in cur.fetchall()]

    def count_belief_entities(self) -> int:
        """Total row count in `belief_entities`. Telemetry / health surface."""
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM belief_entities"
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def _maybe_backfill_entity_index(self) -> int:
        """One-shot backfill of belief_entities for pre-v1.3 stores.

        First open of a v1.3+ binary against any store (fresh or
        legacy) finds no `entity_backfill_complete` row in
        `schema_meta` and re-extracts entities for every existing
        belief. Idempotent: subsequent opens see the stamped marker
        and short-circuit. Re-running by dropping the marker also
        no-ops because `INSERT OR IGNORE` skips duplicates against
        the composite PK.

        Returns the number of new rows inserted (0 on a stamped
        store; positive on the first run against a non-empty
        legacy store; 0 on a fresh v1.3 store).
        """
        if self.get_schema_meta(SCHEMA_META_ENTITY_BACKFILL):
            return 0
        from aelfrice.entity_extractor import extract_entities
        ids = self.list_belief_ids()
        inserted = 0
        for bid in ids:
            cur = self._conn.execute(
                "SELECT content FROM beliefs WHERE id = ?", (bid,)
            )
            row = cur.fetchone()
            if row is None:
                continue
            content = str(row["content"])
            entities = extract_entities(content)
            if not entities:
                continue
            rows = [
                (bid, e.lower, e.raw, e.kind, e.span_start, e.span_end)
                for e in entities
            ]
            cur2 = self._conn.executemany(
                "INSERT OR IGNORE INTO belief_entities "
                "(belief_id, entity_lower, entity_raw, kind, "
                "span_start, span_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            if cur2.rowcount > 0:
                inserted += cur2.rowcount
        # Stamp the marker even on an empty-store run so later opens
        # short-circuit. The value is the ISO timestamp of the run;
        # health.py surfaces it as `entity_index_backfilled_at`.
        self.set_schema_meta(
            SCHEMA_META_ENTITY_BACKFILL,
            datetime.now(timezone.utc).isoformat(),
        )
        return inserted

    def belief_entities_for(self, belief_id: str) -> list[tuple[str, str, str]]:
        """List of (entity_lower, entity_raw, kind) for one belief.

        Used by tests and the future debug CLI; not on the hot path.
        """
        cur = self._conn.execute(
            "SELECT entity_lower, entity_raw, kind FROM belief_entities "
            "WHERE belief_id = ? ORDER BY span_start ASC",
            (belief_id,),
        )
        return [
            (str(r["entity_lower"]), str(r["entity_raw"]), str(r["kind"]))
            for r in cur.fetchall()
        ]

    def search_beliefs(self, query: str, limit: int = 20) -> list[Belief]:
        """FTS5 keyword search over belief content. Ranked by bm25.

        User input is escaped before being passed to FTS5: each
        whitespace-separated token is wrapped in double quotes (with
        embedded `"` doubled per FTS5 syntax) and joined with spaces
        (implicit AND). This protects against FTS5 special characters
        (., -, /, parens, quotes, AND/OR/NEAR keywords) raising
        OperationalError on what looks like an ordinary user query.

        Empty / whitespace-only queries return [] without hitting FTS5
        (which would raise on an empty MATCH expression).
        """
        escaped = _escape_fts5_query(query)
        if not escaped:
            return []
        cur = self._conn.execute(
            """
            SELECT b.*,
                   (SELECT COUNT(*) FROM belief_corroborations bc
                    WHERE bc.belief_id = b.id) AS corroboration_count
            FROM beliefs b
            JOIN beliefs_fts f ON f.id = b.id
            WHERE beliefs_fts MATCH ?
            ORDER BY bm25(beliefs_fts)
            LIMIT ?
            """,
            (escaped, limit),
        )
        return [_row_to_belief(r) for r in cur.fetchall()]

    def search_beliefs_scored(
        self, query: str, limit: int = 20,
    ) -> list[tuple[Belief, float]]:
        """FTS5 keyword search returning `(belief, bm25_score)` pairs.

        Sibling of `search_beliefs`. Same MATCH escaping, same ordering
        (ascending by `bm25(beliefs_fts)`, which SQLite returns as a
        non-positive number — smaller = more relevant). The raw FTS5
        BM25 score is exposed for callers that need to compose it with
        other signals (e.g. v1.3 partial Bayesian-weighted ranking,
        which combines `log(-bm25)` with `log(posterior_mean)` log-
        additively).

        Empty / whitespace-only queries return [] without hitting
        FTS5.
        """
        escaped = _escape_fts5_query(query)
        if not escaped:
            return []
        cur = self._conn.execute(
            """
            SELECT b.*,
                   bm25(beliefs_fts) AS bm25_score,
                   (SELECT COUNT(*) FROM belief_corroborations bc
                    WHERE bc.belief_id = b.id) AS corroboration_count
            FROM beliefs b
            JOIN beliefs_fts f ON f.id = b.id
            WHERE beliefs_fts MATCH ?
            ORDER BY bm25(beliefs_fts)
            LIMIT ?
            """,
            (escaped, limit),
        )
        rows = cur.fetchall()
        out: list[tuple[Belief, float]] = []
        for r in rows:
            score_obj = r["bm25_score"]
            score = float(score_obj) if score_obj is not None else 0.0
            out.append((_row_to_belief(r), score))
        return out

    # --- Retrieval recency -----------------------------------------------

    def stamp_retrieved(
        self,
        belief_ids: Iterable[str],
        ts: str | None = None,
    ) -> int:
        """Mark beliefs as retrieved at `ts` (defaults to UTC now).

        Single batched UPDATE; ids not present in the table are silently
        skipped (UPDATE matches zero rows). Returns the number of rows
        actually updated. Empty input is a no-op returning 0.

        This is the belief-side mirror of feedback_history writes for
        retrieval-driven feedback. The retrieval-audit-loop spec
        (v1.0.1 #127) requires both: feedback_history records the event,
        last_retrieved_at gives downstream consumers (decay moderation,
        recency-aware ranking, telemetry) an O(1) read.
        """
        ids = [bid for bid in belief_ids if bid]
        if not ids:
            return 0
        if ts is None:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"UPDATE beliefs SET last_retrieved_at = ? WHERE id IN ({placeholders})",
            (ts, *ids),
        )
        self._conn.commit()
        return cur.rowcount or 0

    # --- Feedback history ------------------------------------------------

    def insert_feedback_event(
        self,
        belief_id: str,
        valence: float,
        source: str,
        created_at: str,
    ) -> int:
        """Append one row to feedback_history; return its rowid.

        Called by apply_feedback() for every successful Bayesian update.
        """
        cur = self._conn.execute(
            """
            INSERT INTO feedback_history (belief_id, valence, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (belief_id, valence, source, created_at),
        )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("feedback_history insert returned no rowid")
        return rowid

    def list_feedback_events(
        self,
        belief_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Recent feedback events, ordered by id DESC. Filter by belief if given."""
        if belief_id is None:
            cur = self._conn.execute(
                "SELECT * FROM feedback_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT * FROM feedback_history
                WHERE belief_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (belief_id, limit),
            )
        return [_row_to_feedback(r) for r in cur.fetchall()]

    def count_feedback_events(self, belief_id: str | None = None) -> int:
        """Count rows; total or per-belief."""
        if belief_id is None:
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM feedback_history")
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM feedback_history WHERE belief_id = ?",
                (belief_id,),
            )
        row = cur.fetchone()
        if row is None:
            return 0
        return int(row["n"])

    def count_orphan_feedback_events(self) -> int:
        """Count `feedback_history` rows whose belief_id no longer
        exists in `beliefs` (issue #223).

        Pre-#283 re-ingest could create a fresh belief_id for the same
        content_hash and let the old row be deleted while leaving its
        feedback_history rows behind. The UNIQUE content_hash
        constraint and `insert_or_corroborate` plug the source going
        forward; this counter surfaces the residue still in user DBs.
        """
        cur = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM feedback_history fh
            LEFT JOIN beliefs b ON fh.belief_id = b.id
            WHERE b.id IS NULL
            """
        )
        row = cur.fetchone()
        if row is None:
            return 0
        return int(row["n"])

    def delete_orphan_feedback_events(self) -> int:
        """Delete `feedback_history` rows whose belief_id no longer
        resolves in `beliefs`. Returns the number deleted. Issue #223.

        Destructive — caller is responsible for confirmation. The audit
        trail for the deleted rows is unrecoverable: the original
        belief content is already gone and feedback_history stores
        only `belief_id`, not `content_hash`.
        """
        cur = self._conn.execute(
            """
            DELETE FROM feedback_history
            WHERE belief_id IN (
                SELECT fh.belief_id
                FROM feedback_history fh
                LEFT JOIN beliefs b ON fh.belief_id = b.id
                WHERE b.id IS NULL
            )
            """
        )
        self._conn.commit()
        return cur.rowcount if cur.rowcount is not None else 0

    # --- Belief corroborations (v1.5+, #190) -----------------------------

    def insert_or_corroborate(
        self,
        b: Belief,
        *,
        source_type: str,
        session_id: str | None = None,
        source_path_hash: str | None = None,
    ) -> tuple[str, bool]:
        """Insert belief or corroborate existing one with same content_hash.

        Returns (belief_id, was_inserted). On a content_hash hit the
        existing belief row is unchanged and record_corroboration is
        called to record the re-assertion signal. On a miss insert_belief
        is called and (b.id, True) is returned.

        `source_type` must be in CORROBORATION_SOURCE_TYPES; ValueError
        is raised immediately on an unknown value so the caller's test
        suite catches misconfigured mappings early.
        """
        # Validate source_type up-front so the error surfaces at the
        # call site, not inside record_corroboration after the lookup.
        if source_type not in CORROBORATION_SOURCE_TYPES:
            raise ValueError(
                f"Unknown source_type {source_type!r}. "
                f"Must be one of {sorted(CORROBORATION_SOURCE_TYPES)}"
            )
        existing = self.get_belief_by_content_hash(b.content_hash)
        if existing is not None:
            self.record_corroboration(
                existing.id,
                source_type=source_type,
                session_id=session_id,
                source_path_hash=source_path_hash,
            )
            return (existing.id, False)
        # Race / migration guard (#264): same id may already exist under
        # a different content_hash — e.g. legacy migration backfilled
        # the row with a precomputed hash, or a sibling session inserted
        # between the get_belief_by_content_hash check above and our
        # INSERT. Treat as corroboration of the id-collision row so the
        # derivation worker never trips a UNIQUE constraint on id.
        existing_by_id = self.get_belief(b.id)
        if existing_by_id is not None:
            self.record_corroboration(
                existing_by_id.id,
                source_type=source_type,
                session_id=session_id,
                source_path_hash=source_path_hash,
            )
            return (existing_by_id.id, False)
        self.insert_belief(b)
        return (b.id, True)

    def record_corroboration(
        self,
        belief_id: str,
        *,
        source_type: str,
        session_id: str | None = None,
        source_path_hash: str | None = None,
    ) -> None:
        """Record one corroboration row for an already-existing belief.

        Called by the ingest path when an INSERT hits the content_hash
        UNIQUE constraint. Validates `source_type` against
        CORROBORATION_SOURCE_TYPES; raises ValueError on unknown values.

        `session_id` and `source_path_hash` are nullable: pass None
        when unavailable; no exception is raised.
        """
        if source_type not in CORROBORATION_SOURCE_TYPES:
            raise ValueError(
                f"Unknown source_type {source_type!r}. "
                f"Must be one of {sorted(CORROBORATION_SOURCE_TYPES)}"
            )
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO belief_corroborations
                (belief_id, ingested_at, source_type, session_id, source_path_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (belief_id, ts, source_type, session_id, source_path_hash),
        )
        self._conn.commit()

    def count_corroborations(self, belief_id: str) -> int:
        """Return the count of belief_corroborations rows for one belief.

        Used by _row_to_belief to populate Belief.corroboration_count.
        """
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM belief_corroborations "
            "WHERE belief_id = ?",
            (belief_id,),
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def list_corroborations(
        self,
        belief_id: str,
    ) -> list[tuple[str, str, str | None, str | None]]:
        """Return [(ingested_at, source_type, session_id, source_path_hash)]
        for one belief, ordered by ingested_at ASC.

        Used by tests and future debug CLI.
        """
        cur = self._conn.execute(
            """
            SELECT ingested_at, source_type, session_id, source_path_hash
            FROM belief_corroborations
            WHERE belief_id = ?
            ORDER BY ingested_at ASC
            """,
            (belief_id,),
        )
        return [
            (
                str(r["ingested_at"]),
                str(r["source_type"]),
                r["session_id"],
                r["source_path_hash"],
            )
            for r in cur.fetchall()
        ]

    # --- #435 doc linker --------------------------------------------------
    #
    # `belief_documents` rows are 1:1 with (belief_id, doc_uri) pairs.
    # `INSERT OR IGNORE` collapses re-ingest of the same belief from the
    # same source to the first-write row; `get_doc_anchors` returns the
    # canonical row regardless of how many times the writer was called.
    # See `aelfrice.doc_linker` for the public DocAnchor dataclass and
    # spec contract; this module owns the SQL only.

    def link_belief_to_document(
        self,
        *,
        belief_id: str,
        doc_uri: str,
        anchor_type: str,
        position_hint: str | None,
    ) -> "DocAnchor":  # noqa: F821 — forward ref to avoid a circular import
        """Persist one anchor row and return a `DocAnchor`.

        Idempotent on `(belief_id, doc_uri)`: the table's PK + INSERT OR
        IGNORE turns repeats into no-op writes. Returns the canonical
        (first-write) row regardless of whether this call inserted.
        """
        from aelfrice.doc_linker_types import ANCHOR_TYPES, DocAnchor

        if not doc_uri:
            raise ValueError("doc_uri must be non-empty")
        if anchor_type not in ANCHOR_TYPES:
            raise ValueError(
                f"Unknown anchor_type {anchor_type!r}. "
                f"Must be one of {sorted(ANCHOR_TYPES)}"
            )
        ts = datetime.now(timezone.utc).timestamp()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO belief_documents
                (belief_id, doc_uri, anchor_type, position_hint, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (belief_id, doc_uri, anchor_type, position_hint, ts),
        )
        self._conn.commit()
        cur = self._conn.execute(
            """
            SELECT belief_id, doc_uri, anchor_type, position_hint, created_at
            FROM belief_documents
            WHERE belief_id = ? AND doc_uri = ?
            """,
            (belief_id, doc_uri),
        )
        row = cur.fetchone()
        return DocAnchor(
            belief_id=str(row["belief_id"]),
            doc_uri=str(row["doc_uri"]),
            anchor_type=str(row["anchor_type"]),
            position_hint=row["position_hint"],
            created_at=float(row["created_at"]),
        )

    def get_doc_anchors(self, belief_id: str) -> list["DocAnchor"]:  # noqa: F821
        """Return every anchor for one belief, ordered by `created_at` ASC."""
        from aelfrice.doc_linker_types import DocAnchor

        cur = self._conn.execute(
            """
            SELECT belief_id, doc_uri, anchor_type, position_hint, created_at
            FROM belief_documents
            WHERE belief_id = ?
            ORDER BY created_at ASC
            """,
            (belief_id,),
        )
        return [
            DocAnchor(
                belief_id=str(r["belief_id"]),
                doc_uri=str(r["doc_uri"]),
                anchor_type=str(r["anchor_type"]),
                position_hint=r["position_hint"],
                created_at=float(r["created_at"]),
            )
            for r in cur.fetchall()
        ]

    def get_doc_anchors_batch(
        self,
        belief_ids: list[str],
    ) -> dict[str, list["DocAnchor"]]:  # noqa: F821
        """Batched fetch keyed by `belief_id`. Empty list for ids without anchors.

        Used by `retrieve(..., with_doc_anchors=True)` so the projection
        costs one indexed read per call rather than one per surfaced
        belief. Result dict has an entry for every requested id.
        """
        from aelfrice.doc_linker_types import DocAnchor

        out: dict[str, list[DocAnchor]] = {bid: [] for bid in belief_ids}
        if not belief_ids:
            return out
        ph = ",".join("?" * len(belief_ids))
        cur = self._conn.execute(
            f"""
            SELECT belief_id, doc_uri, anchor_type, position_hint, created_at
            FROM belief_documents
            WHERE belief_id IN ({ph})
            ORDER BY belief_id ASC, created_at ASC
            """,
            tuple(belief_ids),
        )
        for r in cur.fetchall():
            out[str(r["belief_id"])].append(
                DocAnchor(
                    belief_id=str(r["belief_id"]),
                    doc_uri=str(r["doc_uri"]),
                    anchor_type=str(r["anchor_type"]),
                    position_hint=r["position_hint"],
                    created_at=float(r["created_at"]),
                )
            )
        return out

    # --- Ingest log (v2.0, #205) -----------------------------------------

    def record_ingest(
        self,
        *,
        source_kind: str,
        raw_text: str,
        source_path: str | None = None,
        raw_meta: dict[str, object] | None = None,
        derived_belief_ids: list[str] | None = None,
        derived_edge_ids: list[tuple[str, str, str]] | None = None,
        classifier_version: str | None = None,
        rule_set_hash: str | None = None,
        session_id: str | None = None,
        ts: str | None = None,
        log_id: str | None = None,
    ) -> str:
        """Append one row to the v2.0 ingest_log. Returns the log id (ULID).

        Per the spec at docs/design/write-log-as-truth.md, every belief
        and edge must trace back to at least one ingest_log row. v2.0
        first slice writes the log in parallel; v2.x flips authority.

        `source_kind` must be one of INGEST_SOURCE_KINDS; raises
        ValueError otherwise. JSON-serializable fields (raw_meta,
        derived_belief_ids, derived_edge_ids) are encoded at write
        time so callers don't have to. `ts` and `log_id` are
        injectable for deterministic tests.
        """
        if source_kind not in INGEST_SOURCE_KINDS:
            raise ValueError(
                f"Unknown source_kind {source_kind!r}. "
                f"Must be one of {sorted(INGEST_SOURCE_KINDS)}"
            )
        log_id = log_id if log_id is not None else ulid()
        ts = ts if ts is not None else datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO ingest_log (
                id, ts, source_kind, source_path, raw_text, raw_meta,
                derived_belief_ids, derived_edge_ids,
                classifier_version, rule_set_hash, session_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id, ts, source_kind, source_path, raw_text,
                json.dumps(raw_meta) if raw_meta is not None else None,
                json.dumps(derived_belief_ids)
                if derived_belief_ids is not None else None,
                json.dumps(derived_edge_ids)
                if derived_edge_ids is not None else None,
                classifier_version, rule_set_hash, session_id,
            ),
        )
        self._bump_log_version(log_id)
        self._conn.commit()
        return log_id

    def update_ingest_derived_ids(
        self,
        log_id: str,
        *,
        derived_belief_ids: list[str] | None = None,
        derived_edge_ids: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Set derived_belief_ids / derived_edge_ids on an existing log row.

        Used when the log row is written before classification produces
        belief/edge ids. Either argument may be None to leave that
        column untouched.
        """
        sets: list[str] = []
        params: list[object] = []
        if derived_belief_ids is not None:
            sets.append("derived_belief_ids = ?")
            params.append(json.dumps(derived_belief_ids))
        if derived_edge_ids is not None:
            sets.append("derived_edge_ids = ?")
            params.append(json.dumps(derived_edge_ids))
        if not sets:
            return
        params.append(log_id)
        self._conn.execute(
            f"UPDATE ingest_log SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def get_ingest_log_entry(self, log_id: str) -> dict[str, object] | None:
        """Return a dict view of one ingest_log row, or None if missing.

        Decodes the JSON-encoded fields (raw_meta, derived_*_ids).
        Used by tests and the v2.0 replay validation harness.
        """
        cur = self._conn.execute(
            "SELECT * FROM ingest_log WHERE id = ?",
            (log_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _ingest_row_to_dict(row)

    def count_ingest_log(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM ingest_log")
        return int(cur.fetchone()["n"])

    def list_unstamped_ingest_log(self) -> list[dict[str, object]]:
        """Return ingest_log rows whose `derived_belief_ids` is unstamped.

        Used by the v2.x derivation worker (#264). A row is considered
        unstamped when `derived_belief_ids` is SQL NULL — i.e. nothing
        has materialized beliefs from it yet. An explicit JSON `[]`
        means the worker visited the row but `derive()` produced no
        belief; that is a stamped no-op and is NOT returned here.

        Rows are returned in ULID order (== insertion / time order),
        same shape as `get_ingest_log_entry`.
        """
        cur = self._conn.execute(
            "SELECT * FROM ingest_log "
            "WHERE derived_belief_ids IS NULL "
            "ORDER BY id"
        )
        return [_ingest_row_to_dict(r) for r in cur.fetchall()]

    def iter_ingest_log_for_belief(
        self, belief_id: str,
    ) -> list[dict[str, object]]:
        """All ingest_log rows whose derived_belief_ids contains belief_id.

        Linear scan — v2.0 first slice has no inverted index. Acceptable
        for the validation harness; revisit if interactive callers appear.
        """
        cur = self._conn.execute(
            "SELECT * FROM ingest_log WHERE derived_belief_ids IS NOT NULL"
        )
        out: list[dict[str, object]] = []
        for row in cur.fetchall():
            d = _ingest_row_to_dict(row)
            ids = d.get("derived_belief_ids") or []
            if isinstance(ids, list) and belief_id in ids:
                out.append(d)
        return out
    # --- Deferred feedback queue (v1.6+, #191) ---------------------------

    def enqueue_deferred_feedback(
        self,
        belief_id: str,
        *,
        event_type: str,
        enqueued_at: str,
    ) -> int:
        """Insert one row into deferred_feedback_queue with status='enqueued'.

        Returns the new rowid. Caller is responsible for grouping bulk
        enqueues (e.g. one retrieve() call producing N surfaced beliefs)
        by sharing a single `enqueued_at` timestamp so the grace-window
        check is well-defined for the batch.
        """
        cur = self._conn.execute(
            """
            INSERT INTO deferred_feedback_queue
                (belief_id, enqueued_at, event_type, status)
            VALUES (?, ?, ?, 'enqueued')
            """,
            (belief_id, enqueued_at, event_type),
        )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError(
                "deferred_feedback_queue insert returned no rowid"
            )
        return rowid

    def list_pending_deferred_feedback(
        self,
        *,
        cutoff_iso: str,
        limit: int = 1000,
    ) -> list[tuple[int, str, str, str]]:
        """Return [(id, belief_id, enqueued_at, event_type)] for queue
        rows with status='enqueued' and enqueued_at <= cutoff_iso.

        Ordered by id ASC for deterministic processing. The caller
        provides the cutoff (typically `now - T_grace`) so the sweeper
        only sees rows whose grace window has elapsed.
        """
        cur = self._conn.execute(
            """
            SELECT id, belief_id, enqueued_at, event_type
            FROM deferred_feedback_queue
            WHERE status = 'enqueued' AND enqueued_at <= ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (cutoff_iso, limit),
        )
        return [
            (
                int(r["id"]),
                str(r["belief_id"]),
                str(r["enqueued_at"]),
                str(r["event_type"]),
            )
            for r in cur.fetchall()
        ]

    def has_explicit_feedback_in_window(
        self,
        belief_id: str,
        *,
        window_start_iso: str,
        window_end_iso: str,
        retrieval_source: str,
    ) -> bool:
        """True if any feedback_history row exists for `belief_id` whose
        `created_at` falls within [window_start_iso, window_end_iso] and
        whose `source` is NOT `retrieval_source`.

        Used by the deferred-feedback sweeper to detect whether an
        explicit user correction or a contradiction-tiebreaker event
        landed during the grace window — either case cancels the
        pending implicit increment per the explicit-beats-implicit
        contract.
        """
        cur = self._conn.execute(
            """
            SELECT 1 FROM feedback_history
            WHERE belief_id = ?
              AND source != ?
              AND created_at >= ? AND created_at <= ?
            LIMIT 1
            """,
            (belief_id, retrieval_source, window_start_iso, window_end_iso),
        )
        return cur.fetchone() is not None

    def count_deferred_feedback_by_status(self) -> dict[str, int]:
        """Return a {status: count} dict over deferred_feedback_queue.

        Used by tests and `aelf doctor` to surface queue health.
        Statuses with zero rows are omitted from the dict.
        """
        cur = self._conn.execute(
            """
            SELECT status, COUNT(*) AS n
            FROM deferred_feedback_queue
            GROUP BY status
            """
        )
        return {str(r["status"]): int(r["n"]) for r in cur.fetchall()}

    # --- Aggregations (used by aelf:health) ------------------------------

    def count_beliefs(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM beliefs")
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def count_edges(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM edges")
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def count_locked(self) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM beliefs WHERE lock_level != 'none'"
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def alpha_beta_pairs(self) -> list[tuple[float, float]]:
        """Return every belief's (alpha, beta) for aggregate computation.

        Used by health.py to compute confidence mean / median and mass
        mean. Memory-bound at large scale (~16 bytes per belief); for
        v1.0 acceptable at the realistic 10^4-10^5 scale. Streams via
        sqlite cursor rather than fetchall to keep peak memory linear.
        """
        cur = self._conn.execute("SELECT alpha, beta FROM beliefs")
        return [(float(r["alpha"]), float(r["beta"])) for r in cur.fetchall()]

    def list_locked_beliefs(self) -> list[Belief]:
        """All beliefs with lock_level != 'none', ordered by locked_at DESC.

        Used by the L0 retrieval layer: locked beliefs are user-asserted
        ground truth and auto-load above any keyword-search results.
        """
        cur = self._conn.execute(
            """
            SELECT b.*,
                   (SELECT COUNT(*) FROM belief_corroborations bc
                    WHERE bc.belief_id = b.id) AS corroboration_count
            FROM beliefs b
            WHERE b.lock_level != 'none'
            ORDER BY b.locked_at DESC, b.id ASC
            """
        )
        return [_row_to_belief(r) for r in cur.fetchall()]

    def find_orphan_beliefs(self, *, max_n: int | None = None) -> list[Belief]:
        """Return beliefs that have no classified type and no feedback signal.

        Orphan definition (two signals, both must be true):
          1. type = 'unknown' OR type IS NULL — never successfully typed by
             onboard or ingest.
          2. alpha + beta <= 2 — never received any feedback event
             (the default prior is alpha=1, beta=1, sum=2; any feedback
             pushes the sum above 2).

        Additional signals (entity_index miss, zero non-CONTAINS edges) are
        planned as future extensions once #143 and edge-type auditing land.

        `max_n` caps the result set. When None, all orphans are returned.
        Callers that want a per-run limit (--max N on the CLI) pass it here
        rather than slicing afterwards, so the SQL does the limiting.
        """
        limit_clause = f"LIMIT {int(max_n)}" if max_n is not None else ""
        cur = self._conn.execute(
            f"""
            SELECT * FROM beliefs
            WHERE (type = 'unknown' OR type IS NULL)
              AND (alpha + beta) <= 2
            ORDER BY created_at ASC
            {limit_clause}
            """
        )
        return [_row_to_belief(r) for r in cur.fetchall()]

    def find_promotable_snapshots(
        self, *, min_corroborations: int = 3, min_sessions: int = 2,
        max_n: int | None = None,
    ) -> list[Belief]:
        """Return snapshot beliefs eligible for promotion to ``fact``.

        Per docs/belief_retention_class.md §4 the promotion rule is:

          retention_class = 'snapshot'
          AND COUNT(corroborations) >= min_corroborations
          AND COUNT(DISTINCT corroborations.session_id) >= min_sessions
          AND no inbound CONTRADICTS edge targets the belief

        ``session_id`` may be NULL on legacy corroboration rows (#192 T3
        backfill not yet landed). NULLs are excluded from the distinct
        count rather than treated as a single anonymous session — that
        avoids letting one un-attributed re-ingest masquerade as
        cross-session reuse.

        ``max_n`` caps the result set; the SQL does the limiting so a
        large store doesn't materialize the full candidate list.
        """
        limit_clause = f"LIMIT {int(max_n)}" if max_n is not None else ""
        cur = self._conn.execute(
            f"""
            SELECT b.* FROM beliefs b
            JOIN (
                SELECT belief_id,
                       COUNT(*) AS n_corr,
                       COUNT(DISTINCT session_id) AS n_sess
                FROM belief_corroborations
                GROUP BY belief_id
            ) bc ON bc.belief_id = b.id
            WHERE b.retention_class = 'snapshot'
              AND bc.n_corr >= ?
              AND bc.n_sess >= ?
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                  WHERE e.dst = b.id AND e.type = 'CONTRADICTS'
              )
            ORDER BY b.created_at ASC
            {limit_clause}
            """,
            (int(min_corroborations), int(min_sessions)),
        )
        return [_row_to_belief(r) for r in cur.fetchall()]

    def set_retention_class(self, belief_id: str, retention_class: str) -> None:
        """Targeted update of a belief's ``retention_class``.

        Phase-3 promotion writes through this rather than ``update_belief``
        because ``update_belief`` rewrites the full row (including FTS5 +
        entity index) and does not currently carry ``retention_class`` in
        its UPDATE clause. A focused setter avoids the broader surgery
        and keeps the write atomic.
        """
        if retention_class not in RETENTION_CLASSES:
            raise ValueError(
                f"Unknown retention_class {retention_class!r}. "
                f"Must be one of {sorted(RETENTION_CLASSES)}"
            )
        self._conn.execute(
            "UPDATE beliefs SET retention_class = ? WHERE id = ?",
            (retention_class, belief_id),
        )
        self._bump_belief_version(belief_id)
        self._conn.commit()
        self._fire_invalidation()

    def count_beliefs_by_type(self) -> dict[str, int]:
        """Return a mapping of belief type → count across all beliefs."""
        cur = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM beliefs GROUP BY type ORDER BY type"
        )
        return {str(r["type"]): int(r["n"]) for r in cur.fetchall()}

    # --- Auditor queries (used by aelf health) ---------------------------

    def count_orphan_edges(self) -> int:
        """Edges whose `src` or `dst` no longer exists in `beliefs`.

        Should always be zero in a healthy store: `delete_belief` cascades
        to incident edges. A nonzero result indicates a foreign-key invariant
        violation (or a partial write the v1.0 schema's lack of FK enforcement
        let through).
        """
        cur = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM edges e
            WHERE NOT EXISTS (SELECT 1 FROM beliefs WHERE id = e.src)
               OR NOT EXISTS (SELECT 1 FROM beliefs WHERE id = e.dst)
            """
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def count_fts_rows(self) -> int:
        """Row count of the `beliefs_fts` virtual table.

        Should equal `count_beliefs()` in a healthy store. Drift indicates
        an FTS5 / `beliefs` write was not properly mirrored — usually a
        crash mid-`update_belief` or a manual schema edit.
        """
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM beliefs_fts")
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def list_locked_contradicts_pairs(self) -> list[tuple[str, str]]:
        """All `(a_id, b_id)` pairs where both beliefs are locked AND a
        `CONTRADICTS` edge exists between them in either direction.

        Pairs are deduplicated and ordered by `(min(a,b), max(a,b))` so the
        result is deterministic. The contradiction tie-breaker (v1.0.1)
        should resolve these via SUPERSEDES; remaining pairs indicate
        unresolved contradictions a reviewer or `aelf resolve` should act on.
        """
        cur = self._conn.execute(
            """
            SELECT DISTINCT
                MIN(e.src, e.dst) AS a,
                MAX(e.src, e.dst) AS b
            FROM edges e
            JOIN beliefs ba ON ba.id = e.src
            JOIN beliefs bb ON bb.id = e.dst
            WHERE e.type = 'CONTRADICTS'
              AND ba.lock_level != 'none'
              AND bb.lock_level != 'none'
            ORDER BY a, b
            """
        )
        return [(str(r["a"]), str(r["b"])) for r in cur.fetchall()]

    def count_edges_by_type(self) -> dict[str, int]:
        """`{edge_type: count}` for every edge type in the store.

        Used by `aelf health` informational output and the v1.1.0 auditor
        feedback-coverage metric.
        """
        cur = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM edges GROUP BY type"
        )
        return {str(r["type"]): int(r["n"]) for r in cur.fetchall()}

    # --- Edge CRUD --------------------------------------------------------

    def insert_edge(self, e: Edge) -> None:
        self._conn.execute(
            "INSERT INTO edges (src, dst, type, weight, anchor_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (e.src, e.dst, e.type, e.weight, e.anchor_text),
        )
        self._bump_edge_version(e.src, e.dst, e.type)
        self._conn.commit()
        self._fire_invalidation()

    def get_edge(self, src: str, dst: str, type_: str) -> Edge | None:
        cur = self._conn.execute(
            "SELECT * FROM edges WHERE src = ? AND dst = ? AND type = ?",
            (src, dst, type_),
        )
        row = cur.fetchone()
        return _row_to_edge(row) if row else None

    def update_edge(self, e: Edge) -> None:
        self._conn.execute(
            "UPDATE edges SET weight = ?, anchor_text = ? "
            "WHERE src = ? AND dst = ? AND type = ?",
            (e.weight, e.anchor_text, e.src, e.dst, e.type),
        )
        self._bump_edge_version(e.src, e.dst, e.type)
        self._conn.commit()
        self._fire_invalidation()

    def delete_edge(self, src: str, dst: str, type_: str) -> None:
        self._conn.execute(
            "DELETE FROM edges WHERE src = ? AND dst = ? AND type = ?",
            (src, dst, type_),
        )
        self._conn.commit()
        self._fire_invalidation()

    def edges_from(self, src: str) -> list[Edge]:
        cur = self._conn.execute(
            "SELECT * FROM edges WHERE src = ?", (src,)
        )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def edges_for_beliefs(self, belief_ids: list[str]) -> list[Edge]:
        """Batched edge fetch for clustering (#436).

        Returns every edge whose `src` OR `dst` is in `belief_ids` —
        the candidate-induced subgraph plus its boundary. The clusterer
        filters down to the candidate-induced subgraph (both endpoints
        in the candidate set); the boundary edges come along for free
        because the SQL is one read.

        Empty input → empty list (no SQL).
        """
        if not belief_ids:
            return []
        ph = ",".join("?" * len(belief_ids))
        params = tuple(belief_ids) + tuple(belief_ids)
        cur = self._conn.execute(
            f"SELECT * FROM edges WHERE src IN ({ph}) OR dst IN ({ph})",
            params,
        )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def edges_to(self, dst: str) -> list[Edge]:
        """Return every edge whose `dst` is `dst`. Symmetric companion
        to `edges_from`. Used by the edge-type-keyed rerank pass
        (#421) to detect marker edges (e.g., POTENTIALLY_STALE)
        targeting a surfaced belief.
        """
        cur = self._conn.execute(
            "SELECT * FROM edges WHERE dst = ?", (dst,)
        )
        return [_row_to_edge(r) for r in cur.fetchall()]

    def iter_all_edges(self) -> Iterator[Edge]:
        """Stream every edge in the store. Ordering is insertion order
        (sqlite ROWID). Used by graph-wide builders such as the signed
        Laplacian eigenbasis (#149)."""
        cur = self._conn.execute("SELECT * FROM edges")
        for r in cur:
            yield _row_to_edge(r)

    def iter_incoming_anchor_text(self) -> Iterable[tuple[str, str]]:
        """Yield `(dst_belief_id, anchor_text)` for every edge whose
        `anchor_text` is non-NULL and non-empty, ordered by
        `(dst, src, type)` for deterministic iteration.

        Used by `aelfrice.bm25.BM25Index.build` to construct each
        belief's BM25F augmented document — the belief's own content
        plus the concatenation of its incoming edges' anchor text
        (replicated by a fixed weight). v1.2 stores already populate
        `Edge.anchor_text` via the commit-ingest hook and triple
        extractor; pre-v1.2 rows have NULL and contribute nothing.

        One row per edge (not per belief); callers group by `dst`.
        """
        cur = self._conn.execute(
            "SELECT dst, anchor_text FROM edges "
            "WHERE anchor_text IS NOT NULL AND anchor_text != '' "
            "ORDER BY dst ASC, src ASC, type ASC"
        )
        for row in cur.fetchall():
            yield (str(row["dst"]), str(row["anchor_text"]))

    def list_beliefs_for_indexing(self) -> list[tuple[str, str]]:
        """Return `[(belief_id, content)]` for every belief, ordered by
        `belief_id ASC` for deterministic indexing.

        Used by `aelfrice.bm25.BM25Index.build` to walk the corpus once
        and assemble the (n_docs × n_terms) sparse term-frequency
        matrix. The id-ASC order is the canonical row order of the
        index — `BM25Index.belief_ids[i]` lines up with row `i` of
        `tf` and entry `i` of `dl`.
        """
        cur = self._conn.execute(
            "SELECT id, content FROM beliefs ORDER BY id ASC"
        )
        return [(str(r["id"]), str(r["content"])) for r in cur.fetchall()]

    # --- Setr: propagate_valence -----------------------------------------

    def propagate_valence(
        self,
        src_id: str,
        valence: float,
        max_hops: int = 3,
        min_threshold: float = 0.05,
    ) -> dict[str, float]:
        """Propagate a valence signal outward through edges, attenuated by
        broker confidence (alpha / (alpha + beta) of intermediate beliefs).

        BFS over outbound edges. At each hop the carried magnitude is
        multiplied by:
            EDGE_VALENCE[edge.type] * broker_confidence(dst)
        Stops when |carried| < min_threshold or hop count exceeds max_hops.

        Returns: dict mapping touched belief_id -> sum of applied deltas.
        The src_id itself is NOT included in the returned map (it's the
        source, not a recipient).
        """
        applied: dict[str, float] = {}
        # Frontier entries: (belief_id, magnitude_carried_into_it, hops_taken)
        # The source contributes its outbound edges at hop 1.
        frontier: list[tuple[str, float, int]] = [(src_id, valence, 0)]
        visited: set[str] = {src_id}

        while frontier:
            next_frontier: list[tuple[str, float, int]] = []
            for current_id, carried, hops in frontier:
                if hops >= max_hops:
                    continue
                if abs(carried) < min_threshold:
                    continue
                for edge in self.edges_from(current_id):
                    multiplier = EDGE_VALENCE.get(edge.type, 0.0)
                    if multiplier == 0.0:
                        continue
                    dst = self.get_belief(edge.dst)
                    if dst is None:
                        continue
                    denom = dst.alpha + dst.beta
                    broker = (dst.alpha / denom) if denom > 0 else 0.0
                    delta = carried * multiplier * broker
                    if abs(delta) < min_threshold:
                        continue
                    applied[edge.dst] = applied.get(edge.dst, 0.0) + delta
                    if edge.dst not in visited:
                        visited.add(edge.dst)
                        next_frontier.append((edge.dst, delta, hops + 1))
            frontier = next_frontier

        return applied

    # --- Onboard sessions -------------------------------------------------

    def insert_onboard_session(self, s: OnboardSession) -> None:
        self._conn.execute(
            """
            INSERT INTO onboard_sessions (
                session_id, repo_path, state, candidates_json,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                s.session_id, s.repo_path, s.state, s.candidates_json,
                s.created_at, s.completed_at,
            ),
        )
        self._conn.commit()

    def get_onboard_session(self, session_id: str) -> OnboardSession | None:
        cur = self._conn.execute(
            "SELECT * FROM onboard_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return _row_to_onboard_session(row) if row else None

    def complete_onboard_session(
        self, session_id: str, completed_at: str,
    ) -> bool:
        """Mark a pending session completed. Returns True if a row was
        updated, False if no pending session with that id existed.

        Idempotent: re-completing an already-completed session updates
        `completed_at` to the new timestamp but the state stays
        `completed`. Caller is responsible for ensuring the session
        wasn't already completed if that matters to it.
        """
        cur = self._conn.execute(
            """
            UPDATE onboard_sessions
            SET state = ?, completed_at = ?
            WHERE session_id = ?
            """,
            (ONBOARD_STATE_COMPLETED, completed_at, session_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def count_onboard_sessions(self, state: str | None = None) -> int:
        if state is None:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM onboard_sessions"
            )
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM onboard_sessions WHERE state = ?",
                (state,),
            )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def list_pending_onboard_sessions(self) -> list[OnboardSession]:
        """All sessions in `pending` state, oldest first.

        Used by the polymorphic onboard handshake to expose `aelf:onboard()`
        with no args as a status call: "what sessions are awaiting host
        classification?".
        """
        cur = self._conn.execute(
            """
            SELECT * FROM onboard_sessions
            WHERE state = ?
            ORDER BY created_at ASC, session_id ASC
            """,
            (ONBOARD_STATE_PENDING,),
        )
        return [_row_to_onboard_session(r) for r in cur.fetchall()]

    def create_session(
        self,
        model: str | None = None,
        project_context: str | None = None,
    ) -> Session:
        """Open an ingest session and persist it to the sessions table.

        Returns a Session handle whose `id` callers pass to ingest_turn /
        ingest_jsonl as `session_id` to tag every belief inserted under
        the session. Pair with `complete_session(id)` at the end of a
        logical group; orphaned sessions are harmless.

        Per the open question in docs/ingest_enrichment.md the session
        row is written immediately rather than lazily on first belief
        insert; the rare empty-session row is left to future GC.
        """
        s = Session(
            id=secrets.token_hex(16),
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            model=model,
            project_context=project_context,
        )
        self._conn.execute(
            """
            INSERT INTO sessions (id, started_at, completed_at, model, project_context)
            VALUES (?, ?, ?, ?, ?)
            """,
            (s.id, s.started_at, s.completed_at, s.model, s.project_context),
        )
        self._conn.commit()
        return s

    def complete_session(self, session_id: str) -> None:
        """Stamp the session row with completed_at. Idempotent.

        A second call on an already-completed session refreshes the
        timestamp rather than failing. Calls on unknown ids are silent
        no-ops (the UPDATE matches zero rows) — this matches the
        spec's idempotency requirement.
        """
        self._conn.execute(
            "UPDATE sessions SET completed_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = cur.fetchone()
        return _row_to_session(row) if row else None
