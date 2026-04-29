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

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Final, Iterable, Iterator

from aelfrice.models import (
    CORROBORATION_SOURCE_TYPES,
    EDGE_VALENCE,
    INGEST_SOURCE_KINDS,
    ONBOARD_STATE_COMPLETED,
    ONBOARD_STATE_PENDING,
    Belief,
    Edge,
    FeedbackEvent,
    OnboardSession,
    Session,
)
from aelfrice.ulid import ulid

# --- Schema ---------------------------------------------------------------

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS beliefs (
        id                  TEXT PRIMARY KEY,
        content             TEXT NOT NULL,
        content_hash        TEXT NOT NULL,
        alpha               REAL NOT NULL,
        beta                REAL NOT NULL,
        type                TEXT NOT NULL,
        lock_level          TEXT NOT NULL,
        locked_at           TEXT,
        demotion_pressure   INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL,
        last_retrieved_at   TEXT,
        session_id          TEXT,
        origin              TEXT NOT NULL DEFAULT 'unknown'
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

# v1.0 -> v1.2 column additions. Each ALTER runs after _SCHEMA. ALTERs
# are idempotent: a duplicate-column OperationalError on a v1.2-fresh
# DB is caught and ignored. The CREATE INDEX after the ALTERs
# references a column that exists only post-migration; placing it here
# (rather than in _SCHEMA) is what lets a v1.0 store open at all.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
    "ALTER TABLE edges ADD COLUMN anchor_text TEXT",
    "ALTER TABLE beliefs ADD COLUMN origin TEXT NOT NULL DEFAULT 'unknown'",
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
        # v2.0 #205 ingest_log version-vector backfill. Same shape
        # as #204 but for the parallel-write log table. Idempotent.
        self._maybe_backfill_log_version_vectors()

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

    def list_belief_ids(self) -> list[str]:
        """All belief ids in insertion-time order. Used by the v1.3
        entity-index backfill to walk every existing belief once."""
        cur = self._conn.execute("SELECT id FROM beliefs ORDER BY id ASC")
        return [str(r["id"]) for r in cur.fetchall()]

    # --- Belief CRUD ------------------------------------------------------

    def insert_belief(self, b: Belief) -> None:
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, demotion_pressure,
                created_at, last_retrieved_at, session_id, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                b.id, b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.session_id, b.origin,
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

    # --- Belief corroborations (v1.5+, #190) -----------------------------

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

    # --- Bulk helpers (used by tests / future modules) -------------------

    def insert_beliefs(self, beliefs: Iterable[Belief]) -> None:
        for b in beliefs:
            self.insert_belief(b)

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            self.insert_edge(e)

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
