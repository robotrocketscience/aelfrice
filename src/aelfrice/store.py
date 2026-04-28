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

import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Iterable

from aelfrice.models import (
    EDGE_VALENCE,
    ONBOARD_STATE_COMPLETED,
    ONBOARD_STATE_PENDING,
    Belief,
    Edge,
    FeedbackEvent,
    OnboardSession,
    Session,
)

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
        session_id          TEXT
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
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_belief ON feedback_history(belief_id)",
    "CREATE INDEX IF NOT EXISTS idx_onboard_state ON onboard_sessions(state)",
)

# v1.0 -> v1.2 column additions. Each ALTER runs after _SCHEMA. ALTERs
# are idempotent: a duplicate-column OperationalError on a v1.2-fresh
# DB is caught and ignored. The CREATE INDEX after the ALTERs
# references a column that exists only post-migration; placing it here
# (rather than in _SCHEMA) is what lets a v1.0 store open at all.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
    "ALTER TABLE edges ADD COLUMN anchor_text TEXT",
)

# Indexes that depend on migrated columns. Run after _MIGRATIONS so
# they see the post-ALTER schema.
_POST_MIGRATION_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_beliefs_session ON beliefs(session_id)",
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
        self._conn.commit()
        self._invalidation_callbacks: list[Callable[[], None]] = []

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

    # --- Belief CRUD ------------------------------------------------------

    def insert_belief(self, b: Belief) -> None:
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, demotion_pressure,
                created_at, last_retrieved_at, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                b.id, b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.session_id,
            ),
        )
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        self._conn.commit()
        self._fire_invalidation()

    def get_belief(self, belief_id: str) -> Belief | None:
        cur = self._conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
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
                session_id = ?
            WHERE id = ?
            """,
            (
                b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.session_id, b.id,
            ),
        )
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (b.id,))
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        self._conn.commit()
        self._fire_invalidation()

    def delete_belief(self, belief_id: str) -> None:
        self._conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (belief_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE src = ? OR dst = ?",
            (belief_id, belief_id),
        )
        self._conn.commit()
        self._fire_invalidation()

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
            SELECT b.* FROM beliefs b
            JOIN beliefs_fts f ON f.id = b.id
            WHERE beliefs_fts MATCH ?
            ORDER BY bm25(beliefs_fts)
            LIMIT ?
            """,
            (escaped, limit),
        )
        return [_row_to_belief(r) for r in cur.fetchall()]

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
            SELECT * FROM beliefs
            WHERE lock_level != 'none'
            ORDER BY locked_at DESC, id ASC
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
