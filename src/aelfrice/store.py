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

import sqlite3
from typing import Iterable

from aelfrice.models import (
    EDGE_VALENCE,
    Belief,
    Edge,
    FeedbackEvent,
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
        last_retrieved_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src     TEXT NOT NULL,
        dst     TEXT NOT NULL,
        type    TEXT NOT NULL,
        weight  REAL NOT NULL,
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
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_belief ON feedback_history(belief_id)",
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
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        src=row["src"],
        dst=row["dst"],
        type=row["type"],
        weight=row["weight"],
    )


def _row_to_feedback(row: sqlite3.Row) -> FeedbackEvent:
    return FeedbackEvent(
        id=row["id"],
        belief_id=row["belief_id"],
        valence=row["valence"],
        source=row["source"],
        created_at=row["created_at"],
    )


class Store:
    """SQLite store. Pass `:memory:` for tests, a path otherwise."""

    def __init__(self, path: str) -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        # WAL only meaningful on-disk; harmless on :memory:.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._conn.execute("PRAGMA foreign_keys=ON")
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Belief CRUD ------------------------------------------------------

    def insert_belief(self, b: Belief) -> None:
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, demotion_pressure,
                created_at, last_retrieved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                b.id, b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at,
            ),
        )
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        self._conn.commit()

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
                last_retrieved_at = ?
            WHERE id = ?
            """,
            (
                b.content, b.content_hash, b.alpha, b.beta, b.type,
                b.lock_level, b.locked_at, b.demotion_pressure,
                b.created_at, b.last_retrieved_at, b.id,
            ),
        )
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (b.id,))
        self._conn.execute(
            "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
            (b.id, b.content),
        )
        self._conn.commit()

    def delete_belief(self, belief_id: str) -> None:
        self._conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self._conn.execute("DELETE FROM beliefs_fts WHERE id = ?", (belief_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE src = ? OR dst = ?",
            (belief_id, belief_id),
        )
        self._conn.commit()

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

    # --- Edge CRUD --------------------------------------------------------

    def insert_edge(self, e: Edge) -> None:
        self._conn.execute(
            "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
            (e.src, e.dst, e.type, e.weight),
        )
        self._conn.commit()

    def get_edge(self, src: str, dst: str, type_: str) -> Edge | None:
        cur = self._conn.execute(
            "SELECT * FROM edges WHERE src = ? AND dst = ? AND type = ?",
            (src, dst, type_),
        )
        row = cur.fetchone()
        return _row_to_edge(row) if row else None

    def update_edge(self, e: Edge) -> None:
        self._conn.execute(
            "UPDATE edges SET weight = ? WHERE src = ? AND dst = ? AND type = ?",
            (e.weight, e.src, e.dst, e.type),
        )
        self._conn.commit()

    def delete_edge(self, src: str, dst: str, type_: str) -> None:
        self._conn.execute(
            "DELETE FROM edges WHERE src = ? AND dst = ? AND type = ?",
            (src, dst, type_),
        )
        self._conn.commit()

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
