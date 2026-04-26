"""SQLite-backed store for Beliefs and Edges, with FTS5 full-text search.

v0.1.0 storage layer. Stdlib-only (sqlite3). WAL journal mode for concurrent
reads. FTS5 virtual table mirrors `beliefs.content` for keyword retrieval.

Future-Rust-port boundary: graph-walk math (propagate_valence next commit;
decay_sweep in v0.2.0) are the candidates for native re-implementation if/
when Python bandwidth becomes a bottleneck. CRUD + FTS5 stay in Python.

demotion_pressure note: this column is BOTH written and read end-to-end here.
v2.0 had a bug where it was persisted but never surfaced; the test suite
locks that behavior in from day one (see tests/test_demotion_pressure.py).
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

from aelfrice.models import (
    EDGE_VALENCE,
    Belief,
    Edge,
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
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
)


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
        """Full-row update. demotion_pressure included -- v2.0 bug fix."""
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
        """FTS5 keyword search over belief content. Ranked by bm25."""
        cur = self._conn.execute(
            """
            SELECT b.* FROM beliefs b
            JOIN beliefs_fts f ON f.id = b.id
            WHERE beliefs_fts MATCH ?
            ORDER BY bm25(beliefs_fts)
            LIMIT ?
            """,
            (query, limit),
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

    # --- Bulk helpers (used by tests / future modules) -------------------

    def insert_beliefs(self, beliefs: Iterable[Belief]) -> None:
        for b in beliefs:
            self.insert_belief(b)

    def insert_edges(self, edges: Iterable[Edge]) -> None:
        for e in edges:
            self.insert_edge(e)
