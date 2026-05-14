"""v1.0 -> v1.2 forward-compat: an old store opens, migrates, round-trips.

Builds a v1.0-shaped SQLite file inline (the legacy schema before
session_id / anchor_text / DERIVED_FROM landed) and verifies that
opening it with the v1.2 MemoryStore picks up the new columns via
ALTER TABLE without losing any existing rows.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.models import EDGE_CITES, EDGE_DERIVED_FROM, Edge, LOCK_NONE
from aelfrice.store import MemoryStore


_V1_0_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE beliefs (
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
    CREATE TABLE edges (
        src     TEXT NOT NULL,
        dst     TEXT NOT NULL,
        type    TEXT NOT NULL,
        weight  REAL NOT NULL,
        PRIMARY KEY (src, dst, type)
    )
    """,
)


def _seed_v1_0_store(path: Path) -> None:
    """Create a v1.0-shaped SQLite file with one belief and one edge."""
    conn = sqlite3.connect(str(path))
    try:
        for stmt in _V1_0_SCHEMA:
            conn.execute(stmt)
        conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, created_at, last_retrieved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("legacy1", "old belief", "h", 1.0, 1.0, "factual",
             LOCK_NONE, None, "2025-01-01T00:00:00+00:00", None),
        )
        conn.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, created_at, last_retrieved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("legacy2", "another", "h2", 1.0, 1.0, "factual",
             LOCK_NONE, None, "2025-01-02T00:00:00+00:00", None),
        )
        conn.execute(
            "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
            ("legacy1", "legacy2", EDGE_CITES, 1.0),
        )
        conn.commit()
    finally:
        conn.close()


def test_v1_0_store_opens_with_v1_2_reader(tmp_path: Path) -> None:
    db = tmp_path / "v1_0.db"
    _seed_v1_0_store(db)
    # Confirm the legacy file lacks the new columns before migration.
    raw = sqlite3.connect(str(db))
    raw.row_factory = sqlite3.Row
    cols = {r["name"] for r in raw.execute("PRAGMA table_info(beliefs)").fetchall()}
    assert "session_id" not in cols
    cols = {r["name"] for r in raw.execute("PRAGMA table_info(edges)").fetchall()}
    assert "anchor_text" not in cols
    raw.close()
    # Open with v1.2 store; ALTER TABLE migration runs.
    store = MemoryStore(str(db))
    try:
        legacy = store.get_belief("legacy1")
        assert legacy is not None
        assert legacy.content == "old belief"
        assert legacy.session_id is None  # nullable column added clean
        edge = store.get_edge("legacy1", "legacy2", EDGE_CITES)
        assert edge is not None
        assert edge.anchor_text is None
    finally:
        store.close()


def test_v1_0_store_accepts_new_writes_after_migration(tmp_path: Path) -> None:
    db = tmp_path / "v1_0.db"
    _seed_v1_0_store(db)
    store = MemoryStore(str(db))
    try:
        store.insert_edge(Edge(
            src="legacy1", dst="legacy2", type=EDGE_DERIVED_FROM,
            weight=1.0, anchor_text="ported edge",
        ))
        got = store.get_edge("legacy1", "legacy2", EDGE_DERIVED_FROM)
        assert got is not None
        assert got.anchor_text == "ported edge"
    finally:
        store.close()


def test_migration_idempotent_on_re_open(tmp_path: Path) -> None:
    db = tmp_path / "v1_0.db"
    _seed_v1_0_store(db)
    # First open: ALTER TABLE adds columns.
    s1 = MemoryStore(str(db))
    s1.close()
    # Second open: ALTER TABLE catches duplicate-column and proceeds.
    s2 = MemoryStore(str(db))
    try:
        b = s2.get_belief("legacy1")
        assert b is not None
    finally:
        s2.close()
