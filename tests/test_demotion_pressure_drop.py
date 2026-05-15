"""Regression for #833: a DB with a `demotion_pressure` column must
remain readable after the v3.1.1+ migration drops it.

Before #833's fix, the v3.1.0 reader had an unconditional
`row["demotion_pressure"]` in `_row_to_belief`. The migration in
`_MIGRATION_STATEMENTS` drops the column on first open by a post-#814
build, leaving the v3.1.0 reader to crash with
`IndexError: No item with that key` on any subsequent `get_belief()`.

This test seeds a DB that still has the column (the v3.1.0 schema
shape), opens it with the current `MemoryStore` so the DROP COLUMN
migration runs, and verifies `get_belief()` round-trips a row whose
`Belief` no longer carries `demotion_pressure`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE
from aelfrice.store import MemoryStore


def _seed_v3_1_0_store(path: Path) -> None:
    """Create a DB with the full pre-DROP belief schema and one row.

    Column set mirrors the v3.1.0 release: every column that any
    `ALTER TABLE ADD COLUMN` entry in `_MIGRATION_STATEMENTS` could
    have created, plus `demotion_pressure` which the post-#814
    migration drops.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE beliefs (
                id                    TEXT PRIMARY KEY,
                content               TEXT NOT NULL,
                content_hash          TEXT NOT NULL,
                alpha                 REAL NOT NULL,
                beta                  REAL NOT NULL,
                type                  TEXT NOT NULL,
                lock_level            TEXT NOT NULL,
                locked_at             TEXT,
                demotion_pressure     INTEGER NOT NULL DEFAULT 0,
                created_at            TEXT NOT NULL,
                last_retrieved_at     TEXT,
                session_id            TEXT,
                origin                TEXT NOT NULL DEFAULT 'unknown',
                corroboration_count   INTEGER NOT NULL DEFAULT 0,
                hibernation_score     REAL,
                activation_condition  TEXT,
                retention_class       TEXT,
                valid_to              TEXT,
                scope                 TEXT NOT NULL DEFAULT 'project'
            )
            """
        )
        conn.execute(
            "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
            "type, lock_level, demotion_pressure, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "b833",
                "demotion-pressure-skew regression",
                "h_b833",
                1.0,
                1.0,
                BELIEF_FACTUAL,
                LOCK_NONE,
                0,
                "2026-05-14T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_demotion_pressure_column_dropped_on_open(tmp_path: Path) -> None:
    db = tmp_path / "v3_1_0.db"
    _seed_v3_1_0_store(db)
    # Sanity: the seeded DB has the column.
    raw = sqlite3.connect(str(db))
    cols = {r[1] for r in raw.execute("PRAGMA table_info(beliefs)").fetchall()}
    assert "demotion_pressure" in cols
    raw.close()

    s = MemoryStore(str(db))
    try:
        # Migration must have dropped the column.
        cols = {
            r[1]
            for r in s._conn.execute(  # noqa: SLF001
                "PRAGMA table_info(beliefs)"
            ).fetchall()
        }
        assert "demotion_pressure" not in cols

        # And get_belief must round-trip without IndexError.
        got = s.get_belief("b833")
        assert got is not None
        assert got.content == "demotion-pressure-skew regression"
    finally:
        s.close()


def test_drop_idempotent_on_reopen(tmp_path: Path) -> None:
    db = tmp_path / "v3_1_0.db"
    _seed_v3_1_0_store(db)
    MemoryStore(str(db)).close()
    # Second open: column is already gone; migration tolerates it.
    s = MemoryStore(str(db))
    try:
        got = s.get_belief("b833")
        assert got is not None
    finally:
        s.close()
