"""Tests for #762: belief_corroborations.belief_id INTEGER -> TEXT retype.

Some legacy DBs were created when the released v1.5.0 package shipped
the column as `INTEGER`. `beliefs.id` is `TEXT PRIMARY KEY`, so the FK
silently misses on every insert of a hex belief id and `aelf lock` on
near-duplicate text raises `FOREIGN KEY constraint failed`.

These tests seed a legacy-shaped DB directly via sqlite3, then open it
with `MemoryStore` and assert (a) the column has been retyped to TEXT,
(b) the marker is stamped, (c) the previously-failing corroboration
insert path now succeeds, and (d) re-opening the store is a no-op.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import (
    SCHEMA_META_CORROBORATIONS_BELIEF_ID_RETYPED,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_legacy_integer_schema(path: str) -> None:
    """Create a DB with the broken `belief_id INTEGER` FK column.

    Mirrors the shape of stores produced by the v1.5.0 release that
    shipped the INTEGER column. The `beliefs` and `schema_meta` tables
    are minimal — just enough for MemoryStore to open the DB and run
    its migration ladder.
    """
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
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
            last_retrieved_at   TEXT,
            session_id          TEXT,
            origin              TEXT NOT NULL DEFAULT 'unknown'
        );
        CREATE VIRTUAL TABLE beliefs_fts
            USING fts5(id UNINDEXED, content, tokenize='porter unicode61');
        CREATE TABLE schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        -- The broken legacy shape: belief_id INTEGER instead of TEXT.
        CREATE TABLE belief_corroborations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id         INTEGER NOT NULL
                REFERENCES beliefs(id) ON DELETE CASCADE,
            ingested_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_type       TEXT NOT NULL,
            session_id        TEXT,
            source_path_hash  TEXT
        );
        CREATE INDEX idx_belief_corroborations_belief_id
            ON belief_corroborations(belief_id);
    """)
    con.commit()
    con.close()


def _belief_id_column_type(store: MemoryStore) -> str:
    """Return the SQL type of `belief_corroborations.belief_id`."""
    cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
        "PRAGMA table_info(belief_corroborations)"
    )
    for row in cur.fetchall():
        if str(row["name"]) == "belief_id":
            return str(row["type"]).upper()
    raise AssertionError("belief_id column missing")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_legacy_integer_schema_is_retyped_on_open(tmp_path: Path) -> None:
    """Opening a legacy-INTEGER DB retypes the column to TEXT."""
    db = str(tmp_path / "legacy.db")
    _seed_legacy_integer_schema(db)

    # Pre-condition: legacy shape is INTEGER.
    pre = sqlite3.connect(db)
    rows = pre.execute(
        "PRAGMA table_info(belief_corroborations)"
    ).fetchall()
    pre.close()
    cols = {r[1]: r[2].upper() for r in rows}
    assert cols.get("belief_id") == "INTEGER", (
        "pre-condition failed: legacy schema did not have INTEGER column"
    )

    store = MemoryStore(db)
    try:
        assert _belief_id_column_type(store) == "TEXT"
        assert store.get_schema_meta(
            SCHEMA_META_CORROBORATIONS_BELIEF_ID_RETYPED
        ) is not None
    finally:
        store.close()


def test_retype_is_idempotent_across_opens(tmp_path: Path) -> None:
    """Re-opening a retyped store does not re-run the migration."""
    db = str(tmp_path / "legacy.db")
    _seed_legacy_integer_schema(db)

    store1 = MemoryStore(db)
    try:
        marker1 = store1.get_schema_meta(
            SCHEMA_META_CORROBORATIONS_BELIEF_ID_RETYPED
        )
        assert marker1 is not None
    finally:
        store1.close()

    store2 = MemoryStore(db)
    try:
        marker2 = store2.get_schema_meta(
            SCHEMA_META_CORROBORATIONS_BELIEF_ID_RETYPED
        )
        assert marker2 == marker1, (
            "marker rewritten on re-open: migration not idempotent"
        )
        assert _belief_id_column_type(store2) == "TEXT"
    finally:
        store2.close()


def test_fresh_store_stamps_marker_without_swap(tmp_path: Path) -> None:
    """A fresh store gets the canonical schema and the marker."""
    db = str(tmp_path / "fresh.db")
    store = MemoryStore(db)
    try:
        assert _belief_id_column_type(store) == "TEXT"
        assert store.get_schema_meta(
            SCHEMA_META_CORROBORATIONS_BELIEF_ID_RETYPED
        ) is not None
    finally:
        store.close()


def test_record_corroboration_succeeds_after_retype(tmp_path: Path) -> None:
    """After retype, `record_corroboration` with a TEXT id no longer
    raises `FOREIGN KEY constraint failed` — the #762 repro.
    """
    db = str(tmp_path / "legacy.db")
    _seed_legacy_integer_schema(db)

    store = MemoryStore(db)
    try:
        b = Belief(
            id="4fadaddd9b67b614",  # hex id, like the real repro
            content="example belief",
            content_hash="ch_example",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-13T00:00:00+00:00",
            last_retrieved_at=None,
        )
        store.insert_belief(b)

        # The previously-failing path: write a corroboration row by id.
        store.record_corroboration(
            b.id,
            source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
            session_id="sess-1",
            source_path_hash=None,
        )

        # Round-trip: the row landed and the count matches.
        assert store.count_corroborations(b.id) == 1
    finally:
        store.close()


def test_existing_corroboration_rows_carry_over(tmp_path: Path) -> None:
    """Rows present before the swap survive it.

    Hex belief ids stored under INTEGER affinity are kept as TEXT
    by SQLite's affinity rules (digits-only literals would be coerced,
    but hex strings are not valid INTEGER literals). The straight copy
    in the migration preserves them.
    """
    db = str(tmp_path / "legacy.db")
    _seed_legacy_integer_schema(db)

    # Seed one belief and one corroboration row directly under the
    # legacy schema. PRAGMA foreign_keys=OFF lets us insert despite
    # the broken FK affinity (mirrors the affinity-coercion bug
    # that lets some writes land before the runtime check trips).
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
        "type, lock_level, demotion_pressure, created_at, origin) "
        "VALUES (?, ?, ?, 1.0, 1.0, 'factual', 'none', 0, ?, "
        "'agent_remembered')",
        ("4fadaddd9b67b614", "example", "ch_x", "2026-05-13T00:00:00Z"),
    )
    con.execute(
        "INSERT INTO belief_corroborations "
        "(belief_id, ingested_at, source_type, session_id, source_path_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "4fadaddd9b67b614",
            "2026-05-13T00:00:01Z",
            CORROBORATION_SOURCE_COMMIT_INGEST,
            "sess-pre",
            None,
        ),
    )
    con.commit()
    con.close()

    store = MemoryStore(db)
    try:
        assert _belief_id_column_type(store) == "TEXT"
        # The pre-existing row carried over with its TEXT belief_id.
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT belief_id, source_type FROM belief_corroborations "
            "WHERE session_id = ?",
            ("sess-pre",),
        )
        row = cur.fetchone()
        assert row is not None
        assert str(row["belief_id"]) == "4fadaddd9b67b614"
        assert str(row["source_type"]) == CORROBORATION_SOURCE_COMMIT_INGEST
    finally:
        store.close()
