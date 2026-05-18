"""Acceptance tests for v3.2 #858 — project_context field on beliefs.

Defect 3 of #855: beliefs from one within-repo project context keep
surfacing when the user is working in a re-scoped version of the same
project. Cross-repo isolation already exists via db_path()'s git-common-
dir routing; this column is what catches the within-repo re-scope case.

Covers in this commit (foundation, schema only):

* `project_context` column exists on fresh DB.
* `project_context` column is added by migration on a pre-#858 DB
  opened for the first time (idempotency check).
* `idx_beliefs_project_context` index is present after migration.
* Default value is '' (the empty string), preserving pre-migration
  retrieval semantics.

Write-side stamping, hook filter, and the env-var resolver are
exercised by later commits in the same PR.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.store import MemoryStore


def _column_names(db_path: Path, table: str) -> list[str]:
    con = sqlite3.connect(str(db_path))
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    finally:
        con.close()


def _index_names(db_path: Path, table: str) -> list[str]:
    con = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name=?", (table,),
        ).fetchall()]
    finally:
        con.close()


def test_project_context_column_present_on_fresh_db(tmp_path: Path) -> None:
    """A freshly initialized store carries project_context with default ''."""
    p = tmp_path / "fresh.db"
    store = MemoryStore(str(p))
    try:
        cols = _column_names(p, "beliefs")
        assert "project_context" in cols, (
            f"project_context missing from fresh beliefs schema: {cols!r}"
        )
    finally:
        store.close()


def test_project_context_index_present_on_fresh_db(tmp_path: Path) -> None:
    """idx_beliefs_project_context is created by _POST_MIGRATION_INDEXES."""
    p = tmp_path / "fresh.db"
    store = MemoryStore(str(p))
    try:
        idxs = _index_names(p, "beliefs")
        assert "idx_beliefs_project_context" in idxs, (
            f"idx_beliefs_project_context missing: {idxs!r}"
        )
    finally:
        store.close()


def test_project_context_default_empty_string(tmp_path: Path) -> None:
    """Default value of project_context is '' (empty string), not NULL.

    Empty string is the "cross-context — no filter" marker. NULL would
    add a three-valued-logic wrinkle to the hook filter; the NOT NULL
    DEFAULT '' lets the filter stay in plain SQL.
    """
    p = tmp_path / "fresh.db"
    store = MemoryStore(str(p))
    try:
        # Direct SQL insert that omits project_context — the column
        # default should fire.
        store._conn.execute(  # type: ignore[attr-defined]
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, created_at, last_retrieved_at
            ) VALUES ('b1', 'hello', 'h1', 1.0, 1.0, 'factual',
                      'none', NULL, '2026-05-18T00:00:00Z', NULL)
            """,
        )
        row = store._conn.execute(  # type: ignore[attr-defined]
            "SELECT project_context FROM beliefs WHERE id='b1'",
        ).fetchone()
        assert row[0] == "", f"expected '' default, got {row[0]!r}"
    finally:
        store.close()


def test_migration_idempotent_on_reopen(tmp_path: Path) -> None:
    """Opening the same DB twice does not error or duplicate the column.

    The migration loop catches "duplicate column name" on re-runs; this
    test pins that contract for project_context specifically.
    """
    p = tmp_path / "twice.db"
    store1 = MemoryStore(str(p))
    store1.close()
    # Second open re-runs the migration tuple. Must not raise.
    store2 = MemoryStore(str(p))
    try:
        cols = _column_names(p, "beliefs")
        # Single column, not duplicated.
        assert cols.count("project_context") == 1
    finally:
        store2.close()


def test_migration_adds_column_to_legacy_db(tmp_path: Path) -> None:
    """A pre-#858 DB (project_context-less beliefs table) gains the column.

    Synthesizes a beliefs table missing the column, opens via
    MemoryStore, and confirms the migration runs.
    """
    p = tmp_path / "legacy.db"
    con = sqlite3.connect(str(p))
    try:
        # Minimal pre-#858 beliefs schema. Just enough columns for the
        # migration to see the table and skip the create-from-scratch
        # branch in _SCHEMA.
        con.execute(
            """
            CREATE TABLE beliefs (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                type TEXT NOT NULL,
                lock_level TEXT NOT NULL,
                locked_at TEXT,
                created_at TEXT NOT NULL,
                last_retrieved_at TEXT
            )
            """,
        )
        con.commit()
    finally:
        con.close()

    # Pre-condition: column not present.
    cols_before = _column_names(p, "beliefs")
    assert "project_context" not in cols_before

    # Open through MemoryStore — runs the migration tuple.
    store = MemoryStore(str(p))
    try:
        cols_after = _column_names(p, "beliefs")
        assert "project_context" in cols_after, (
            f"migration did not add project_context: {cols_after!r}"
        )
    finally:
        store.close()
