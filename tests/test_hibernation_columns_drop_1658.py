"""#1658: the #196 hibernation columns are dropped on open.

`beliefs.hibernation_score` and `beliefs.activation_condition` were
added in v2.0 (#196) as the storage half of a hibernation lifecycle
that was never built. No code path wrote a non-NULL value. #1658 drops
both columns through two `ALTER TABLE ... DROP COLUMN` entries in
`_MIGRATIONS`, the same shape as the #814 `demotion_pressure` drop
(see tests/test_demotion_pressure_drop.py).

These tests seed a store with the pre-#1658 schema and non-NULL values
in both columns, so the drop has data to discard, then check that the
store opens, that the columns are gone, and that every other column of
the row survives the table rewrite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.store import _SCHEMA, MemoryStore

_DROPPED = ("hibernation_score", "activation_condition")


def _seed_pre_1658_store(path: Path) -> None:
    """Create a DB with the pre-#1658 belief schema and two rows.

    Both rows carry non-NULL values in the two hibernation columns and
    non-default values in the columns that must survive the rewrite.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE beliefs (
                id                    TEXT PRIMARY KEY,
                content               TEXT NOT NULL,
                content_hash          TEXT NOT NULL UNIQUE,
                alpha                 REAL NOT NULL,
                beta                  REAL NOT NULL,
                type                  TEXT NOT NULL,
                lock_level            TEXT NOT NULL,
                locked_at             TEXT,
                created_at            TEXT NOT NULL,
                last_retrieved_at     TEXT,
                session_id            TEXT,
                origin                TEXT NOT NULL DEFAULT 'unknown',
                hibernation_score     REAL,
                activation_condition  TEXT,
                retention_class       TEXT NOT NULL DEFAULT 'unknown',
                valid_to              TEXT,
                scope                 TEXT NOT NULL DEFAULT 'project',
                project_context       TEXT NOT NULL DEFAULT '',
                last_confirmed_at     TEXT,
                lock_tier             TEXT NOT NULL DEFAULT 'frozen',
                lock_expires_at       TEXT
            )
            """
        )
        rows = [
            (
                "b1658a", "the build uses uv for every python command",
                "h_b1658a", 4.5, 1.25, BELIEF_FACTUAL, LOCK_USER,
                "2026-05-01T00:00:00Z", "2026-04-30T00:00:00Z",
                "2026-05-02T00:00:00Z", "sess-1", "user_stated",
                0.42, '{"on": "next_retrieval"}',
                "fact", None, "global", "ctx-a",
                "2026-05-03T00:00:00Z", "reference",
                "2027-01-01T00:00:00Z",
            ),
            (
                "b1658b", "release notes live under the changelog folder",
                "h_b1658b", 1.0, 3.0, BELIEF_FACTUAL, "none",
                None, "2026-04-29T00:00:00Z",
                None, None, "agent_inferred",
                0.0, "{}",
                "snapshot", None, "project", "",
                None, "frozen", None,
            ),
        ]
        conn.executemany(
            "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
            "type, lock_level, locked_at, created_at, last_retrieved_at, "
            "session_id, origin, hibernation_score, activation_condition, "
            "retention_class, valid_to, scope, project_context, "
            "last_confirmed_at, lock_tier, lock_expires_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _belief_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(beliefs)")}


def test_seed_has_both_columns_with_values(tmp_path: Path) -> None:
    """Guard the guard: the drop tests below are vacuous if the seed
    never held the columns or held only NULLs."""
    db = tmp_path / "pre_1658.db"
    _seed_pre_1658_store(db)
    raw = sqlite3.connect(str(db))
    try:
        assert set(_DROPPED) <= _belief_columns(raw)
        n = raw.execute(
            "SELECT COUNT(hibernation_score), COUNT(activation_condition) "
            "FROM beliefs"
        ).fetchone()
        assert n == (2, 2)
    finally:
        raw.close()


def test_open_drops_both_columns(tmp_path: Path) -> None:
    db = tmp_path / "pre_1658.db"
    _seed_pre_1658_store(db)
    MemoryStore(str(db)).close()
    raw = sqlite3.connect(str(db))
    try:
        cols = _belief_columns(raw)
    finally:
        raw.close()
    for name in _DROPPED:
        assert name not in cols, name


def test_other_columns_survive_the_drop(tmp_path: Path) -> None:
    db = tmp_path / "pre_1658.db"
    _seed_pre_1658_store(db)
    s = MemoryStore(str(db))
    try:
        a = s.get_belief("b1658a")
        b = s.get_belief("b1658b")
    finally:
        s.close()
    assert a is not None and b is not None
    assert a.content == "the build uses uv for every python command"
    assert (a.alpha, a.beta) == (4.5, 1.25)
    assert a.lock_level == LOCK_USER
    assert a.locked_at == "2026-05-01T00:00:00Z"
    assert a.created_at == "2026-04-30T00:00:00Z"
    assert a.last_retrieved_at == "2026-05-02T00:00:00Z"
    assert a.session_id == "sess-1"
    assert a.origin == "user_stated"
    assert a.retention_class == "fact"
    assert a.scope == "global"
    assert a.project_context == "ctx-a"
    assert a.last_confirmed_at == "2026-05-03T00:00:00Z"
    assert a.lock_tier == "reference"
    assert a.lock_expires_at == "2027-01-01T00:00:00Z"
    assert b.content == "release notes live under the changelog folder"
    assert (b.alpha, b.beta) == (1.0, 3.0)
    assert b.origin == "agent_inferred"
    assert b.retention_class == "snapshot"


def test_drop_is_idempotent_on_reopen(tmp_path: Path) -> None:
    db = tmp_path / "pre_1658.db"
    _seed_pre_1658_store(db)
    MemoryStore(str(db)).close()
    s = MemoryStore(str(db))
    try:
        assert s.get_belief("b1658a") is not None
    finally:
        s.close()


def _schema_version(path: Path) -> int:
    raw = sqlite3.connect(str(path))
    try:
        return int(raw.execute("PRAGMA schema_version").fetchone()[0])
    finally:
        raw.close()


def test_reopen_leaves_the_schema_alone(tmp_path: Path) -> None:
    """A migrated store must not re-add and re-drop the columns on every
    open. With the old ADD COLUMN entries still in `_MIGRATIONS`, each
    open added both columns and the DROP entries removed them again,
    which rewrote `beliefs` twice per open and bumped `schema_version`
    by 4 each time."""
    db = tmp_path / "pre_1658.db"
    _seed_pre_1658_store(db)
    MemoryStore(str(db)).close()
    before = _schema_version(db)
    assert before > 0  # non-vacuous: the store has a schema
    MemoryStore(str(db)).close()
    MemoryStore(str(db)).close()
    assert _schema_version(db) == before


def test_create_ddl_never_has_the_columns() -> None:
    """The CREATE DDL alone must not create either column.

    A fresh `MemoryStore` cannot show this: the trailing DROP entries in
    `_MIGRATIONS` run on every open and would remove a column that the
    CREATE DDL had added back. So this runs `_SCHEMA` by itself, with no
    migrations, on a bare connection.
    """
    raw = sqlite3.connect(":memory:")
    try:
        for stmt in _SCHEMA:
            raw.execute(stmt)
        cols = _belief_columns(raw)
    finally:
        raw.close()
    assert "content" in cols  # non-vacuous: the beliefs table was created
    for name in _DROPPED:
        assert name not in cols, name


def test_belief_has_no_hibernation_fields() -> None:
    names = set(Belief.__dataclass_fields__)
    assert "content" in names
    for name in _DROPPED:
        assert name not in names, name
