"""Tests for _maybe_consolidate_content_hash_duplicates (#219).

Verifies that pre-existing duplicate content_hash rows are merged onto
the canonical (oldest) belief, with correct alpha/beta summing, FK
rewriting, origin/lock precedence, and idempotence.

All tests use sqlite3 directly to seed duplicate rows bypassing the
dedup guard, then open a MemoryStore so the migration fires.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aelfrice.store import (
    MemoryStore,
    SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_duplicates(path: str) -> None:
    """Seed two duplicate content_hash rows directly via sqlite3.

    Bypasses MemoryStore so the dedup guard never runs. Seeds the
    minimum schema required by the migration pass.
    """
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    # Use the same schema as MemoryStore but without UNIQUE on content_hash.
    con.executescript("""
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
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts
        USING fts5(id UNINDEXED, content, tokenize='porter unicode61');
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id   TEXT NOT NULL,
            valence     REAL NOT NULL,
            source      TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS belief_corroborations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id         TEXT    NOT NULL,
            ingested_at       TEXT    NOT NULL,
            source_type       TEXT    NOT NULL,
            session_id        TEXT,
            source_path_hash  TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            src         TEXT NOT NULL,
            dst         TEXT NOT NULL,
            type        TEXT NOT NULL,
            weight      REAL NOT NULL,
            anchor_text TEXT,
            PRIMARY KEY (src, dst, type)
        );
        CREATE TABLE IF NOT EXISTS belief_entities (
            belief_id    TEXT NOT NULL,
            entity_lower TEXT NOT NULL,
            entity_raw   TEXT NOT NULL,
            kind         TEXT NOT NULL,
            span_start   INTEGER NOT NULL,
            span_end     INTEGER NOT NULL,
            PRIMARY KEY (belief_id, entity_lower, span_start)
        );
        CREATE TABLE IF NOT EXISTS belief_versions (
            belief_id TEXT NOT NULL,
            scope_id  TEXT NOT NULL,
            counter   INTEGER NOT NULL,
            PRIMARY KEY (belief_id, scope_id)
        );
        CREATE TABLE IF NOT EXISTS edge_versions (
            src       TEXT NOT NULL,
            dst       TEXT NOT NULL,
            type      TEXT NOT NULL,
            scope_id  TEXT NOT NULL,
            counter   INTEGER NOT NULL,
            PRIMARY KEY (src, dst, type, scope_id)
        );
        CREATE TABLE IF NOT EXISTS onboard_sessions (
            session_id      TEXT PRIMARY KEY,
            repo_path       TEXT NOT NULL,
            state           TEXT NOT NULL,
            candidates_json TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            completed_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id              TEXT PRIMARY KEY,
            started_at      TEXT NOT NULL,
            completed_at    TEXT,
            model           TEXT,
            project_context TEXT
        );
        CREATE TABLE IF NOT EXISTS log_versions (
            log_id   TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            counter  INTEGER NOT NULL,
            PRIMARY KEY (log_id, scope_id)
        );
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
        );
    """)
    # Canonical (older created_at).
    con.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, type, "
        "lock_level, demotion_pressure, created_at, origin) "
        "VALUES ('can-001', 'Sky is blue.', 'hash-sky', 2.0, 3.0, 'factual', "
        "'none', 0, '2026-01-01T00:00:00Z', 'agent_inferred')"
    )
    con.execute(
        "INSERT INTO beliefs_fts (id, content) VALUES ('can-001', 'Sky is blue.')"
    )
    # Duplicate (newer created_at).
    con.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, type, "
        "lock_level, demotion_pressure, created_at, origin) "
        "VALUES ('dup-001', 'Sky is blue.', 'hash-sky', 1.0, 1.0, 'factual', "
        "'user', 0, '2026-02-01T00:00:00Z', 'user_stated')"
    )
    con.execute(
        "INSERT INTO beliefs_fts (id, content) VALUES ('dup-001', 'Sky is blue.')"
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Alpha/beta sum
# ---------------------------------------------------------------------------


def test_alpha_beta_summed_across_group(tmp_path: Path) -> None:
    """After consolidation canonical alpha = sum(group alphas)."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        canonical = store.get_belief("can-001")
        assert canonical is not None
        assert canonical.alpha == pytest.approx(3.0)  # 2.0 + 1.0
        assert canonical.beta == pytest.approx(4.0)   # 3.0 + 1.0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Duplicate row removed
# ---------------------------------------------------------------------------


def test_duplicate_row_deleted(tmp_path: Path) -> None:
    """The duplicate belief row must not exist after consolidation."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        assert store.get_belief("dup-001") is None
    finally:
        store.close()


def test_only_canonical_row_survives(tmp_path: Path) -> None:
    """Exactly one belief row per content_hash after consolidation."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        ids = store.list_belief_ids()
        assert "can-001" in ids
        assert "dup-001" not in ids
    finally:
        store.close()


# ---------------------------------------------------------------------------
# FK rewrite: feedback_history
# ---------------------------------------------------------------------------


def test_feedback_history_rewritten(tmp_path: Path) -> None:
    """feedback_history rows pointing at the dupe are rewritten to canonical."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    # Insert a feedback row pointing at the dupe before MemoryStore opens.
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO feedback_history (belief_id, valence, source, created_at) "
        "VALUES ('dup-001', 1.0, 'test', '2026-01-15T00:00:00Z')"
    )
    con.commit()
    con.close()

    store = MemoryStore(db)
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT belief_id FROM feedback_history"
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["belief_id"] == "can-001"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Synthetic corroboration row
# ---------------------------------------------------------------------------


def test_synthetic_corroboration_inserted(tmp_path: Path) -> None:
    """One consolidation_migration corroboration row per duplicate."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT source_type FROM belief_corroborations "
            "WHERE belief_id = 'can-001'"
        )
        rows = cur.fetchall()
        migration_rows = [r for r in rows if r["source_type"] == "consolidation_migration"]
        assert len(migration_rows) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_idempotent_second_open_is_noop(tmp_path: Path) -> None:
    """Opening the store a second time does not change belief count."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)

    store = MemoryStore(db)
    try:
        ids_first = set(store.list_belief_ids())
    finally:
        store.close()

    store2 = MemoryStore(db)
    try:
        ids_second = set(store2.list_belief_ids())
    finally:
        store2.close()

    assert ids_first == ids_second


def test_schema_meta_marker_set(tmp_path: Path) -> None:
    """schema_meta marker is stamped after consolidation."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        marker = store.get_schema_meta(SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE)
        assert marker is not None
        assert marker != ""
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Origin and lock precedence
# ---------------------------------------------------------------------------


def test_origin_precedence_user_beats_agent(tmp_path: Path) -> None:
    """user_stated beats agent_inferred in origin consolidation."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        canonical = store.get_belief("can-001")
        assert canonical is not None
        assert canonical.origin == "user_stated"
    finally:
        store.close()


def test_lock_level_precedence_user_wins(tmp_path: Path) -> None:
    """lock_level='user' on any group member propagates to canonical."""
    db = str(tmp_path / "mem.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        canonical = store.get_belief("can-001")
        assert canonical is not None
        assert canonical.lock_level == "user"
    finally:
        store.close()
