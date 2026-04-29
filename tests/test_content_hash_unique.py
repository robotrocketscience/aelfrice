"""Tests for UNIQUE(content_hash) on beliefs table (#219).

Verifies that fresh stores enforce the constraint at the DDL level and
that the migration applies it to existing stores via the table-swap pass.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore, SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belief(bid: str, content_hash: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=content_hash,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# ---------------------------------------------------------------------------
# Fresh store: UNIQUE constraint present in DDL
# ---------------------------------------------------------------------------


def test_fresh_store_rejects_duplicate_content_hash() -> None:
    """A fresh store rejects a raw INSERT with a duplicate content_hash."""
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_belief("id-001", "hash-aaa"))
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(  # type: ignore[reportPrivateUsage]
                "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
                "type, lock_level, demotion_pressure, created_at, origin) "
                "VALUES (?, ?, ?, 1.0, 1.0, 'factual', 'none', 0, "
                "'2026-04-28T00:00:00Z', 'unknown')",
                ("id-002", "different content", "hash-aaa"),
            )
    finally:
        store.close()


def test_fresh_store_unique_constraint_in_sqlite_master() -> None:
    """sqlite_master DDL for beliefs includes UNIQUE on content_hash."""
    store = MemoryStore(":memory:")
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='beliefs'"
        )
        row = cur.fetchone()
        assert row is not None
        ddl = str(row["sql"])
        assert "UNIQUE" in ddl
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Migration: UNIQUE applied to existing store
# ---------------------------------------------------------------------------


def test_existing_store_gets_unique_after_migration(tmp_path: Path) -> None:
    """After opening an old-format store, UNIQUE is present in sqlite_master."""
    # Import the consolidation test helper to seed a DB without UNIQUE.
    from tests.test_content_hash_consolidation import _seed_duplicates

    db = str(tmp_path / "legacy.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='beliefs'"
        )
        row = cur.fetchone()
        assert row is not None
        ddl = str(row["sql"])
        assert "UNIQUE" in ddl
    finally:
        store.close()


def test_unique_schema_meta_marker_set(tmp_path: Path) -> None:
    """SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED is stamped after migration."""
    from tests.test_content_hash_consolidation import _seed_duplicates

    db = str(tmp_path / "legacy.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        marker = store.get_schema_meta(SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED)
        assert marker is not None
        assert marker != ""
    finally:
        store.close()


def test_unique_enforced_after_migration(tmp_path: Path) -> None:
    """After migration, a raw INSERT with duplicate content_hash is rejected."""
    from tests.test_content_hash_consolidation import _seed_duplicates

    db = str(tmp_path / "legacy.db")
    _seed_duplicates(db)
    store = MemoryStore(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(  # type: ignore[reportPrivateUsage]
                "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
                "type, lock_level, demotion_pressure, created_at, origin) "
                "VALUES ('new-999', 'x', 'hash-sky', 1.0, 1.0, "
                "'factual', 'none', 0, '2026-04-28T00:00:00Z', 'unknown')"
            )
    finally:
        store.close()
