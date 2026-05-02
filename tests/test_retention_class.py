"""Tests for the #290 phase-1 retention_class schema axis.

Covers:
* default retention_class on insert is 'unknown' (the migration value)
* explicit retention_class round-trips via insert + read
* invalid retention_class is rejected at the python boundary
* the SQLite CHECK constraint rejects garbage on fresh stores
* a pre-#290 store opens cleanly: ALTER TABLE backfills 'unknown'
  on every existing row, and reads still work
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    RETENTION_CLASSES,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_TRANSIENT,
    RETENTION_UNKNOWN,
    Belief,
)
from aelfrice.store import MemoryStore


def _mk_belief(
    bid: str = "b1",
    *,
    retention_class: str = RETENTION_UNKNOWN,
) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-02T00:00:00Z",
        last_retrieved_at=None,
        retention_class=retention_class,
    )


# ---- enum surface -------------------------------------------------------


def test_retention_classes_has_four_values() -> None:
    assert RETENTION_CLASSES == frozenset({
        RETENTION_FACT,
        RETENTION_SNAPSHOT,
        RETENTION_TRANSIENT,
        RETENTION_UNKNOWN,
    })


def test_belief_default_retention_class_is_unknown() -> None:
    b = Belief(
        id="x", content="c", content_hash="h", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        demotion_pressure=0, created_at="2026-05-02T00:00:00Z",
        last_retrieved_at=None,
    )
    # Default exists so callers that don't yet know about retention_class
    # land on the migration-default value, not on something like None.
    assert b.retention_class == RETENTION_UNKNOWN


# ---- store round-trip ---------------------------------------------------


def test_retention_class_round_trips_for_each_value(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    cases = {
        "f": RETENTION_FACT,
        "s": RETENTION_SNAPSHOT,
        "t": RETENTION_TRANSIENT,
        "u": RETENTION_UNKNOWN,
    }
    for bid, rc in cases.items():
        store.insert_belief(_mk_belief(bid, retention_class=rc))
    for bid, rc in cases.items():
        b = store.get_belief(bid)
        assert b is not None
        assert b.retention_class == rc


def test_invalid_retention_class_rejected_at_python_boundary(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    bad = _mk_belief("bad", retention_class="garbage")
    with pytest.raises(ValueError, match="invalid retention_class"):
        store.insert_belief(bad)


def test_check_constraint_rejects_direct_sql_garbage(
    tmp_path: Path,
) -> None:
    """Bypassing the python boundary still trips the CHECK on a fresh
    store. Migrated stores don't have the CHECK (ALTER TABLE limitation)
    — that path is covered by the python-side validator."""
    db_path = tmp_path / "m.db"
    MemoryStore(str(db_path))  # initialize schema with CHECK
    raw = sqlite3.connect(str(db_path))
    raw.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            """
            INSERT INTO beliefs (
                id, content, content_hash, alpha, beta, type,
                lock_level, locked_at, demotion_pressure,
                created_at, last_retrieved_at, retention_class
            ) VALUES (
                'raw', 'c', 'h_raw', 1.0, 1.0, 'factual',
                'none', NULL, 0, '2026-05-02T00:00:00Z', NULL,
                'totally_invalid'
            )
            """
        )
    raw.close()


# ---- migration path -----------------------------------------------------


def test_pre_290_store_opens_cleanly_and_backfills_unknown(
    tmp_path: Path,
) -> None:
    """Build a store WITHOUT the retention_class column (simulating a
    pre-#290 DB), insert a row, then re-open with the current
    MemoryStore. The ALTER TABLE in _MIGRATIONS must add the column
    with default 'unknown' on every existing row, and reads must
    still produce a valid Belief."""
    db_path = tmp_path / "m.db"

    raw = sqlite3.connect(str(db_path))
    # Mirror enough of the v1.5 schema to be openable but without
    # retention_class. Other columns the row reader needs are
    # included explicitly.
    raw.execute(
        """
        CREATE TABLE beliefs (
            id                  TEXT PRIMARY KEY,
            content             TEXT NOT NULL,
            content_hash        TEXT NOT NULL UNIQUE,
            alpha               REAL NOT NULL,
            beta                REAL NOT NULL,
            type                TEXT NOT NULL,
            lock_level          TEXT NOT NULL,
            locked_at           TEXT,
            demotion_pressure   INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT NOT NULL,
            last_retrieved_at   TEXT,
            session_id          TEXT,
            origin              TEXT NOT NULL DEFAULT 'unknown',
            hibernation_score   REAL,
            activation_condition TEXT
        )
        """
    )
    raw.execute(
        """
        INSERT INTO beliefs (
            id, content, content_hash, alpha, beta, type,
            lock_level, locked_at, demotion_pressure,
            created_at, last_retrieved_at, origin
        ) VALUES (
            'pre', 'old content', 'h_pre', 1.0, 1.0, 'factual',
            'none', NULL, 0, '2026-04-01T00:00:00Z', NULL, 'unknown'
        )
        """
    )
    raw.commit()
    raw.close()

    store = MemoryStore(str(db_path))
    cur = store._conn.execute(
        "SELECT retention_class FROM beliefs WHERE id = 'pre'"
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] == RETENTION_UNKNOWN

    b = store.get_belief("pre")
    assert b is not None
    assert b.retention_class == RETENTION_UNKNOWN

    fresh = _mk_belief("post", retention_class=RETENTION_FACT)
    store.insert_belief(fresh)
    got = store.get_belief("post")
    assert got is not None
    assert got.retention_class == RETENTION_FACT
