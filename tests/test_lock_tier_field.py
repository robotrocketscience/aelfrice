"""Acceptance tests for v3.7 #1016-B PR1 — the lock_tier field.

Covers the data-model foundation (no injection change yet):
* Schema: lock_tier column on a fresh DB and added to a pre-#1016 DB
  (existing rows → 'frozen', the no-silent-loss migration); idempotent.
* Belief dataclass default 'frozen'; round-trips through the store.
* Write-time validation: a malformed lock_tier raises ValueError
  (insert + update).
* CLI: `aelf lock --reference` sets the tier; default is frozen;
  re-locking with --reference demotes; `aelf locked` annotates
  reference locks.
"""
from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    DEFAULT_LOCK_TIER,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


def _lock(id_: str, content: str, *, tier: str = DEFAULT_LOCK_TIER) -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-hash",
        alpha=9.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at="2026-06-01T00:00:00+00:00",
        created_at="2026-06-01T00:00:00+00:00",
        last_retrieved_at=None,
        lock_tier=tier,
    )


# --- 1. Schema migration ----------------------------------------------


def test_fresh_db_has_lock_tier_column() -> None:
    store = MemoryStore(":memory:")
    try:
        cols = {
            r["name"]
            for r in store._conn.execute(
                "PRAGMA table_info(beliefs)"
            ).fetchall()
        }
        assert "lock_tier" in cols
    finally:
        store.close()


def test_migration_adds_lock_tier_to_existing_db(tmp_path: Path) -> None:
    """A pre-#1016 DB (no lock_tier) gains the column; existing locks
    default to 'frozen' — the no-silent-loss migration (#379/#1016-B)."""
    db_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        """
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            type TEXT NOT NULL,
            lock_level TEXT NOT NULL,
            locked_at TEXT,
            created_at TEXT NOT NULL,
            last_retrieved_at TEXT,
            session_id TEXT,
            origin TEXT NOT NULL DEFAULT 'unknown',
            hibernation_score REAL,
            activation_condition TEXT,
            retention_class TEXT NOT NULL DEFAULT 'unknown',
            valid_to TEXT
        )
        """
    )
    raw.execute(
        "INSERT INTO beliefs "
        "(id, content, content_hash, alpha, beta, type, lock_level, "
        "created_at, origin, retention_class) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old-lock", "a locked truth", "old-lock-hash",
         9.0, 0.5, "factual", "user",
         "2026-01-01T00:00:00+00:00", "user_stated", "fact"),
    )
    raw.commit()
    raw.close()

    store = MemoryStore(str(db_path))
    try:
        cols = {
            r["name"]
            for r in store._conn.execute(
                "PRAGMA table_info(beliefs)"
            ).fetchall()
        }
        assert "lock_tier" in cols
        got = store.get_belief("old-lock")
        assert got is not None
        assert got.lock_tier == LOCK_TIER_FROZEN
    finally:
        store.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "modern.db"
    MemoryStore(str(db_path)).close()
    MemoryStore(str(db_path)).close()  # second open must not raise


# --- 2. Dataclass default + round-trip --------------------------------


def test_belief_lock_tier_defaults_to_frozen() -> None:
    assert _lock("b1", "x").lock_tier == LOCK_TIER_FROZEN
    assert DEFAULT_LOCK_TIER == LOCK_TIER_FROZEN


def test_lock_tier_round_trips_through_store() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_lock("L1", "ref lock", tier=LOCK_TIER_REFERENCE))
        got = store.get_belief("L1")
        assert got is not None
        assert got.lock_tier == LOCK_TIER_REFERENCE
    finally:
        store.close()


# --- 3. Validation -----------------------------------------------------


def test_insert_belief_rejects_invalid_lock_tier() -> None:
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="invalid lock_tier"):
            store.insert_belief(_lock("bad", "x", tier="bogus"))
    finally:
        store.close()


def test_update_belief_rejects_invalid_lock_tier() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_lock("L1", "x"))
        got = store.get_belief("L1")
        assert got is not None
        bad = Belief(**{**got.__dict__, "lock_tier": "nope"})
        with pytest.raises(ValueError, match="invalid lock_tier"):
            store.update_belief(bad)
    finally:
        store.close()


# --- 4. CLI ------------------------------------------------------------


def _setup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "brain.db"))


def _locked_tier(db_env_id: str) -> str:
    from aelfrice.db_paths import db_path
    s = MemoryStore(str(db_path()))
    try:
        b = s.get_belief(db_env_id)
        return b.lock_tier if b is not None else ""
    finally:
        s.close()


def test_cli_lock_reference_sets_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.cli import main
    _setup_db(tmp_path, monkeypatch)
    out = io.StringIO()
    rc = main(["lock", "deploy via the merge-train label", "--reference"], out=out)
    assert rc == 0
    text = out.getvalue()
    assert "tier: reference" in text
    bid = text.split("locked: ")[1].split()[0].strip()
    assert _locked_tier(bid) == LOCK_TIER_REFERENCE


def test_cli_lock_defaults_to_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.cli import main
    _setup_db(tmp_path, monkeypatch)
    out = io.StringIO()
    rc = main(["lock", "the store lives under dot-git aelfrice"], out=out)
    assert rc == 0
    text = out.getvalue()
    assert "tier:" not in text  # no tier line when defaulting
    bid = text.split("locked: ")[1].split()[0].strip()
    assert _locked_tier(bid) == LOCK_TIER_FROZEN


def test_cli_relock_reference_demotes_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.cli import main
    _setup_db(tmp_path, monkeypatch)
    stmt = "always sign public-repo commits with the rrs key"
    main(["lock", stmt], out=io.StringIO())  # frozen
    out = io.StringIO()
    rc = main(["lock", stmt, "--reference"], out=out)  # demote
    assert rc == 0
    assert "tier: reference" in out.getvalue()


def test_cli_locked_annotates_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.cli import main
    _setup_db(tmp_path, monkeypatch)
    main(["lock", "frozen identity fact"], out=io.StringIO())
    main(["lock", "bounded reference fact", "--reference"], out=io.StringIO())
    listing = io.StringIO()
    rc = main(["locked"], out=listing)
    assert rc == 0
    text = listing.getvalue()
    assert "[reference]" in text
    # exactly one line carries the marker
    assert sum("[reference]" in ln for ln in text.splitlines()) == 1
