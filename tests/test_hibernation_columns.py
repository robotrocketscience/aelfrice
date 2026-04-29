"""Storage round-trip for the v2.0 #196 hibernation columns.

Behavior (when to set hibernation_score, predicate evaluator,
sweeper integration) is deferred to a follow-up issue. This file
locks in the storage shape ratified in
docs/substrate_decision.md § Decision asks #3 + #4: both columns
nullable, `None` means active, `activation_condition` is JSON-
encoded TEXT.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


def _mk_belief(
    bid: str = "b1",
    *,
    hibernation_score: float | None = None,
    activation_condition: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content="the sky is blue",
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-29T00:00:00Z",
        last_retrieved_at=None,
        hibernation_score=hibernation_score,
        activation_condition=activation_condition,
    )


def test_default_hibernation_columns_are_null() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief())
    got = s.get_belief("b1")
    assert got is not None
    assert got.hibernation_score is None
    assert got.activation_condition is None


def test_hibernation_columns_round_trip() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(
        _mk_belief(
            hibernation_score=0.42,
            activation_condition='{"kind": "retrieval_count_gte", "n": 3}',
        )
    )
    got = s.get_belief("b1")
    assert got is not None
    assert got.hibernation_score == 0.42
    assert got.activation_condition == (
        '{"kind": "retrieval_count_gte", "n": 3}'
    )


def test_hibernation_score_accepts_zero_and_one() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("low", hibernation_score=0.0))
    s.insert_belief(_mk_belief("high", hibernation_score=1.0))
    assert s.get_belief("low").hibernation_score == 0.0  # type: ignore[union-attr]
    assert s.get_belief("high").hibernation_score == 1.0  # type: ignore[union-attr]


def test_get_belief_by_content_hash_returns_hibernation_fields() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(
        _mk_belief(hibernation_score=0.7, activation_condition="{}")
    )
    got = s.get_belief_by_content_hash("h_b1")
    assert got is not None
    assert got.hibernation_score == 0.7
    assert got.activation_condition == "{}"


def test_legacy_db_without_hibernation_columns_migrates(tmp_path: Path) -> None:
    """A pre-#196 DB (only the v1.2 column set) must gain the new columns
    on next open. Idempotent — second open is a no-op. Existing rows
    return NULL for both."""
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
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
            last_retrieved_at   TEXT,
            session_id          TEXT,
            origin              TEXT NOT NULL DEFAULT 'unknown'
        )
        """
    )
    conn.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
        "type, lock_level, demotion_pressure, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "old1",
            "pre-#196 row",
            "h_old1",
            1.0,
            1.0,
            BELIEF_FACTUAL,
            LOCK_NONE,
            0,
            "2026-04-01T00:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    # Open via MemoryStore — migration runs.
    s = MemoryStore(str(db_path))
    cols = {
        r[1]
        for r in s._conn.execute("PRAGMA table_info(beliefs)").fetchall()  # noqa: SLF001
    }
    assert "hibernation_score" in cols
    assert "activation_condition" in cols

    legacy = s.get_belief("old1")
    assert legacy is not None
    assert legacy.hibernation_score is None
    assert legacy.activation_condition is None

    # New row writes both fields after migration.
    s.insert_belief(
        _mk_belief("new1", hibernation_score=0.5, activation_condition="{}")
    )
    fresh = s.get_belief("new1")
    assert fresh is not None
    assert fresh.hibernation_score == 0.5
    assert fresh.activation_condition == "{}"
    s.close()

    # Re-open: migration is idempotent.
    s2 = MemoryStore(str(db_path))
    again = s2.get_belief("new1")
    assert again is not None
    assert again.hibernation_score == 0.5
    s2.close()
