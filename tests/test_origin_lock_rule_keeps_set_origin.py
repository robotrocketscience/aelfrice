"""#2013: the lock rule changes only a locked belief whose origin is `unknown`.

The origin backfill in `store.py` and the row conversion in `migrate.py`
give a locked belief whose origin is `unknown` the origin `user_stated`.
A locked belief whose origin is already set keeps that origin on both
paths. The tests drive a real store file, and each one deletes the
`origin_backfill_complete` marker first, so the next writable open runs
the backfill.
"""
from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import SCHEMA_META_ORIGIN_BACKFILL, MemoryStore

LOCKED_CORRECTED = "l0ckedc0rrected1"
LOCKED_INFERRED = "l0ckedinferred01"
LOCKED_UNKNOWN = "l0ckedunkn0wn001"

_SEEDED: dict[str, str] = {
    LOCKED_CORRECTED: ORIGIN_USER_CORRECTED,
    LOCKED_INFERRED: ORIGIN_AGENT_INFERRED,
    LOCKED_UNKNOWN: ORIGIN_UNKNOWN,
}


def _locked(bid: str, origin: str) -> Belief:
    return Belief(
        id=bid,
        content=f"locked belief {bid} for the #2013 lock rule test",
        content_hash=f"h-{bid}",
        alpha=9.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
        last_retrieved_at=None,
        origin=origin,
    )


def _raw_origins(db: Path) -> dict[str, str]:
    raw = sqlite3.connect(str(db))
    try:
        return dict(raw.execute("SELECT id, origin FROM beliefs").fetchall())
    finally:
        raw.close()


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """A store of locked beliefs with no backfill marker.

    The `unknown` row is inserted as `user_stated`, then set back to
    `unknown` with raw SQL, so the store looks like one from before the
    backfill. It shows that the lock rule still runs on each path.
    """
    db = tmp_path / "pre_backfill.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_locked(LOCKED_CORRECTED, ORIGIN_USER_CORRECTED))
        store.insert_belief(_locked(LOCKED_INFERRED, ORIGIN_AGENT_INFERRED))
        store.insert_belief(_locked(LOCKED_UNKNOWN, ORIGIN_USER_STATED))
    finally:
        store.close()
    raw = sqlite3.connect(str(db))
    try:
        raw.execute(
            "UPDATE beliefs SET origin = ? WHERE id = ?",
            (ORIGIN_UNKNOWN, LOCKED_UNKNOWN),
        )
        raw.execute(
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_ORIGIN_BACKFILL,),
        )
        raw.commit()
    finally:
        raw.close()
    # Confirms the seed, so a later pass proves the path did the work.
    assert _raw_origins(db) == _SEEDED
    return db


def test_backfill_keeps_a_set_origin_on_a_locked_belief(
    seeded_db: Path,
) -> None:
    MemoryStore(str(seeded_db)).close()
    origins = _raw_origins(seeded_db)
    assert origins[LOCKED_CORRECTED] == ORIGIN_USER_CORRECTED
    assert origins[LOCKED_INFERRED] == ORIGIN_AGENT_INFERRED
    assert origins[LOCKED_UNKNOWN] == ORIGIN_USER_STATED


def test_migrate_keeps_a_set_origin_on_a_locked_belief(
    seeded_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target" / "memory.db"
    target.parent.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(target))
    out = io.StringIO()
    code = main(
        ["migrate", "--from", str(seeded_db), "--apply", "--all", "--yes"],
        out=out,
    )
    assert code == 0, out.getvalue()
    # The source is opened read-only, so its rows are unchanged and only
    # the migrate conversion can set an origin on the copied rows.
    assert _raw_origins(seeded_db) == _SEEDED
    origins = _raw_origins(target)
    assert set(origins) == set(_SEEDED), out.getvalue()
    assert origins[LOCKED_CORRECTED] == ORIGIN_USER_CORRECTED
    assert origins[LOCKED_INFERRED] == ORIGIN_AGENT_INFERRED
    assert origins[LOCKED_UNKNOWN] == ORIGIN_USER_STATED
