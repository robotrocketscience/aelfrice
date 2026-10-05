"""Tests for the #779 ``injection_events`` store API.

``record_injection_event`` writes one row per injected belief, and
exploration reads the table as its never-shown pool. The relevance
detector and sweeper that once scored these rows were removed in #1655.
"""
from __future__ import annotations

import json

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    RETENTION_FACT,
    Belief,
)
from aelfrice.store import MemoryStore


# --- Helpers ----------------------------------------------------------

def _mk_belief(bid: str, content: str = "x") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-14T00:00:00+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _store_with_belief(bid: str = "B1") -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief(bid))
    return s


# --- Schema presence (covers commit 1's migration) --------------------

def test_injection_events_table_present_on_fresh_store() -> None:
    s = MemoryStore(":memory:")
    rows = s._conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='injection_events'"
    ).fetchall()
    assert rows, "injection_events table missing"


def test_injection_events_columns_match_schema() -> None:
    s = MemoryStore(":memory:")
    cols = {r[1] for r in s._conn.execute(
        "PRAGMA table_info(injection_events)"
    ).fetchall()}
    assert cols == {
        "id", "session_id", "turn_id", "belief_id", "injected_at",
        "source", "active_consumers", "referenced", "referenced_at",
    }


def test_injection_events_indexes_present() -> None:
    s = MemoryStore(":memory:")
    idxs = {r[1] for r in s._conn.execute(
        "SELECT * FROM sqlite_master "
        "WHERE type='index' AND tbl_name='injection_events'"
    ).fetchall()}
    assert "idx_injection_events_session_turn" in idxs
    assert "idx_injection_events_belief" in idxs
    assert "idx_injection_events_pending" in idxs


# --- record_injection_event ------------------------------------------

def test_record_event_writes_one_row() -> None:
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s1",
        turn_id="t1",
        belief_id="B1",
        injected_at="2026-05-14T00:00:01+00:00",
        source="ups",
        active_consumers=["meta:retrieval.temporal_half_life_seconds"],
    )
    assert rowid > 0
    row = s._conn.execute(
        "SELECT session_id, turn_id, belief_id, source, "
        "active_consumers, referenced, referenced_at "
        "FROM injection_events WHERE id = ?", (rowid,)
    ).fetchone()
    assert dict(row) == {
        "session_id": "s1",
        "turn_id": "t1",
        "belief_id": "B1",
        "source": "ups",
        "active_consumers": (
            '["meta:retrieval.temporal_half_life_seconds"]'
        ),
        "referenced": None,
        "referenced_at": None,
    }


def test_record_event_canonical_consumer_order() -> None:
    """Two records with the same consumers in different orders produce
    byte-identical ``active_consumers`` column values (determinism)."""
    s = _store_with_belief()
    r1 = s.record_injection_event(
        session_id="s", turn_id="t1", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["meta:b", "meta:a", "meta:c"],
    )
    r2 = s.record_injection_event(
        session_id="s", turn_id="t2", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["meta:c", "meta:a", "meta:b"],
    )
    v1 = s._conn.execute(
        "SELECT active_consumers FROM injection_events WHERE id = ?",
        (r1,),
    ).fetchone()["active_consumers"]
    v2 = s._conn.execute(
        "SELECT active_consumers FROM injection_events WHERE id = ?",
        (r2,),
    ).fetchone()["active_consumers"]
    assert v1 == v2 == '["meta:a","meta:b","meta:c"]'


def test_record_event_dedupes_repeated_consumers() -> None:
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["meta:a", "meta:a", "meta:b"],
    )
    raw = s._conn.execute(
        "SELECT active_consumers FROM injection_events WHERE id = ?",
        (rowid,),
    ).fetchone()["active_consumers"]
    assert json.loads(raw) == ["meta:a", "meta:b"]


def test_record_event_empty_consumers_default() -> None:
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    raw = s._conn.execute(
        "SELECT active_consumers FROM injection_events WHERE id = ?",
        (rowid,),
    ).fetchone()["active_consumers"]
    assert raw == "[]"


def test_record_event_rejects_empty_source() -> None:
    s = _store_with_belief()
    with pytest.raises(ValueError):
        s.record_injection_event(
            session_id="s", turn_id="t", belief_id="B1",
            injected_at="x", source="", active_consumers=[],
        )


def test_record_event_cascade_on_belief_delete() -> None:
    """FK ON DELETE CASCADE: deleting the belief removes its events."""
    s = _store_with_belief()
    s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    assert s._conn.execute(
        "SELECT COUNT(*) FROM injection_events"
    ).fetchone()[0] == 1
    s._conn.execute("PRAGMA foreign_keys = ON")
    s._conn.execute("DELETE FROM beliefs WHERE id = 'B1'")
    s._conn.commit()
    assert s._conn.execute(
        "SELECT COUNT(*) FROM injection_events"
    ).fetchone()[0] == 0
