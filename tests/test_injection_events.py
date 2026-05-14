"""Tests for the #779 ``injection_events`` store API.

Layer 1 of the close-the-loop relevance-signal infrastructure: the
``record_injection_event`` / ``list_pending_injection_events`` /
``update_injection_referenced`` triad. Layer 2 (detection) and Layer 3
(sweeper integration into UPS) ship in their own test modules.
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
        demotion_pressure=0,
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


# --- list_pending_injection_events ------------------------------------

def test_list_pending_returns_only_unscored_rows() -> None:
    s = _store_with_belief()
    e1 = s.record_injection_event(
        session_id="s", turn_id="t1", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["m"],
    )
    e2 = s.record_injection_event(
        session_id="s", turn_id="t1", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["m"],
    )
    s.update_injection_referenced(e1, referenced=1, referenced_at="y")

    pending = s.list_pending_injection_events("s")
    assert [r[0] for r in pending] == [e2]


def test_list_pending_filters_by_session() -> None:
    s = _store_with_belief()
    s.record_injection_event(
        session_id="s1", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    s.record_injection_event(
        session_id="s2", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    assert len(s.list_pending_injection_events("s1")) == 1
    assert len(s.list_pending_injection_events("s2")) == 1
    assert s.list_pending_injection_events("nonexistent") == []


def test_list_pending_before_turn_id_slices_prior_turns() -> None:
    """``turn_id`` shape is ``{utc-compact-ts}-{hex4}`` — lexicographic
    sort is chronological. The sweeper passes the current turn's id to
    only score prior turns."""
    s = _store_with_belief()
    old = s.record_injection_event(
        session_id="s", turn_id="20260514T000000000000Z-aaaa",
        belief_id="B1", injected_at="x", source="ups",
        active_consumers=[],
    )
    same = s.record_injection_event(
        session_id="s", turn_id="20260514T000100000000Z-bbbb",
        belief_id="B1", injected_at="x", source="ups",
        active_consumers=[],
    )
    cutoff = "20260514T000100000000Z-bbbb"
    pending = s.list_pending_injection_events(
        "s", before_turn_id=cutoff,
    )
    assert [r[0] for r in pending] == [old]
    assert same not in [r[0] for r in pending]


def test_list_pending_returns_decoded_consumers() -> None:
    s = _store_with_belief()
    s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups",
        active_consumers=["meta:a", "meta:b"],
    )
    pending = s.list_pending_injection_events("s")
    assert len(pending) == 1
    _id, turn_id, belief_id, injected_at, source, consumers = pending[0]
    assert consumers == ["meta:a", "meta:b"]


def test_list_pending_deterministic_order() -> None:
    """Same insert sequence on two stores → same returned id list."""
    def play() -> list[int]:
        s = _store_with_belief()
        ids: list[int] = []
        for i in range(5):
            ids.append(s.record_injection_event(
                session_id="s", turn_id=f"t{i:02d}",
                belief_id="B1", injected_at="x", source="ups",
                active_consumers=[],
            ))
        return [r[0] for r in s.list_pending_injection_events("s")]
    assert play() == play()


def test_list_pending_respects_limit() -> None:
    s = _store_with_belief()
    for i in range(10):
        s.record_injection_event(
            session_id="s", turn_id=f"t{i:02d}",
            belief_id="B1", injected_at="x", source="ups",
            active_consumers=[],
        )
    assert len(s.list_pending_injection_events("s", limit=3)) == 3


# --- update_injection_referenced --------------------------------------

def test_update_referenced_first_call_returns_true() -> None:
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    assert s.update_injection_referenced(
        rowid, referenced=1, referenced_at="2026-05-14T00:01:00+00:00",
    ) is True
    row = s._conn.execute(
        "SELECT referenced, referenced_at FROM injection_events "
        "WHERE id = ?", (rowid,),
    ).fetchone()
    assert row["referenced"] == 1
    assert row["referenced_at"] == "2026-05-14T00:01:00+00:00"


def test_update_referenced_idempotent_on_second_call() -> None:
    """Already-scored rows don't get re-stamped; the second update
    returns False and leaves the prior score intact."""
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    assert s.update_injection_referenced(
        rowid, referenced=1, referenced_at="t1",
    ) is True
    assert s.update_injection_referenced(
        rowid, referenced=0, referenced_at="t2",
    ) is False
    row = s._conn.execute(
        "SELECT referenced, referenced_at FROM injection_events "
        "WHERE id = ?", (rowid,),
    ).fetchone()
    assert row["referenced"] == 1
    assert row["referenced_at"] == "t1"


def test_update_referenced_rejects_bad_value() -> None:
    s = _store_with_belief()
    rowid = s.record_injection_event(
        session_id="s", turn_id="t", belief_id="B1",
        injected_at="x", source="ups", active_consumers=[],
    )
    with pytest.raises(ValueError):
        s.update_injection_referenced(
            rowid, referenced=2, referenced_at="x",
        )
    with pytest.raises(ValueError):
        s.update_injection_referenced(
            rowid, referenced=-1, referenced_at="x",
        )


def test_update_referenced_unknown_id_returns_false() -> None:
    s = _store_with_belief()
    assert s.update_injection_referenced(
        999999, referenced=1, referenced_at="x",
    ) is False
