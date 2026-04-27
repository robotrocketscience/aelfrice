"""Smoke tests for the feedback_history schema and MemoryStore helpers.

The audit table records every apply_feedback call so a project's
feedback regime is recoverable after the fact. Pre-commit #5 in scope.

Tests are split per the deterministic-atomic-short policy: one
property per test, each :memory: store, all run in milliseconds.
"""
from __future__ import annotations

from aelfrice.models import FeedbackEvent
from aelfrice.store import MemoryStore


def test_feedback_history_starts_empty() -> None:
    s = MemoryStore(":memory:")
    assert s.list_feedback_events() == []
    assert s.count_feedback_events() == 0


def test_insert_feedback_event_returns_positive_rowid() -> None:
    s = MemoryStore(":memory:")
    rowid = s.insert_feedback_event(
        belief_id="b1",
        valence=1.0,
        source="user",
        created_at="2026-04-26T18:00:00Z",
    )
    assert rowid > 0


def test_insert_two_events_yields_two_distinct_rowids() -> None:
    s = MemoryStore(":memory:")
    r1 = s.insert_feedback_event("b1", 1.0, "user", "2026-04-26T18:00:00Z")
    r2 = s.insert_feedback_event("b1", -1.0, "system", "2026-04-26T18:01:00Z")
    assert r1 != r2


def test_count_feedback_events_total_after_three_inserts() -> None:
    s = MemoryStore(":memory:")
    s.insert_feedback_event("b1", 1.0, "user", "2026-04-26T18:00:00Z")
    s.insert_feedback_event("b1", -1.0, "system", "2026-04-26T18:01:00Z")
    s.insert_feedback_event("b2", 1.0, "user", "2026-04-26T18:02:00Z")
    assert s.count_feedback_events() == 3


def test_count_feedback_events_per_belief_filter() -> None:
    s = MemoryStore(":memory:")
    s.insert_feedback_event("b1", 1.0, "user", "2026-04-26T18:00:00Z")
    s.insert_feedback_event("b1", -1.0, "system", "2026-04-26T18:01:00Z")
    s.insert_feedback_event("b2", 1.0, "user", "2026-04-26T18:02:00Z")
    assert s.count_feedback_events("b1") == 2
    assert s.count_feedback_events("b2") == 1
    assert s.count_feedback_events("nonexistent") == 0


def test_list_feedback_events_round_trip_returns_typed_object() -> None:
    s = MemoryStore(":memory:")
    s.insert_feedback_event("b1", 0.7, "user", "2026-04-26T18:00:00Z")
    events = s.list_feedback_events()
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, FeedbackEvent)
    assert e.belief_id == "b1"
    assert e.valence == 0.7
    assert e.source == "user"
    assert e.created_at == "2026-04-26T18:00:00Z"


def test_list_feedback_events_orders_by_id_desc() -> None:
    s = MemoryStore(":memory:")
    r1 = s.insert_feedback_event("b1", 1.0, "user", "2026-04-26T18:00:00Z")
    r2 = s.insert_feedback_event("b1", -1.0, "user", "2026-04-26T18:01:00Z")
    r3 = s.insert_feedback_event("b2", 1.0, "user", "2026-04-26T18:02:00Z")
    events = s.list_feedback_events()
    ids = [e.id for e in events]
    assert ids == [r3, r2, r1]


def test_list_feedback_events_limit_caps_result_size() -> None:
    s = MemoryStore(":memory:")
    for i in range(7):
        s.insert_feedback_event("b1", 1.0, "user", f"2026-04-26T18:0{i}:00Z")
    events = s.list_feedback_events(limit=3)
    assert len(events) == 3


def test_list_feedback_events_belief_filter_excludes_others() -> None:
    s = MemoryStore(":memory:")
    s.insert_feedback_event("b1", 1.0, "user", "2026-04-26T18:00:00Z")
    s.insert_feedback_event("b2", 1.0, "user", "2026-04-26T18:01:00Z")
    s.insert_feedback_event("b1", -1.0, "user", "2026-04-26T18:02:00Z")
    events = s.list_feedback_events(belief_id="b1")
    belief_ids = {e.belief_id for e in events}
    assert belief_ids == {"b1"}
