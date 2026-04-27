"""Atomic CRUD tests for the onboard_sessions table.

One property per test, deterministic, in-memory store, no fixtures
beyond a fresh `Store(":memory:")`. The polymorphic onboard state
machine builds on these primitives in v0.6.0; this file only locks the
storage layer.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    ONBOARD_STATE_COMPLETED,
    ONBOARD_STATE_PENDING,
    OnboardSession,
)
from aelfrice.store import Store


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _session(
    *,
    session_id: str = "sess-1",
    state: str = ONBOARD_STATE_PENDING,
    completed_at: str | None = None,
) -> OnboardSession:
    return OnboardSession(
        session_id=session_id,
        repo_path="/tmp/repo",
        state=state,
        candidates_json='[{"text": "x", "source_meta": {}}]',
        created_at="2026-04-26T00:00:00Z",
        completed_at=completed_at,
    )


def test_insert_then_get_round_trips_all_fields(store: Store) -> None:
    s = _session()
    store.insert_onboard_session(s)
    got = store.get_onboard_session("sess-1")
    assert got is not None
    assert got == s


def test_get_unknown_session_returns_none(store: Store) -> None:
    assert store.get_onboard_session("missing") is None


def test_insert_duplicate_session_id_raises(store: Store) -> None:
    store.insert_onboard_session(_session())
    with pytest.raises(Exception):
        store.insert_onboard_session(_session())


def test_complete_returns_true_for_pending_session(store: Store) -> None:
    store.insert_onboard_session(_session())
    assert store.complete_onboard_session("sess-1", "2026-04-26T01:00:00Z") is True


def test_complete_returns_false_for_unknown_session(store: Store) -> None:
    assert store.complete_onboard_session("nope", "2026-04-26T01:00:00Z") is False


def test_complete_sets_state_to_completed(store: Store) -> None:
    store.insert_onboard_session(_session())
    store.complete_onboard_session("sess-1", "2026-04-26T01:00:00Z")
    got = store.get_onboard_session("sess-1")
    assert got is not None
    assert got.state == ONBOARD_STATE_COMPLETED


def test_complete_writes_completed_at_timestamp(store: Store) -> None:
    store.insert_onboard_session(_session())
    store.complete_onboard_session("sess-1", "2026-04-26T01:00:00Z")
    got = store.get_onboard_session("sess-1")
    assert got is not None
    assert got.completed_at == "2026-04-26T01:00:00Z"


def test_count_no_filter_counts_all_states(store: Store) -> None:
    store.insert_onboard_session(_session(session_id="a"))
    store.insert_onboard_session(
        _session(session_id="b", state=ONBOARD_STATE_COMPLETED,
                 completed_at="2026-04-26T02:00:00Z")
    )
    assert store.count_onboard_sessions() == 2


def test_count_filtered_by_pending_excludes_completed(store: Store) -> None:
    store.insert_onboard_session(_session(session_id="a"))
    store.insert_onboard_session(
        _session(session_id="b", state=ONBOARD_STATE_COMPLETED,
                 completed_at="2026-04-26T02:00:00Z")
    )
    assert store.count_onboard_sessions(ONBOARD_STATE_PENDING) == 1


def test_count_filtered_by_completed_excludes_pending(store: Store) -> None:
    store.insert_onboard_session(_session(session_id="a"))
    store.insert_onboard_session(
        _session(session_id="b", state=ONBOARD_STATE_COMPLETED,
                 completed_at="2026-04-26T02:00:00Z")
    )
    assert store.count_onboard_sessions(ONBOARD_STATE_COMPLETED) == 1


def test_count_on_empty_store_is_zero(store: Store) -> None:
    assert store.count_onboard_sessions() == 0


def test_list_pending_returns_only_pending_sessions(store: Store) -> None:
    store.insert_onboard_session(_session(session_id="a"))
    store.insert_onboard_session(
        _session(session_id="b", state=ONBOARD_STATE_COMPLETED,
                 completed_at="2026-04-26T02:00:00Z")
    )
    result = store.list_pending_onboard_sessions()
    assert [s.session_id for s in result] == ["a"]


def test_list_pending_orders_by_created_at_then_id(store: Store) -> None:
    store.insert_onboard_session(OnboardSession(
        session_id="b", repo_path="/r", state=ONBOARD_STATE_PENDING,
        candidates_json="[]", created_at="2026-04-26T01:00:00Z",
        completed_at=None,
    ))
    store.insert_onboard_session(OnboardSession(
        session_id="a", repo_path="/r", state=ONBOARD_STATE_PENDING,
        candidates_json="[]", created_at="2026-04-26T00:00:00Z",
        completed_at=None,
    ))
    result = store.list_pending_onboard_sessions()
    assert [s.session_id for s in result] == ["a", "b"]


def test_list_pending_on_empty_store_is_empty(store: Store) -> None:
    assert store.list_pending_onboard_sessions() == []


def test_complete_then_list_pending_no_longer_returns_session(store: Store) -> None:
    store.insert_onboard_session(_session())
    store.complete_onboard_session("sess-1", "2026-04-26T01:00:00Z")
    assert store.list_pending_onboard_sessions() == []


def test_candidates_json_blob_round_trips_unchanged(store: Store) -> None:
    blob = '[{"text": "alpha", "source_meta": {"path": "x.py", "line": 3}}]'
    store.insert_onboard_session(OnboardSession(
        session_id="sess-1", repo_path="/r", state=ONBOARD_STATE_PENDING,
        candidates_json=blob, created_at="2026-04-26T00:00:00Z",
        completed_at=None,
    ))
    got = store.get_onboard_session("sess-1")
    assert got is not None
    assert got.candidates_json == blob
