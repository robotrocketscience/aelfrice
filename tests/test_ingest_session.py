"""Belief.session_id round-trips and ingest_turn persists it."""
from __future__ import annotations

from aelfrice.ingest import ingest_turn
from aelfrice.models import Belief, LOCK_NONE
from aelfrice.store import MemoryStore


def _b(id_: str, session_id: str | None = None) -> Belief:
    return Belief(
        id=id_,
        content=id_ + "-content",
        content_hash=id_,
        alpha=1.0,
        beta=1.0,
        type="factual",
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-27T00:00:00+00:00",
        last_retrieved_at=None,
        session_id=session_id,
    )


def test_default_session_id_is_none() -> None:
    assert _b("a").session_id is None


def test_session_id_persists_through_round_trip() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("a", session_id="sess-1"))
        got = store.get_belief("a")
        assert got is not None
        assert got.session_id == "sess-1"
    finally:
        store.close()


def test_create_session_returns_fresh_non_empty_id() -> None:
    store = MemoryStore(":memory:")
    try:
        s1 = store.create_session(model="haiku")
        s2 = store.create_session(model="haiku")
        assert s1.id and s2.id
        assert s1.id != s2.id
    finally:
        store.close()


def test_create_session_persists_to_sessions_table() -> None:
    store = MemoryStore(":memory:")
    try:
        s = store.create_session(model="opus", project_context="aelfrice")
        got = store.get_session(s.id)
        assert got is not None
        assert got.id == s.id
        assert got.model == "opus"
        assert got.project_context == "aelfrice"
        assert got.completed_at is None
    finally:
        store.close()


def test_complete_session_stamps_completed_at() -> None:
    store = MemoryStore(":memory:")
    try:
        s = store.create_session()
        store.complete_session(s.id)
        got = store.get_session(s.id)
        assert got is not None
        assert got.completed_at is not None
    finally:
        store.close()


def test_complete_session_idempotent_on_already_completed() -> None:
    store = MemoryStore(":memory:")
    try:
        s = store.create_session()
        store.complete_session(s.id)
        store.complete_session(s.id)  # must not raise
        got = store.get_session(s.id)
        assert got is not None
        assert got.completed_at is not None
    finally:
        store.close()


def test_complete_session_silent_on_unknown_id() -> None:
    store = MemoryStore(":memory:")
    try:
        store.complete_session("nonexistent")  # must not raise
    finally:
        store.close()


def test_ingest_turn_writes_session_id_on_each_belief() -> None:
    store = MemoryStore(":memory:")
    try:
        text = "Pi is 3.14. Water boils at 100C."
        n = ingest_turn(
            store=store, text=text, source="test", session_id="sess-X",
        )
        assert n >= 1
        all_beliefs = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT id, session_id FROM beliefs"
        ).fetchall()
        assert all_beliefs
        for row in all_beliefs:
            assert row["session_id"] == "sess-X"
    finally:
        store.close()


def test_ingest_turn_without_session_leaves_session_id_null() -> None:
    store = MemoryStore(":memory:")
    try:
        n = ingest_turn(store=store, text="The sky is blue.", source="test")
        assert n >= 1
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        for row in rows:
            assert row["session_id"] is None
    finally:
        store.close()
