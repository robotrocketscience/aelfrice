"""End-to-end smoke tests for the v1.0.x ingest_turn shim."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_turn
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "ingest.db"))
    yield s
    s.close()


def test_ingest_turn_returns_zero_for_empty_text(store: MemoryStore) -> None:
    assert ingest_turn(store, "", source="user") == 0


def test_ingest_turn_inserts_factual_sentences(store: MemoryStore) -> None:
    text = (
        "The configuration file lives at /etc/aelfrice/conf. "
        "The default port is 8080 for the dashboard."
    )
    n = ingest_turn(store, text, source="user")
    assert n == 2
    assert store.count_beliefs() == 2


def test_ingest_turn_skips_questions(store: MemoryStore) -> None:
    text = (
        "What is the default port for the dashboard service? "
        "The answer lives in the configuration file at /etc."
    )
    n = ingest_turn(store, text, source="user")
    # Question is skipped (persist=False); statement is kept.
    assert n == 1


def test_ingest_turn_search_finds_ingested_content(store: MemoryStore) -> None:
    ingest_turn(
        store,
        "The Hubble telescope observes galactic supernovae nightly. "
        "Astronomers process the imagery using cluster computing.",
        source="user",
    )
    hits = store.search_beliefs("Hubble", limit=10)
    assert len(hits) == 1
    assert "Hubble" in hits[0].content


def test_ingest_turn_idempotent_on_same_text(store: MemoryStore) -> None:
    text = "The configuration file lives at the default location here."
    ingest_turn(store, text, source="user")
    before = store.count_beliefs()
    ingest_turn(store, text, source="user")
    after = store.count_beliefs()
    assert before == after  # INSERT OR IGNORE prevents duplicates


def test_ingest_turn_accepts_session_id_and_source_id(store: MemoryStore) -> None:
    """session_id and source_id are accepted for adapter parity but
    not persisted at v1.0.x — calls must succeed without error."""
    n = ingest_turn(
        store,
        "A meaningful statement here that should be classified factual.",
        source="user",
        session_id="some-session-uuid",
        source_id="S0:turn:3",
    )
    assert n == 1


def test_ingest_turn_stamps_origin_agent_inferred(store: MemoryStore) -> None:
    """Ingested-via-classifier beliefs land with origin=agent_inferred,
    not the model default ORIGIN_UNKNOWN. Regression for #224."""
    ingest_turn(
        store,
        "The default port is 8080 for the dashboard service.",
        source="user",
    )
    hits = store.search_beliefs("dashboard", limit=10)
    assert len(hits) == 1
    assert hits[0].origin == "agent_inferred"


def test_ingest_turn_uses_provided_created_at(store: MemoryStore) -> None:
    ingest_turn(
        store,
        "A meaningful statement here that should be classified factual.",
        source="user",
        created_at="2026-04-27T08:00:00+00:00",
    )
    beliefs = store.search_beliefs("meaningful", limit=10)
    assert len(beliefs) == 1
    assert beliefs[0].created_at == "2026-04-27T08:00:00+00:00"


def test_create_session_returns_handle_with_id(store: MemoryStore) -> None:
    session = store.create_session(model="test", project_context="ctx")
    assert session.id  # non-empty
    assert session.completed_at is None
    assert session.model == "test"


def test_complete_session_is_no_op(store: MemoryStore) -> None:
    """Public v1.0.x complete_session is a terminator with no schema effect."""
    session = store.create_session()
    # Should not raise even though the session is not in any table.
    store.complete_session(session.id)
