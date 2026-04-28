"""Tests for belief_corroborations — schema, ingest recorder, retrieval count.

Issue #190: phantom-prereqs T1.

All tests use an in-memory store (MemoryStore(":memory:")) and are
deterministic (no LLM calls, no randomness, no filesystem I/O).
Each test must complete well under 1 second.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from aelfrice.ingest import ingest_turn
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_HOOK_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_TYPES,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store() -> MemoryStore:
    return MemoryStore(":memory:")


def _belief(bid: str, content: str, content_hash: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=content_hash,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# ---------------------------------------------------------------------------
# Schema: table created on fresh store; idempotent on reopen
# ---------------------------------------------------------------------------


def test_table_created_on_fresh_store() -> None:
    """belief_corroborations table exists on a fresh in-memory store."""
    store = _fresh_store()
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='belief_corroborations'"
        )
        assert cur.fetchone() is not None, "table must exist"
    finally:
        store.close()


def test_index_created_on_fresh_store() -> None:
    """idx_belief_corroborations_belief_id index exists."""
    store = _fresh_store()
    try:
        cur = store._conn.execute(  # type: ignore[reportPrivateUsage]
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_belief_corroborations_belief_id'"
        )
        assert cur.fetchone() is not None, "index must exist"
    finally:
        store.close()


def test_migration_idempotent_on_rerun(tmp_path: Path) -> None:
    """Opening the same on-disk store twice does not error — schema is
    CREATE IF NOT EXISTS throughout."""
    db = str(tmp_path / "idem.db")
    s1 = MemoryStore(db)
    s1.close()
    s2 = MemoryStore(db)
    s2.close()


# ---------------------------------------------------------------------------
# Re-ingest records exactly one corroboration; belief row unchanged
# ---------------------------------------------------------------------------


def test_reingest_records_corroboration_row(tmp_path: Path) -> None:
    """Re-ingesting the same (source, sentence) records exactly one new
    corroboration row; the canonical belief row is unchanged."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The default port is 8080 for the service."
        source = "user"
        n1 = ingest_turn(store, text, source=source)
        assert n1 >= 1
        before_id = store.search_beliefs("default port", limit=5)
        assert len(before_id) == 1
        bid = before_id[0].id
        original_content_hash = before_id[0].content_hash

        # Re-ingest: same text, same source
        n2 = ingest_turn(store, text, source=source)
        assert n2 == 0  # no new beliefs

        # Belief row is unchanged
        after = store.get_belief(bid)
        assert after is not None
        assert after.content_hash == original_content_hash

        # One corroboration recorded
        count = store.count_corroborations(bid)
        assert count == 1
    finally:
        store.close()


def test_reingest_twice_records_two_corroborations(tmp_path: Path) -> None:
    """Each subsequent ingest adds exactly one row to belief_corroborations."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The configuration directory is /etc/aelfrice."
        source = "user"
        ingest_turn(store, text, source=source)
        beliefs = store.search_beliefs("configuration directory", limit=5)
        assert len(beliefs) >= 1
        bid = beliefs[0].id

        ingest_turn(store, text, source=source)
        ingest_turn(store, text, source=source)

        assert store.count_corroborations(bid) == 2
    finally:
        store.close()


def test_belief_id_and_content_hash_unchanged_after_corroboration(
    tmp_path: Path,
) -> None:
    """The canonical belief row's id and content_hash must not change
    after a re-ingest that triggers corroboration recording."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "Aelfrice stores beliefs in a SQLite database."
        source = "user"
        ingest_turn(store, text, source=source)
        beliefs = store.search_beliefs("Aelfrice", limit=5)
        assert beliefs
        bid = beliefs[0].id
        chash = beliefs[0].content_hash

        ingest_turn(store, text, source=source)

        after = store.get_belief(bid)
        assert after is not None
        assert after.id == bid
        assert after.content_hash == chash
    finally:
        store.close()


# ---------------------------------------------------------------------------
# corroboration_count on retrieval
# ---------------------------------------------------------------------------


def test_corroboration_count_zero_on_fresh_belief(tmp_path: Path) -> None:
    """A newly inserted belief has corroboration_count == 0."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The API rate limit is 100 requests per minute."
        ingest_turn(store, text, source="user")
        beliefs = store.search_beliefs("rate limit", limit=5)
        assert beliefs
        assert beliefs[0].corroboration_count == 0
    finally:
        store.close()


def test_corroboration_count_increments_on_retrieval(tmp_path: Path) -> None:
    """corroboration_count on a retrieved belief reflects the number of
    corroboration rows, not zero."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The scheduler runs every five minutes by default."
        source = "user"
        ingest_turn(store, text, source=source)
        beliefs = store.search_beliefs("scheduler", limit=5)
        assert beliefs
        bid = beliefs[0].id

        # Two re-ingests => two corroboration rows
        ingest_turn(store, text, source=source)
        ingest_turn(store, text, source=source)

        after = store.get_belief(bid)
        assert after is not None
        assert after.corroboration_count == 2
    finally:
        store.close()


def test_corroboration_count_via_search_beliefs(tmp_path: Path) -> None:
    """search_beliefs results also expose corroboration_count."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The log level is DEBUG in development environments."
        source = "user"
        ingest_turn(store, text, source=source)
        beliefs = store.search_beliefs("log level", limit=5)
        assert beliefs
        bid = beliefs[0].id

        ingest_turn(store, text, source=source)

        results = store.search_beliefs("log level", limit=5)
        assert results
        match = next((b for b in results if b.id == bid), None)
        assert match is not None
        assert match.corroboration_count == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Different source_type values record distinctly
# ---------------------------------------------------------------------------


def test_different_source_types_record_distinctly() -> None:
    """Corroboration rows for different source_types are each stored and
    counted independently. The count groups correctly."""
    store = _fresh_store()
    try:
        b = _belief("b1", "X uses Y.", "hash-b1")
        store.insert_belief(b)

        store.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST)
        store.record_corroboration("b1", source_type=CORROBORATION_SOURCE_COMMIT_INGEST)
        store.record_corroboration("b1", source_type=CORROBORATION_SOURCE_MCP_REMEMBER)
        store.record_corroboration("b1", source_type=CORROBORATION_SOURCE_HOOK_INGEST)

        assert store.count_corroborations("b1") == 4

        rows = store.list_corroborations("b1")
        assert len(rows) == 4
        # rows are (ingested_at, source_type, session_id, source_path_hash)
        recorded_types = {r[1] for r in rows}
        assert recorded_types == {
            CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            CORROBORATION_SOURCE_COMMIT_INGEST,
            CORROBORATION_SOURCE_MCP_REMEMBER,
            CORROBORATION_SOURCE_HOOK_INGEST,
        }
    finally:
        store.close()


def test_source_type_enum_covers_all_variants() -> None:
    """CORROBORATION_SOURCE_TYPES contains exactly the four documented
    values."""
    expected = {
        "commit_ingest",
        "transcript_ingest",
        "mcp_remember",
        "hook_ingest",
    }
    assert CORROBORATION_SOURCE_TYPES == expected


# ---------------------------------------------------------------------------
# session_id and source_path_hash nullable
# ---------------------------------------------------------------------------


def test_null_session_id_and_source_path_hash_accepted() -> None:
    """record_corroboration with session_id=None and source_path_hash=None
    must not raise, and the stored row reflects NULLs."""
    store = _fresh_store()
    try:
        b = _belief("b2", "Y implies Z.", "hash-b2")
        store.insert_belief(b)

        store.record_corroboration(
            "b2",
            source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            session_id=None,
            source_path_hash=None,
        )

        rows = store.list_corroborations("b2")
        assert len(rows) == 1
        # rows are (ingested_at, source_type, session_id, source_path_hash)
        assert rows[0][2] is None  # session_id
        assert rows[0][3] is None  # source_path_hash
    finally:
        store.close()


def test_session_id_stamped_when_provided() -> None:
    """session_id is persisted when supplied."""
    store = _fresh_store()
    try:
        b = _belief("b3", "Z follows W.", "hash-b3")
        store.insert_belief(b)

        store.record_corroboration(
            "b3",
            source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            session_id="sess-abc",
            source_path_hash="pathsha256",
        )

        rows = store.list_corroborations("b3")
        assert len(rows) == 1
        # rows are (ingested_at, source_type, session_id, source_path_hash)
        assert rows[0][2] == "sess-abc"    # session_id
        assert rows[0][3] == "pathsha256"  # source_path_hash
    finally:
        store.close()


def test_ingest_turn_passes_session_id_to_corroboration(tmp_path: Path) -> None:
    """When ingest_turn is called with session_id and a duplicate is found,
    the corroboration row is stamped with that session_id."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The memory limit is four gigabytes per process."
        source = "user"
        ingest_turn(store, text, source=source, session_id="sess-1")
        beliefs = store.search_beliefs("memory limit", limit=5)
        assert beliefs
        bid = beliefs[0].id

        # Re-ingest with a different session_id
        ingest_turn(store, text, source=source, session_id="sess-2")

        rows = store.list_corroborations(bid)
        assert len(rows) == 1
        # rows are (ingested_at, source_type, session_id, source_path_hash)
        assert rows[0][2] == "sess-2"  # session_id
    finally:
        store.close()


# ---------------------------------------------------------------------------
# ON DELETE CASCADE
# ---------------------------------------------------------------------------


def test_belief_deletion_cascades_to_corroborations(tmp_path: Path) -> None:
    """Deleting a belief removes its corroboration rows via ON DELETE CASCADE."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The retry backoff is exponential with a cap of sixty seconds."
        source = "user"
        ingest_turn(store, text, source=source)
        beliefs = store.search_beliefs("retry backoff", limit=5)
        assert beliefs
        bid = beliefs[0].id

        ingest_turn(store, text, source=source)
        ingest_turn(store, text, source=source)
        assert store.count_corroborations(bid) == 2

        store.delete_belief(bid)

        assert store.get_belief(bid) is None
        assert store.count_corroborations(bid) == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# record_corroboration directly (store API)
# ---------------------------------------------------------------------------


def test_record_corroboration_returns_none() -> None:
    """record_corroboration returns None (side-effect only, no rowid exposed)."""
    store = _fresh_store()
    try:
        b = _belief("b4", "A causes B.", "hash-b4")
        store.insert_belief(b)

        result = store.record_corroboration("b4", source_type=CORROBORATION_SOURCE_COMMIT_INGEST)
        assert result is None
    finally:
        store.close()


def test_count_corroborations_per_belief() -> None:
    """count_corroborations(belief_id) returns the count for that belief only."""
    store = _fresh_store()
    try:
        for i in range(3):
            bid = f"b{i}"
            b = _belief(bid, f"Statement {i}.", f"hash-{i}")
            store.insert_belief(b)
            store.record_corroboration(bid, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST)
            store.record_corroboration(bid, source_type=CORROBORATION_SOURCE_COMMIT_INGEST)

        # Each belief has 2 corroborations
        for i in range(3):
            assert store.count_corroborations(f"b{i}") == 2
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Spec-required test: record_corroboration_adds_one_row
# ---------------------------------------------------------------------------


def test_record_corroboration_adds_one_row() -> None:
    """record_corroboration adds exactly one row to belief_corroborations
    per call (direct store API contract)."""
    store = _fresh_store()
    try:
        b = _belief("b_rcar", "Z precedes W.", "hash-rcar")
        store.insert_belief(b)

        assert store.count_corroborations("b_rcar") == 0
        store.record_corroboration(
            "b_rcar", source_type=CORROBORATION_SOURCE_COMMIT_INGEST
        )
        assert store.count_corroborations("b_rcar") == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Spec-required test: test_unknown_source_type_raises_valueerror
# ---------------------------------------------------------------------------


def test_unknown_source_type_raises_valueerror() -> None:
    """record_corroboration raises ValueError for an unknown source_type."""
    store = _fresh_store()
    try:
        b = _belief("b_err", "A causes B.", "hash-b-err")
        store.insert_belief(b)

        with pytest.raises(ValueError, match="Unknown source_type"):
            store.record_corroboration("b_err", source_type="not_a_real_source")
    finally:
        store.close()
