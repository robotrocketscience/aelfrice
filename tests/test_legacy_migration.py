"""Tests for the v2.x #263 legacy_unknown ingest_log synthesis migration.

Every test states a falsifiable hypothesis. The migration synthesizes one
`source_kind=legacy_unknown` ingest_log row per belief that has no log
coverage at open time.  After migration the reachability check must report
zero orphans.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.replay import check_log_reachability
from aelfrice.store import (
    SCHEMA_META_LEGACY_LOG_SYNTH,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_belief_direct(conn: sqlite3.Connection, bid: str, content: str,
                           session_id: str | None = None,
                           created_at: str = "2024-01-01T00:00:00+00:00",
                           ) -> None:
    """Insert a belief row directly — bypassing MemoryStore.insert_belief —
    so no version-vector or ingest_log rows are created. Simulates
    pre-v2.0 belief data."""
    conn.execute(
        """
        INSERT INTO beliefs (
            id, content, content_hash, alpha, beta, type, lock_level,
            created_at, origin, session_id
        ) VALUES (?, ?, ?, 1.0, 1.0, 'factual', 'none', ?, 'unknown', ?)
        """,
        (bid, content, f"hash-{bid}", created_at, session_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "migration.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Test 1: fresh store — no beliefs, no synth rows, marker stamped
# ---------------------------------------------------------------------------


def test_fresh_store_no_beliefs_marker_stamped(tmp_path: Path) -> None:
    """Hypothesis: opening a fresh store (zero beliefs) produces no
    synthesized log rows and stamps the migration marker. Falsifiable by
    any synth rows being inserted or the marker being absent."""
    db = tmp_path / "fresh.db"
    s = MemoryStore(str(db))
    try:
        assert s.count_ingest_log() == 0
        marker = s.get_schema_meta(SCHEMA_META_LEGACY_LOG_SYNTH)
        assert marker is not None, "marker should be stamped even on empty store"
    finally:
        s.close()


def test_fresh_store_second_open_noop(tmp_path: Path) -> None:
    """Hypothesis: re-opening a migrated empty store is a no-op — no rows
    added and marker unchanged. Falsifiable by any row count increase or
    marker disappearing."""
    db = tmp_path / "fresh2.db"
    s1 = MemoryStore(str(db))
    marker_first = s1.get_schema_meta(SCHEMA_META_LEGACY_LOG_SYNTH)
    s1.close()

    s2 = MemoryStore(str(db))
    try:
        assert s2.count_ingest_log() == 0
        marker_second = s2.get_schema_meta(SCHEMA_META_LEGACY_LOG_SYNTH)
        assert marker_second == marker_first, "marker must not change on re-open"
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# Test 2: legacy store with N beliefs, 0 log rows → N synth rows, 0 orphans
# ---------------------------------------------------------------------------


def test_legacy_store_n_beliefs_produces_n_synth_rows(tmp_path: Path) -> None:
    """Hypothesis: a store with N beliefs and zero ingest_log rows produces
    exactly N legacy_unknown synth rows after the first open and reports
    zero orphans. Falsifiable by any other synth count or any remaining
    orphan."""
    db = tmp_path / "legacy.db"

    # Bootstrap the schema without triggering migration by temporarily
    # removing the migration logic: we open, inject beliefs directly,
    # then simulate old state by wiping the marker and the log.
    s = MemoryStore(str(db))
    n_beliefs = 5
    for i in range(n_beliefs):
        _insert_belief_direct(
            s._conn,  # pyright: ignore[reportPrivateUsage]
            bid=f"legacy-b-{i:04d}",
            content=f"pre-v2.0 fact number {i}",
            created_at="2024-06-01T12:00:00+00:00",
        )
    # Simulate pre-migration state: wipe log rows + marker.
    s._conn.execute("DELETE FROM ingest_log")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute("DELETE FROM log_versions")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    # Re-open — migration should fire.
    s2 = MemoryStore(str(db))
    try:
        assert s2.count_ingest_log() == n_beliefs, (
            f"expected {n_beliefs} synth rows, got {s2.count_ingest_log()}"
        )
        # All synth rows must be legacy_unknown.
        rows = s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT source_kind FROM ingest_log"
        ).fetchall()
        assert all(r["source_kind"] == INGEST_SOURCE_LEGACY_UNKNOWN for r in rows)
        # Reachability check must report zero orphans.
        report = check_log_reachability(s2)
        assert report.total_beliefs == n_beliefs
        assert report.all_reachable, f"orphans remain: {report.orphan_belief_ids}"
    finally:
        s2.close()


def test_synth_rows_have_correct_fields(tmp_path: Path) -> None:
    """Hypothesis: each synthesized row carries ts=belief.created_at,
    raw_text=belief.content, source_path=NULL, session_id=belief.session_id,
    derived_belief_ids=[belief.id]. Falsifiable by any field mismatch."""
    db = tmp_path / "fields.db"
    s = MemoryStore(str(db))
    bid = "field-check-b001"
    _insert_belief_direct(
        s._conn,  # pyright: ignore[reportPrivateUsage]
        bid=bid,
        content="a pre-v2.0 belief about the system",
        session_id="sess-legacy-42",
        created_at="2023-11-15T08:30:00+00:00",
    )
    s._conn.execute("DELETE FROM ingest_log")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute("DELETE FROM log_versions")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    s2 = MemoryStore(str(db))
    try:
        row = s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM ingest_log"
        ).fetchone()
        assert row is not None
        assert row["source_kind"] == INGEST_SOURCE_LEGACY_UNKNOWN
        assert row["raw_text"] == "a pre-v2.0 belief about the system"
        assert row["source_path"] is None
        assert row["ts"] == "2023-11-15T08:30:00+00:00"
        assert row["session_id"] == "sess-legacy-42"
        assert json.loads(row["derived_belief_ids"]) == [bid]
        assert row["raw_meta"] is None
        assert row["derived_edge_ids"] is None
    finally:
        s2.close()


def test_synth_row_null_session_id_when_belief_has_none(tmp_path: Path) -> None:
    """Hypothesis: when a belief has no session_id, the synthesized row
    also has NULL session_id. Falsifiable by any non-NULL value."""
    db = tmp_path / "null_session.db"
    s = MemoryStore(str(db))
    _insert_belief_direct(
        s._conn,  # pyright: ignore[reportPrivateUsage]
        bid="no-session-b001",
        content="belief without a session",
        session_id=None,
    )
    s._conn.execute("DELETE FROM ingest_log")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute("DELETE FROM log_versions")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    s2 = MemoryStore(str(db))
    try:
        row = s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM ingest_log"
        ).fetchone()
        assert row is not None
        assert row["session_id"] is None
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# Test 3: mixed store — M covered + N orphan beliefs → exactly N synth rows
# ---------------------------------------------------------------------------


def test_mixed_store_only_orphans_get_synth_rows(tmp_path: Path) -> None:
    """Hypothesis: in a store where M beliefs already have log coverage and
    N do not, the migration inserts exactly N synth rows and leaves the
    M existing log rows untouched. Falsifiable by any different count or
    any mutation to existing rows."""
    db = tmp_path / "mixed.db"
    s = MemoryStore(str(db))

    # Insert M covered beliefs via the normal record_ingest path.
    m_covered = 3
    covered_ids = []
    for i in range(m_covered):
        bid = f"covered-b-{i:04d}"
        covered_ids.append(bid)
        # Insert belief directly then record a log row for it.
        _insert_belief_direct(
            s._conn,  # pyright: ignore[reportPrivateUsage]
            bid=bid,
            content=f"covered belief {i}",
        )
        s.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            raw_text=f"covered belief {i}",
            derived_belief_ids=[bid],
        )

    # Insert N orphan beliefs with no log rows.
    n_orphan = 4
    orphan_ids = []
    for i in range(n_orphan):
        bid = f"orphan-b-{i:04d}"
        orphan_ids.append(bid)
        _insert_belief_direct(
            s._conn,  # pyright: ignore[reportPrivateUsage]
            bid=bid,
            content=f"orphan belief {i}",
        )

    # Record log count before wiping the migration marker.
    pre_log_count = s.count_ingest_log()
    assert pre_log_count == m_covered

    # Strip only the migration marker so the migration fires on re-open.
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    # Also remove the log VV backfill marker so it doesn't short-circuit.
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = 'log_version_vector_backfill_complete'"
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    s2 = MemoryStore(str(db))
    try:
        total_log = s2.count_ingest_log()
        assert total_log == m_covered + n_orphan, (
            f"expected {m_covered + n_orphan} log rows, got {total_log}"
        )
        # Count only synth rows.
        synth_count = s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) AS n FROM ingest_log WHERE source_kind = ?",
            (INGEST_SOURCE_LEGACY_UNKNOWN,),
        ).fetchone()["n"]
        assert synth_count == n_orphan, (
            f"expected {n_orphan} synth rows, got {synth_count}"
        )
        # Covered belief ids must not appear in any synth row.
        for cid in covered_ids:
            synth_rows = [
                r for r in s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
                    "SELECT derived_belief_ids FROM ingest_log "
                    "WHERE source_kind = ?",
                    (INGEST_SOURCE_LEGACY_UNKNOWN,),
                ).fetchall()
                if cid in (json.loads(r["derived_belief_ids"] or "[]"))
            ]
            assert not synth_rows, (
                f"covered belief {cid} got an unexpected synth row"
            )
        # Reachability must be clean.
        report = check_log_reachability(s2)
        assert report.all_reachable, f"orphans remain: {report.orphan_belief_ids}"
    finally:
        s2.close()


# ---------------------------------------------------------------------------
# Test 4: re-open after migration is a no-op (marker prevents re-run)
# ---------------------------------------------------------------------------


def test_reopen_after_migration_is_noop(tmp_path: Path) -> None:
    """Hypothesis: re-opening a store after the legacy migration has run
    returns 0 from _maybe_synthesize_legacy_log_rows and does not add new
    rows. Falsifiable by any row count increase or return value != 0."""
    db = tmp_path / "reopen.db"

    s = MemoryStore(str(db))
    bid = "reopen-b001"
    _insert_belief_direct(
        s._conn,  # pyright: ignore[reportPrivateUsage]
        bid=bid,
        content="fact for reopen test",
    )
    s._conn.execute("DELETE FROM ingest_log")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute("DELETE FROM log_versions")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    # First re-open: migration fires.
    s2 = MemoryStore(str(db))
    count_after_first = s2.count_ingest_log()
    assert count_after_first == 1
    # Call directly to get return value.
    second_call_return = s2._maybe_synthesize_legacy_log_rows()  # pyright: ignore[reportPrivateUsage]
    assert second_call_return == 0, (
        f"second call should return 0 (marker present), got {second_call_return}"
    )
    s2.close()

    # Second re-open: row count must not change.
    s3 = MemoryStore(str(db))
    try:
        assert s3.count_ingest_log() == count_after_first, (
            "row count must not increase on second open"
        )
    finally:
        s3.close()


# ---------------------------------------------------------------------------
# Test 5: synthesized rows survive the version-vector log backfill
# ---------------------------------------------------------------------------


def test_synth_rows_get_version_vector_stamped(tmp_path: Path) -> None:
    """Hypothesis: synthesized log rows have a log_versions entry after
    open, because _maybe_synthesize_legacy_log_rows runs before
    _maybe_backfill_log_version_vectors. Falsifiable by any synth log row
    without a matching log_versions row."""
    db = tmp_path / "vv.db"
    s = MemoryStore(str(db))
    for i in range(3):
        _insert_belief_direct(
            s._conn,  # pyright: ignore[reportPrivateUsage]
            bid=f"vv-b-{i:04d}",
            content=f"vv test belief {i}",
        )
    s._conn.execute("DELETE FROM ingest_log")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute("DELETE FROM log_versions")  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = ?", (SCHEMA_META_LEGACY_LOG_SYNTH,)
    )
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = 'log_version_vector_backfill_complete'"
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()

    s2 = MemoryStore(str(db))
    try:
        synth_ids = [
            r["id"] for r in s2._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "SELECT id FROM ingest_log WHERE source_kind = ?",
                (INGEST_SOURCE_LEGACY_UNKNOWN,),
            ).fetchall()
        ]
        assert len(synth_ids) == 3
        for log_id in synth_ids:
            vv = s2.get_log_version_vector(log_id)
            assert vv, (
                f"synth log row {log_id} has no version-vector entry"
            )
            assert list(vv.values()) == [1], (
                f"expected counter=1, got {vv}"
            )
    finally:
        s2.close()
