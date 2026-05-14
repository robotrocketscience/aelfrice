"""Acceptance tests for `aelf doctor --gc-orphan-feedback` (issue #223).

The pre-#283 re-ingest path could leave a `feedback_history` row
pointing at a `belief_id` that was later deleted (because a fresh
duplicate took its place). The pass counts those rows; with
`--apply`, deletes them.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.doctor import (
    OrphanFeedbackReport,
    format_orphan_feedback_report,
    gc_orphan_feedback,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk(store: MemoryStore, bid: str, content: str) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-04-26T00:00:00Z",
            last_retrieved_at=None,
        )
    )


def _seed_orphans(store: MemoryStore) -> None:
    """One live belief with two feedback rows, plus three orphan
    feedback rows pointing at deleted belief_ids."""
    _mk(store, "LIVE", "still here")
    _fb = lambda bid, v: store.insert_feedback_event(  # noqa: E731
        bid, valence=v, source="test", created_at="2026-04-26T00:00:00Z"
    )
    _ = _fb("LIVE", 1.0)
    _ = _fb("LIVE", 0.0)
    _ = _fb("DEAD1", 1.0)
    _ = _fb("DEAD2", 1.0)
    _ = _fb("DEAD2", 0.0)


# --- store helpers ------------------------------------------------------


def test_count_orphan_feedback_events_zero_on_clean_store(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _mk(store, "B1", "hello")
        _ = store.insert_feedback_event(
            "B1", valence=1.0, source="test",
            created_at="2026-04-26T00:00:00Z",
        )
        assert store.count_orphan_feedback_events() == 0
    finally:
        store.close()


def test_count_orphan_feedback_events_finds_dangling(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed_orphans(store)
        # 3 orphan rows: DEAD1 (1) + DEAD2 (2). LIVE rows must not count.
        assert store.count_orphan_feedback_events() == 3
    finally:
        store.close()


def test_delete_orphan_feedback_events_removes_only_dangling(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed_orphans(store)
        deleted = store.delete_orphan_feedback_events()
        assert deleted == 3
        assert store.count_orphan_feedback_events() == 0
        # The two rows for LIVE must remain.
        assert store.count_feedback_events(belief_id="LIVE") == 2
    finally:
        store.close()


# --- doctor function ----------------------------------------------------


def test_gc_orphan_feedback_dry_run_counts_without_deleting(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed_orphans(store)
        report = gc_orphan_feedback(store, dry_run=True)
        assert report.orphans_found == 3
        assert report.deleted == 0
        assert report.dry_run is True
        assert store.count_orphan_feedback_events() == 3
    finally:
        store.close()


def test_gc_orphan_feedback_apply_deletes_and_reports(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed_orphans(store)
        report = gc_orphan_feedback(store, dry_run=False)
        assert report.orphans_found == 3
        assert report.deleted == 3
        assert report.dry_run is False
        assert store.count_orphan_feedback_events() == 0
    finally:
        store.close()


def test_gc_orphan_feedback_noop_on_clean_store(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _mk(store, "B1", "hello")
        report = gc_orphan_feedback(store, dry_run=False)
        assert report.orphans_found == 0
        assert report.deleted == 0
    finally:
        store.close()


# --- format ------------------------------------------------------------


def test_format_orphan_feedback_report_dry_run() -> None:
    out = format_orphan_feedback_report(
        OrphanFeedbackReport(orphans_found=3, deleted=0, dry_run=True)
    )
    assert "orphan feedback rows: 3" in out
    assert "--apply" in out


def test_format_orphan_feedback_report_apply() -> None:
    out = format_orphan_feedback_report(
        OrphanFeedbackReport(orphans_found=3, deleted=3, dry_run=False)
    )
    assert "orphan feedback rows: 3" in out
    assert "deleted: 3" in out


def test_format_orphan_feedback_report_clean_dry_run() -> None:
    out = format_orphan_feedback_report(
        OrphanFeedbackReport(orphans_found=0, deleted=0, dry_run=True)
    )
    assert "nothing to do" in out


# --- CLI wire-up -------------------------------------------------------


def _run_cli_with_db(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_cli_doctor_gc_orphan_feedback_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        _seed_orphans(store)
    finally:
        store.close()
    code, output = _run_cli_with_db(
        monkeypatch, db, "doctor", "--gc-orphan-feedback"
    )
    assert code == 0
    assert "orphan feedback rows: 3" in output
    assert "--apply" in output
    # No deletion under dry-run.
    store = MemoryStore(str(db))
    try:
        assert store.count_orphan_feedback_events() == 3
    finally:
        store.close()


def test_cli_doctor_gc_orphan_feedback_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        _seed_orphans(store)
    finally:
        store.close()
    code, output = _run_cli_with_db(
        monkeypatch, db, "doctor", "--gc-orphan-feedback", "--apply"
    )
    assert code == 0
    assert "deleted: 3" in output
    store = MemoryStore(str(db))
    try:
        assert store.count_orphan_feedback_events() == 0
        # Live feedback survives.
        assert store.count_feedback_events(belief_id="LIVE") == 2
    finally:
        store.close()
