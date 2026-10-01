"""#1669: `aelf doctor --gc-filesystem-corroboration` removes the rows
filesystem ingest wrote before #1615, and reports who leaves core.

Before #1615, each onboard or repository scan wrote a corroboration row
for every paragraph it read again. The rows still count toward `aelf
core`. The pass deletes only those rows, and reports the beliefs that
lose core membership because of it.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.doctor import (
    format_filesystem_corroboration_report,
    gc_filesystem_corroboration,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

_FS = CORROBORATION_SOURCE_FILESYSTEM_INGEST
_TX = CORROBORATION_SOURCE_TRANSCRIPT_INGEST


def _mk(store: MemoryStore, bid: str, *, locked: bool = False) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=f"belief {bid} states one fact",
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER if locked else LOCK_NONE,
            locked_at="2026-01-01T00:00:00Z" if locked else None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        )
    )


def _row(store: MemoryStore, bid: str, source: str, day: int) -> None:
    # A day apart, so each row is its own episode under #1635.
    store.record_corroboration(
        bid, source_type=source, session_id=f"s{day}",
        ts=f"2026-01-{day:02d}T00:00:00Z",
    )


def _seed(store: MemoryStore) -> None:
    _mk(store, "FS_ONLY")  # in core only through filesystem rows
    _row(store, "FS_ONLY", _FS, 2)
    _row(store, "FS_ONLY", _FS, 3)
    _mk(store, "MIXED")  # keeps two transcript rows, so stays in core
    _row(store, "MIXED", _FS, 2)
    _row(store, "MIXED", _TX, 3)
    _row(store, "MIXED", _TX, 4)
    _mk(store, "TX_ONLY")  # never touched
    _row(store, "TX_ONLY", _TX, 2)
    _row(store, "TX_ONLY", _TX, 3)
    _mk(store, "LOCKED", locked=True)  # in core through the lock
    _row(store, "LOCKED", _FS, 2)
    _row(store, "LOCKED", _FS, 3)
    _mk(store, "WEAK")  # one row: never in core
    _row(store, "WEAK", _FS, 2)


def _all_rows(store: MemoryStore) -> list[tuple[object, ...]]:
    cur = store._conn.execute(  # noqa: SLF001 - read-only probe
        "SELECT belief_id, ingested_at, source_type, session_id, "
        "source_path_hash FROM belief_corroborations ORDER BY 1, 2, 3"
    )
    return [tuple(r) for r in cur.fetchall()]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "memory.db"))
    _seed(s)
    yield s
    s.close()


def test_dry_run_reports_rows_and_core_loss_and_changes_nothing(
    store: MemoryStore,
) -> None:
    before = _all_rows(store)
    report = gc_filesystem_corroboration(store, dry_run=True)
    assert report.rows_found == 6
    assert report.beliefs_affected == 4
    assert report.leaving_core == ["FS_ONLY"]
    assert report.deleted == 0
    assert _all_rows(store) == before


def test_apply_deletes_only_filesystem_rows(store: MemoryStore) -> None:
    other_before = [r for r in _all_rows(store) if r[2] != _FS]
    report = gc_filesystem_corroboration(store, dry_run=False)
    assert report.deleted == 6
    assert report.leaving_core == ["FS_ONLY"]
    after = _all_rows(store)
    assert [r for r in after if r[2] == _FS] == []
    assert after == other_before


def test_apply_reports_what_the_dry_run_reported(store: MemoryStore) -> None:
    dry = gc_filesystem_corroboration(store, dry_run=True)
    applied = gc_filesystem_corroboration(store, dry_run=False)
    assert applied.leaving_core == dry.leaving_core
    assert applied.rows_found == dry.rows_found


def test_second_apply_is_a_no_op(store: MemoryStore) -> None:
    gc_filesystem_corroboration(store, dry_run=False)
    again = gc_filesystem_corroboration(store, dry_run=False)
    assert (again.rows_found, again.deleted, again.leaving_core) == (0, 0, [])


def test_format_names_the_beliefs_leaving_core(store: MemoryStore) -> None:
    text = format_filesystem_corroboration_report(
        gc_filesystem_corroboration(store, dry_run=True)
    )
    assert "filesystem corroboration rows: 6 on 4 belief(s)" in text
    assert "beliefs leaving `aelf core`: 1" in text
    assert "  FS_ONLY" in text
    assert "--apply" in text


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_cli_dry_run_then_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "cli.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    code, out = _run_cli(monkeypatch, db, "doctor", "--gc-filesystem-corroboration")
    assert code == 0
    assert "filesystem corroboration rows: 6" in out
    assert "dry-run" in out
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--gc-filesystem-corroboration", "--apply"
    )
    assert code == 0
    assert "deleted: 6" in out
    s = MemoryStore(str(db))
    try:
        assert s.count_corroborations_by_source({_FS}) == {}
    finally:
        s.close()
