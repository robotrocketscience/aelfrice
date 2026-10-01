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


def _row(
    store: MemoryStore, bid: str, source: str, day: int, minute: int = 0,
) -> None:
    # Rows a day apart are separate episodes under #1635; rows within an
    # hour of each other are one.
    store.record_corroboration(
        bid, source_type=source, session_id=f"s{day}-{minute}",
        ts=f"2026-01-{day:02d}T00:{minute:02d}:00Z",
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
    # Keeps two rows after the cleanup, but both fall in the creation
    # hour, so it drops to one episode and leaves core on episodes alone.
    _mk(store, "EP_LOSS")
    _row(store, "EP_LOSS", _TX, 1, minute=10)
    _row(store, "EP_LOSS", _TX, 1, minute=20)
    _row(store, "EP_LOSS", _FS, 5)
    _mk(store, "RETIRED")  # in core by its rows, but retired: not listed
    _row(store, "RETIRED", _FS, 2)
    _row(store, "RETIRED", _FS, 3)
    store.soft_delete_belief("RETIRED")


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
    report = gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    assert report.rows_found == 9
    assert report.beliefs_affected == 6
    assert report.leaving_core == ["EP_LOSS", "FS_ONLY"]
    assert report.deleted == 0
    assert _all_rows(store) == before


def test_apply_deletes_only_filesystem_rows(store: MemoryStore) -> None:
    other_before = [r for r in _all_rows(store) if r[2] != _FS]
    report = gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=False)
    assert report.deleted == 9
    assert report.leaving_core == ["EP_LOSS", "FS_ONLY"]
    after = _all_rows(store)
    assert [r for r in after if r[2] == _FS] == []
    assert after == other_before


def test_apply_reports_what_the_dry_run_reported(store: MemoryStore) -> None:
    dry = gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    applied = gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=False)
    assert applied.leaving_core == dry.leaving_core
    assert applied.rows_found == dry.rows_found


def test_apply_bumps_the_store_generation(store: MemoryStore) -> None:
    """`corr=` is rendered into injected beliefs, so caches keyed on the
    generation must see the delete."""
    gen = store.store_generation()
    gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    assert store.store_generation() == gen
    gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=False)
    assert store.store_generation() > gen


def test_second_apply_is_a_no_op(store: MemoryStore) -> None:
    gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=False)
    again = gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=False)
    assert (again.rows_found, again.deleted, again.leaving_core) == (0, 0, [])


def test_refuses_inside_an_open_transaction(store: MemoryStore) -> None:
    """A nested dry run can't roll back, so it must not run at all."""
    before = _all_rows(store)
    with store.transaction():
        with pytest.raises(RuntimeError, match="own transaction"):
            gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    assert _all_rows(store) == before


def test_refuses_with_pending_writes(store: MemoryStore) -> None:
    """Its rollback would discard a caller's uncommitted write."""
    store._conn.execute(  # noqa: SLF001 - plant an uncommitted write
        "UPDATE beliefs SET alpha = 5.0 WHERE id = 'TX_ONLY'"
    )
    with pytest.raises(RuntimeError, match="own transaction"):
        gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    store._conn.commit()  # noqa: SLF001
    b = store.get_belief("TX_ONLY")
    assert b is not None and b.alpha == 5.0


def test_format_names_the_beliefs_leaving_core(store: MemoryStore) -> None:
    text = format_filesystem_corroboration_report(
        gc_filesystem_corroboration(store, qualifies=cli_module.default_core_rule, dry_run=True)
    )
    assert "filesystem corroboration rows: 9 on 6 belief(s)" in text
    assert "beliefs leaving `aelf core`: 2" in text
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
    assert "filesystem corroboration rows: 9" in out
    assert "dry-run" in out
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--gc-filesystem-corroboration", "--apply"
    )
    assert code == 0
    assert "deleted: 9" in out
    s = MemoryStore(str(db))
    try:
        assert s.count_corroborations_by_source({_FS}) == {}
    finally:
        s.close()


def test_cli_refuses_both_gc_passes_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "cli.db"
    MemoryStore(str(db)).close()
    code, _ = _run_cli(
        monkeypatch, db, "doctor", "--gc-orphan-feedback",
        "--gc-filesystem-corroboration", "--apply",
    )
    assert code == 2
    assert "one at a time" in capsys.readouterr().err
