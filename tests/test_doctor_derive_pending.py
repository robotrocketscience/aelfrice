"""Tests for `aelf doctor --derive-pending` (#264 slice 2).

The manual escape for orphan ingest_log rows: when a worker died
between batches and left rows with NULL derived_belief_ids, the next
ingest call would normally pick them up — but until then, the operator
can force a catch-up pass.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.derivation_worker import run_worker
from aelfrice.ingest import ingest_turn
from aelfrice.models import INGEST_SOURCE_FILESYSTEM
from aelfrice.store import MemoryStore


@pytest.fixture
def isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    db_path = tmp_path / "doctor-derive-pending.db"
    monkeypatch.setenv("AELFRICE_DB", str(db_path))
    monkeypatch.setenv("AELFRICE_SESSION_ID", "sess-test")
    yield db_path


def _capture(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        rc = cli_main(list(argv))
    return rc, buf.getvalue()


def test_derive_pending_on_clean_store_returns_zero(isolated_db: Path) -> None:
    """Hypothesis: a store with no orphan log rows (the steady-state
    after every well-behaved ingest) reports zero scanned and exits 0.
    Falsifiable if exit code is non-zero or any "rows scanned" is
    non-zero."""
    rc, out = _capture("doctor", "--derive-pending")
    assert rc == 0
    assert "unstamped before:        0" in out
    assert "rows scanned:            0" in out
    assert "unstamped after:         0" in out


def test_derive_pending_stamps_orphan_log_row(isolated_db: Path) -> None:
    """Hypothesis: an orphan log row inserted directly to ingest_log
    (simulating the 'worker died mid-batch' scenario) is stamped after
    --derive-pending runs. Exit 0 on success.

    Falsifiable if `derived_belief_ids` remains NULL OR if the worker
    counters don't reflect the recovery (rows_scanned == 0)."""
    store = MemoryStore(str(isolated_db))
    try:
        store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path="orphan.md",
            raw_text="The orphan port is 9090.",
            session_id="sess-orphan",
            ts="2026-04-26T00:00:00Z",
            raw_meta={"call_site": "filesystem_ingest"},
        )
        # confirm orphan state
        assert len(store.list_unstamped_ingest_log()) == 1
    finally:
        store.close()

    rc, out = _capture("doctor", "--derive-pending")
    assert rc == 0
    assert "unstamped before:        1" in out
    assert "rows scanned:            1" in out
    assert "rows stamped:            1" in out
    assert "unstamped after:         0" in out

    store = MemoryStore(str(isolated_db))
    try:
        assert store.list_unstamped_ingest_log() == []
    finally:
        store.close()


def test_derive_pending_idempotent(isolated_db: Path) -> None:
    """Hypothesis: running --derive-pending twice over the same store
    produces a no-op on the second invocation (rows scanned == 0 the
    second time). Falsifiable if the second run touches any rows."""
    # Seed with an ingest so there's data, then verify no orphans.
    store = MemoryStore(str(isolated_db))
    try:
        ingest_turn(store, "Atomic commits beat batched commits.", source="user")
    finally:
        store.close()

    rc1, out1 = _capture("doctor", "--derive-pending")
    rc2, out2 = _capture("doctor", "--derive-pending")
    assert rc1 == 0
    assert rc2 == 0
    # Second run: nothing to do.
    assert "unstamped before:        0" in out2
    assert "rows scanned:            0" in out2


def test_run_worker_returns_aggregate_counts(isolated_db: Path) -> None:
    """Hypothesis: run_worker's WorkerResult correctly aggregates
    counts across multiple orphan rows. Falsifiable by any counter
    mismatch."""
    store = MemoryStore(str(isolated_db))
    try:
        for i in range(3):
            store.record_ingest(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                source_path=f"orphan-{i}.md",
                raw_text=f"Distinct fact number {i}.",
                session_id="sess-orphan",
                ts="2026-04-26T00:00:00Z",
                raw_meta={"call_site": "filesystem_ingest"},
            )
        result = run_worker(store)
        assert result.rows_scanned == 3
        assert result.rows_stamped == 3
        assert result.beliefs_inserted == 3
        assert result.beliefs_corroborated == 0
    finally:
        store.close()
