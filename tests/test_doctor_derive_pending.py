"""Tests for `aelf doctor --derive-pending` (#264 escape hatch).

Falsifiable hypotheses:

    The sweep stamps every previously-unstamped log row by invoking
    `run_worker(store)`, idempotent on a clean store, and reports
    accurate before/after counts.
"""
from __future__ import annotations

import argparse
import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_doctor_derive_pending
from aelfrice.models import (
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_FILESYSTEM,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    db = tmp_path / "doctor-derive-pending.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    yield s
    s.close()


def test_sweep_clean_store_is_no_op(
    store: MemoryStore,  # noqa: ARG001
) -> None:
    """Hypothesis: invoking --derive-pending on a store with no unstamped
    rows scans 0 rows, stamps 0 rows, and returns 0."""
    out = io.StringIO()
    rc = _cmd_doctor_derive_pending(argparse.Namespace(), out)
    assert rc == 0
    body = out.getvalue()
    assert "rows scanned:        0" in body
    assert "rows stamped:        0" in body
    assert "unstamped before:    0" in body
    assert "unstamped after:     0" in body


def test_sweep_stamps_orphan_rows(
    store: MemoryStore,
) -> None:
    """Hypothesis: rows hand-inserted via record_ingest with no
    derived_belief_ids stamp count as `unstamped before`. After the
    sweep, all rows are stamped and `unstamped after` is 0. Falsifiable
    if any row remains unstamped."""
    # Two unstamped rows simulating a pre-#264 worker crash mid-batch.
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="user",
        raw_text="The default port is 8080.",
        ts="2026-05-06T00:00:00Z",
        raw_meta={"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
    )
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="user",
        raw_text="Aelfrice stores beliefs in SQLite.",
        ts="2026-05-06T00:00:01Z",
        raw_meta={"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
    )
    assert len(store.list_unstamped_ingest_log()) == 2

    out = io.StringIO()
    rc = _cmd_doctor_derive_pending(argparse.Namespace(), out)
    assert rc == 0
    assert "unstamped before:    2" in out.getvalue()
    assert "rows stamped:        2" in out.getvalue()
    assert "unstamped after:     0" in out.getvalue()
    assert store.list_unstamped_ingest_log() == []


def test_sweep_is_idempotent(store: MemoryStore) -> None:
    """Hypothesis: running the sweep a second time on an already-stamped
    store reports zero scanned, zero stamped, and exits 0."""
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="user",
        raw_text="One log row to seed the sweep.",
        ts="2026-05-06T00:00:00Z",
        raw_meta={"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
    )
    out1 = io.StringIO()
    _cmd_doctor_derive_pending(argparse.Namespace(), out1)
    assert "rows stamped:        1" in out1.getvalue()

    out2 = io.StringIO()
    rc = _cmd_doctor_derive_pending(argparse.Namespace(), out2)
    assert rc == 0
    body2 = out2.getvalue()
    assert "rows scanned:        0" in body2
    assert "rows stamped:        0" in body2
    assert "unstamped before:    0" in body2
    assert "unstamped after:     0" in body2
