"""Integration tests for the post-#264 cli `aelf lock` →
derivation_worker call shape.

Falsifiable hypothesis: aelf lock writes one ingest_log row with
`call_site=cli_remember`, invokes run_worker once, and reads the
stamped belief id back. Re-lock semantic still lives at the entry
point.
"""
from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_lock
from aelfrice.models import LOCK_USER, ORIGIN_USER_STATED
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    db = tmp_path / "cli-lock-via-worker.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    yield s
    s.close()


def _ns(statement: str, session_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(statement=statement, session_id=session_id)


def test_lock_writes_log_row_with_call_site(
    store: MemoryStore, capsys: pytest.CaptureFixture[str],
) -> None:
    """Hypothesis: aelf lock writes exactly one log row with
    raw_meta.call_site = cli_remember. Falsifiable if the row is
    missing or carries a different call_site."""
    out = io.StringIO()
    rc = _cmd_lock(_ns("atomic commits beat batched"), out)
    assert rc == 0
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta, source_kind FROM ingest_log"
    ).fetchall()
    assert len(rows) == 1
    meta = json.loads(rows[0]["raw_meta"]) if rows[0]["raw_meta"] else None
    assert isinstance(meta, dict)
    assert meta.get("call_site") == "cli_remember"
    assert rows[0]["source_kind"] == "cli_remember"


def test_log_row_is_stamped_after_lock(store: MemoryStore) -> None:
    """Hypothesis: the worker ran end-of-call and stamped the log row.
    Falsifiable if any unstamped row remains after _cmd_lock."""
    _cmd_lock(_ns("the sky is blue"), io.StringIO())
    unstamped = store.list_unstamped_ingest_log()
    assert unstamped == []


def test_relock_upgrades_existing_lock_id(store: MemoryStore) -> None:
    """Hypothesis: re-locking refreshes lock_level/locked_at on the
    existing lock-id belief via update_belief. Falsifiable if the
    second call inserts a duplicate or leaves the prior belief's
    lock fields stale."""
    out1 = io.StringIO()
    _cmd_lock(_ns("we always sign commits"), out1)
    msg1 = out1.getvalue()
    assert msg1.startswith("locked: ")
    bid = msg1.split("locked: ")[1].strip()
    # Demote so the upgrade is observable.
    b = store.get_belief(bid)
    assert b is not None
    b.lock_level = 0
    b.locked_at = None
    b.demotion_pressure = 5
    store.update_belief(b)

    out2 = io.StringIO()
    _cmd_lock(_ns("we always sign commits"), out2)
    msg2 = out2.getvalue()
    assert msg2.startswith("upgraded existing belief to lock: ")
    refreshed = store.get_belief(bid)
    assert refreshed is not None
    assert refreshed.lock_level == LOCK_USER
    assert refreshed.locked_at is not None
    assert refreshed.demotion_pressure == 0
    assert refreshed.origin == ORIGIN_USER_STATED


def test_replay_full_equality_passes_after_lock(store: MemoryStore) -> None:
    """Hypothesis (CI gate): an aelf lock workload passes the full-equality
    replay probe with zero drift. Falsifiable by any non-zero counter."""
    _cmd_lock(_ns("we always sign commits with ssh"), io.StringIO())
    _cmd_lock(_ns("aelfrice stores beliefs in sqlite"), io.StringIO())
    # Re-lock to exercise the upgrade path.
    _cmd_lock(_ns("we always sign commits with ssh"), io.StringIO())

    report = replay_full_equality(store)
    assert report.total_log_rows > 0
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0


def test_lock_idempotent_on_canonical_state(store: MemoryStore) -> None:
    """Hypothesis: re-locking is idempotent on the canonical belief set
    even though it adds a new log row per call."""
    out1 = io.StringIO()
    _cmd_lock(_ns("atomic commits beat batched"), out1)
    bid_1 = out1.getvalue().split("locked: ")[1].strip()
    n_before = store.count_beliefs()

    _cmd_lock(_ns("atomic commits beat batched"), io.StringIO())
    assert store.count_beliefs() == n_before
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log WHERE raw_text = ?",
        ("atomic commits beat batched",),
    ).fetchone()
    assert rows["n"] == 2
    # Same lock-id resolves both calls.
    assert store.get_belief(bid_1) is not None
