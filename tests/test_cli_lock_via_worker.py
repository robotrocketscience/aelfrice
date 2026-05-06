"""Integration tests for the post-#264 `aelf lock` → derivation_worker
call shape.

Mirrors test_ingest_via_worker.py for the CLI lock entry point. After
slice 2 ships, every `aelf lock <statement>` invocation appends one
unstamped log row (with `raw_meta.call_site = cli_remember`) and
invokes `run_worker(store)` once at end-of-call. Re-lock semantics
(applying lock-upgrade to a pre-existing lock-id belief) are layered
on top of the worker's stamped output, not used as a parallel-write
short-circuit.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Fresh DB per test under tmp_path."""
    db_path = tmp_path / "lock-via-worker.db"
    monkeypatch.setenv("AELFRICE_DB", str(db_path))
    monkeypatch.setenv("AELFRICE_SESSION_ID", "sess-test")
    yield db_path


def _open_store(db_path: Path) -> MemoryStore:
    return MemoryStore(str(db_path))


def test_every_log_row_is_stamped_after_lock(isolated_db: Path) -> None:
    """Hypothesis: after `aelf lock` returns 0, no log row remains
    unstamped — the worker ran end-of-call. Falsifiable by any row
    whose `derived_belief_ids IS NULL`."""
    rc = cli_main(["lock", "we always sign commits with ssh"])
    assert rc == 0
    store = _open_store(isolated_db)
    try:
        assert store.list_unstamped_ingest_log() == []
    finally:
        store.close()


def test_log_row_carries_call_site_metadata(isolated_db: Path) -> None:
    """Hypothesis: `aelf lock` stamps `raw_meta.call_site=cli_remember`
    so the worker resolves the corroboration source unambiguously.
    Falsifiable if `raw_meta` is missing or carries a different
    call_site."""
    rc = cli_main(["lock", "the canonical port is 8080"])
    assert rc == 0
    store = _open_store(isolated_db)
    try:
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT raw_meta FROM ingest_log"
        ).fetchall()
        assert rows
        import json
        for row in rows:
            meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
            assert isinstance(meta, dict)
            assert meta.get("call_site") == "cli_remember"
    finally:
        store.close()


def test_re_lock_appends_log_row_and_keeps_lock(isolated_db: Path) -> None:
    """Hypothesis: re-locking the same statement appends a second log
    row (canonical history shows both invocations) AND re-applies
    lock-upgrade semantics (the belief stays LOCK_USER with refreshed
    timestamp). Falsifiable if the second invocation drops the row OR
    leaves the belief in a non-LOCK_USER state."""
    cli_main(["lock", "atomic commits beat batched commits"])
    rc = cli_main(["lock", "atomic commits beat batched commits"])
    assert rc == 0
    store = _open_store(isolated_db)
    try:
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) AS n FROM ingest_log",
        ).fetchone()
        assert rows["n"] == 2
        # Still exactly one locked belief (no double-insert).
        assert store.count_locked() == 1
    finally:
        store.close()


def test_replay_full_equality_passes_after_lock(isolated_db: Path) -> None:
    """Hypothesis (CI gate): after a representative lock invocation,
    the full-equality replay probe reports zero drift. Falsifiable by
    any non-zero drift counter — the `cli_remember` call site must
    re-derive to the same canonical belief on replay."""
    cli_main(["lock", "the dashboard runs on port 8080"])
    cli_main(["lock", "always use uv to manage python envs"])
    store = _open_store(isolated_db)
    try:
        report = replay_full_equality(store)
        assert report.total_log_rows > 0
        assert report.matched == report.total_log_rows, (
            f"replay drift: matched={report.matched}, "
            f"mismatched={report.mismatched}, "
            f"derived_orphan={report.derived_orphan}, "
            f"canonical_orphan={report.canonical_orphan}, "
            f"examples={report.drift_examples}"
        )
    finally:
        store.close()
