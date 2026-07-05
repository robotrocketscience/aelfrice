"""Acceptance tests for `aelf doctor --prune-noise` (#1029).

Re-applies the current ingest filters (is_transcript_noise + classifier
persist gate) to existing beliefs and soft-deletes the matches. Safety:
only agent_inferred, non-locked, active beliefs are eligible; soft-delete
is reversible (sets valid_to).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore


def _mk(
    store: MemoryStore,
    bid: str,
    content: str,
    *,
    origin: str = ORIGIN_AGENT_INFERRED,
    lock_level: str = LOCK_NONE,
) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=lock_level,
            locked_at="2026-06-01T00:00:00Z" if lock_level == LOCK_USER else None,
            created_at="2026-06-01T00:00:00Z",
            last_retrieved_at=None,
            origin=origin,
        )
    )


def _seed(store: MemoryStore) -> None:
    # Noise (agent_inferred, non-locked) — should be pruned.
    _mk(store, "n_tag", "<tool-use-id>toolu_abc</tool-use-id>")
    _mk(store, "n_q", "Want me to run it?")
    _mk(store, "n_ack", "Polling.")
    # Durable agent_inferred fact — must survive.
    _mk(store, "keep_fact", "The project stores beliefs in SQLite under .git.")
    # Safety: noise-shaped but user-sourced or locked — must survive.
    _mk(store, "user_q", "Which way?", origin=ORIGIN_USER_STATED)
    _mk(store, "locked_q", "Want me to run it?", lock_level=LOCK_USER)


def _run(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _active_ids(db: Path) -> set[str]:
    store = MemoryStore(str(db))
    try:
        return {b.id for b in store.list_active_beliefs()}
    finally:
        store.close()


def test_prune_noise_dry_run_reports_without_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    code, out = _run(monkeypatch, db, "doctor", "--prune-noise")
    assert code == 0
    assert "3 active agent_inferred non-locked belief(s)" in out
    assert "dry-run" in out
    # nothing soft-deleted
    assert _active_ids(db) == {
        "n_tag", "n_q", "n_ack", "keep_fact", "user_q", "locked_q",
    }


def test_prune_noise_apply_soft_deletes_only_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    code, out = _run(monkeypatch, db, "doctor", "--prune-noise", "--apply")
    assert code == 0
    assert "soft-deleted 3 belief(s)" in out
    # noise gone; fact + user-sourced + locked survive.
    assert _active_ids(db) == {"keep_fact", "user_q", "locked_q"}


def test_prune_noise_removes_stranded_capture_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1081: a stored orphan section header ("Recommendation:") and a
    shell-output echo ("$ …") are stranded standalone rows the ingest-time
    sub-floor detector never got to anchor. The GC prunes them while a
    real assertion that merely contains a colon survives."""
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _mk(s, "hdr", "Recommendation:")
        _mk(s, "hdr2", "Two paths forward:")
        _mk(s, "echo", "$ python run_all.py")
        _mk(s, "keep_colon", "Recommendation: use the read-through cache.")
        _mk(s, "keep_fact", "The pipeline orders feedback ahead of corroboration.")
    finally:
        s.close()
    code, out = _run(monkeypatch, db, "doctor", "--prune-noise", "--apply")
    assert code == 0
    assert "soft-deleted 3 belief(s)" in out
    assert _active_ids(db) == {"keep_colon", "keep_fact"}


def test_prune_noise_max_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    code, out = _run(
        monkeypatch, db, "doctor", "--prune-noise", "--apply", "--max", "1"
    )
    assert code == 0
    assert "soft-deleted 1 belief(s)" in out
    assert len(_active_ids(db)) == 5  # 6 seeded - 1 pruned


def test_prune_noise_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    _run(monkeypatch, db, "doctor", "--prune-noise", "--apply")
    code, out = _run(monkeypatch, db, "doctor", "--prune-noise")
    assert code == 0
    assert "0 active agent_inferred non-locked belief(s)" in out


def test_prune_noise_keeps_fts_sync_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After pruning, the fts_sync audit must stay in sync (it counts
    active beliefs, not all rows — #1029 companion fix)."""
    from aelfrice.auditor import CHECK_FTS_SYNC, SEVERITY_INFO, audit
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    _run(monkeypatch, db, "doctor", "--prune-noise", "--apply")
    s = MemoryStore(str(db))
    try:
        report = audit(s)
    finally:
        s.close()
    fts = next(f for f in report.findings if f.check == CHECK_FTS_SYNC)
    assert fts.severity == SEVERITY_INFO, fts.detail
