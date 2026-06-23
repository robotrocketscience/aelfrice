"""Opt-in phantom auto-GC on SessionStart (#980 item 2).

The wonder GC exit is wired and correct but had never run in any store
(the #980 audit: 0 phantoms GC'd, ever). This lane makes GC actually run,
once per session, behind a default-off env flag — covering the helper
parsing, the no-op-when-disabled contract, and the full session_start()
integration (soft-delete + `wonder.gc` feed emission + stderr notice).
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.hook import (
    ENV_WONDER_AUTOGC,
    ENV_WONDER_AUTOGC_TTL_DAYS,
    _WONDER_AUTOGC_DEFAULT_TTL_DAYS,
    _maybe_run_wonder_autogc,
    _wonder_autogc_enabled,
    _wonder_autogc_ttl_days,
    session_start,
)
from aelfrice.models import (
    BELIEF_SPECULATIVE,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _old_ts(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _insert_stale_phantom(db_path: Path, bid: str, days_ago: int = 15) -> None:
    """Insert one stale speculative belief into the on-disk store."""
    store = MemoryStore(str(db_path))
    try:
        store.insert_belief(
            Belief(
                id=bid,
                content=f"speculative content {bid}",
                content_hash=f"spec_hash_{bid}",
                alpha=0.3,
                beta=1.0,
                type=BELIEF_SPECULATIVE,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at=_old_ts(days_ago),
                last_retrieved_at=None,
                origin=ORIGIN_SPECULATIVE,
                retention_class="snapshot",
            )
        )
    finally:
        store.close()


def _read_feed_events(db_dir: Path) -> list[dict[str, object]]:
    feed = db_dir / "feed.jsonl"
    if not feed.exists():
        return []
    return [json.loads(line) for line in feed.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# _wonder_autogc_enabled — default-off, truthy parsing
# ---------------------------------------------------------------------------


def test_autogc_disabled_by_default() -> None:
    assert _wonder_autogc_enabled(env={}) is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " On "])
def test_autogc_enabled_truthy(val: str) -> None:
    assert _wonder_autogc_enabled(env={ENV_WONDER_AUTOGC: val}) is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "maybe"])
def test_autogc_disabled_falsy(val: str) -> None:
    assert _wonder_autogc_enabled(env={ENV_WONDER_AUTOGC: val}) is False


# ---------------------------------------------------------------------------
# _wonder_autogc_ttl_days — default 14, override, malformed fallback
# ---------------------------------------------------------------------------


def test_ttl_days_default_when_unset() -> None:
    assert _wonder_autogc_ttl_days(env={}) == _WONDER_AUTOGC_DEFAULT_TTL_DAYS


def test_ttl_days_honors_override() -> None:
    assert _wonder_autogc_ttl_days(env={ENV_WONDER_AUTOGC_TTL_DAYS: "30"}) == 30


@pytest.mark.parametrize("val", ["", "abc", "0", "-3", "  "])
def test_ttl_days_falls_back_on_bad_value(val: str) -> None:
    got = _wonder_autogc_ttl_days(env={ENV_WONDER_AUTOGC_TTL_DAYS: val})
    assert got == _WONDER_AUTOGC_DEFAULT_TTL_DAYS


# ---------------------------------------------------------------------------
# _maybe_run_wonder_autogc — no-op when disabled
# ---------------------------------------------------------------------------


def test_autogc_noop_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _insert_stale_phantom(db, "spec1")
    # Flag unset → disabled.
    monkeypatch.delenv(ENV_WONDER_AUTOGC, raising=False)

    serr = io.StringIO()
    _maybe_run_wonder_autogc(serr)

    store = MemoryStore(str(db))
    try:
        b = store.get_belief("spec1")
    finally:
        store.close()
    assert b is not None
    assert b.valid_to is None  # untouched
    assert serr.getvalue() == ""
    assert _read_feed_events(db_dir) == []


# ---------------------------------------------------------------------------
# _maybe_run_wonder_autogc — sweeps + emits feed row + stderr when enabled
# ---------------------------------------------------------------------------


def test_autogc_sweeps_stale_phantom_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _insert_stale_phantom(db, "spec1")
    monkeypatch.setenv(ENV_WONDER_AUTOGC, "1")

    serr = io.StringIO()
    _maybe_run_wonder_autogc(serr)

    store = MemoryStore(str(db))
    try:
        b = store.get_belief("spec1")
    finally:
        store.close()
    assert b is not None
    assert b.valid_to is not None  # soft-deleted

    events = _read_feed_events(db_dir)
    gc_rows = [r for r in events if r.get("event") == "wonder.gc"]
    assert len(gc_rows) == 1
    assert gc_rows[0]["deleted"] == 1
    assert gc_rows[0]["trigger"] == "sessionstart_autogc"
    assert "swept 1 stale phantom" in serr.getvalue()


def test_autogc_no_feed_row_when_nothing_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _insert_stale_phantom(db, "fresh", days_ago=1)  # too young to GC
    monkeypatch.setenv(ENV_WONDER_AUTOGC, "1")

    serr = io.StringIO()
    _maybe_run_wonder_autogc(serr)

    store = MemoryStore(str(db))
    try:
        b = store.get_belief("fresh")
    finally:
        store.close()
    assert b is not None
    assert b.valid_to is None  # survived
    assert _read_feed_events(db_dir) == []
    assert serr.getvalue() == ""


# ---------------------------------------------------------------------------
# Full session_start() integration
# ---------------------------------------------------------------------------


def test_session_start_runs_autogc_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _insert_stale_phantom(db, "spec1")
    monkeypatch.setenv(ENV_WONDER_AUTOGC, "1")
    # Keep the recap lane quiet so it can't interfere with the assertion.
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")

    sout, serr = io.StringIO(), io.StringIO()
    rc = session_start(
        stdin=io.StringIO('{"session_id": "test-session"}'),
        stdout=sout,
        stderr=serr,
    )
    assert rc == 0

    store = MemoryStore(str(db))
    try:
        b = store.get_belief("spec1")
    finally:
        store.close()
    assert b is not None
    assert b.valid_to is not None  # GC ran during SessionStart
    # GC notice goes to stderr, never to the injected stdout block.
    assert "wonder auto-GC" in serr.getvalue()
    assert "wonder auto-GC" not in sout.getvalue()


def test_session_start_skips_autogc_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _insert_stale_phantom(db, "spec1")
    monkeypatch.delenv(ENV_WONDER_AUTOGC, raising=False)

    rc = session_start(
        stdin=io.StringIO('{"session_id": "test-session"}'),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0

    store = MemoryStore(str(db))
    try:
        b = store.get_belief("spec1")
    finally:
        store.close()
    assert b is not None
    assert b.valid_to is None  # untouched — opt-in stays off by default
