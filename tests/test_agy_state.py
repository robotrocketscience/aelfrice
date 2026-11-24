"""Per-conversation agy state (`aelfrice.agy_state`)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from aelfrice import agy_state

pytestmark = pytest.mark.timeout(30)


@pytest.fixture(autouse=True)
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(agy_state.STATE_DIR_ENV, str(tmp_path / "state"))
    return tmp_path / "state"


def test_pending_round_trip_is_consumed_once() -> None:
    assert agy_state.append_pending("c1", "first")
    assert agy_state.append_pending("c1", "second")
    assert agy_state.take_pending("c1") == "first\n\nsecond"
    assert agy_state.take_pending("c1") == ""


def test_pending_is_per_conversation() -> None:
    agy_state.append_pending("c1", "one")
    agy_state.append_pending("c2", "two")
    assert agy_state.take_pending("c2") == "two"
    assert agy_state.take_pending("c1") == "one"


def test_blank_context_is_not_queued() -> None:
    assert not agy_state.append_pending("c1", "  \n")
    assert agy_state.take_pending("c1") == ""


def test_pending_keeps_the_newest_text_within_the_cap() -> None:
    big = "x" * agy_state.MAX_PENDING_CHARS
    agy_state.append_pending("c1", big)
    agy_state.append_pending("c1", "NEWEST")
    out = agy_state.take_pending("c1")
    assert len(out) == agy_state.MAX_PENDING_CHARS
    assert out.endswith("NEWEST")


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "", "x" * 200])
def test_unsafe_ids_write_nothing(bad: str, state_root: Path) -> None:
    assert not agy_state.append_pending(bad, "ctx")
    agy_state.mark_baseline(bad)
    assert not agy_state.has_baseline(bad)
    assert not any(state_root.rglob("*")) or all(p.is_dir() for p in state_root.rglob("*"))


def test_baseline_marker() -> None:
    assert not agy_state.has_baseline("c1")
    agy_state.mark_baseline("c1")
    assert agy_state.has_baseline("c1")
    assert not agy_state.has_baseline("c2")


def test_sweep_removes_only_stale_files() -> None:
    agy_state.mark_baseline("old")
    agy_state.mark_baseline("new")
    agy_state.append_pending("old", "ctx")
    old_t = time.time() - agy_state.STALE_AFTER_S - 60
    root = agy_state.state_dir()
    for path in (root / "baseline" / "old", root / "pending" / "old.txt"):
        os.utime(path, (old_t, old_t))
    assert agy_state.sweep_stale() == 2
    assert agy_state.has_baseline("new")
    assert not agy_state.has_baseline("old")
    assert agy_state.take_pending("old") == ""
