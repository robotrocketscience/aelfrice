"""Tests for `scripts/replay_soak_streak.py` (#403 C).

The streak computation is the merge-gate primitive; tests cover the
boundary cases that would actually mask drift on real history.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

# Importable path for the script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import replay_soak_streak  # type: ignore[import-not-found]


def _write(rows: list[dict], path: Path) -> None:  # type: ignore[type-arg]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _pass(date: str, sha: str = "abc") -> dict:  # type: ignore[type-arg]
    return {
        "date": date, "sha": sha,
        "replay_full_equality_result": "pass",
        "total_log_rows": 60, "mismatched": 0, "derived_orphan": 0,
    }


def _fail(date: str, sha: str = "abc") -> dict:  # type: ignore[type-arg]
    return {
        "date": date, "sha": sha,
        "replay_full_equality_result": "fail",
        "total_log_rows": 60, "mismatched": 1, "derived_orphan": 0,
    }


def test_empty_status_file_streak_zero(tmp_path: Path) -> None:
    """Hypothesis: an empty file (no entries yet) returns streak=0.
    Falsifiable if streak is ever non-zero on an empty input."""
    p = tmp_path / "status.json"
    p.write_text("")
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 0


def test_missing_status_file_streak_zero(tmp_path: Path) -> None:
    """Hypothesis: a non-existent file is treated as zero entries.
    Falsifiable if `load_rows` raises or returns garbage."""
    p = tmp_path / "absent.json"
    assert replay_soak_streak.load_rows(p) == []
    assert replay_soak_streak.streak([]) == 0


def test_seven_consecutive_pass(tmp_path: Path) -> None:
    """Hypothesis: 7 consecutive passes return streak=7.
    Falsifiable if the count is off by one or breaks early."""
    rows = [_pass(f"2026-05-{n:02d}") for n in range(1, 8)]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 7


def test_streak_breaks_on_fail(tmp_path: Path) -> None:
    """Hypothesis: a `fail` row breaks the streak count at the first
    fail walking backwards from the tail. Falsifiable if a fail in
    the middle is ignored, or if the count includes the fail."""
    rows = [
        _pass("2026-05-01"),
        _pass("2026-05-02"),
        _fail("2026-05-03"),  # break
        _pass("2026-05-04"),
        _pass("2026-05-05"),
        _pass("2026-05-06"),
    ]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 3


def test_streak_breaks_on_drift_count(tmp_path: Path) -> None:
    """Hypothesis: a row whose `mismatched + derived_orphan != 0`
    breaks the streak even if `replay_full_equality_result == "pass"`.
    Falsifiable if the drift counters are ignored."""
    rows = [
        _pass("2026-05-01"),
        _pass("2026-05-02"),
        {**_pass("2026-05-03"), "mismatched": 1},  # drift but result=pass
        _pass("2026-05-04"),
    ]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 1


def test_malformed_jsonl_raises(tmp_path: Path) -> None:
    """Hypothesis: a non-JSON line raises SystemExit (exit code 2 in main)."""
    p = tmp_path / "status.json"
    p.write_text("not-json\n")
    with pytest.raises(SystemExit):
        replay_soak_streak.load_rows(p)
