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


def _pass(date: str, sha: str | None = None) -> dict:  # type: ignore[type-arg]
    """A green row. `sha` defaults to one derived from `date`, i.e. a
    moving `main` — pass an explicit `sha` to model an idle one."""
    return {
        "date": date, "sha": sha if sha is not None else f"sha-{date}",
        "replay_full_equality_result": "pass",
        "total_log_rows": 60, "mismatched": 0, "derived_orphan": 0,
    }


def _fail(date: str, sha: str | None = None) -> dict:  # type: ignore[type-arg]
    return {
        "date": date, "sha": sha if sha is not None else f"sha-{date}",
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


# --- distinct-commit counting (#1239) --------------------------------------


def test_repeated_sha_counts_once(tmp_path: Path) -> None:
    """Hypothesis: consecutive rows for the same commit are one measurement.

    The soak is deterministic, so replaying an unchanged tree reproduces the
    previous result and adds no evidence. Falsifiable if the streak counts
    rows: seven identical-sha rows would then return 7.
    """
    rows = [_pass(f"2026-07-{n:02d}", sha="018eb88a") for n in range(23, 30)]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 1


def test_the_idle_week_no_longer_satisfies_the_threshold(tmp_path: Path) -> None:
    """Regression for the real history that motivated #1239.

    `main` did not advance 2026-07-22 -> 2026-07-29 and the cron recorded
    `018eb88a` on seven consecutive days. Those seven rows cleared the
    threshold of 7 on their own. They must not.
    """
    rows = [
        _pass("2026-07-22", sha="1a4f41b6"),
        *[_pass(f"2026-07-{n:02d}", sha="018eb88a") for n in range(23, 30)],
    ]
    p = tmp_path / "status.json"
    _write(rows, p)
    n = replay_soak_streak.streak(replay_soak_streak.load_rows(p))
    assert n == 2, f"idle week should contribute one commit, not seven (got {n})"
    assert n < 7, "the idle week must not satisfy the gate on its own"


def test_alternating_shas_are_not_collapsed(tmp_path: Path) -> None:
    """Only *consecutive* repeats collapse.

    A -> B -> A is three measurements of two trees; the middle entry proves
    the tree changed and changed back, so the third is not a repeat of the
    first. Falsifiable if the implementation de-duplicates set-wise.
    """
    rows = [_pass("2026-05-01", sha="a"),
            _pass("2026-05-02", sha="b"),
            _pass("2026-05-03", sha="a")]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 3


def test_rows_without_a_sha_each_count(tmp_path: Path) -> None:
    """Absent provenance is not evidence of sameness, so no collapse.

    Pre-schema or hand-edited rows carry no `sha`. Counting them separately
    preserves the old behaviour for exactly those rows rather than silently
    merging entries that cannot be shown to be the same commit.
    """
    rows = [{k: v for k, v in _pass(f"2026-05-{n:02d}").items() if k != "sha"}
            for n in range(1, 4)]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert all("sha" not in r for r in rows)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 3


def test_a_repeat_does_not_break_the_streak(tmp_path: Path) -> None:
    """A repeated commit is not counted, but it is also not a break.

    Falsifiable if the dedupe were implemented as a `break`: the streak would
    stop at the repeat and report 1 instead of counting the earlier commits.
    """
    rows = [_pass("2026-05-01", sha="a"),
            _pass("2026-05-02", sha="b"),
            _pass("2026-05-03", sha="b")]
    p = tmp_path / "status.json"
    _write(rows, p)
    assert replay_soak_streak.streak(replay_soak_streak.load_rows(p)) == 2
