"""#1650: `benchmarks/supports_log_latency.py` runs and measures real matches.

Driven as a subprocess: the script points `HOME` and the store paths at
a sandbox when it is imported, and that must not leak into this session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "supports_log_latency.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True, text=True, timeout=110,
    )


@pytest.mark.timeout(120)
def test_dry_run_builds_nothing_and_exits_zero() -> None:
    proc = _run("--dry-run", "--beliefs", "50", "--turns", "4")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("would build 50 beliefs")


@pytest.mark.timeout(120)
def test_a_small_run_logs_matches_only_in_the_on_arm() -> None:
    proc = _run("--beliefs", "300", "--turns", "6")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["off"]["logged_rows"] == 0
    assert "pass_calls" not in report["off"]
    assert report["on"]["logged_rows"] > 0
    assert report["on"]["pass_calls"] == 6
    for key in ("turn_median_ms", "turn_p99_ms", "pass_median_ms", "pass_p99_ms"):
        assert report["on"][key] >= 0


@pytest.mark.timeout(120)
def test_a_uniform_short_turn_run_reports_its_corpus_and_logs_matches() -> None:
    proc = _run("--beliefs", "300", "--turns", "6", "--vocabulary", "60",
                "--uniform", "--turn-words", "5")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert (report["vocabulary"], report["uniform"], report["turn_words"]) == (
        60, True, 5)
    assert report["on"]["logged_rows"] > 0
