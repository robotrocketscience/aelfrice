"""The pattern scan fails closed on a pattern it cannot run.

Audit finding CI-02. Both pattern jobs in `staging-gate.yml` looped over
the rules decoded from `SCAN_RULES_B64` with `if grep -E "$p" ...`. grep
exits 2 when a pattern does not compile, and `if` reads every non-zero
status as "no hit", so a malformed rule switched itself off and the job
still passed. `history-scan` also never checked that the decoded file
was non-empty, so a missing secret scanned nothing and passed.

The loop now lives in `scripts/scan_patterns.sh`, which both jobs call.
These tests run that script against dummy pattern files and pin the
workflow wiring as text, the way `tests/test_ci_manual_dispatch.py`
reads the same file.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(60)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "scan_patterns.sh"

# Dummy rules only. None of these resembles a real scan rule.
_HIT = "dummy-needle-[0-9]+"
_MISS = "absent-dummy-[a-z]{40}"
_BROKEN = "dummyopen("


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *(str(a) for a in args)],
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text("first line\nhas dummy-needle-42 in it\nlast line\n", encoding="utf-8")
    return path


def _patterns(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "patterns"
    path.write_text(body, encoding="utf-8")
    return path


def test_valid_pattern_with_a_hit_fails_and_reports_it(tmp_path: Path, corpus: Path) -> None:
    r = _run("--mode", "tree", _patterns(tmp_path, f"{_HIT}\n"), corpus)
    assert r.returncode == 1, r
    assert "::error::pattern hit" in r.stdout
    assert f"{corpus}:2:has dummy-needle-42 in it" in r.stdout


def test_valid_pattern_with_no_hit_passes(tmp_path: Path, corpus: Path) -> None:
    r = _run("--mode", "tree", _patterns(tmp_path, f"# a comment\n\n{_MISS}\n"), corpus)
    assert r.returncode == 0, r
    assert "::error::" not in r.stdout


@pytest.mark.parametrize("mode", ["tree", "history"])
def test_pattern_that_does_not_compile_fails_without_echoing_it(
    tmp_path: Path, corpus: Path, mode: str
) -> None:
    r = _run("--mode", mode, _patterns(tmp_path, f"{_MISS}\n{_BROKEN}\n"), corpus)
    assert r.returncode == 2, r
    assert "::error::pattern 2 did not compile" in r.stdout
    assert "dummyopen" not in r.stdout + r.stderr


def test_a_broken_pattern_is_not_masked_by_a_hit(tmp_path: Path, corpus: Path) -> None:
    """A hit elsewhere must not turn the run into a plain exit 1."""
    r = _run("--mode", "tree", _patterns(tmp_path, f"{_BROKEN}\n{_HIT}\n"), corpus)
    assert r.returncode == 2, r
    assert "::error::pattern 1 did not compile" in r.stdout
    assert "::error::pattern hit" in r.stdout


@pytest.mark.parametrize("body", ["", "# only a comment\n\n"], ids=["empty", "comments-only"])
def test_pattern_file_with_no_patterns_fails(tmp_path: Path, corpus: Path, body: str) -> None:
    r = _run("--mode", "tree", _patterns(tmp_path, body), corpus)
    assert r.returncode == 2, r
    assert "::error::pattern file" in r.stdout


def test_missing_pattern_file_fails(tmp_path: Path, corpus: Path) -> None:
    r = _run("--mode", "tree", tmp_path / "nope", corpus)
    assert r.returncode == 2, r
    assert "::error::pattern file is missing or empty" in r.stdout


def test_last_pattern_without_a_trailing_newline_is_scanned(
    tmp_path: Path, corpus: Path
) -> None:
    r = _run("--mode", "tree", _patterns(tmp_path, f"{_MISS}\n{_HIT}"), corpus)
    assert r.returncode == 1, r


def test_pattern_starting_with_a_dash_is_a_pattern_not_an_option(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    target.write_text("x -dummy-flag y\n", encoding="utf-8")
    r = _run("--mode", "tree", _patterns(tmp_path, "-dummy-flag\n"), target)
    assert r.returncode == 1, r


def test_unreadable_input_fails(tmp_path: Path, corpus: Path) -> None:
    r = _run("--mode", "history", _patterns(tmp_path, f"{_MISS}\n"), corpus, tmp_path)
    assert r.returncode == 2, r
    assert "::error::pattern 1: grep could not read the input" in r.stdout


def test_history_mode_reports_a_hit_without_the_matched_text(
    tmp_path: Path, corpus: Path
) -> None:
    r = _run("--mode", "history", _patterns(tmp_path, f"{_HIT}\n"), corpus)
    assert r.returncode == 1, r
    assert "::error::pattern hit in history" in r.stdout
    assert "dummy-needle-42" not in r.stdout + r.stderr


def test_dry_run_scans_nothing_but_still_rejects_a_bad_file(
    tmp_path: Path, corpus: Path
) -> None:
    ok = _run("--dry-run", "--mode", "tree", _patterns(tmp_path, f"{_HIT}\n"), corpus)
    assert ok.returncode == 0, ok
    assert "dry run: 1 patterns, 1 input files, nothing scanned" in ok.stdout
    assert "dummy-needle-42" not in ok.stdout
    bad = _run("--dry-run", "--mode", "tree", _patterns(tmp_path, f"{_BROKEN}\n"), corpus)
    assert bad.returncode == 2, bad


@pytest.mark.parametrize("args", [[], ["--mode", "tree"], ["--mode", "other", "p", "f"]])
def test_usage_errors_exit_non_zero(args: list[str]) -> None:
    r = _run(*args)
    assert r.returncode == 2, r
    assert "Usage:" in r.stderr

