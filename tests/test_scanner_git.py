"""extract_git_log: read commit subjects via subprocess git log.

Tests use tmp_path + a real `git init` + `git commit` so the extractor
runs against actual git output (not a mock). Identity is set per
invocation via `git -c user.name=... -c user.email=...` so the tests
work without depending on the runner's git config.

If `git` is not on PATH, the relevant tests skip via pytest.importorskip
on shutil.which.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aelfrice.scanner import SentenceCandidate, extract_git_log

_GIT_AVAILABLE = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not _GIT_AVAILABLE, reason="git binary not on PATH")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand against repo with hermetic identity."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=check,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")


def _make_commit(repo: Path, filename: str, subject: str) -> None:
    (repo / filename).write_text("x", encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", subject)


# --- Empty / missing inputs ---------------------------------------------


def test_missing_path_returns_empty(tmp_path: Path) -> None:
    bogus = tmp_path / "nope"
    assert extract_git_log(bogus) == []


def test_file_path_instead_of_directory_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    assert extract_git_log(f) == []


def test_non_git_directory_returns_empty(tmp_path: Path) -> None:
    """Directory exists but has no .git — not a repo."""
    assert extract_git_log(tmp_path) == []


@needs_git
def test_repo_with_no_commits_returns_empty(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert extract_git_log(tmp_path) == []


# --- Single + multi-commit extraction -----------------------------------


@needs_git
def test_single_commit_yields_one_candidate(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: initial scaffold")
    candidates = extract_git_log(tmp_path)
    assert len(candidates) == 1


@needs_git
def test_three_commits_yield_three_candidates(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: first")
    _make_commit(tmp_path, "b.txt", "feat: second")
    _make_commit(tmp_path, "c.txt", "feat: third")
    candidates = extract_git_log(tmp_path)
    assert len(candidates) == 3


@needs_git
def test_commit_subject_is_candidate_text(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: add retrieval module")
    c = extract_git_log(tmp_path)[0]
    assert c.text == "feat: add retrieval module"


# --- Source format ------------------------------------------------------


@needs_git
def test_source_starts_with_git_commit_prefix(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: x")
    c = extract_git_log(tmp_path)[0]
    assert c.source.startswith("git:commit:")


@needs_git
def test_source_includes_seven_char_short_hash(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: x")
    c = extract_git_log(tmp_path)[0]
    short_hash = c.source.split(":")[-1]
    assert len(short_hash) == 7


@needs_git
def test_distinct_commits_yield_distinct_short_hashes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: a")
    _make_commit(tmp_path, "b.txt", "feat: b")
    candidates = extract_git_log(tmp_path)
    hashes = [c.source for c in candidates]
    assert len(set(hashes)) == 2


# --- Ordering -----------------------------------------------------------


@needs_git
def test_most_recent_commit_first(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: oldest")
    _make_commit(tmp_path, "b.txt", "feat: middle")
    _make_commit(tmp_path, "c.txt", "feat: newest")
    candidates = extract_git_log(tmp_path)
    assert candidates[0].text == "feat: newest"
    assert candidates[-1].text == "feat: oldest"


# --- Limit clamp --------------------------------------------------------


@needs_git
def test_limit_caps_returned_count(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    for i in range(5):
        _make_commit(tmp_path, f"f{i}.txt", f"feat: c{i}")
    candidates = extract_git_log(tmp_path, limit=2)
    assert len(candidates) == 2


@needs_git
def test_limit_one_returns_only_newest(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: oldest")
    _make_commit(tmp_path, "b.txt", "feat: newest")
    candidates = extract_git_log(tmp_path, limit=1)
    assert len(candidates) == 1
    assert candidates[0].text == "feat: newest"


# --- Result types -------------------------------------------------------


@needs_git
def test_results_are_sentence_candidates(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _make_commit(tmp_path, "a.txt", "feat: x")
    candidates = extract_git_log(tmp_path)
    assert all(isinstance(c, SentenceCandidate) for c in candidates)
