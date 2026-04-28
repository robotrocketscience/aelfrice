"""Tests for the v1.1.0 git-recency weighting (#94, Tier 2 of CORE_RANKING).

The scanner records the most-recent author date of the commit that
touched each source file. `scan_repo` uses that as `belief.created_at`
so the existing decay mechanism penalises pre-migration content from
old branches.

Files outside any git work-tree, untracked files, and the entire
fallback when `git` is unavailable continue to use the wall-clock
`now` parameter — preserving the v1.0 behaviour.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aelfrice.scanner import (
    SentenceCandidate,
    _build_file_recency_map,
    extract_ast,
    extract_filesystem,
    extract_git_log,
    scan_repo,
)
from aelfrice.store import MemoryStore


def _git_init(repo: Path, env_extra: dict[str, str] | None = None) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)


def _git_commit(
    repo: Path,
    *files: str,
    message: str,
    author_date: str = "2024-01-01T00:00:00+00:00",
) -> None:
    subprocess.run(["git", "add", *files], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c", "user.email=t@t",
            "-c", "user.name=t",
            "-c", "commit.gpgsign=false",
            "commit", "-q", "-m", message,
            "--date", author_date,
        ],
        cwd=repo,
        check=True,
        env={
            **__import__("os").environ,
            "GIT_AUTHOR_DATE": author_date,
            "GIT_COMMITTER_DATE": author_date,
        },
    )


# --- _build_file_recency_map ---------------------------------------


def test_recency_map_empty_outside_git(tmp_path: Path) -> None:
    """A non-git directory yields no recency entries."""
    assert _build_file_recency_map(tmp_path) == {}


def test_recency_map_records_one_commit(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("hello", encoding="utf-8")
    _git_commit(repo, "README.md", message="add readme",
                author_date="2024-06-15T12:00:00+00:00")
    rec = _build_file_recency_map(repo)
    assert "README.md" in rec
    assert rec["README.md"].startswith("2024-06-15")


def test_recency_map_keeps_most_recent_when_file_touched_twice(
    tmp_path: Path,
) -> None:
    """A file modified across two commits is recorded with the newer date."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "f.md").write_text("v1", encoding="utf-8")
    _git_commit(repo, "f.md", message="v1",
                author_date="2023-01-01T00:00:00+00:00")
    (repo / "f.md").write_text("v2", encoding="utf-8")
    _git_commit(repo, "f.md", message="v2",
                author_date="2025-06-01T00:00:00+00:00")
    rec = _build_file_recency_map(repo)
    # newer commit comes first in `git log`, so first-seen wins
    assert rec["f.md"].startswith("2025-06-01")


# --- extract_filesystem ---------------------------------------------


def test_extract_filesystem_passes_commit_date_through(
    tmp_path: Path,
) -> None:
    """Doc candidates carry the recency map's date in commit_date."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "DOC.md").write_text(
        "this is a paragraph long enough to pass the minimum char filter",
        encoding="utf-8",
    )
    recency = {"DOC.md": "2024-09-01T00:00:00+00:00"}
    cands = extract_filesystem(repo, recency=recency)
    assert cands
    for c in cands:
        assert c.commit_date == "2024-09-01T00:00:00+00:00"


def test_extract_filesystem_no_recency_means_no_commit_date(
    tmp_path: Path,
) -> None:
    """No recency arg -> commit_date stays None (back-compat path)."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "DOC.md").write_text(
        "this is a paragraph long enough to pass the minimum char filter",
        encoding="utf-8",
    )
    cands = extract_filesystem(repo)
    assert cands
    for c in cands:
        assert c.commit_date is None


# --- extract_ast ----------------------------------------------------


def test_extract_ast_passes_commit_date_through(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text(
        '"""module-level docstring describing what this module does."""\n',
        encoding="utf-8",
    )
    recency = {"m.py": "2024-04-01T00:00:00+00:00"}
    cands = extract_ast(repo, recency=recency)
    assert cands
    for c in cands:
        assert c.commit_date == "2024-04-01T00:00:00+00:00"


# --- extract_git_log ------------------------------------------------


def test_git_log_candidate_carries_commit_date(tmp_path: Path) -> None:
    """Each git:commit:* candidate carries the commit's own author date."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "f").write_text("x", encoding="utf-8")
    _git_commit(repo, "f", message="atomic descriptive subject for the test",
                author_date="2024-12-25T08:00:00+00:00")
    cands = extract_git_log(repo)
    assert cands
    assert cands[0].commit_date is not None
    assert cands[0].commit_date.startswith("2024-12-25")


# --- scan_repo end-to-end ------------------------------------------


def test_scan_repo_uses_commit_date_for_committed_doc(tmp_path: Path) -> None:
    """A doc file with a git commit lands in the store with
    `belief.created_at == commit author date` rather than the wall-clock
    `now` argument."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "RULES.md").write_text(
        "lock fast and ship small commits with conventional prefixes",
        encoding="utf-8",
    )
    _git_commit(repo, "RULES.md", message="add rules",
                author_date="2024-08-01T00:00:00+00:00")

    db = tmp_path / "db.sqlite"
    store = MemoryStore(str(db))
    try:
        scan_repo(store, repo, now="2099-01-01T00:00:00Z")
        # Find the doc-derived belief and check its created_at
        cur = store._conn.execute(  # type: ignore[attr-defined]
            "SELECT created_at FROM beliefs WHERE id IN ("
            "  SELECT id FROM beliefs WHERE content LIKE '%lock fast%'"
            ") LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None
        # Must be the commit date, not the 2099 wall-clock
        assert row["created_at"].startswith("2024-08-01")
    finally:
        store.close()


def test_scan_repo_falls_back_to_wallclock_outside_git(
    tmp_path: Path,
) -> None:
    """Non-git directory: every belief gets `created_at = now`."""
    project = tmp_path / "p"
    project.mkdir()
    (project / "DOC.md").write_text(
        "the project ships only the regex fallback at version zero point five",
        encoding="utf-8",
    )
    db = tmp_path / "db.sqlite"
    store = MemoryStore(str(db))
    try:
        scan_repo(store, project, now="2030-05-05T00:00:00Z")
        cur = store._conn.execute(  # type: ignore[attr-defined]
            "SELECT created_at FROM beliefs"
        )
        rows = cur.fetchall()
        assert rows
        for row in rows:
            assert row["created_at"] == "2030-05-05T00:00:00Z"
    finally:
        store.close()
