"""#1629 and #1621: the git-recency parse reads git's framing, not line content.

Before, `_build_file_recency_map` split `git log --name-only` output by
guessing: a line starting with four digits and a hyphen was a date
(#1629), and every other line was a literal path, although git C-quotes
non-ASCII names, quotes, and backslashes (#1621). Both guesses are gone:
the parse runs on `-z` output, where paths are NUL-terminated and raw.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from aelfrice.scanner import (
    _build_file_recency_map,  # pyright: ignore[reportPrivateUsage]
    scan_repo,
)
from aelfrice.store import MemoryStore

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the filename is not legal on Windows",
)

_OLD = "2020-01-01T00:00:00Z"
_NEW = "2021-06-01T12:00:00Z"


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = dict(os.environ)
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        [
            "git",
            "-c", "user.email=t@t",
            "-c", "user.name=t",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        env=env,
        timeout=30,
    )


def _repo_with(tmp_path: Path, *names: str, date: str = _OLD) -> Path:
    """One commit that adds every name, each with a distinct paragraph."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    for i, name in enumerate(names):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"paragraph number {i} records a fact about the file it lives in",
            encoding="utf-8",
        )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add files", date=date)
    return repo


# --- #1629: a path that looks like a date is a path --------------------


@pytest.mark.timeout(30)
def test_date_shaped_paths_are_dated_not_read_as_dates(tmp_path: Path) -> None:
    repo = _repo_with(
        tmp_path, "2026-01-15-retro.md", "zeta.md", "2026-notes/x.md",
    )
    assert _build_file_recency_map(repo) == {
        "2026-01-15-retro.md": _OLD,
        "2026-notes/x.md": _OLD,
        "zeta.md": _OLD,
    }


@_POSIX_ONLY
@pytest.mark.timeout(30)
def test_a_path_that_is_a_whole_aware_timestamp_is_a_path(tmp_path: Path) -> None:
    """The strongest form of #1629: the path parses as a date on its own."""
    name = "2026-01-01T10:00:00-08:00"
    repo = _repo_with(tmp_path, name, "zeta.md")
    assert _build_file_recency_map(repo) == {name: _OLD, "zeta.md": _OLD}


@pytest.mark.timeout(60)
def test_scan_repo_never_writes_a_created_at_that_is_not_a_date(
    tmp_path: Path,
) -> None:
    repo = _repo_with(tmp_path, "2026-01-15-retro.md", "zeta.md")
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        scan_repo(store, repo, now="2099-01-01T00:00:00Z")
        created = [
            r[0]
            for r in store._conn.execute(  # noqa: SLF001 - read-only probe
                "SELECT created_at FROM beliefs WHERE content LIKE 'paragraph number%'"
            ).fetchall()
        ]
    finally:
        store.close()
    assert len(created) == 2
    for value in created:
        datetime.fromisoformat(value)  # raises on '2026-01-15-retro.md'
    assert created == [_OLD, _OLD]


# --- #1621: git's quoting never reaches the key -------------------------


@pytest.mark.timeout(30)
def test_non_ascii_path_is_keyed_as_the_filesystem_names_it(tmp_path: Path) -> None:
    repo = _repo_with(tmp_path, "résumé.md", "日本語.md")
    assert _build_file_recency_map(repo) == {"résumé.md": _OLD, "日本語.md": _OLD}


@pytest.mark.timeout(60)
def test_scan_repo_dates_a_non_ascii_file_from_git(tmp_path: Path) -> None:
    """#1621 AC1: the belief carries the git date, not the ingest time."""
    repo = _repo_with(tmp_path, "résumé.md")
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        scan_repo(store, repo, now="2099-01-01T00:00:00Z")
        row = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT created_at FROM beliefs WHERE content LIKE 'paragraph number%'"
        ).fetchone()
    finally:
        store.close()
    assert row is not None
    assert row[0] == _OLD


@pytest.mark.timeout(30)
def test_path_with_a_space_is_keyed_verbatim(tmp_path: Path) -> None:
    repo = _repo_with(tmp_path, "has space.md")
    assert _build_file_recency_map(repo) == {"has space.md": _OLD}


@_POSIX_ONLY
@pytest.mark.timeout(30)
def test_path_with_quote_and_backslash_is_keyed_verbatim(tmp_path: Path) -> None:
    """#1621 AC3: the characters git's quoting escapes."""
    name = 'we"ird\\name.md'
    repo = _repo_with(tmp_path, name)
    assert _build_file_recency_map(repo) == {name: _OLD}


@_POSIX_ONLY
@pytest.mark.timeout(30)
def test_path_with_a_newline_is_keyed_verbatim(tmp_path: Path) -> None:
    """A newline in the FIRST path of a commit shares a token with the date."""
    name = "a\nb.md"
    repo = _repo_with(tmp_path, name)
    assert _build_file_recency_map(repo) == {name: _OLD}


# --- commit boundaries --------------------------------------------------


@pytest.mark.timeout(30)
def test_newest_commit_wins_across_an_empty_commit(tmp_path: Path) -> None:
    """A commit that lists no paths must not shift the dates around it."""
    repo = _repo_with(tmp_path, "old.md", "both.md")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "empty",
         date="2020-06-01T00:00:00Z")
    (repo / "both.md").write_text("changed later", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "touch both", date=_NEW)
    assert _build_file_recency_map(repo) == {"both.md": _NEW, "old.md": _OLD}
