"""#1612: a merge commit dates the paths it changed relative to every parent.

Without `--cc`, `git log --name-only` lists no paths for a merge, so a
file that arrived only through a merge was undated, and a conflict
resolved to new content kept the date of the last edit before the merge.
Each fixture is checked against the per-path answer,
`git log -1 --format=%aI -- <path>`, which is what the issue asks the
single pass to match.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from aelfrice.scanner import (
    _build_file_recency_map,  # pyright: ignore[reportPrivateUsage]
    scan_repo,
)
from aelfrice.store import MemoryStore


def _git(repo: Path, *args: str, date: str | None = None, check: bool = True) -> None:
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
        check=check,
        env=env,
        capture_output=True,
        timeout=30,
    )


def _write(repo: Path, name: str, text: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _per_path(repo: Path) -> dict[str, str]:
    """The reference answer: one `git log -1` per tracked path."""
    names = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True,
        check=True, timeout=30,
    ).stdout.split(b"\x00")
    out: dict[str, str] = {}
    for raw in names:
        if not raw:
            continue
        path = os.fsdecode(raw)
        date = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", path], cwd=repo,
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        out[path] = date
    return out


def _tracked_dates(repo: Path) -> dict[str, str]:
    """The recency map restricted to tracked paths, the only keys looked up.

    The map also holds paths that exist only in history, such as a file
    at its pre-import location.
    """
    tracked = _per_path(repo).keys()
    return {p: d for p, d in _build_file_recency_map(repo).items() if p in tracked}


def _init(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    return repo


def _history_import(tmp_path: Path) -> Path:
    """The issue's shape: a file that exists on `main` only via a merge."""
    repo = _init(tmp_path)
    _write(repo, "main.md", "main paragraph that states a fact about the main line")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "m1", date="2020-01-01T00:00:00Z")
    _git(repo, "checkout", "-q", "--orphan", "side")
    _git(repo, "rm", "-r", "-f", "-q", "--", "main.md")
    _write(repo, "orig/f.md", "imported paragraph that states a fact about history")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "s1", date="2020-02-01T00:00:00Z")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-commit",
         "--allow-unrelated-histories", "-s", "ours", "side")
    _write(repo, "lab/f.md", "imported paragraph that states a fact about history")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "import history", date="2020-03-01T00:00:00Z")
    return repo


@pytest.mark.timeout(60)
def test_a_file_that_arrived_only_through_a_merge_is_dated(tmp_path: Path) -> None:
    repo = _history_import(tmp_path)
    rec = _tracked_dates(repo)
    assert rec == {"lab/f.md": "2020-03-01T00:00:00Z", "main.md": "2020-01-01T00:00:00Z"}
    assert rec == {p: d.replace("+00:00", "Z") for p, d in _per_path(repo).items()}


@pytest.mark.timeout(60)
def test_a_merge_that_took_a_path_from_one_parent_does_not_redate_it(
    tmp_path: Path,
) -> None:
    """The over-attribution the issue measured for `-m` / first-parent."""
    repo = _init(tmp_path)
    _write(repo, "g.md", "0")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b0", date="2020-01-01T00:00:00Z")
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo, "h.md", "side")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "side h", date="2020-02-01T00:00:00Z")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "g.md", "main")
    _git(repo, "commit", "-q", "-am", "main g", date="2020-03-01T00:00:00Z")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side",
         date="2020-04-01T00:00:00Z")
    rec = _tracked_dates(repo)
    assert rec == {"g.md": "2020-03-01T00:00:00Z", "h.md": "2020-02-01T00:00:00Z"}
    assert rec == {p: d.replace("+00:00", "Z") for p, d in _per_path(repo).items()}


@pytest.mark.timeout(60)
def test_a_conflict_resolved_to_new_content_is_dated_to_the_merge(
    tmp_path: Path,
) -> None:
    repo = _init(tmp_path)
    _write(repo, "k.md", "0")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c0", date="2020-01-01T00:00:00Z")
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo, "k.md", "side")
    _git(repo, "commit", "-q", "-am", "side k", date="2020-02-01T00:00:00Z")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "k.md", "main")
    _git(repo, "commit", "-q", "-am", "main k", date="2020-03-01T00:00:00Z")
    _git(repo, "merge", "-q", "--no-ff", "side", check=False)
    _write(repo, "k.md", "resolved")
    _git(repo, "add", "k.md")
    _git(repo, "commit", "-q", "-m", "resolve", date="2020-04-01T00:00:00Z")
    rec = _tracked_dates(repo)
    assert rec == {"k.md": "2020-04-01T00:00:00Z"}
    assert rec == {p: d.replace("+00:00", "Z") for p, d in _per_path(repo).items()}


@pytest.mark.xfail(
    strict=True,
    reason="known limit: a side edit the merge discarded is still seen, "
    "because only a pathspec lets git prune it (see the docstring)",
)
@pytest.mark.timeout(60)
def test_a_side_edit_the_merge_discarded_does_not_date_the_file(
    tmp_path: Path,
) -> None:
    repo = _init(tmp_path)
    _write(repo, "q.md", "0")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "d0", date="2020-01-01T00:00:00Z")
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo, "q.md", "side")
    _git(repo, "commit", "-q", "-am", "side q", date="2020-05-01T00:00:00Z")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-s", "ours", "-m", "ours", "side",
         date="2020-06-01T00:00:00Z")
    assert _build_file_recency_map(repo) == {"q.md": "2020-01-01T00:00:00Z"}


@pytest.mark.timeout(60)
def test_scan_repo_dates_the_imported_file(tmp_path: Path) -> None:
    """End to end: the issue's symptom was a wall-clock created_at."""
    repo = _history_import(tmp_path)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        scan_repo(store, repo, now="2099-01-01T00:00:00Z")
        row = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT created_at FROM beliefs WHERE content LIKE 'imported paragraph%'"
        ).fetchone()
    finally:
        store.close()
    assert row is not None
    assert row[0] == "2020-03-01T00:00:00Z"
