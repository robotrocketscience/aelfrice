"""Branch + upstream resolution for <recent-work> sub-block (#887)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from aelfrice.hook import _resolve_branch


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run(path, "init", "-q", "-b", "main")
    _run(path, "config", "user.email", "t@t")
    _run(path, "config", "user.name", "t")
    _run(path, "config", "commit.gpgsign", "false")
    (path / "f").write_text("x")
    _run(path, "add", "f")
    _run(path, "commit", "-q", "-m", "init")


def test_resolve_branch_on_main(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    branch, upstream = _resolve_branch(repo)
    assert branch == "main"
    assert upstream is None


def test_resolve_branch_with_upstream(tmp_path: Path) -> None:
    upstream_repo = tmp_path / "upstream.git"
    _run(tmp_path, "init", "-q", "--bare", str(upstream_repo))
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run(repo, "remote", "add", "origin", str(upstream_repo))
    _run(repo, "push", "-q", "-u", "origin", "main")
    branch, upstream = _resolve_branch(repo)
    assert branch == "main"
    assert upstream == "origin/main"


def test_resolve_branch_feature_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run(repo, "checkout", "-q", "-b", "feat/issue-887-recent-work")
    branch, _upstream = _resolve_branch(repo)
    assert branch == "feat/issue-887-recent-work"


def test_resolve_branch_detached_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    _run(repo, "checkout", "-q", sha)
    branch, upstream = _resolve_branch(repo)
    assert branch is None
    assert upstream is None


def test_resolve_branch_non_git_dir(tmp_path: Path) -> None:
    not_git = tmp_path / "elsewhere"
    not_git.mkdir()
    branch, upstream = _resolve_branch(not_git)
    assert branch is None
    assert upstream is None
