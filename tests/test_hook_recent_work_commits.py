"""Recent-commits resolution for <recent-work> sub-block (#887)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from aelfrice.hook import _resolve_recent_commits


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run(path, "init", "-q", "-b", "main")
    _run(path, "config", "user.email", "t@t")
    _run(path, "config", "user.name", "t")
    _run(path, "config", "commit.gpgsign", "false")


def _commit(repo: Path, name: str, subject: str) -> None:
    (repo / name).write_text(name)
    _run(repo, "add", name)
    _run(repo, "commit", "-q", "-m", subject)


def test_empty_repo_returns_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert _resolve_recent_commits(repo, 5) == []


def test_main_branch_returns_recent_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    for i in range(3):
        _commit(repo, f"f{i}", f"feat: thing {i}")
    out = _resolve_recent_commits(repo, 5)
    assert len(out) == 3
    # newest first
    assert out[0][1] == "feat: thing 2"
    assert out[2][1] == "feat: thing 0"
    # short SHA looks like a short sha
    assert all(len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha)
               for sha, _ in out)


def test_feature_branch_returns_only_ahead_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "base", "feat: base")
    _run(repo, "checkout", "-q", "-b", "feat/issue-887")
    _commit(repo, "a", "feat: A on branch (#887)")
    _commit(repo, "b", "feat: B on branch (#887)")
    out = _resolve_recent_commits(repo, 5)
    subjects = [s for _, s in out]
    assert subjects == [
        "feat: B on branch (#887)",
        "feat: A on branch (#887)",
    ]
    assert "feat: base" not in subjects


def test_limit_caps_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    for i in range(10):
        _commit(repo, f"f{i}", f"feat: {i}")
    out = _resolve_recent_commits(repo, 4)
    assert len(out) == 4
    assert out[0][1] == "feat: 9"


def test_zero_limit_returns_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "x", "init")
    assert _resolve_recent_commits(repo, 0) == []


def test_branch_at_main_falls_back_to_last_n(tmp_path: Path) -> None:
    """On main with no commits ahead, fall back to last-N on HEAD."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "a", "feat: A")
    _commit(repo, "b", "feat: B")
    out = _resolve_recent_commits(repo, 5)
    subjects = [s for _, s in out]
    assert subjects == ["feat: B", "feat: A"]


def test_non_git_dir_returns_empty(tmp_path: Path) -> None:
    not_git = tmp_path / "elsewhere"
    not_git.mkdir()
    assert _resolve_recent_commits(not_git, 5) == []


def test_no_main_branch_falls_back_to_head(tmp_path: Path) -> None:
    """Repo with main renamed: still returns last-N HEAD commits."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "a", "init")
    _run(repo, "branch", "-m", "main", "trunk")
    _commit(repo, "b", "feat: B")
    out = _resolve_recent_commits(repo, 5)
    subjects = [s for _, s in out]
    assert "feat: B" in subjects
    assert "init" in subjects
