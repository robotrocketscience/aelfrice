"""Builder for <recent-work> sub-block (#887)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from aelfrice.hook import _build_recent_work_subblock


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


def test_non_git_returns_empty(tmp_path: Path) -> None:
    not_git = tmp_path / "x"
    not_git.mkdir()
    assert _build_recent_work_subblock(not_git) == ""


def test_renders_branch_and_commits(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "base", "feat: base")
    _run(repo, "checkout", "-q", "-b", "feat/issue-887-recent-work")
    _commit(repo, "a", "feat: implement A (#887)")
    block = _build_recent_work_subblock(repo)
    assert block.startswith("<recent-work>")
    assert block.endswith("</recent-work>")
    assert "<branch>feat/issue-887-recent-work</branch>" in block
    assert "<commits>" in block
    assert ">feat: implement A (#887)</commit>" in block
    assert "<linked-issues>#887</linked-issues>" in block


def test_emits_upstream_when_present(tmp_path: Path) -> None:
    bare = tmp_path / "u.git"
    _run(tmp_path, "init", "-q", "--bare", str(bare))
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "x", "feat: x")
    _run(repo, "remote", "add", "github", str(bare))
    _run(repo, "push", "-q", "-u", "github", "main")
    block = _build_recent_work_subblock(repo)
    assert "<upstream>github/main</upstream>" in block


def test_omits_upstream_when_absent(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "x", "feat: x")
    block = _build_recent_work_subblock(repo)
    assert "<upstream>" not in block


def test_omits_commits_section_on_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    block = _build_recent_work_subblock(repo)
    # empty repo: branch may still be 'main' but log is empty
    if block:
        assert "<commits>" not in block


def test_omits_linked_issues_when_none(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "x", "docs: README")
    block = _build_recent_work_subblock(repo)
    assert "<linked-issues>" not in block


def test_escapes_subject_that_looks_like_tag(tmp_path: Path) -> None:
    """A commit subject containing a framing tag must be entity-escaped."""
    repo = tmp_path / "r"
    _init_repo(repo)
    _commit(repo, "x", "feat: see <recent-work> in old design")
    block = _build_recent_work_subblock(repo)
    assert "feat: see &lt;recent-work&gt; in old design" in block
    # The raw injectable token should NOT appear inside the rendered subject.
    # (The wrapping <recent-work> open tag itself is allowed at line start.)
    assert block.count("<recent-work>") == 1


def test_commit_limit_capped(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_repo(repo)
    for i in range(20):
        _commit(repo, f"f{i}", f"feat: c{i}")
    block = _build_recent_work_subblock(repo, commit_limit=3)
    assert block.count("<commit ") == 3
    assert "feat: c19" in block
    assert "feat: c0" not in block
