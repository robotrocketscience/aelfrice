"""#1657: the claude-memory mirror never got consent from a worktree.

Consent is recorded at `aelf setup` only when the project's memory
directory exists, and `derive_memory_dir` encoded the working directory
itself. The host keeps a worktree session's memory under the main
checkout, so from a linked worktree the directory never existed and
consent was deferred forever, silently. `aelf doctor` printed only an
info line. Now a worktree maps to its main checkout, and doctor warns
when the hook is installed but nothing has turned the mirror on.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aelfrice import claude_memory
from aelfrice.claude_memory import derive_memory_dir, mirror_consent_missing
from aelfrice.db_paths import main_checkout_path


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=30)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    home = tmp_path / "home"
    repo = home / "projects" / "app"
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "init", cwd=repo)
    wt = repo / ".claude" / "worktrees" / "7"
    wt.parent.mkdir(parents=True)
    _git("worktree", "add", "-q", str(wt), cwd=repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AELFRICE_MIRROR_CLAUDE_MEMORY", raising=False)
    store = repo / ".git" / "aelfrice" / "memory.db"
    store.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(store))
    return {"home": home, "repo": repo.resolve(), "wt": wt.resolve(), "store": store}


def _memory_dir_for(home: Path, root: Path) -> Path:
    return home.resolve() / ".claude" / "projects" / claude_memory.encode_project_path(str(root)) / "memory"


def test_a_worktree_maps_to_the_main_checkouts_memory(env: dict[str, Path]) -> None:
    assert derive_memory_dir(env["wt"]) == _memory_dir_for(env["home"], env["repo"])


def test_a_worktree_subdirectory_keeps_its_relative_part(env: dict[str, Path]) -> None:
    sub = env["wt"] / "src"
    sub.mkdir()
    assert derive_memory_dir(sub) == _memory_dir_for(env["home"], env["repo"] / "src")


def test_the_main_checkout_maps_to_itself(env: dict[str, Path]) -> None:
    assert derive_memory_dir(env["repo"]) == _memory_dir_for(env["home"], env["repo"])


def test_a_gitfile_without_commondir_is_left_alone(tmp_path: Path) -> None:
    # A submodule's `.git` file points at `.git/modules/x`, which has no
    # `commondir`, so it is not a linked worktree.
    sub = tmp_path / "sub"
    (tmp_path / "modules" / "x").mkdir(parents=True)
    sub.mkdir()
    (sub / ".git").write_text(f"gitdir: {tmp_path / 'modules' / 'x'}\n")
    assert main_checkout_path(sub.resolve()) == sub.resolve()


def test_a_broken_gitfile_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("not a gitdir line\n")
    assert main_checkout_path(tmp_path.resolve()) == tmp_path.resolve()


def test_setup_from_a_worktree_records_consent(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem = _memory_dir_for(env["home"], env["repo"])
    mem.mkdir(parents=True)
    (mem / "a.md").write_text(
        "---\nname: a\ndescription: x\nmetadata:\n  type: project\n---\nThe build uses uv.\n")
    monkeypatch.chdir(env["wt"])
    from aelfrice.cli import _maybe_reconcile_claude_memory_at_setup

    _maybe_reconcile_claude_memory_at_setup()
    assert claude_memory.reconcile_sentinel_path(env["store"]).exists()
    assert claude_memory.is_mirror_enabled() is True


def test_consent_missing_only_when_nothing_decides(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mirror_consent_missing(start=env["repo"]) is True
    monkeypatch.setenv("AELFRICE_MIRROR_CLAUDE_MEMORY", "0")
    assert mirror_consent_missing(start=env["repo"]) is False
    monkeypatch.delenv("AELFRICE_MIRROR_CLAUDE_MEMORY")
    (env["repo"] / ".aelfrice.toml").write_text("[memory]\nmirror_claude_memory = false\n")
    assert mirror_consent_missing(start=env["repo"]) is False
    (env["repo"] / ".aelfrice.toml").unlink()
    claude_memory.reconcile_sentinel_path(env["store"]).write_text("ok\n")
    assert mirror_consent_missing(start=env["repo"]) is False


def _doctor(env: dict[str, Path], tmp_path: Path, *, hook: bool):  # noqa: ANN202
    import json

    from aelfrice.doctor import diagnose, format_report

    settings = tmp_path / "settings.json"
    cmd = "aelf-claude-memory-mirror" if hook else "aelf-hook"
    settings.write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Write", "hooks": [{"type": "command", "command": cmd}]}]}}))
    report = diagnose(user_settings=settings, project_root=env["repo"],
                      hook_failures_log=tmp_path / "none.log",
                      aelfrice_projects_dir=tmp_path / "none")
    return report, format_report(report)


def test_doctor_warns_when_the_hook_has_no_consent(
    env: dict[str, Path], tmp_path: Path,
) -> None:
    _memory_dir_for(env["home"], env["repo"]).mkdir(parents=True)
    report, text = _doctor(env, tmp_path, hook=True)
    assert report.mirror_consent_missing is True
    assert "mirror hook is installed but the mirror is off" in text
    assert "aelf reconcile-claude-memory" in text


@pytest.mark.parametrize("case", ["no hook", "no memory dir", "consented", "opted out"])
def test_doctor_is_quiet_otherwise(
    env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    if case != "no memory dir":
        _memory_dir_for(env["home"], env["repo"]).mkdir(parents=True)
    if case == "consented":
        claude_memory.reconcile_sentinel_path(env["store"]).write_text("ok\n")
    if case == "opted out":
        monkeypatch.setenv("AELFRICE_MIRROR_CLAUDE_MEMORY", "0")
    report, text = _doctor(env, tmp_path, hook=case != "no hook")
    assert report.mirror_consent_missing is False
    assert "mirror hook is installed but the mirror is off" not in text


def test_a_worktree_of_a_bare_repo_is_left_alone(tmp_path: Path) -> None:
    # A bare repository's common dir is `repo.git`, not `<checkout>/.git`,
    # so there is no main checkout to map to.
    bare = tmp_path / "repo.git"
    wtgit = bare / "worktrees" / "w"
    wtgit.mkdir(parents=True)
    (wtgit / "commondir").write_text("../..\n")
    wt = tmp_path / "w"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wtgit}\n")
    assert main_checkout_path(wt.resolve()) == wt.resolve()


@pytest.mark.timeout(60)
def test_a_repo_nested_inside_a_worktree_maps_to_itself(env: dict[str, Path]) -> None:
    # The walk stops at the first `.git` directory, so a repository nested
    # inside a worktree is its own project, not part of the main checkout.
    nested = env["wt"] / "vendor" / "lib"
    nested.mkdir(parents=True)
    _git("init", "-q", cwd=nested)
    assert main_checkout_path(nested.resolve()) == nested.resolve()
