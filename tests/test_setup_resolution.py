"""Unit tests for the per-project install routing primitives.

Covers `resolve_hook_command`, `detect_default_scope`, and
`clean_dangling_shims` from `aelfrice.setup`. Each test patches the
relevant filesystem / interpreter introspection so it runs the same
inside or outside a venv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aelfrice.setup import (
    DanglingShimCleanup,
    clean_dangling_shims,
    detect_default_scope,
    resolve_hook_command,
)


def _make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    path.chmod(0o755)


# --- resolve_hook_command -----------------------------------------------


def test_resolve_hook_command_project_prefers_sys_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    hook = venv_bin / "aelf-hook"
    _make_executable(hook)
    monkeypatch.setattr(sys, "prefix", str(venv_bin.parent))
    # PATH does not have anything resolvable: project scope must still
    # find the venv hook.
    monkeypatch.setenv("PATH", "/nonexistent")

    assert resolve_hook_command("project") == str(hook)


def test_resolve_hook_command_user_prefers_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_hook = venv_bin / "aelf-hook"
    _make_executable(venv_hook)
    pipx_dir = tmp_path / "pipx" / "bin"
    pipx_hook = pipx_dir / "aelf-hook"
    _make_executable(pipx_hook)
    monkeypatch.setattr(sys, "prefix", str(venv_bin.parent))
    monkeypatch.setenv("PATH", str(pipx_dir))

    assert resolve_hook_command("user") == str(pipx_hook)


def test_resolve_hook_command_user_falls_back_to_venv_when_path_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_hook = venv_bin / "aelf-hook"
    _make_executable(venv_hook)
    monkeypatch.setattr(sys, "prefix", str(venv_bin.parent))
    monkeypatch.setenv("PATH", "/nonexistent")

    assert resolve_hook_command("user") == str(venv_hook)


def test_resolve_hook_command_last_resort_is_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_bin = tmp_path / "empty" / "bin"
    empty_bin.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(empty_bin.parent))
    monkeypatch.setenv("PATH", "/nonexistent")

    # Nothing on disk, nothing on PATH: fall back to bare name so doctor
    # can flag it instead of crashing.
    assert resolve_hook_command("project") == "aelf-hook"
    assert resolve_hook_command("user") == "aelf-hook"


# --- detect_default_scope -----------------------------------------------


def test_detect_default_scope_project_when_prefix_matches_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "myproj"
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))

    assert detect_default_scope(cwd=project) == "project"


def test_detect_default_scope_user_when_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "noventer"
    project.mkdir()
    monkeypatch.setattr(sys, "prefix", "/something/else")

    assert detect_default_scope(cwd=project) == "user"


def test_detect_default_scope_user_when_prefix_mismatches_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "wrongvenv"
    (project / ".venv" / "bin").mkdir(parents=True)
    other_venv = tmp_path / "otherproj" / ".venv"
    (other_venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(other_venv))

    assert detect_default_scope(cwd=project) == "user"


# --- clean_dangling_shims -----------------------------------------------


def test_clean_dangling_shims_removes_dangling_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "deleted-target"
    shim = tmp_path / "shim"
    target.write_text("alive")
    shim.symlink_to(target)
    target.unlink()  # now shim is dangling

    result = clean_dangling_shims(candidates=(shim,))

    assert isinstance(result, DanglingShimCleanup)
    assert shim in result.removed
    assert not shim.exists()
    assert not shim.is_symlink()


def test_clean_dangling_shims_skips_live_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "alive-target"
    target.write_text("hello")
    shim = tmp_path / "shim"
    shim.symlink_to(target)

    result = clean_dangling_shims(candidates=(shim,))

    assert shim in result.skipped
    assert shim.exists()


def test_clean_dangling_shims_skips_real_files(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.write_text("not a symlink")

    result = clean_dangling_shims(candidates=(real,))

    assert real in result.skipped
    assert real.exists()


def test_clean_dangling_shims_idempotent_on_missing_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "never-existed"
    result = clean_dangling_shims(candidates=(missing,))

    assert result.removed == ()
    # not in skipped either: nothing was there to start with.
    assert missing not in result.skipped
