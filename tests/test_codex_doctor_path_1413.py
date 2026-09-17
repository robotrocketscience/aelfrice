"""`aelf doctor --host codex` must fail when `aelf` is off PATH (#1413).

The generated `$aelf-*` skills and the installed Codex hooks both reach
the CLI by name. On a machine where the name does not resolve, every
dispatch dies at exec and doctor was the only surface that could have
said so — it exited 0.

The operator ruling of 2026-08-12 settled the contradiction between the
issue body ("add a doctor **warning**") and the acceptance matrix
("yields a doctor **failure**") in favour of the matrix, scoped by
population:

    `aelf doctor --host codex` emits `[FAIL]` and exits 1 when
    `launcher.which_on_path("aelf")` is None **and**
    `report.owned_handler_count > 0`; otherwise exit 0.

That mirrors the #1430 gate on this same command: absent wiring stays
exit 0, so a source checkout and this repo's own CI — neither of which
has a `uv tool` install — stay green.

Everything here is a `PATH` and filesystem state, so none of it needs the
reachable Codex host that parks the rest of #1413.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pytest

from aelfrice import launcher
from aelfrice.cli import _cmd_doctor_codex
from aelfrice.host_codex import install_codex_hooks, install_codex_skills


def _doctor(codex_dir: Path, skills: Path) -> tuple[int, str]:
    out = io.StringIO()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"),
        out,
        codex_dir=codex_dir,
        skills_dest=skills,
    )
    return rc, out.getvalue()


@pytest.fixture
def installed(tmp_path: Path) -> tuple[Path, Path]:
    """A complete, healthy Codex install — hooks plus skills."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json")
    skills = tmp_path / "skills"
    install_codex_skills(skills)
    return codex_dir, skills


@pytest.fixture
def empty_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Path:
    """A `PATH` holding exactly one directory, and nothing in it.

    Set rather than cleared: an empty `PATH` falls back to `os.defpath`
    inside `which_on_path`, which on this machine is a real system path
    that could hold a real `aelf`. The point is determinism, not an
    approximation of it.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("PATH", str(bindir))
    return bindir


def _shim(directory: Path, name: str) -> Path:
    """An executable file that stands in for the installed `aelf`."""
    target = directory / name
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    return target


class TestWiringInstalled:
    """The population the ruling names: wiring present."""

    def test_aelf_off_path_fails_and_exits_one(
        self, installed: tuple[Path, Path], empty_path: Path,
    ) -> None:
        assert launcher.which_on_path("aelf") is None, "fixture precondition"
        rc, out = _doctor(*installed)
        assert rc == 1, out
        assert "[FAIL]" in out, out
        assert "`aelf` is not on PATH" in out, out

    def test_the_failure_names_the_install_and_path_remedy(
        self, installed: tuple[Path, Path], empty_path: Path,
    ) -> None:
        """AC5 asks for guidance, not just a verdict."""
        _rc, out = _doctor(*installed)
        assert "uv tool install aelfrice" in out, out
        assert "uv tool update-shell" in out, out

    def test_a_posix_shim_on_path_stays_green(
        self, installed: tuple[Path, Path], empty_path: Path,
    ) -> None:
        _shim(empty_path, "aelf")
        assert launcher.which_on_path("aelf") is not None
        rc, out = _doctor(*installed)
        assert rc == 0, out
        assert "not on PATH" not in out, out

    def test_a_windows_launcher_on_path_stays_green(
        self,
        installed: tuple[Path, Path],
        empty_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Windows arm, simulated so it runs on POSIX CI.

        `shutil.which` consults `PATHEXT` under `os.name == "nt"` only, and
        that branch is CPython's own — not a seam this repo can flip. So
        the resolver is driven to return the `aelf.exe` a Windows `uv tool`
        install would put on `PATH`, and the assertion is that the gate
        accepts it rather than demanding an extensionless name.
        """
        exe = _shim(empty_path, "aelf.exe")
        monkeypatch.setattr(
            launcher.shutil, "which",
            lambda name, mode=os.X_OK, path=None: (
                str(exe) if name == "aelf" else None
            ),
        )
        rc, out = _doctor(*installed)
        assert rc == 0, out
        assert "not on PATH" not in out, out


class TestWiringAbsent:
    """The population the ruling excludes: no Codex wiring installed.

    This is the arm that keeps `uv run aelf doctor --host codex` green in
    this repo's own CI, where no `uv tool` install exists.
    """

    def test_an_empty_codex_home_stays_green_with_aelf_off_path(
        self, tmp_path: Path, empty_path: Path,
    ) -> None:
        codex_dir = tmp_path / "codex"
        codex_dir.mkdir()
        assert launcher.which_on_path("aelf") is None
        rc, out = _doctor(codex_dir, tmp_path / "skills")
        assert rc == 0, out
        assert "not on PATH" not in out, out

    def test_skills_without_hooks_stay_green_with_aelf_off_path(
        self, tmp_path: Path, empty_path: Path,
    ) -> None:
        """Skills alone are not wiring — `owned_handler_count` is 0."""
        codex_dir = tmp_path / "codex"
        codex_dir.mkdir()
        skills = tmp_path / "skills"
        install_codex_skills(skills)
        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
        assert "not on PATH" not in out, out

