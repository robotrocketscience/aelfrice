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

    def test_the_failure_claims_no_surface_that_this_build_lacks(
        self, installed: tuple[Path, Path], empty_path: Path,
    ) -> None:
        """The diagnosis may not outrun the tree it ships with.

        A `[FAIL]` line is a diagnosis, and this one lands on a stable CI
        exit contract (#1430), so it may only name mechanisms this build
        actually has. It originally had neither of the two the obvious
        wording reaches for, and this guard said so and promised to
        retire when the skills converted.

        Half of it has now retired. Since the rest of #1413 landed, every
        generated `$aelf-*` skill DOES invoke `aelf` by name, so a line
        naming that surface is accurate. The other half stands unchanged:
        `aelf setup --host codex` still pins each hook handler to an
        absolute `aelf-*` path whenever it can resolve one, so the
        handlers do not invoke `aelf` by name and the diagnosis still may
        not say they do. Both preconditions are re-derived here rather
        than assumed, so this reds if either surface changes again.
        """
        codex_dir, skills = installed
        bodies = [
            path.read_text(encoding="utf-8")
            for path in sorted(skills.rglob("SKILL.md"))
        ]
        assert bodies, "fixture precondition: skills are installed"
        assert not any("uv run aelf" in body for body in bodies), (
            "a generated skill routes through `uv run` again; #1413 "
            "converted them to invoke `aelf` by name"
        )
        assert all("`aelf " in body for body in bodies), (
            "precondition gone: a generated skill no longer invokes "
            "`aelf` by name, so re-read this guard before trusting it"
        )
        hooks_doc = (codex_dir / "hooks.json").read_text(encoding="utf-8")
        assert '"aelf"' not in hooks_doc, hooks_doc
        _rc, out = _doctor(*installed)
        assert "hook handler" not in out or "by name" not in out, out

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


class TestHooksFeatureOffIsAFailure:
    """The behaviour the doc caveat has to describe.

    `[features].hooks = false` with our handlers installed is a
    `CodexDoctorReport.tampering()` reason, so doctor prints a `[warn]`
    *and* a `[FAIL]` and exits 1. Calling it a warning in the docs would
    invite an operator to gate on this command and be surprised by a red
    light. This test anchors the doc guard below to real behaviour rather
    than to a string.
    """

    def test_the_feature_off_state_fails_and_exits_one(
        self, installed: tuple[Path, Path], empty_path: Path,
    ) -> None:
        _shim(empty_path, "aelf")  # isolate from the #1413 PATH fault
        codex_dir, skills = installed
        (codex_dir / "config.toml").write_text(
            "[features]\nhooks = false\n", encoding="utf-8",
        )
        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "[warn]" in out, out
        assert "[FAIL]" in out, out
        assert "not on PATH" not in out, out


_REPO = Path(__file__).resolve().parents[1]
_SLASH_COMMANDS_DOC = _REPO / "docs" / "user" / "SLASH_COMMANDS.md"


class TestCodexFeatureKeyIsDocumented:
    """AC6: the docs must name the current feature key, not the retired one.

    `codex features list` on codex-cli 0.145.0 reports one row for this
    feature, `hooks  stable  true`, and no `codex_hooks` row at all. The
    doc told you to turn on a `codex_hooks` flag that no longer exists.
    """

    def test_the_caveat_names_the_current_hooks_feature(self) -> None:
        text = _SLASH_COMMANDS_DOC.read_text(encoding="utf-8")
        assert "with the `hooks` feature on" in text, text[:400]
        assert "`[features].hooks = false`" in text, text[:400]

    def test_the_caveat_no_longer_names_codex_hooks_as_the_live_flag(
        self,
    ) -> None:
        """The distinguishing assert.

        `codex_hooks` may still appear as the named predecessor — doctor
        honors it when an old `config.toml` carries it — so the guard is
        on the instruction, not on the string.
        """
        text = _SLASH_COMMANDS_DOC.read_text(encoding="utf-8")
        assert "`codex_hooks` feature flag on" not in text, text[:400]

    def test_the_caveat_calls_the_feature_off_state_a_failure(self) -> None:
        """It is a `[FAIL]` and exit 1, not a warning.

        `TestHooksFeatureOffIsAFailure` proves the behaviour; this pins the
        prose to it, because the caveat is what an operator reads before
        deciding whether to gate on this command.
        """
        text = _SLASH_COMMANDS_DOC.read_text(encoding="utf-8")
        caveat = text.split("Two caveats are specific to the Codex host:")[1]
        caveat = caveat.split("\n2. ")[0]
        assert "`[features].hooks = false`" in caveat, caveat
        assert "`[FAIL]`" in caveat, caveat
        assert "exits 1" in caveat, caveat
        assert "reports as a warning" not in caveat, caveat
