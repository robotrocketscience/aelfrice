"""`aelf doctor --host codex` must not exit 0 on broken wiring (#1430).

The issue reported that doctor returns success for an installation that
cannot provide the requested memory surface. The operator ruling on it
narrows the scope in one direction and keeps it in the other:

* **Absent** wiring stays exit 0. A machine that has never run
  ``aelf setup --host codex`` is a normal machine; failing doctor there
  would make the Codex host stricter than the Claude host for no stated
  reason, and "not selected" versus "missing" is not distinguishable
  without a persisted setup plan that does not exist.
* **Tampered or partially-broken** wiring is still a false green and is
  still in scope.

Everything asserted here is a filesystem or config state, so none of it
needs the reachable Codex model that parked the rest of the issue.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_doctor_codex
from aelfrice.host_codex import (
    _SKILL_FILENAME,
    _bundled_codex_skills,
    install_codex_hooks,
    install_codex_skills,
)


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
    """A complete, healthy Codex install."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json")
    skills = tmp_path / "skills"
    install_codex_skills(skills)
    return codex_dir, skills


def _read_hooks(codex_dir: Path) -> dict[str, object]:
    return json.loads(
        (codex_dir / "hooks.json").read_text(encoding="utf-8"),
    )["hooks"]


def _write_hooks(codex_dir: Path, hooks: dict[str, object]) -> None:
    (codex_dir / "hooks.json").write_text(
        json.dumps({"hooks": hooks}, indent=2) + "\n", encoding="utf-8",
    )


class TestAbsentWiringStaysGreen:
    """The half the ruling took out of scope. These pin it stays out."""

    def test_an_empty_codex_home_exits_zero(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / "codex"
        codex_dir.mkdir()
        rc, out = _doctor(codex_dir, tmp_path / "skills")
        assert rc == 0, out

    def test_no_codex_home_at_all_exits_zero(self, tmp_path: Path) -> None:
        rc, out = _doctor(tmp_path / "nope", tmp_path / "skills")
        assert rc == 0, out

    def test_a_complete_install_exits_zero(
        self, installed: tuple[Path, Path],
    ) -> None:
        codex_dir, skills = installed
        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
        assert "[FAIL]" not in out


class TestTamperedWiringFails:
    def test_a_handler_pointing_at_a_missing_command_fails(
        self, installed: tuple[Path, Path],
    ) -> None:
        """The classic broken install: setup ran, then the venv went away."""
        codex_dir, skills = installed
        hooks = _read_hooks(codex_dir)
        event = next(iter(hooks))
        groups = hooks[event]
        groups[0]["hooks"][0]["command"] = "/nonexistent/aelf-hook"  # type: ignore[index]
        _write_hooks(codex_dir, hooks)

        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "not present on disk" in out

    def test_a_partial_install_fails(
        self, installed: tuple[Path, Path],
    ) -> None:
        """Some groups removed by hand — the surface cannot do its job."""
        codex_dir, skills = installed
        hooks = _read_hooks(codex_dir)
        assert len(hooks) > 1
        del hooks[next(iter(hooks))]
        _write_hooks(codex_dir, hooks)

        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "partially installed" in out

    def test_handlers_installed_with_the_feature_disabled_fails(
        self, installed: tuple[Path, Path],
    ) -> None:
        """Every hook is present and none of them will ever run."""
        codex_dir, skills = installed
        (codex_dir / "config.toml").write_text(
            "[features]\nhooks = false\n", encoding="utf-8",
        )
        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "none of them run" in out

    def test_a_modified_skill_fails(
        self, installed: tuple[Path, Path],
    ) -> None:
        """Ownership is not integrity.

        `_is_owned_skill_dir` gates on the name prefix and the marker, so an
        edited skill still counted as installed and healthy and doctor
        reported a surface that no longer does what it claims.
        """
        codex_dir, skills = installed
        name = sorted(_bundled_codex_skills())[0]
        target = skills / name / _SKILL_FILENAME
        target.write_text(
            target.read_text(encoding="utf-8") + "\nrm -rf /\n",
            encoding="utf-8",
        )

        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "no longer match" in out
        assert name in out


class TestTheDeliberateExclusions:
    def test_zero_approvals_alone_does_not_fail(
        self, installed: tuple[Path, Path],
    ) -> None:
        """Approval keying is positional and slated to change upstream.

        A count of zero does not distinguish "unapproved" from "keyed
        differently", and the issue asks for `unknown` rather than unhealthy
        when trust cannot be established authoritatively. It stays a warning.
        """
        codex_dir, skills = installed
        assert not (codex_dir / "config.toml").exists()

        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
        assert "no approved [hooks.state] entries" in out

    def test_an_uninstalled_skill_is_absent_not_modified(
        self, installed: tuple[Path, Path],
    ) -> None:
        """Deleting a skill is absent wiring, which the ruling exempts."""
        codex_dir, skills = installed
        name = sorted(_bundled_codex_skills())[0]
        (skills / name / _SKILL_FILENAME).unlink()
        (skills / name).rmdir()

        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out

    def test_a_foreign_skill_is_never_our_fault(
        self, installed: tuple[Path, Path],
    ) -> None:
        """No marker means not ours, so its bytes are not ours to judge."""
        codex_dir, skills = installed
        foreign = skills / "aelf-someone-elses"
        foreign.mkdir()
        (foreign / _SKILL_FILENAME).write_text(
            "---\nname: aelf-someone-elses\n---\nmine\n", encoding="utf-8",
        )

        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
