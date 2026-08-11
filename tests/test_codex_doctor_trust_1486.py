"""The trust half of #1430, carried by #1486.

#1476 shipped the fixture-provable half of #1430 — doctor now exits
nonzero on installed-but-broken Codex wiring. Two of the things it left:

* `modified_codex_skills()` called with no argument raised `NameError`.
  #1427 deleted the module-level ``AGENTS_SKILLS_DIR`` constant in favour
  of the late-bound `resolve_agents_skills_dir()`; #1476 then wrote a
  fresh reference to the deleted name into the new function's default-dir
  branch. Every production caller passes a directory, so the branch was
  unreachable — but nothing in the tree caught it either.
* An `aelf-*` skill directory carrying our marker but no longer in the
  bundle was counted as installed and never examined, because
  `modified_codex_skills` iterates bundled names only. The count doctor
  printed was therefore one higher than the number of skills it judged,
  and a renamed-away slash command stayed invokable by the model.

Everything here is a filesystem or config state; none of it needs the
reachable Codex model that keeps the rest of #1430 blocked externally.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_doctor_codex
from aelfrice.host_codex import (
    _SKILL_FILENAME,
    _SKILL_MARKER,
    _bundled_codex_skills,
    count_installed_codex_skills,
    install_codex_hooks,
    install_codex_skills,
    modified_codex_skills,
    orphaned_codex_skills,
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
    """A complete, healthy Codex install: hooks wired, skills written."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json")
    skills = tmp_path / "skills"
    install_codex_skills(skills)
    return codex_dir, skills


def _plant_orphan(
    skills: Path, name: str = "aelf-gone", *, marker: bool = True,
) -> None:
    """An ``aelf-*`` skill directory for a command no longer in the bundle."""
    assert name not in _bundled_codex_skills()
    body = f"---\nname: {name}\n---\n"
    if marker:
        body += f"<!-- {_SKILL_MARKER}: auto-generated -->\n"
    (skills / name).mkdir(parents=True)
    (skills / name / _SKILL_FILENAME).write_text(body, encoding="utf-8")


class TestDefaultSkillsDir:
    """`modified_codex_skills()` must resolve its own default directory."""

    def test_modified_codex_skills_resolves_its_default_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-argument call finds the real `~/.agents/skills`, not a NameError.

        The assertion is deliberately positive rather than "does not raise":
        a default that resolved to some other directory would also not raise,
        and would silently judge nothing.
        """
        home = tmp_path / "profile"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        skills = home / ".agents" / "skills"
        install_codex_skills(skills)
        name = sorted(_bundled_codex_skills())[0]
        target = skills / name / _SKILL_FILENAME
        target.write_text(
            target.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8",
        )

        assert modified_codex_skills() == [name]

    def test_orphaned_codex_skills_resolves_its_default_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Its sibling gets the same treatment, for the same reason."""
        home = tmp_path / "profile"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        skills = home / ".agents" / "skills"
        install_codex_skills(skills)
        _plant_orphan(skills)

        assert orphaned_codex_skills() == ["aelf-gone"]


class TestOrphanedSkills:
    """AC2: an orphaned marker-carrying skill is reported, and the
    printed installed-count agrees with what doctor judged."""

    def test_an_orphaned_marked_skill_is_reported(
        self, installed: tuple[Path, Path],
    ) -> None:
        codex_dir, skills = installed
        _plant_orphan(skills)

        rc, out = _doctor(codex_dir, skills)
        assert rc == 1, out
        assert "aelf-gone" in out
        assert "no longer part of this build" in out

    def test_an_unmarked_aelf_dir_is_never_our_orphan(
        self, installed: tuple[Path, Path],
    ) -> None:
        """No marker means not ours, so its presence is not ours to judge.

        This is the distinguishing assert. An implementation that globs
        ``aelf-*`` and drops the `_is_owned_skill_dir` call passes the test
        above and fails doctor on a user's hand-authored `aelf-mine` skill —
        the same trap `_is_owned_skill_dir` guards on the prune path, where
        getting it wrong deletes the user's file rather than nagging about it.
        """
        codex_dir, skills = installed
        _plant_orphan(skills, "aelf-mine", marker=False)

        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
        assert "aelf-mine" not in out

    def test_the_reported_count_agrees_with_what_doctor_judges(
        self, installed: tuple[Path, Path],
    ) -> None:
        """AC2's second half: the installed count and the judged set match.

        `count_installed_codex_skills` globs and counts the orphan; the
        bundle-driven scan cannot see it. Printing only the glob total left
        the two numbers disagreeing with nothing on screen to reconcile them.
        """
        codex_dir, skills = installed
        _plant_orphan(skills)
        n_bundled = len(_bundled_codex_skills())
        assert count_installed_codex_skills(skills) == n_bundled + 1

        _, out = _doctor(codex_dir, skills)
        assert f"{n_bundled + 1} installed (1 orphaned)" in out

    def test_orphans_are_not_judged_when_no_wiring_is_installed(
        self, tmp_path: Path,
    ) -> None:
        """Absent wiring stays exit 0, per the ruling on #1430.

        Skills left behind by a machine that never wired Codex hooks — or
        never removed them — must not turn doctor red. The informational
        count still reports them, because that line describes the directory
        rather than passing judgement on it.
        """
        codex_dir = tmp_path / "codex"
        codex_dir.mkdir()
        skills = tmp_path / "skills"
        install_codex_skills(skills)
        _plant_orphan(skills)

        rc, out = _doctor(codex_dir, skills)
        assert rc == 0, out
        assert "(1 orphaned)" in out
        assert "no longer part of this build" not in out
