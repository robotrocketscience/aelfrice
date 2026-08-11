"""The trust half of #1430, carried by #1486.

`modified_codex_skills()` called with no argument raised `NameError`.
#1427 deleted the module-level ``AGENTS_SKILLS_DIR`` constant in favour of
the late-bound `resolve_agents_skills_dir()`, because a constant froze
``$HOME`` at import time; #1476 then wrote a fresh reference to the deleted
name into the new function's default-dir branch. Every production caller
passes a directory explicitly, so the branch was unreachable — but nothing
in the tree caught it either, and the next caller to take the default would
have crashed doctor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.host_codex import (
    _SKILL_FILENAME,
    _bundled_codex_skills,
    install_codex_skills,
    modified_codex_skills,
)


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
