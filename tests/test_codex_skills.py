"""Codex agent-skills port of the /aelf:* slash commands.

The `$aelf-*` skills are DERIVED from the same bundle the Claude installer
ships (`src/aelfrice/slash_commands/*.md`), so parity with the slash
surface is automatic. All install/remove paths are tmp_path-scoped;
nothing touches a real ~/.agents/skills.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.host_codex import (
    _SKILL_MARKER,
    _bundled_codex_skills,
    _is_owned_skill_dir,
    codex_skill_from_slash,
    count_installed_codex_skills,
    install_codex_skills,
    remove_codex_skills,
)
from aelfrice.setup import bundled_slash_files as _bundled_slash_files

# --- transform -------------------------------------------------------------


def test_transform_renames_colon_to_hyphen() -> None:
    name, text = codex_skill_from_slash("search.md", _bundled_slash_files()["search.md"])
    assert name == "aelf-search"
    assert "name: aelf-search" in text
    # No colon form leaks into the skill name (invalid in dir names).
    assert "name: aelf:search" not in text


def test_transform_keeps_only_name_and_description_frontmatter() -> None:
    _, text = codex_skill_from_slash("search.md", _bundled_slash_files()["search.md"])
    head = text.split("---", 2)[1]
    assert "name:" in head
    assert "description:" in head
    # allowed-tools / argument-hint must not survive into the frontmatter.
    assert "allowed-tools" not in head
    assert "argument-hint" not in head


def test_transform_carries_marker_and_invocation() -> None:
    _, text = codex_skill_from_slash("status.md", _bundled_slash_files()["status.md"])
    assert _SKILL_MARKER in text
    assert "$aelf-status" in text


def test_transform_body_is_verbatim() -> None:
    src = _bundled_slash_files()["status.md"]
    _, text = codex_skill_from_slash("status.md", src)
    # The <objective>/<process> body is copied through unchanged.
    body = src.split("---", 2)[2].strip()
    assert body in text


def test_argument_hint_folds_into_adapter() -> None:
    _, text = codex_skill_from_slash("search.md", _bundled_slash_files()["search.md"])
    assert "Arguments:" in text
    assert '$ARGUMENTS' in text  # source body still references it
    assert "substitute the text" in text


def test_no_arguments_command_omits_arg_note() -> None:
    _, text = codex_skill_from_slash("status.md", _bundled_slash_files()["status.md"])
    # `status` takes no args -> no argument-hint, no $ARGUMENTS note.
    assert "Arguments:" not in text
    assert "substitute the text" not in text


def test_task_mapping_only_on_subagent_commands() -> None:
    skills = _bundled_codex_skills()
    tagged = {n for n, t in skills.items() if "Codex's own subagent" in t}
    assert tagged == {"aelf-onboard", "aelf-reason", "aelf-wonder"}


def test_transform_is_deterministic() -> None:
    src = _bundled_slash_files()["reason.md"]
    a = codex_skill_from_slash("reason.md", src)
    b = codex_skill_from_slash("reason.md", src)
    assert a == b


# --- parity with the slash bundle -----------------------------------------


def test_skill_count_matches_bundle_no_collisions() -> None:
    files = _bundled_slash_files()
    skills = _bundled_codex_skills()
    assert len(skills) == len(files)  # every command -> exactly one skill
    for name in skills:
        assert name.startswith("aelf-")


# --- installer -------------------------------------------------------------


def test_install_writes_one_skill_dir_per_command(tmp_path: Path) -> None:
    result = install_codex_skills(tmp_path)
    assert len(result.written) == len(_bundled_codex_skills())
    assert not result.already
    assert not result.pruned
    sample = tmp_path / "aelf-search" / "SKILL.md"
    assert sample.is_file()
    assert _SKILL_MARKER in sample.read_text(encoding="utf-8")


def test_install_is_idempotent(tmp_path: Path) -> None:
    install_codex_skills(tmp_path)
    again = install_codex_skills(tmp_path)
    assert not again.written
    assert len(again.already) == len(_bundled_codex_skills())
    assert not again.pruned


def test_install_prunes_stale_owned_skill(tmp_path: Path) -> None:
    install_codex_skills(tmp_path)
    # A stale aelfrice skill (renamed/removed command) carrying our marker.
    stale = tmp_path / "aelf-gone"
    stale.mkdir()
    (stale / "SKILL.md").write_text(
        f"---\nname: aelf-gone\ndescription: x\n---\n<!-- {_SKILL_MARKER} -->\n",
        encoding="utf-8",
    )
    result = install_codex_skills(tmp_path)
    assert "aelf-gone" in result.pruned
    assert not stale.exists()


def test_install_never_prunes_foreign_or_unmarked(tmp_path: Path) -> None:
    # A user's own aelf-* skill (no marker) and a non-aelf skill.
    foreign = tmp_path / "aelf-mine"
    foreign.mkdir()
    (foreign / "SKILL.md").write_text(
        "---\nname: aelf-mine\ndescription: hand made\n---\nbody\n",
        encoding="utf-8",
    )
    other = tmp_path / "dual-matrix"
    other.mkdir()
    (other / "SKILL.md").write_text("x", encoding="utf-8")

    install_codex_skills(tmp_path)
    assert foreign.exists()
    assert not _is_owned_skill_dir(foreign)
    assert other.exists()


def test_count_installed(tmp_path: Path) -> None:
    assert count_installed_codex_skills(tmp_path) == 0
    install_codex_skills(tmp_path)
    assert count_installed_codex_skills(tmp_path) == len(_bundled_codex_skills())


# --- remove ----------------------------------------------------------------


def test_remove_deletes_only_owned(tmp_path: Path) -> None:
    foreign = tmp_path / "aelf-mine"
    foreign.mkdir()
    (foreign / "SKILL.md").write_text(
        "---\nname: aelf-mine\ndescription: mine\n---\nbody\n", encoding="utf-8"
    )
    install_codex_skills(tmp_path)
    result = remove_codex_skills(tmp_path)
    assert len(result.pruned) == len(_bundled_codex_skills())
    assert count_installed_codex_skills(tmp_path) == 0
    # The user's own skill is left standing.
    assert foreign.exists()


def test_remove_on_empty_dir_is_noop(tmp_path: Path) -> None:
    result = remove_codex_skills(tmp_path)
    assert not result.pruned


def test_remove_on_missing_dir_is_noop(tmp_path: Path) -> None:
    result = remove_codex_skills(tmp_path / "does-not-exist")
    assert not result.pruned
