"""Codex agent-skills port of the /aelf:* slash commands.

The `$aelf-*` skills are DERIVED from the same bundle the Claude installer
ships (`src/aelfrice/slash_commands/*.md`), so parity with the slash
surface is automatic. All install/remove paths are tmp_path-scoped;
nothing touches a real ~/.agents/skills.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from aelfrice.cli import (
    _cmd_doctor_codex,
    _cmd_setup_codex,
    _cmd_unsetup_codex,
)
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


def test_low_tier_classifier_note_for_codex_onboard() -> None:
    # #1153: onboard's classification fan-out defaults to a low-cost
    # model tier (tier-abstract since #1155) — on Codex the cheap tier is
    # a `-mini`-class model. The transform must steer the fan-out to that
    # tier rather than let it fall through to the session's default model.
    skills = _bundled_codex_skills()
    onboard = skills["aelf-onboard"]
    assert "`-mini`-class model" in onboard
    assert "not the session's default model" in onboard
    # The note is onboard-specific: other subagent skills (reason/wonder)
    # have no model directive and must not carry it.
    for name in ("aelf-reason", "aelf-wonder"):
        assert "`-mini`-class model" not in skills[name]


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


# --- host-management steering (#1136) ---------------------------------------

_HOST_MGMT = ("aelf-setup", "aelf-doctor", "aelf-uninstall", "aelf-upgrade")


def test_host_management_skills_carry_codex_steering() -> None:
    skills = _bundled_codex_skills()
    for name in _HOST_MGMT:
        text = skills[name]
        assert "<host-adapter>" in text, name
        assert "--host codex" in text, name
        # The note must precede the body so it governs the Run: lines.
        assert text.index("<host-adapter>") < text.index("<objective>"), name


def test_ordinary_skills_lack_codex_steering() -> None:
    skills = _bundled_codex_skills()
    for name, text in skills.items():
        if name in _HOST_MGMT:
            continue
        assert "<host-adapter>" not in text, name
        assert "--host codex" not in text, name


def test_setup_description_rewritten_for_codex_host() -> None:
    _, text = codex_skill_from_slash(
        "setup.md", _bundled_slash_files()["setup.md"]
    )
    head = text.split("---", 2)[1]
    # Describes the codex-host effect, not another host's artifacts.
    assert "hooks.json" in head
    assert ".agents/skills" in head
    assert "settings.json" not in head
    assert "statusline" not in head


def test_non_setup_descriptions_pass_through() -> None:
    src = _bundled_slash_files()["doctor.md"]
    _, text = codex_skill_from_slash("doctor.md", src)
    head = text.split("---", 2)[1]
    src_desc = [
        line for line in src.splitlines() if line.startswith("description:")
    ][0]
    assert src_desc in head


# --- replace path is marker-gated (#1136) -----------------------------------


def test_replace_never_clobbers_foreign_same_name_skill(tmp_path: Path) -> None:
    # A hand-authored (unmarked) skill whose name collides with a bundled one.
    foreign = tmp_path / "aelf-search"
    foreign.mkdir()
    original = "---\nname: aelf-search\ndescription: mine\n---\nhand made\n"
    (foreign / "SKILL.md").write_text(original, encoding="utf-8")

    result = install_codex_skills(tmp_path)
    assert "aelf-search" in result.skipped
    assert "aelf-search" not in result.written
    assert (foreign / "SKILL.md").read_text(encoding="utf-8") == original


def test_replace_updates_stale_owned_skill(tmp_path: Path) -> None:
    install_codex_skills(tmp_path)
    target = tmp_path / "aelf-search" / "SKILL.md"
    stale = target.read_text(encoding="utf-8") + "\nstale tail\n"
    target.write_text(stale, encoding="utf-8")

    result = install_codex_skills(tmp_path)
    assert "aelf-search" in result.written
    assert not result.skipped
    assert "stale tail" not in target.read_text(encoding="utf-8")


# --- prune/remove failures surface (#1136) ----------------------------------


def test_remove_reports_leftover_dir_and_deletes_nothing_else(
    tmp_path: Path,
) -> None:
    install_codex_skills(tmp_path)
    stray = tmp_path / "aelf-search" / "notes.txt"
    stray.write_text("keep me", encoding="utf-8")

    result = remove_codex_skills(tmp_path)
    # SKILL.md is gone -> the skill counts as removed ...
    assert "aelf-search" in result.pruned
    assert not (tmp_path / "aelf-search" / "SKILL.md").exists()
    # ... but the half-removal is surfaced, and the stray file survives
    # (nothing is deleted recursively).
    assert any(msg.startswith("aelf-search:") for msg in result.failed)
    assert stray.exists()


def test_install_prune_reports_leftover_dir(tmp_path: Path) -> None:
    install_codex_skills(tmp_path)
    stale = tmp_path / "aelf-gone"
    stale.mkdir()
    (stale / "SKILL.md").write_text(
        f"---\nname: aelf-gone\ndescription: x\n---\n<!-- {_SKILL_MARKER} -->\n",
        encoding="utf-8",
    )
    (stale / "extra.txt").write_text("x", encoding="utf-8")

    result = install_codex_skills(tmp_path)
    assert "aelf-gone" in result.pruned
    assert any(msg.startswith("aelf-gone:") for msg in result.failed)
    assert (stale / "extra.txt").exists()


def test_clean_remove_has_no_failures(tmp_path: Path) -> None:
    install_codex_skills(tmp_path)
    result = remove_codex_skills(tmp_path)
    assert result.failed == ()


# --- CLI layer: setup/unsetup --host codex via injectable dests (#1136) -----


def _setup_args(**overrides: object) -> "argparse.Namespace":
    ns = argparse.Namespace(host="codex", force=False, codex_skills=True)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _neutralize_opt_out(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Keep the #1053 opt-out side effects away from real state."""
    import aelfrice.auto_install as auto_install
    import aelfrice.host_codex as hc

    monkeypatch.setattr(hc, "claude_host_has_aelfrice_hooks", lambda _p: False)
    monkeypatch.setattr(
        auto_install, "read_host_opt_outs", lambda *a, **k: frozenset({"claude"})
    )
    monkeypatch.setattr(
        auto_install,
        "add_host_opt_out",
        lambda *a, **k: pytest.fail("must not write the real opt-out file"),
    )


def test_cli_setup_codex_installs_into_injected_dests(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    _neutralize_opt_out(monkeypatch)
    out = io.StringIO()
    rc = _cmd_setup_codex(
        _setup_args(),
        out,
        hooks_path=tmp_path / "codex" / "hooks.json",
        skills_dest=tmp_path / "skills",
    )
    assert rc == 0
    assert (tmp_path / "codex" / "hooks.json").is_file()
    assert (tmp_path / "skills" / "aelf-search" / "SKILL.md").is_file()
    assert "installed" in out.getvalue()


def test_cli_setup_codex_skills_oserror_exits_1_cleanly(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
    capsys: "pytest.CaptureFixture[str]",
) -> None:
    import aelfrice.host_codex as hc
    _neutralize_opt_out(monkeypatch)

    def boom(dest_dir: Path | None = None) -> object:
        raise PermissionError("skills dir is read-only")

    monkeypatch.setattr(hc, "install_codex_skills", boom)
    out = io.StringIO()
    rc = _cmd_setup_codex(
        _setup_args(),
        out,
        hooks_path=tmp_path / "hooks.json",
        skills_dest=tmp_path / "skills",
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "skills install failed" in err
    assert "Traceback" not in err
    # hooks.json was written before the skills half failed.
    assert (tmp_path / "hooks.json").is_file()


def test_cli_setup_codex_reports_skipped_foreign_collision(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    _neutralize_opt_out(monkeypatch)
    skills = tmp_path / "skills"
    foreign = skills / "aelf-search"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text(
        "---\nname: aelf-search\ndescription: mine\n---\nhand made\n",
        encoding="utf-8",
    )
    out = io.StringIO()
    rc = _cmd_setup_codex(
        _setup_args(),
        out,
        hooks_path=tmp_path / "hooks.json",
        skills_dest=skills,
    )
    assert rc == 0
    assert "skipped" in out.getvalue()
    assert "aelf-search" in out.getvalue()


def test_cli_unsetup_codex_removes_from_injected_dests(tmp_path: Path) -> None:
    from aelfrice.host_codex import install_codex_hooks

    hooks = tmp_path / "hooks.json"
    install_codex_hooks(hooks)
    install_codex_skills(tmp_path / "skills")

    out = io.StringIO()
    rc = _cmd_unsetup_codex(
        argparse.Namespace(host="codex"),
        out,
        hooks_path=hooks,
        skills_dest=tmp_path / "skills",
    )
    assert rc == 0
    assert count_installed_codex_skills(tmp_path / "skills") == 0
    assert "removed" in out.getvalue()


def test_cli_unsetup_codex_warns_on_leftover(tmp_path: Path) -> None:
    install_codex_skills(tmp_path / "skills")
    (tmp_path / "skills" / "aelf-search" / "notes.txt").write_text(
        "keep me", encoding="utf-8"
    )
    out = io.StringIO()
    rc = _cmd_unsetup_codex(
        argparse.Namespace(host="codex"),
        out,
        hooks_path=tmp_path / "hooks.json",
        skills_dest=tmp_path / "skills",
    )
    assert rc == 0
    assert "[warn] codex skill: aelf-search:" in out.getvalue()


def test_cli_doctor_codex_counts_injected_skills(tmp_path: Path) -> None:
    from aelfrice.host_codex import install_codex_hooks

    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json")
    install_codex_skills(tmp_path / "skills")

    out = io.StringIO()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"),
        out,
        codex_dir=codex_dir,
        skills_dest=tmp_path / "skills",
    )
    assert rc == 0
    n = len(_bundled_codex_skills())
    assert f"{n} installed" in out.getvalue()
