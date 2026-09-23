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


def test_transform_body_is_verbatim_apart_from_the_cli_prefix() -> None:
    """The body is carried through unchanged except for `uv run` (#1413).

    Before #1413 the body was byte-identical to its source. It no longer
    is, and that is the change rather than a workaround: the generated
    skill must invoke `aelf` directly, while the slash source keeps the
    `uv run` form that is correct for a source checkout on other hosts.
    Undoing the rewrite is the only difference, so this still pins that
    nothing else in the body is edited, reordered, or dropped.
    """
    src = _bundled_slash_files()["status.md"]
    _, text = codex_skill_from_slash("status.md", src)
    body = src.split("---", 2)[2].strip()
    assert "uv run aelf " in body, (
        "fixture no longer exercises the rewrite; pick a slash command "
        "whose body still invokes `uv run aelf`"
    )
    assert body not in text
    assert body.replace("uv run aelf ", "aelf ") in text


def test_no_generated_skill_invokes_uv_run() -> None:
    """AC1 (#1413): `uv run aelf` must not survive into any skill.

    Measured on the bundle before the fix: 96 occurrences across 31 of
    31 skills. `uv run` initializes and locks a cache before aelfrice
    starts, so a Codex sandbox with a read-only cache fails at the
    wrapper — `uv run aelf --version` exits 2 where `aelf --version`
    exits 0.
    """
    skills = _bundled_codex_skills()
    assert skills, "bundle is empty; the assertion below would be vacuous"
    offenders = {n: t.count("uv run") for n, t in skills.items() if "uv run" in t}
    assert offenders == {}, f"generated skills still wrap the CLI: {offenders}"


def test_every_bundled_uv_run_is_a_rewritable_prefix() -> None:
    """The literal replace is exact only while the bundle has one shape.

    A slash body that line-breaks the invocation, writes `uv run
    aelfrice`, or ends a line on `uv run aelf` would be rewritten
    wrongly or not at all, and the skill would ship half-converted. Pin
    the property against the source bundle so such an edit reds here
    rather than in a user's sandbox.
    """
    import re

    bad: dict[str, list[str]] = {}
    for filename, text in _bundled_slash_files().items():
        hits = [
            m.group(0)
            for m in re.finditer(r"uv\s+run\s+\S*", text)
            if m.group(0) != "uv run aelf"
        ]
        # `uv run aelf` with nothing after it is also unrewritable.
        hits += [m.group(0) for m in re.finditer(r"uv run aelf(?![ \t])", text)]
        if hits:
            bad[filename] = hits
    assert bad == {}, f"unrewritable `uv run` forms in the slash bundle: {bad}"


@pytest.mark.timeout(240)
def test_a_generated_command_runs_against_a_read_only_uv_cache(
    tmp_path: Path,
) -> None:
    """AC2 (#1413): the failure reproduces, and the fix clears it.

    Both arms run the command as the generated skill spells it, against
    a PATH holding only a fake `aelf` shim and a uv cache directory
    stripped of write permission. The wrapped form is the control: if it
    ever stops failing, this test proves nothing and the assertion below
    says so rather than passing quietly.
    """
    import os
    import re
    import shutil
    import stat
    import subprocess

    if os.name != "posix":
        pytest.skip("shim is a POSIX script; Windows resolution is covered below")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; nothing to wrap")

    text = _bundled_codex_skills()["aelf-status"]
    commands = [
        c for c in re.findall(r"`(aelf [^`]+)`", text) if not c.endswith("...")
    ]
    assert commands, "no runnable command found in the generated skill"
    command = commands[0]

    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "aelf"
    shim.write_text("#!/bin/sh\necho SHIM-OK \"$@\"\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    cache = tmp_path / "ro-cache"
    cache.mkdir()
    cache.chmod(stat.S_IRUSR | stat.S_IXUSR)

    env = dict(os.environ)
    env["PATH"] = str(bindir)
    env["UV_CACHE_DIR"] = str(cache)

    try:
        direct = subprocess.run(
            command.split(), env=env, capture_output=True, text=True, timeout=60,
        )
        wrapped = subprocess.run(
            [uv, "run", *command.split()],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
        )
    finally:
        cache.chmod(stat.S_IRWXU)

    assert wrapped.returncode != 0, (
        "the control passed: a read-only uv cache no longer breaks the "
        "wrapped form, so this test cannot distinguish the fix"
    )
    assert direct.returncode == 0, direct.stderr
    assert "SHIM-OK" in direct.stdout


def test_the_generated_command_cannot_discover_a_project_environment() -> None:
    """AC3 (#1413): no shadowing by an unactivated checkout.

    `uv run` walks up for a `pyproject.toml` and selects that project's
    environment, so a checkout holding a different aelfrice version wins
    over the `uv tool` install that generated the skill. A bare command
    resolves through PATH only, which satisfies this by construction —
    so what there is to test is that no invocation shape capable of
    project discovery survives into the bundle.
    """
    import re

    discovering = re.compile(r"\b(uv run|uvx|python -m aelfrice|poetry run|pipenv run)\b")
    offenders = {
        name: sorted(set(discovering.findall(text)))
        for name, text in _bundled_codex_skills().items()
        if discovering.search(text)
    }
    assert offenders == {}, f"project-discovering invocations survive: {offenders}"


def test_generated_commands_resolve_on_both_posix_and_windows_shims() -> None:
    """AC4 (#1413): one spelling has to serve both shim families.

    A `uv tool` install puts `aelf` on PATH as an executable on POSIX and
    as `aelf.exe`/`aelf.cmd` on Windows. A bare `aelf` resolves to either
    — the extension is supplied by PATHEXT — while any spelling carrying
    an extension, a directory part, or a `./` prefix binds the skill to
    one platform. Assert the bundle only ever uses the bare form.
    """
    import re

    bad: dict[str, list[str]] = {}
    for name, text in _bundled_codex_skills().items():
        hits = re.findall(r"`([^`\n]*\baelf(?:\.exe|\.cmd|\.bat)?\b[^`\n]*)`", text)
        offending = [
            h for h in hits
            if re.search(r"\baelf\.(exe|cmd|bat)\b", h)
            # A directory part on the COMMAND: `./aelf`, `~/bin/aelf`,
            # `/usr/local/bin/aelf`. The trailing class keeps out
            # `/aelf:search` (a slash-command name) and
            # `/tmp/aelf-wonder-dispatch.jsonl` (an argument path that
            # merely starts with the same four letters), and a directory
            # such as `~/.../commands/aelf/`, which ends in a separator.
            or re.search(r"(?:\.|~|[\w.-])/(?:[\w.~-]+/)*aelf(?![\w:./-])", h)
        ]
        if offending:
            bad[name] = offending
    assert bad == {}, f"platform-bound spellings in the bundle: {bad}"


def test_generated_skills_bake_in_no_absolute_path() -> None:
    """AC (#1413, 2026-08-06 fork): no machine-specific path.

    The skill must resolve `aelf` through the invoking shell's PATH, so
    a bundle generated on one machine stays valid on another. Guard the
    two shapes an absolute launcher path would take.
    """
    import re

    for name, text in _bundled_codex_skills().items():
        assert not re.search(r"/\S*/bin/aelf\b", text), f"{name} bakes a POSIX path"
        assert not re.search(r"[A-Za-z]:\\\\\S*aelf\.(exe|cmd)", text), (
            f"{name} bakes a Windows path"
        )


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
