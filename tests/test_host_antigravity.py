"""Antigravity host install, skills, and doctor (`aelfrice.host_antigravity`)."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from aelfrice import host_antigravity as ha

pytestmark = pytest.mark.timeout(30)


@pytest.fixture
def hook_cmd(tmp_path: Path) -> str:
    exe = tmp_path / "bin" / "aelf-agy-hook"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_table_covers_the_five_events_with_one_command(hook_cmd: str) -> None:
    table = ha.desired_agy_hooks(hook_cmd)
    assert list(table) == ["SessionStart", "PreInvocation", "PreToolUse", "PostToolUse", "Stop"]
    assert table["PreToolUse"] == [{"matcher": "run_command|invoke_subagent", "hooks": [
        {"type": "command", "command": hook_cmd, "timeout": 45}]}]
    assert table["PostToolUse"] == [{"matcher": "run_command", "hooks": [
        {"type": "command", "command": hook_cmd, "timeout": 15}]}]
    assert table["SessionStart"] == [{"type": "command", "command": hook_cmd, "timeout": 15}]


def test_timeouts_sum_the_roles_an_event_runs(hook_cmd: str) -> None:
    table = ha.desired_agy_hooks(hook_cmd, timeouts_s={
        "session_start": 15, "transcript_ingest": 30, "user_prompt_submit": 15,
        "search_tool_bash": 15, "stop": 30,
    })
    pre_inv = table["PreInvocation"]
    assert isinstance(pre_inv, list) and pre_inv[0]["timeout"] == 75  # type: ignore[index]
    stop = table["Stop"]
    assert isinstance(stop, list) and stop[0]["timeout"] == 60  # type: ignore[index]


def test_skip_passes_roles_and_drops_empty_events(hook_cmd: str) -> None:
    table = ha.desired_agy_hooks(hook_cmd, skip=frozenset({
        "commit_ingest", "agent_context", "session_start",
    }))
    assert "PostToolUse" not in table
    assert "SessionStart" not in table
    group = json.loads(json.dumps(table["PreToolUse"]))[0]
    assert group["matcher"] == "run_command"
    assert group["hooks"][0]["command"] == (
        f"{hook_cmd} --skip agent_context,commit_ingest,session_start"
    )


def test_install_keeps_foreign_keys_and_is_idempotent(tmp_path: Path, hook_cmd: str) -> None:
    path = tmp_path / "hooks.json"
    foreign = {"lint": {"PostToolUse": [{"matcher": "*", "hooks": [{"command": "./x.sh"}]}]}}
    path.write_text(json.dumps(foreign))
    first = ha.install_agy_hooks(path, hook_cmd)
    second = ha.install_agy_hooks(path, hook_cmd)
    data = _load(path)
    assert first.changed and not second.changed
    assert data["lint"] == foreign["lint"]
    assert data[ha.OWNED_HOOK_KEY] == ha.desired_agy_hooks(hook_cmd)


def test_uninstall_removes_only_the_owned_key(tmp_path: Path, hook_cmd: str) -> None:
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps({"lint": {"Stop": []}}))
    ha.install_agy_hooks(path, hook_cmd)
    result = ha.uninstall_agy_hooks(path)
    assert result.changed
    assert _load(path) == {"lint": {"Stop": []}}
    assert not ha.uninstall_agy_hooks(path).changed
    assert not ha.uninstall_agy_hooks(tmp_path / "missing.json").changed


def test_uninstall_leaves_an_empty_object(tmp_path: Path, hook_cmd: str) -> None:
    path = tmp_path / "hooks.json"
    ha.install_agy_hooks(path, hook_cmd)
    ha.uninstall_agy_hooks(path)
    assert _load(path) == {}


def test_skills_cover_the_bundle_minus_claude_memory() -> None:
    from aelfrice.setup import bundled_slash_files

    skills = ha.bundled_agy_skills()
    assert "aelf-audit-claude-memory" not in skills
    assert "aelf-reconcile-claude-memory" not in skills
    expected = {
        "aelf-" + name.removesuffix(".md") for name in bundled_slash_files()
        if name != "audit-claude-memory.md"
    }
    assert set(skills) == expected


def test_skill_frontmatter_quotes_the_description() -> None:
    text = "---\nname: aelf:doctor\ndescription: Diagnose hooks: exits 1 on failure.\n---\nRun `uv run aelf doctor`.\n"
    name, skill = ha.antigravity_skill_from_slash("doctor.md", text)
    assert name == "aelf-doctor"
    lines = skill.splitlines()
    assert lines[1] == "name: aelf-doctor"
    assert lines[2] == 'description: "Diagnose hooks: exits 1 on failure."'
    assert "--host antigravity" in skill
    assert "aelf doctor" in skill and "uv run aelf" not in skill


def test_skills_install_prune_and_protect_user_skills(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    user = dest / "aelf-search"
    user.mkdir(parents=True)
    (user / "SKILL.md").write_text("---\nname: aelf-search\n---\nmine\n")
    orphan = dest / "aelf-gone"
    orphan.mkdir()
    (orphan / "SKILL.md").write_text("<!-- AELFRICE-ANTIGRAVITY-SKILL -->")
    result = ha.install_agy_skills(dest)
    assert "aelf-search" in result.skipped
    assert (user / "SKILL.md").read_text().endswith("mine\n")
    assert result.pruned == ("aelf-gone",)
    assert not orphan.exists()
    again = ha.install_agy_skills(dest)
    assert again.written == ()
    removed = ha.remove_agy_skills(dest)
    assert "aelf-search" not in removed.pruned
    assert user.exists()
    assert not (dest / "aelf-doctor").exists()


def test_doctor_passes_a_fresh_install(tmp_path: Path, hook_cmd: str) -> None:
    path = tmp_path / "hooks.json"
    ha.install_agy_hooks(path, hook_cmd)
    ha.install_agy_skills(tmp_path / "skills")
    result = ha.doctor_agy(path, tmp_path / "skills")
    assert not result.broken, result.lines
    assert any(line.startswith("unsupported on antigravity:") for line in result.lines)


@pytest.mark.parametrize("case", ["missing", "disabled", "bad_event", "no_program", "ms_timeout"])
def test_doctor_flags_broken_entries(tmp_path: Path, hook_cmd: str, case: str) -> None:
    path = tmp_path / "hooks.json"
    ha.install_agy_hooks(path, hook_cmd)
    data = _load(path)
    entry = data[ha.OWNED_HOOK_KEY]
    assert isinstance(entry, dict)
    if case == "missing":
        data = {}
    elif case == "disabled":
        entry["enabled"] = False
    elif case == "bad_event":
        entry["UserPromptSubmit"] = [{"command": hook_cmd}]
    elif case == "no_program":
        entry["Stop"] = [{"command": str(tmp_path / "nope"), "timeout": 30}]
    else:
        entry["Stop"] = [{"command": hook_cmd, "timeout": 30000}]
    path.write_text(json.dumps(data))
    assert ha.doctor_agy(path, tmp_path / "skills").broken
