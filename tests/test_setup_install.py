"""Idempotent install/uninstall of UserPromptSubmit hooks in settings.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.setup import (
    PROJECT_SETTINGS_RELPATH,
    USER_SETTINGS_PATH,
    default_settings_path,
    install_user_prompt_submit_hook,
    uninstall_user_prompt_submit_hook,
)

_HOOK_CMD = "aelf retrieve --hook"
_OTHER_CMD = "/some/other/script.sh"


def _read_json(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)  # pyright: ignore[reportAny]
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _user_prompt_submit_list(data: dict[str, object]) -> list[dict[str, object]]:
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    entries = hooks_typed["UserPromptSubmit"]
    assert isinstance(entries, list)
    return cast(list[dict[str, object]], entries)


def _entry_command(entry: dict[str, object], idx: int = 0) -> str:
    inner = entry["hooks"]
    assert isinstance(inner, list)
    inner_typed = cast(list[dict[str, object]], inner)
    cmd = inner_typed[idx]["command"]
    assert isinstance(cmd, str)
    return cmd


def test_default_settings_path_user() -> None:
    assert default_settings_path("user") == USER_SETTINGS_PATH


def test_default_settings_path_project_uses_cwd_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_settings_path("project") == tmp_path / PROJECT_SETTINGS_RELPATH


def test_default_settings_path_project_explicit_root(tmp_path: Path) -> None:
    got = default_settings_path("project", project_root=tmp_path)
    assert got == tmp_path / PROJECT_SETTINGS_RELPATH


def test_install_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    assert not settings.exists()
    result = install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert result.installed is True
    assert result.already_present is False
    assert result.path == settings
    data = _read_json(settings)
    entries = _user_prompt_submit_list(data)
    assert len(entries) == 1
    inner = entries[0]["hooks"]
    assert isinstance(inner, list)
    inner_typed = cast(list[dict[str, object]], inner)
    assert len(inner_typed) == 1
    assert inner_typed[0] == {"type": "command", "command": _HOOK_CMD}


def test_install_preserves_unrelated_keys(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "claude-opus-4-7",
                "permissions": {"allow": ["Bash(ls *)"]},
            }
        ),
        encoding="utf-8",
    )
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    data = _read_json(settings)
    assert data["model"] == "claude-opus-4-7"
    assert data["permissions"] == {"allow": ["Bash(ls *)"]}
    assert "hooks" in data


def test_install_preserves_other_hook_events(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "echo hi"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    data = _read_json(settings)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    assert "SessionStart" in hooks_typed
    assert "UserPromptSubmit" in hooks_typed
    session_start = hooks_typed["SessionStart"]
    assert isinstance(session_start, list)
    session_start_typed = cast(list[object], session_start)
    assert len(session_start_typed) == 1


def test_install_appends_when_other_user_prompt_entries_exist(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": _OTHER_CMD}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    data = _read_json(settings)
    entries = _user_prompt_submit_list(data)
    assert len(entries) == 2
    commands = [_entry_command(entry) for entry in entries]
    assert _OTHER_CMD in commands
    assert _HOOK_CMD in commands


def test_install_is_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    first = install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert first.installed is True
    assert first.already_present is False
    second = install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert second.installed is False
    assert second.already_present is True
    data = _read_json(settings)
    entries = _user_prompt_submit_list(data)
    assert len(entries) == 1


def test_install_records_optional_timeout_and_status(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(
        settings,
        command=_HOOK_CMD,
        timeout=7,
        status_message="searching aelfrice...",
    )
    data = _read_json(settings)
    entries = _user_prompt_submit_list(data)
    inner = entries[0]["hooks"]
    assert isinstance(inner, list)
    inner_typed = cast(list[dict[str, object]], inner)
    assert inner_typed[0] == {
        "type": "command",
        "command": _HOOK_CMD,
        "timeout": 7,
        "statusMessage": "searching aelfrice...",
    }


def test_install_rejects_empty_command(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        install_user_prompt_submit_hook(settings, command="")


def test_install_rejects_non_object_root(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        install_user_prompt_submit_hook(settings, command=_HOOK_CMD)


def test_install_rejects_non_object_hooks(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        install_user_prompt_submit_hook(settings, command=_HOOK_CMD)


def test_install_rejects_non_list_event(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"hooks": {"UserPromptSubmit": {}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        install_user_prompt_submit_hook(settings, command=_HOOK_CMD)


def test_install_handles_empty_file(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("", encoding="utf-8")
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    data = _read_json(settings)
    assert "hooks" in data


def test_uninstall_returns_zero_when_file_missing(tmp_path: Path) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    result = uninstall_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert result.removed == 0
    assert result.path == settings
    assert not settings.exists()


def test_uninstall_returns_zero_when_event_missing(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"hooks": {"SessionStart": []}}), encoding="utf-8"
    )
    result = uninstall_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert result.removed == 0


def test_uninstall_removes_matching_entry_only(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(settings, command=_OTHER_CMD)
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    result = uninstall_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert result.removed == 1
    data = _read_json(settings)
    entries = _user_prompt_submit_list(data)
    assert len(entries) == 1
    assert _entry_command(entries[0]) == _OTHER_CMD


def test_uninstall_is_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    first = uninstall_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert first.removed == 1
    second = uninstall_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    assert second.removed == 0


def test_uninstall_rejects_empty_command(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        uninstall_user_prompt_submit_hook(settings, command="")


def test_uninstall_basename_match_removes_bare_and_absolute(
    tmp_path: Path,
) -> None:
    """Basename mode catches multiple stacked stale-path entries.

    Settings files from before #781's install-dedup landing may carry
    several UserPromptSubmit entries with the same basename but
    different paths (uv/pipx/venv churn). install_user_prompt_submit_hook
    now collapses those on append, so this test writes the stale shape
    directly.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "aelf-hook"}]},
                {"hooks": [{
                    "type": "command",
                    "command": "/Users/me/.venv/bin/aelf-hook",
                }]},
                {"hooks": [{"type": "command", "command": "/keep/me"}]},
            ],
        },
    }))

    result = uninstall_user_prompt_submit_hook(
        settings, command_basename="aelf-hook"
    )

    assert result.removed == 2
    data = _read_json(settings)
    cmds = [
        cast(list[dict[str, object]], entry["hooks"])[0]["command"]
        for entry in _user_prompt_submit_list(data)
    ]
    assert cmds == ["/keep/me"]


def test_uninstall_requires_exactly_one_match_kind(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    with pytest.raises(ValueError):
        uninstall_user_prompt_submit_hook(settings)
    with pytest.raises(ValueError):
        uninstall_user_prompt_submit_hook(
            settings, command="x", command_basename="x"
        )
    with pytest.raises(ValueError):
        uninstall_user_prompt_submit_hook(settings, command_basename="")


def test_install_atomic_no_partial_file_on_simulated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "settings.json"
    original_payload: dict[str, object] = {"hooks": {"SessionStart": []}}
    settings.write_text(json.dumps(original_payload), encoding="utf-8")
    import aelfrice.setup as setup_mod

    def boom(_src: str | Path, _dst: str | Path) -> None:
        raise RuntimeError("simulated rename failure")

    monkeypatch.setattr(setup_mod.os, "replace", boom)
    with pytest.raises(RuntimeError, match="simulated rename failure"):
        install_user_prompt_submit_hook(settings, command=_HOOK_CMD)
    surviving = _read_json(settings)
    assert surviving == original_payload
    leftover = list(tmp_path.glob("settings.json.*.tmp"))
    assert leftover == []
