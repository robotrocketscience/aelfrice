"""install_session_start_hook / uninstall_session_start_hook idempotency."""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.setup import (
    install_session_start_hook,
    install_user_prompt_submit_hook,
    uninstall_session_start_hook,
    uninstall_user_prompt_submit_hook,
)


_SS_CMD = "/usr/local/bin/aelf-session-start-hook"
_UPS_CMD = "/usr/local/bin/aelf-hook"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


# ---- install -----------------------------------------------------------


def test_install_session_start_writes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = install_session_start_hook(p, command=_SS_CMD)
    assert result.installed is True
    assert result.already_present is False
    data = _read(p)
    assert "SessionStart" in data["hooks"]  # type: ignore[index]
    entry_list = data["hooks"]["SessionStart"]  # type: ignore[index]
    assert isinstance(entry_list, list) and len(entry_list) == 1
    inner = entry_list[0]["hooks"][0]
    assert inner["type"] == "command"
    assert inner["command"] == _SS_CMD


def test_install_session_start_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_session_start_hook(p, command=_SS_CMD)
    second = install_session_start_hook(p, command=_SS_CMD)
    assert second.installed is False
    assert second.already_present is True
    data = _read(p)
    entry_list = data["hooks"]["SessionStart"]  # type: ignore[index]
    assert len(entry_list) == 1


def test_install_session_start_does_not_touch_other_events(
    tmp_path: Path,
) -> None:
    """Three events live independently in the same settings.json."""
    p = tmp_path / "settings.json"
    install_user_prompt_submit_hook(p, command=_UPS_CMD)
    install_session_start_hook(p, command=_SS_CMD)
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "UserPromptSubmit" in hooks
    assert "SessionStart" in hooks
    assert len(hooks["UserPromptSubmit"]) == 1
    assert len(hooks["SessionStart"]) == 1


# ---- uninstall ---------------------------------------------------------


def test_uninstall_session_start_removes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_session_start_hook(p, command=_SS_CMD)
    result = uninstall_session_start_hook(p, command=_SS_CMD)
    assert result.removed == 1
    data = _read(p)
    assert data["hooks"]["SessionStart"] == []  # type: ignore[index]


def test_uninstall_session_start_no_match(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_session_start_hook(p, command=_SS_CMD)
    result = uninstall_session_start_hook(p, command="/different/binary")
    assert result.removed == 0


def test_uninstall_session_start_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "nope.json"
    result = uninstall_session_start_hook(p, command=_SS_CMD)
    assert result.removed == 0


def test_uninstall_session_start_does_not_touch_user_prompt_submit(
    tmp_path: Path,
) -> None:
    p = tmp_path / "settings.json"
    install_user_prompt_submit_hook(p, command=_UPS_CMD)
    install_session_start_hook(p, command=_SS_CMD)
    uninstall_session_start_hook(p, command=_SS_CMD)
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "UserPromptSubmit" in hooks
    assert len(hooks["UserPromptSubmit"]) == 1


def test_uninstall_session_start_by_basename(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_session_start_hook(p, command=_SS_CMD)
    install_session_start_hook(
        p, command="/different/path/aelf-session-start-hook"
    )
    data = _read(p)
    assert len(data["hooks"]["SessionStart"]) == 2  # type: ignore[index]
    result = uninstall_session_start_hook(
        p, command_basename="aelf-session-start-hook"
    )
    assert result.removed == 2
    data = _read(p)
    assert data["hooks"]["SessionStart"] == []  # type: ignore[index]
