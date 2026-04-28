"""install_pre_compact_hook / uninstall_pre_compact_hook idempotency."""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.setup import (
    install_pre_compact_hook,
    install_user_prompt_submit_hook,
    uninstall_pre_compact_hook,
    uninstall_user_prompt_submit_hook,
)


_PC_CMD = "/usr/local/bin/aelf-pre-compact-hook"
_UPS_CMD = "/usr/local/bin/aelf-hook"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


# ---- install ------------------------------------------------------------


def test_install_pre_compact_writes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = install_pre_compact_hook(p, command=_PC_CMD)
    assert result.installed is True
    assert result.already_present is False
    data = _read(p)
    assert "PreCompact" in data["hooks"]  # type: ignore[index]
    entry_list = data["hooks"]["PreCompact"]  # type: ignore[index]
    assert isinstance(entry_list, list) and len(entry_list) == 1
    inner = entry_list[0]["hooks"][0]
    assert inner["type"] == "command"
    assert inner["command"] == _PC_CMD


def test_install_pre_compact_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_pre_compact_hook(p, command=_PC_CMD)
    second = install_pre_compact_hook(p, command=_PC_CMD)
    assert second.installed is False
    assert second.already_present is True
    data = _read(p)
    entry_list = data["hooks"]["PreCompact"]  # type: ignore[index]
    assert len(entry_list) == 1


def test_install_pre_compact_does_not_touch_user_prompt_submit(
    tmp_path: Path,
) -> None:
    """Two events live independently in the same settings.json."""
    p = tmp_path / "settings.json"
    install_user_prompt_submit_hook(p, command=_UPS_CMD)
    install_pre_compact_hook(p, command=_PC_CMD)
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "UserPromptSubmit" in hooks
    assert "PreCompact" in hooks
    assert len(hooks["UserPromptSubmit"]) == 1
    assert len(hooks["PreCompact"]) == 1


# ---- uninstall ----------------------------------------------------------


def test_uninstall_pre_compact_removes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_pre_compact_hook(p, command=_PC_CMD)
    result = uninstall_pre_compact_hook(p, command=_PC_CMD)
    assert result.removed == 1
    data = _read(p)
    assert data["hooks"]["PreCompact"] == []  # type: ignore[index]


def test_uninstall_pre_compact_no_match(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_pre_compact_hook(p, command=_PC_CMD)
    other = "/different/binary"
    result = uninstall_pre_compact_hook(p, command=other)
    assert result.removed == 0


def test_uninstall_pre_compact_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "nope.json"
    result = uninstall_pre_compact_hook(p, command=_PC_CMD)
    assert result.removed == 0


def test_uninstall_pre_compact_does_not_touch_user_prompt_submit(
    tmp_path: Path,
) -> None:
    p = tmp_path / "settings.json"
    install_user_prompt_submit_hook(p, command=_UPS_CMD)
    install_pre_compact_hook(p, command=_PC_CMD)
    uninstall_pre_compact_hook(p, command=_PC_CMD)
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "UserPromptSubmit" in hooks
    assert len(hooks["UserPromptSubmit"]) == 1
    # PreCompact entry list is empty but key may persist.


def test_uninstall_pre_compact_by_basename(tmp_path: Path) -> None:
    """Basename match cleans up across abs / bare installs."""
    p = tmp_path / "settings.json"
    install_pre_compact_hook(p, command=_PC_CMD)
    install_pre_compact_hook(
        p, command="/different/path/aelf-pre-compact-hook"
    )
    data = _read(p)
    assert len(data["hooks"]["PreCompact"]) == 2  # type: ignore[index]
    result = uninstall_pre_compact_hook(
        p, command_basename="aelf-pre-compact-hook"
    )
    assert result.removed == 2
    data = _read(p)
    assert data["hooks"]["PreCompact"] == []  # type: ignore[index]
