"""install_stop_hook / uninstall_stop_hook idempotency (#582).

Mirrors test_setup_session_start.py one-for-one against the Stop event.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.setup import (
    install_stop_hook,
    install_transcript_ingest_hooks,
    install_user_prompt_submit_hook,
    uninstall_stop_hook,
)


_STOP_CMD = "/usr/local/bin/aelf-stop-hook"
_TI_CMD = "/usr/local/bin/aelf-transcript-logger"
_UPS_CMD = "/usr/local/bin/aelf-hook"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


# ---- install -----------------------------------------------------------


def test_install_stop_hook_writes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = install_stop_hook(p, command=_STOP_CMD)
    assert result.installed is True
    assert result.already_present is False
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "Stop" in hooks
    entry_list = hooks["Stop"]  # type: ignore[index]
    assert isinstance(entry_list, list) and len(entry_list) == 1
    inner = entry_list[0]["hooks"][0]
    assert inner["type"] == "command"
    assert inner["command"] == _STOP_CMD


def test_install_stop_hook_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_stop_hook(p, command=_STOP_CMD)
    second = install_stop_hook(p, command=_STOP_CMD)
    assert second.installed is False
    assert second.already_present is True
    data = _read(p)
    entry_list = data["hooks"]["Stop"]  # type: ignore[index]
    assert len(entry_list) == 1


def test_install_stop_hook_coexists_with_transcript_ingest_stop(
    tmp_path: Path,
) -> None:
    """Both Stop entries (transcript-logger + stop-hook) must live
    independently under the same Stop event key. Critical: the
    transcript-ingest install must not be displaced when the stop-hook
    install runs (and vice versa)."""
    p = tmp_path / "settings.json"
    install_transcript_ingest_hooks(p, command=_TI_CMD)
    install_stop_hook(p, command=_STOP_CMD)
    data = _read(p)
    entries = data["hooks"]["Stop"]  # type: ignore[index]
    commands = [e["hooks"][0]["command"] for e in entries]
    assert _TI_CMD in commands
    assert _STOP_CMD in commands
    assert len(entries) == 2


def test_install_stop_hook_does_not_touch_other_events(
    tmp_path: Path,
) -> None:
    p = tmp_path / "settings.json"
    install_user_prompt_submit_hook(p, command=_UPS_CMD)
    install_stop_hook(p, command=_STOP_CMD)
    data = _read(p)
    hooks = data["hooks"]  # type: ignore[index]
    assert "UserPromptSubmit" in hooks
    assert "Stop" in hooks
    assert len(hooks["UserPromptSubmit"]) == 1
    assert len(hooks["Stop"]) == 1


# ---- uninstall ---------------------------------------------------------


def test_uninstall_stop_hook_removes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_stop_hook(p, command=_STOP_CMD)
    result = uninstall_stop_hook(p, command=_STOP_CMD)
    assert result.removed == 1
    data = _read(p)
    assert data["hooks"]["Stop"] == []  # type: ignore[index]


def test_uninstall_stop_hook_no_match(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_stop_hook(p, command=_STOP_CMD)
    result = uninstall_stop_hook(p, command="/different/binary")
    assert result.removed == 0


def test_uninstall_stop_hook_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "nope.json"
    result = uninstall_stop_hook(p, command=_STOP_CMD)
    assert result.removed == 0


def test_uninstall_stop_hook_basename_does_not_remove_transcript_ingest(
    tmp_path: Path,
) -> None:
    """Basename uninstall of stop-hook must NOT match the transcript-logger
    entry under the same Stop key — the two share the event but have
    distinct basenames."""
    p = tmp_path / "settings.json"
    install_transcript_ingest_hooks(p, command=_TI_CMD)
    install_stop_hook(p, command=_STOP_CMD)
    result = uninstall_stop_hook(p, command_basename="aelf-stop-hook")
    assert result.removed == 1
    data = _read(p)
    remaining = data["hooks"]["Stop"]  # type: ignore[index]
    assert len(remaining) == 1
    assert remaining[0]["hooks"][0]["command"] == _TI_CMD


def test_uninstall_stop_hook_by_basename(tmp_path: Path) -> None:
    """Uninstall-by-basename clears multiple stale-path entries.

    Pre-existing settings.json files may carry stacked stale-path
    entries from before the #781 install-dedup fix landed. Write
    that shape directly here — install_stop_hook now deduplicates by
    basename and would collapse the two stale entries on append.
    """
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({
        "hooks": {
            "Stop": [
                {"hooks": [{"type": "command", "command": _STOP_CMD}]},
                {"hooks": [{
                    "type": "command",
                    "command": "/different/path/aelf-stop-hook",
                }]},
            ],
        },
    }))
    data = _read(p)
    assert len(data["hooks"]["Stop"]) == 2  # type: ignore[index]
    result = uninstall_stop_hook(
        p, command_basename="aelf-stop-hook"
    )
    assert result.removed == 2
    data = _read(p)
    assert data["hooks"]["Stop"] == []  # type: ignore[index]
