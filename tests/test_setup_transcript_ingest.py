"""install_transcript_ingest_hooks: idempotency + four-event coverage."""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.setup import (
    TRANSCRIPT_INGEST_EVENTS,
    install_transcript_ingest_hooks,
    install_user_prompt_submit_hook,
    uninstall_transcript_ingest_hooks,
)


def _read_settings(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_install_writes_all_four_events(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    r = install_transcript_ingest_hooks(settings, command="aelf-transcript-logger")
    assert set(r.installed) == set(TRANSCRIPT_INGEST_EVENTS)
    assert not r.already
    data = _read_settings(settings)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    for event in TRANSCRIPT_INGEST_EVENTS:
        assert event in hooks
        entries = hooks[event]
        assert isinstance(entries, list)
        assert len(entries) == 1


def test_install_idempotent(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_transcript_ingest_hooks(settings, command="aelf-transcript-logger")
    r2 = install_transcript_ingest_hooks(settings, command="aelf-transcript-logger")
    assert not r2.installed
    assert set(r2.already) == set(TRANSCRIPT_INGEST_EVENTS)


def test_install_does_not_disturb_existing_user_prompt_submit_hook(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(settings, command="aelf-hook")
    install_transcript_ingest_hooks(settings, command="aelf-transcript-logger")
    data = _read_settings(settings)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    # UserPromptSubmit now has two entries: aelf-hook and aelf-transcript-logger.
    ups = hooks["UserPromptSubmit"]
    assert isinstance(ups, list)
    commands = {
        h["command"]
        for entry in ups
        for h in entry["hooks"]
    }
    assert commands == {"aelf-hook", "aelf-transcript-logger"}


def test_uninstall_strips_all_events_by_basename(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_transcript_ingest_hooks(
        settings, command="/abs/path/to/aelf-transcript-logger",
    )
    r = uninstall_transcript_ingest_hooks(
        settings, command_basename="aelf-transcript-logger",
    )
    assert sum(r.removed.values()) == len(TRANSCRIPT_INGEST_EVENTS)
    data = _read_settings(settings)
    # Each event list should be empty after uninstall.
    hooks = data.get("hooks", {})
    if isinstance(hooks, dict):
        for event in TRANSCRIPT_INGEST_EVENTS:
            entries = hooks.get(event, [])
            assert entries == []


def test_uninstall_idempotent_on_missing_file(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"  # never created
    r = uninstall_transcript_ingest_hooks(
        settings, command_basename="aelf-transcript-logger",
    )
    assert r.removed == {}
    assert not settings.exists()


def test_uninstall_leaves_aelf_hook_alone(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(settings, command="aelf-hook")
    install_transcript_ingest_hooks(settings, command="aelf-transcript-logger")
    uninstall_transcript_ingest_hooks(
        settings, command_basename="aelf-transcript-logger",
    )
    data = _read_settings(settings)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    ups = hooks["UserPromptSubmit"]
    assert isinstance(ups, list)
    commands = {
        h["command"] for entry in ups for h in entry["hooks"]
    }
    assert commands == {"aelf-hook"}


def test_install_with_explicit_command_string_match(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_transcript_ingest_hooks(settings, command="custom-cmd-1")
    install_transcript_ingest_hooks(settings, command="custom-cmd-2")
    # Now uninstall by exact command should only strip the matching one.
    r = uninstall_transcript_ingest_hooks(settings, command="custom-cmd-1")
    assert sum(r.removed.values()) == len(TRANSCRIPT_INGEST_EVENTS)
    data = _read_settings(settings)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    for event in TRANSCRIPT_INGEST_EVENTS:
        entries = hooks[event]
        assert isinstance(entries, list)
        commands = {
            h["command"] for e in entries for h in e["hooks"]
        }
        assert commands == {"custom-cmd-2"}
