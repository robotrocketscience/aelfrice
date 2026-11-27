"""#1956: `aelf doctor` reports a logger entry in both settings files.

The host runs a handler defined in two settings files once only when the
two definitions match. Setup writes an absolute logger path per scope, so
a user-scope and a project-scope install carry two different commands,
and both fire on every event. The per-file duplicate check (#1161) sees
one entry in each file and reports nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.doctor import diagnose, format_report
from aelfrice.setup import (
    PROJECT_SETTINGS_RELPATH,
    TRANSCRIPT_INGEST_EVENTS,
    TRANSCRIPT_LOGGER_SCRIPT_NAME,
)


def _exec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _logger_settings(command: str, events: tuple[str, ...]) -> dict[str, object]:
    return {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": command}]}]
            for event in events
        },
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_logger_in_both_settings_files_is_reported(tmp_path: Path) -> None:
    user_logger = _exec(tmp_path / "user-bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    project_logger = _exec(
        tmp_path / "proj-venv" / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME
    )
    user_path = tmp_path / "home" / ".claude" / "settings.json"
    project_root = tmp_path / "proj"
    _write(user_path, _logger_settings(str(user_logger), TRANSCRIPT_INGEST_EVENTS))
    _write(
        project_root / PROJECT_SETTINGS_RELPATH,
        _logger_settings(str(project_logger), TRANSCRIPT_INGEST_EVENTS),
    )

    report = diagnose(user_settings=user_path, project_root=project_root)

    # The per-file check finds nothing: each file holds one entry per event.
    assert report.duplicate_hook_entries == []
    assert sorted(e.event for e in report.cross_scope_logger_entries) == sorted(
        TRANSCRIPT_INGEST_EVENTS
    )
    for entry in report.cross_scope_logger_entries:
        assert entry.user_path == user_path
        assert entry.project_path == project_root / PROJECT_SETTINGS_RELPATH
    text = format_report(report)
    assert "both the user and the project settings file" in text
    assert "UserPromptSubmit" in text
    # Informational: a second registration is not a broken hook.
    assert report.broken == []


def test_identical_logger_command_in_both_files_is_not_reported(
    tmp_path: Path,
) -> None:
    """The host runs one handler for a command both files define the same way."""
    logger = _exec(tmp_path / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    user_path = tmp_path / "home" / ".claude" / "settings.json"
    project_root = tmp_path / "proj"
    _write(user_path, _logger_settings(str(logger), TRANSCRIPT_INGEST_EVENTS))
    _write(
        project_root / PROJECT_SETTINGS_RELPATH,
        _logger_settings(str(logger), TRANSCRIPT_INGEST_EVENTS),
    )

    report = diagnose(user_settings=user_path, project_root=project_root)

    assert report.cross_scope_logger_entries == []


def test_only_events_with_differing_commands_are_reported(tmp_path: Path) -> None:
    shared = _exec(tmp_path / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    other = _exec(tmp_path / "venv" / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    user_path = tmp_path / "home" / ".claude" / "settings.json"
    project_root = tmp_path / "proj"
    _write(user_path, _logger_settings(str(shared), TRANSCRIPT_INGEST_EVENTS))
    project = _logger_settings(str(shared), TRANSCRIPT_INGEST_EVENTS)
    project["hooks"]["Stop"] = [  # type: ignore[index]
        {"hooks": [{"type": "command", "command": str(other)}]},
    ]
    _write(project_root / PROJECT_SETTINGS_RELPATH, project)

    report = diagnose(user_settings=user_path, project_root=project_root)

    assert [e.event for e in report.cross_scope_logger_entries] == ["Stop"]


def test_logger_in_one_settings_file_is_not_reported(tmp_path: Path) -> None:
    logger = _exec(tmp_path / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    other = _exec(tmp_path / "bin" / "aelf-hook")
    user_path = tmp_path / "home" / ".claude" / "settings.json"
    project_root = tmp_path / "proj"
    _write(user_path, _logger_settings(str(logger), TRANSCRIPT_INGEST_EVENTS))
    _write(
        project_root / PROJECT_SETTINGS_RELPATH,
        _logger_settings(str(other), ("UserPromptSubmit",)),
    )

    report = diagnose(user_settings=user_path, project_root=project_root)

    assert report.cross_scope_logger_entries == []
    assert "both the user and the project settings file" not in format_report(
        report
    )


def test_project_settings_that_are_the_user_settings_are_not_reported(
    tmp_path: Path,
) -> None:
    """Doctor run from the home folder scans one file under both names.

    The file holds two different logger commands on Stop, which the
    per-file duplicate check reports. Read as two files, it would also
    look like a logger wired in both scopes.
    """
    logger = _exec(tmp_path / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    other = _exec(tmp_path / "venv" / "bin" / TRANSCRIPT_LOGGER_SCRIPT_NAME)
    home = tmp_path / "home"
    user_path = home / PROJECT_SETTINGS_RELPATH
    settings = _logger_settings(str(logger), TRANSCRIPT_INGEST_EVENTS)
    settings["hooks"]["Stop"].append(  # type: ignore[index]
        {"hooks": [{"type": "command", "command": str(other)}]},
    )
    _write(user_path, settings)

    report = diagnose(user_settings=user_path, project_root=home)

    assert report.cross_scope_logger_entries == []
