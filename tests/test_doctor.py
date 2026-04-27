"""Tests for `aelf doctor` and the underlying `aelfrice.doctor` module."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.doctor import diagnose, format_report


def _write_settings(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _exec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_diagnose_returns_empty_report_when_no_settings(
    tmp_path: Path,
) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
    )
    assert report.scopes_scanned == []
    assert report.findings == []
    assert "nothing to check" in format_report(report)


def test_diagnose_flags_missing_absolute_path(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{
                    "type": "command",
                    "command": "/no/such/aelf-hook",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    broken = report.broken
    assert len(broken) == 1
    assert broken[0].program == "/no/such/aelf-hook"
    assert "does not exist" in broken[0].detail


def test_diagnose_flags_bare_name_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "definitely-not-installed"}],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    broken = report.broken
    assert len(broken) == 1
    assert "not on $PATH" in broken[0].detail


def test_diagnose_passes_when_program_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "aelf-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": str(hook)}],
            }],
        },
        "statusLine": {"type": "command", "command": str(hook)},
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert report.ok_count == 2


def test_diagnose_skips_shell_pipe_commands(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": "true | echo skipped && exit 0",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert report.skipped_count == 1


def test_diagnose_inspects_script_under_interpreter(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "settings.json"
    # `bash <missing-script>` should report the SCRIPT, not bash.
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": f"bash {tmp_path / 'gone.sh'}",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert len(report.broken) == 1
    assert "gone.sh" in report.broken[0].program


def test_doctor_cli_exit_1_when_broken(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": "/no/such"}],
            }],
        },
    })
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--user-settings", str(user_path),
            "--project-root", str(tmp_path / "noproj"),
        ],
        out=buf,
    )
    assert code == 1
    assert "broken" in buf.getvalue()


def test_doctor_cli_exit_0_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "aelf-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": str(hook)}],
            }],
        },
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--user-settings", str(user_path),
            "--project-root", str(tmp_path / "noproj"),
        ],
        out=buf,
    )
    assert code == 0
    assert "1 ok" in buf.getvalue()
