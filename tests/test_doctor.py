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


def test_diagnose_inspects_script_through_silent_failure_wrapper(
    tmp_path: Path,
) -> None:
    """Issue #113: a stale `bash <missing>.sh 2>/dev/null || true`
    hook silently skipped past doctor because the shell-meta check
    short-circuited before we extracted the script path. We now
    extract the script even when the wrapper appears."""
    user_path = tmp_path / "settings.json"
    missing = tmp_path / "hook-aelf-search.sh"
    _write_settings(user_path, {
        "hooks": {
            "PostToolUse": [{
                "matcher": "Grep|Glob",
                "hooks": [{
                    "type": "command",
                    "command": f"bash {missing} 2>/dev/null || true",
                    "timeout": 5,
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert len(report.broken) == 1, report.findings
    assert "hook-aelf-search.sh" in report.broken[0].program
    assert report.broken[0].silent_failure is True


def test_diagnose_flags_silent_failure_pattern_on_resolved_script(
    tmp_path: Path,
) -> None:
    """Even when the script exists, the wrapper itself is the
    anti-feature (issue #114). Surface a soft warning."""
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "ok-script.sh")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PostToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": f"bash {hook} 2>/dev/null || true",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert len(report.silent_failure) == 1
    rendered = format_report(report)
    assert "silent-failure pattern" in rendered


def test_diagnose_surfaces_recent_hook_failures(tmp_path: Path) -> None:
    """When ~/.aelfrice/logs/hook-failures.log exists, doctor tails
    it into the report (issue #114)."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    log = tmp_path / "hook-failures.log"
    log.write_text(
        "\n".join([
            "2026-04-27T22:00:00 hook-aelf-search: file not found",
            "2026-04-27T22:01:00 hook-aelf-search: file not found",
            "",
        ]),
        encoding="utf-8",
    )
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        hook_failures_log=log,
    )
    assert len(report.hook_failures_tail) == 2
    rendered = format_report(report)
    assert "recent hook failures" in rendered
    assert "hook-aelf-search" in rendered


def test_diagnose_no_log_no_section(tmp_path: Path) -> None:
    """Missing log → no log block in the report (no false noise)."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        hook_failures_log=tmp_path / "no-log.log",
    )
    assert report.hook_failures_tail == ()
    rendered = format_report(report)
    assert "recent hook failures" not in rendered


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
