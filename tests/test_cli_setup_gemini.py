"""`aelf setup` / `aelf unsetup` CLI subcommand integration tests for `--host gemini`.

Tests drive `cli.main(argv=...)` end-to-end against a tmp settings.json
chosen via `--scope project --project-root <tmp_path> --host gemini`
so no real ~/.gemini/settings.json is ever touched.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import DEFAULT_HOOK_COMMAND, main
from aelfrice.setup import (
    resolve_hook_command,
    USER_SETTINGS_PATH_GEMINI,
    PROJECT_SETTINGS_RELPATH_GEMINI,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "aelf.db"))


@pytest.fixture(autouse=True)
def isolated_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import cli as cli_mod
    from aelfrice.lifecycle import MigrationResult

    monkeypatch.setattr(
        cli_mod,
        "maybe_migrate_to_uv",
        lambda: MigrationResult(False, False, "already on uv tool"),
    )


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _read_settings(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _project_settings(tmp_path: Path) -> Path:
    return tmp_path / ".gemini" / "settings.json"


_OPT_OUT_AUTO_CAPTURE = (
    "--no-transcript-ingest",
    "--no-commit-ingest",
    "--no-session-start",
    "--no-stop-hook",
    "--no-claude-memory-mirror",
)


def _hook_commands(settings_path: Path, event: str = "BeforeAgent") -> list[str]:
    data = _read_settings(settings_path)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    entries = hooks_typed.get(event)
    if entries is None:
        return []
    assert isinstance(entries, list)
    out: list[str] = []
    for entry in cast(list[dict[str, object]], entries):
        inner = entry["hooks"]
        assert isinstance(inner, list)
        inner_typed = cast(list[dict[str, object]], inner)
        cmd = inner_typed[0]["command"]
        assert isinstance(cmd, str)
        out.append(cmd)
    return out


def test_setup_default_command_writes_project_settings_gemini(tmp_path: Path) -> None:
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    settings = _project_settings(tmp_path)
    assert settings.exists()
    
    # Under gemini host, the command starts with AELFRICE_HOST=gemini
    expected = resolve_hook_command("project")
    expected_gemini = f"AELFRICE_HOST=gemini {expected}"
    
    assert _hook_commands(settings, "BeforeAgent") == [expected_gemini]
    assert "installed" in output
    assert DEFAULT_HOOK_COMMAND in output


def test_setup_idempotent_reports_already_present_gemini(tmp_path: Path) -> None:
    _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
        *_OPT_OUT_AUTO_CAPTURE,
    )
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    assert "already installed" in output
    expected = resolve_hook_command("project")
    expected_gemini = f"AELFRICE_HOST=gemini {expected}"
    assert _hook_commands(_project_settings(tmp_path), "BeforeAgent") == [expected_gemini]


def test_unsetup_removes_default_command_gemini(tmp_path: Path) -> None:
    _run("setup", "--scope", "project", "--project-root", str(tmp_path), "--host", "gemini")
    code, output = _run(
        "unsetup", "--scope", "project", "--project-root", str(tmp_path), "--host", "gemini"
    )
    assert code == 0
    assert "removed 1" in output
    assert _hook_commands(_project_settings(tmp_path), "BeforeAgent") == []


def test_user_scope_writes_into_monkeypatched_user_path_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_user_settings = tmp_path / "user-gemini" / "settings.json"
    import aelfrice.setup as setup_mod

    monkeypatch.setattr(setup_mod, "USER_SETTINGS_PATH_GEMINI", fake_user_settings)
    code, _ = _run("setup", "--scope", "user", "--host", "gemini", *_OPT_OUT_AUTO_CAPTURE)
    assert code == 0
    assert fake_user_settings.exists()
    expected = resolve_hook_command("user")
    expected_gemini = f"AELFRICE_HOST=gemini {expected}"
    assert _hook_commands(fake_user_settings, "BeforeAgent") == [expected_gemini]


def test_setup_default_on_auto_capture_writes_all_hooks_gemini(
    tmp_path: Path,
) -> None:
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path), "--host", "gemini"
    )
    assert code == 0
    data = _read_settings(_project_settings(tmp_path))
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    
    # In Gemini host, UserPromptSubmit maps to BeforeAgent, Stop maps to SessionEnd,
    # and PreCompact maps to PreCompress. No PostCompact equivalent.
    for event in ("BeforeAgent", "SessionEnd", "PreCompress"):
        assert event in hooks_typed, f"missing {event} in default gemini setup"
    
    # SessionStart is mapped to SessionStart.
    assert "SessionStart" in hooks_typed, "SessionStart not default-on"
    
    # PostToolUse:Bash is mapped to AfterTool with run_shell_command.
    assert "AfterTool" in hooks_typed, "commit-ingest AfterTool not default-on"
    after_tool_entries = hooks_typed["AfterTool"]
    assert isinstance(after_tool_entries, list)
    
    # PreToolUse:Grep|Glob is mapped to BeforeTool with grep_search|glob.
    assert "BeforeTool" in hooks_typed, "BeforeTool not default-on"


def test_doctor_cli_exit_0_when_clean_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Setup stubs for hooks on PATH so doctor reports clean
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    for basename in (
        "aelf-hook", "aelf-transcript-logger", "aelf-commit-ingest",
        "aelf-session-start-hook", "aelf-stop-hook", "aelf-search-tool-hook",
        "aelf-pre-issue-hook", "aelf-claude-memory-mirror", "aelf-agent-context-hook"
    ):
        dummy = bin_dir / basename
        dummy.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
        dummy.chmod(0o755)
        
    monkeypatch.setenv("PATH", str(bin_dir))
    
    # Wire the settings file
    _run("setup", "--scope", "project", "--project-root", str(tmp_path), "--host", "gemini")
    
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--host", "gemini",
            "--project-root", str(tmp_path),
        ],
        out=buf,
    )
    assert code == 0
    assert "0 broken" in buf.getvalue()


def test_doctor_cli_exit_1_when_broken_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_dir = tmp_path / ".gemini"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / "settings.json"
    
    # Write settings directly with nonexistent path
    settings_file.write_text(json.dumps({
        "hooks": {
            "BeforeAgent": [{
                "hooks": [{"type": "command", "command": "AELFRICE_HOST=gemini /no/such/aelf-hook"}],
            }],
        },
    }), encoding="utf-8")
    
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--host", "gemini",
            "--project-root", str(tmp_path),
        ],
        out=buf,
    )
    assert code == 1
    assert "broken" in buf.getvalue()


def test_audit_gemini_memory_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_HOST", "gemini")
    
    from aelfrice.claude_memory import derive_memory_dir
    memory_dir = derive_memory_dir(tmp_path)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    memory_md = memory_dir / "MEMORY.md"
    memory_md.write_text("- bullet: subject predicate value", encoding="utf-8")
    
    buf = io.StringIO()
    code = main(
        argv=[
            "audit-gemini-memory",
            "--project", str(tmp_path),
        ],
        out=buf,
    )
    assert code == 0
    output = buf.getvalue()
    assert "audit-gemini-memory" in output


def test_reconcile_gemini_memory_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_HOST", "gemini")
    
    from aelfrice.claude_memory import derive_memory_dir
    memory_dir = derive_memory_dir(tmp_path)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    fact_md = memory_dir / "fact.md"
    fact_md.write_text("- some fact text", encoding="utf-8")
    
    buf = io.StringIO()
    code = main(
        argv=[
            "reconcile-gemini-memory",
            "--project", str(tmp_path),
        ],
        out=buf,
    )
    assert code == 0
    output = buf.getvalue()
    assert "reconcile-gemini-memory" in output


def test_setup_installs_gemini_slash_commands(tmp_path: Path) -> None:
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    
    slash_dir = tmp_path / ".gemini" / "commands" / "aelf"
    assert slash_dir.is_dir()
    
    doctor_toml = slash_dir / "doctor.toml"
    assert doctor_toml.exists()
    
    content = doctor_toml.read_text(encoding="utf-8")
    assert "description =" in content
    assert "prompt =" in content
    assert "AELFRICE_HOST=gemini" in content
    assert "doctor" in content


def test_unsetup_uninstalls_gemini_slash_commands(tmp_path: Path) -> None:
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    
    slash_dir = tmp_path / ".gemini" / "commands" / "aelf"
    assert (slash_dir / "doctor.toml").exists()
    
    code_un, output_un = _run(
        "unsetup", "--scope", "project", "--project-root", str(tmp_path),
        "--host", "gemini",
    )
    assert code_un == 0
    assert not (slash_dir / "doctor.toml").exists()


def test_setup_auto_detects_multiple_hosts(tmp_path: Path) -> None:
    # Create both .claude and .gemini directories under the project root
    (tmp_path / ".claude").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".gemini").mkdir(parents=True, exist_ok=True)
    
    # Run setup with host='auto' (the default)
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    
    # Verify both settings.json files were written
    claude_settings = tmp_path / ".claude" / "settings.json"
    gemini_settings = tmp_path / ".gemini" / "settings.json"
    
    assert claude_settings.exists()
    assert gemini_settings.exists()
    
    # Verify local Gemini slash commands were installed
    assert (tmp_path / ".gemini" / "commands" / "aelf").is_dir()
