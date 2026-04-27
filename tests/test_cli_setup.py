"""`aelf setup` / `aelf unsetup` CLI subcommand integration tests.

Tests drive `cli.main(argv=...)` end-to-end against a tmp settings.json
chosen via `--scope project --project-root <tmp_path>` so no real
~/.claude/settings.json is ever touched.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import DEFAULT_HOOK_COMMAND, main
from aelfrice.setup import resolve_hook_command


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "aelf.db"))


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _read_settings(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)  # pyright: ignore[reportAny]
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _project_settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def _hook_commands(settings_path: Path) -> list[str]:
    data = _read_settings(settings_path)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    entries = hooks_typed["UserPromptSubmit"]
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


def test_setup_default_command_writes_project_settings(tmp_path: Path) -> None:
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    settings = _project_settings(tmp_path)
    assert settings.exists()
    expected = resolve_hook_command("project")
    assert _hook_commands(settings) == [expected]
    assert "installed" in output
    # The basename is always present, whether we wrote bare or absolute.
    assert DEFAULT_HOOK_COMMAND in output


def test_setup_idempotent_reports_already_present(tmp_path: Path) -> None:
    _run("setup", "--scope", "project", "--project-root", str(tmp_path))
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert "already installed" in output
    expected = resolve_hook_command("project")
    assert _hook_commands(_project_settings(tmp_path)) == [expected]


def test_setup_custom_command_timeout_and_status_message(
    tmp_path: Path,
) -> None:
    code, _ = _run(
        "setup",
        "--scope", "project",
        "--project-root", str(tmp_path),
        "--command", "/usr/local/bin/my-hook.sh",
        "--timeout", "7",
        "--status-message", "thinking...",
    )
    assert code == 0
    data = _read_settings(_project_settings(tmp_path))
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    entries = hooks_typed["UserPromptSubmit"]
    assert isinstance(entries, list)
    inner = cast(list[dict[str, object]], entries)[0]["hooks"]
    assert isinstance(inner, list)
    inner_typed = cast(list[dict[str, object]], inner)
    assert inner_typed[0] == {
        "type": "command",
        "command": "/usr/local/bin/my-hook.sh",
        "timeout": 7,
        "statusMessage": "thinking...",
    }


def test_setup_explicit_settings_path_overrides_scope(tmp_path: Path) -> None:
    explicit = tmp_path / "weird-place" / "claude.json"
    code, _ = _run("setup", "--settings-path", str(explicit))
    assert code == 0
    assert explicit.exists()
    # Without an explicit --scope, auto-detect from cwd. Whichever scope
    # auto-detect picks, the resolver returns the same command for that
    # scope, so we just check basename ends up correct.
    cmds = _hook_commands(explicit)
    assert len(cmds) == 1
    assert Path(cmds[0]).name == DEFAULT_HOOK_COMMAND


def test_unsetup_removes_default_command(tmp_path: Path) -> None:
    _run("setup", "--scope", "project", "--project-root", str(tmp_path))
    code, output = _run(
        "unsetup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert "removed 1" in output
    assert _hook_commands(_project_settings(tmp_path)) == []


def test_unsetup_no_op_when_missing_reports_zero(tmp_path: Path) -> None:
    code, output = _run(
        "unsetup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert "no matching hook" in output


def test_unsetup_keeps_other_entries(tmp_path: Path) -> None:
    _run(
        "setup",
        "--scope", "project",
        "--project-root", str(tmp_path),
        "--command", "/keep/me",
    )
    _run("setup", "--scope", "project", "--project-root", str(tmp_path))
    code, output = _run(
        "unsetup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert "removed 1" in output
    assert _hook_commands(_project_settings(tmp_path)) == ["/keep/me"]


def test_user_scope_writes_into_monkeypatched_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_user_settings = tmp_path / "user-claude" / "settings.json"
    import aelfrice.setup as setup_mod

    monkeypatch.setattr(setup_mod, "USER_SETTINGS_PATH", fake_user_settings)
    code, _ = _run("setup", "--scope", "user")
    assert code == 0
    assert fake_user_settings.exists()
    expected = resolve_hook_command("user")
    assert _hook_commands(fake_user_settings) == [expected]
