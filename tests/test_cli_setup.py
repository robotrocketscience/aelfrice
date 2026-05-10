"""`aelf setup` / `aelf unsetup` CLI subcommand integration tests.

Tests drive `cli.main(argv=...)` end-to-end against a tmp settings.json
chosen via `--scope project --project-root <tmp_path>` so no real
~/.claude/settings.json is ever touched.

Also covers install_slash_commands / uninstall_slash_commands (unit) and
the end-to-end path where `aelf setup` populates a tmp slash dir and
`aelf doctor` reports no orphan slash commands.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import DEFAULT_HOOK_COMMAND, main
from aelfrice.doctor import SLASH_COMMANDS_DIR_DEFAULT as DOCTOR_SLASH_DIR
from aelfrice.setup import (
    install_slash_commands,
    uninstall_slash_commands,
    resolve_hook_command,
    SLASH_COMMANDS_DIR_DEFAULT,
)
from test_slash_commands import EXPECTED_COMMANDS


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


_OPT_OUT_AUTO_CAPTURE = (
    "--no-transcript-ingest",
    "--no-commit-ingest",
    "--no-session-start",
    "--no-stop-hook",
)


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
    # Auto-capture hooks default-on as of #529; this test scopes itself to
    # the UserPromptSubmit (read-side) wiring path by opting them out.
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        *_OPT_OUT_AUTO_CAPTURE,
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
    _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        *_OPT_OUT_AUTO_CAPTURE,
    )
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        *_OPT_OUT_AUTO_CAPTURE,
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
    code, _ = _run(
        "setup", "--settings-path", str(explicit), *_OPT_OUT_AUTO_CAPTURE,
    )
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
    code, _ = _run("setup", "--scope", "user", *_OPT_OUT_AUTO_CAPTURE)
    assert code == 0
    assert fake_user_settings.exists()
    expected = resolve_hook_command("user")
    assert _hook_commands(fake_user_settings) == [expected]


def test_setup_default_on_auto_capture_writes_all_hooks(
    tmp_path: Path,
) -> None:
    """Bare `aelf setup` wires the v1.2.0 auto-capture pipeline (#529).

    Asserts UserPromptSubmit + Stop + PreCompact + PostCompact (transcript-
    ingest), PostToolUse:Bash (commit-ingest), and SessionStart land
    without any opt-in flag.
    """
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    data = _read_settings(_project_settings(tmp_path))
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    # All four transcript-ingest events present.
    for event in ("UserPromptSubmit", "Stop", "PreCompact", "PostCompact"):
        assert event in hooks_typed, f"missing {event} in default setup"
    # SessionStart and PostToolUse (commit-ingest) present.
    assert "SessionStart" in hooks_typed, "SessionStart not default-on"
    assert "PostToolUse" in hooks_typed, "commit-ingest PostToolUse not default-on"


def test_setup_no_flags_skip_auto_capture(tmp_path: Path) -> None:
    """Opt-out flags suppress each auto-capture hook independently (#529)."""
    code, _ = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path),
        *_OPT_OUT_AUTO_CAPTURE,
    )
    assert code == 0
    data = _read_settings(_project_settings(tmp_path))
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    # UserPromptSubmit (read-side) is unaffected by these flags.
    assert "UserPromptSubmit" in hooks_typed
    # Auto-capture event types should be absent.
    for event in ("Stop", "PreCompact", "PostCompact", "SessionStart", "PostToolUse"):
        assert event not in hooks_typed, (
            f"{event} present despite --no-* opt-out"
        )


# ---------------------------------------------------------------------------
# install_slash_commands / uninstall_slash_commands unit tests
# ---------------------------------------------------------------------------


def test_install_slash_commands_writes_all_bundled_files(
    tmp_path: Path,
) -> None:
    """All bundled slash files end up in dest_dir."""
    dest = tmp_path / "slash"
    result = install_slash_commands(dest)
    assert result.dest_dir == dest
    assert dest.is_dir()
    written_names = set(result.written)
    expected_names = {f"{cmd}.md" for cmd in EXPECTED_COMMANDS}
    assert expected_names == written_names, (
        f"written={sorted(written_names)} expected={sorted(expected_names)}"
    )
    assert result.already == ()
    assert result.pruned == ()
    # Files are on disk.
    for name in expected_names:
        assert (dest / name).exists()


def test_install_slash_commands_idempotent(tmp_path: Path) -> None:
    """Second install with identical content produces no writes."""
    dest = tmp_path / "slash"
    install_slash_commands(dest)
    result2 = install_slash_commands(dest)
    assert result2.written == ()
    assert result2.pruned == ()
    assert set(result2.already) == {f"{cmd}.md" for cmd in EXPECTED_COMMANDS}


def test_install_slash_commands_prunes_orphans(tmp_path: Path) -> None:
    """Stale .md files not in the bundle are removed."""
    dest = tmp_path / "slash"
    dest.mkdir(parents=True)
    stale = dest / "stats.md"
    stale.write_text("old content", encoding="utf-8")
    result = install_slash_commands(dest)
    assert "stats.md" in result.pruned
    assert not stale.exists()


def test_install_slash_commands_updates_changed_file(tmp_path: Path) -> None:
    """A file whose content differs from the bundle is overwritten."""
    dest = tmp_path / "slash"
    dest.mkdir(parents=True)
    target = dest / "setup.md"
    target.write_text("stale content", encoding="utf-8")
    result = install_slash_commands(dest)
    assert "setup.md" in result.written
    # Content now matches bundle.
    from aelfrice.setup import _bundled_slash_files
    bundle = _bundled_slash_files()
    assert target.read_text(encoding="utf-8") == bundle["setup.md"]


def test_uninstall_slash_commands_removes_bundle(tmp_path: Path) -> None:
    """uninstall_slash_commands removes all installed files."""
    dest = tmp_path / "slash"
    install_slash_commands(dest)
    result = uninstall_slash_commands(dest)
    expected_names = {f"{cmd}.md" for cmd in EXPECTED_COMMANDS}
    assert set(result.pruned) == expected_names
    for name in expected_names:
        assert not (dest / name).exists()
    assert result.written == ()
    assert result.already == ()


def test_uninstall_slash_commands_noop_when_missing(tmp_path: Path) -> None:
    """uninstall_slash_commands is a no-op when the dir does not exist."""
    dest = tmp_path / "nonexistent"
    result = uninstall_slash_commands(dest)
    assert result.pruned == ()


def test_uninstall_slash_commands_only_removes_bundle_files(
    tmp_path: Path,
) -> None:
    """uninstall_slash_commands removes only bundle files, not arbitrary .md files."""
    dest = tmp_path / "slash"
    dest.mkdir(parents=True)
    # Plant a non-bundle file directly (bypass install so it isn't pruned).
    user_file = dest / "my-custom.md"
    user_file.write_text("user content", encoding="utf-8")
    # Manually write one bundle file to confirm only that gets removed.
    bundle_file = dest / "setup.md"
    bundle_file.write_text("some content", encoding="utf-8")
    result = uninstall_slash_commands(dest)
    assert "setup.md" in result.pruned
    assert not bundle_file.exists()
    # The non-bundle file is untouched.
    assert user_file.exists()


# ---------------------------------------------------------------------------
# End-to-end: aelf setup populates slash dir; doctor orphan check is green
# ---------------------------------------------------------------------------


def test_setup_cli_installs_slash_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aelf setup writes slash files and outputs a confirmation line."""
    slash_dir = tmp_path / "slash"
    import aelfrice.setup as setup_mod
    import aelfrice.cli as cli_mod

    monkeypatch.setattr(setup_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    monkeypatch.setattr(cli_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert slash_dir.is_dir()
    expected_names = {f"{cmd}.md" for cmd in EXPECTED_COMMANDS}
    on_disk = {p.name for p in slash_dir.glob("*.md")}
    assert expected_names == on_disk
    assert "slash command" in output


def test_setup_cli_idempotent_slash_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running aelf setup twice reports commands as already up to date."""
    slash_dir = tmp_path / "slash"
    import aelfrice.setup as setup_mod
    import aelfrice.cli as cli_mod

    monkeypatch.setattr(setup_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    monkeypatch.setattr(cli_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    _run("setup", "--scope", "project", "--project-root", str(tmp_path))
    code, output = _run(
        "setup", "--scope", "project", "--project-root", str(tmp_path)
    )
    assert code == 0
    assert "already up to date" in output


def test_doctor_orphan_check_green_after_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After aelf setup the doctor orphan_slash_commands check is empty."""
    slash_dir = tmp_path / "slash"
    import aelfrice.setup as setup_mod
    import aelfrice.cli as cli_mod
    import aelfrice.doctor as doctor_mod

    monkeypatch.setattr(setup_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    monkeypatch.setattr(cli_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)
    monkeypatch.setattr(doctor_mod, "SLASH_COMMANDS_DIR_DEFAULT", slash_dir)

    _run("setup", "--scope", "project", "--project-root", str(tmp_path))

    from aelfrice.cli import _known_cli_subcommands
    from aelfrice.doctor import diagnose

    report = diagnose(
        slash_commands_dir=slash_dir,
        known_cli_subcommands=_known_cli_subcommands(),
    )
    assert report.orphan_slash_commands == [], (
        f"unexpected orphans: {report.orphan_slash_commands}"
    )
