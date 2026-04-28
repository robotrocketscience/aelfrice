"""Verify the slash-command markdown files match the CLI surface.

These files ship inside the aelfrice package so `setup.py` (v0.7.0) can
copy them into `~/.claude/commands/aelf/`. The tests check structural
properties — frontmatter, required fields, and that each command's
shell action references the matching CLI subcommand.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# The CLI's user-facing surface -- must match cli.py's subparsers exactly.
# Lifecycle commands (upgrade/uninstall/statusline) added in v1.0.1.
# `resolve` joins at v1.0.1 alongside the contradiction tie-breaker.
# `status` (alias for health), `regime` (preserved v1.0 classifier),
# and `migrate` (legacy-DB import) join at v1.1.0.
# `rebuild` (context rebuilder MVP) joins at v1.1.0 alpha.
# v1.3 user-facing surface — must match cli.py's non-SUPPRESS subparsers.
# Hidden subparsers (rebuild, statusline, bench, regime, migrate, unsetup,
# health, stats) keep working but ship no slash file.
EXPECTED_COMMANDS = (
    "onboard",
    "search",
    "lock",
    "locked",
    "demote",
    "validate",
    "resolve",
    "feedback",
    "status",
    "doctor",
    "setup",
    "upgrade",
    "uninstall",
    "ingest-transcript",
)


def _slash_dir() -> Path:
    """Locate src/aelfrice/slash_commands without importing aelfrice."""
    import aelfrice
    return Path(aelfrice.__file__).parent / "slash_commands"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_file_exists(cmd: str) -> None:
    path = _slash_dir() / f"{cmd}.md"
    assert path.exists(), f"missing slash command: {path}"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_file_nonempty(cmd: str) -> None:
    path = _slash_dir() / f"{cmd}.md"
    assert path.stat().st_size > 0


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_has_frontmatter(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    assert text.startswith("---\n"), f"{cmd}.md missing opening frontmatter delimiter"
    assert "\n---\n" in text[4:], f"{cmd}.md missing closing frontmatter delimiter"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_name_matches_filename(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    m = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
    assert m is not None, f"{cmd}.md missing name: line"
    assert m.group(1) == f"aelf:{cmd}", f"{cmd}.md name mismatch: {m.group(1)}"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_has_description(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m is not None, f"{cmd}.md missing description: line"
    assert len(m.group(1).strip()) >= 20, f"{cmd}.md description too short"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_invokes_matching_cli(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    assert f"uv run aelf {cmd}" in text, (
        f"{cmd}.md does not invoke `uv run aelf {cmd}`"
    )


def test_no_extra_files_in_slash_commands_dir() -> None:
    """Catches typos like `loocked.md` and accidental .md.bak files."""
    files = sorted(p.name for p in _slash_dir().iterdir() if p.is_file())
    expected = sorted(f"{c}.md" for c in EXPECTED_COMMANDS)
    assert files == expected


# Subparsers intentionally hidden from --help at v1.3. Aliases live one
# minor and are deleted at v1.4 (`health`, `stats`); the rest stay
# callable indefinitely as scripting / hook entry points.
HIDDEN_SUBCOMMANDS = frozenset({
    "rebuild", "statusline", "bench", "regime", "migrate", "unsetup",
    "health", "stats", "project-warm", "session-delta",
})


def test_slash_commands_match_visible_cli_subcommands() -> None:
    """The CLI's full subparser set must equal EXPECTED ∪ HIDDEN, and the
    slash directory must equal EXPECTED. Catches drift in either
    direction."""
    from aelfrice.cli import build_parser
    parser = build_parser()
    sub_actions = [a for a in parser._subparsers._actions  # type: ignore[union-attr]
                    if a.__class__.__name__ == "_SubParsersAction"]
    assert sub_actions, "cli has no subparsers"
    cli_names = set(sub_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected_total = set(EXPECTED_COMMANDS) | HIDDEN_SUBCOMMANDS
    assert cli_names == expected_total, (
        f"cli subcommands {sorted(cli_names)} != "
        f"EXPECTED ∪ HIDDEN {sorted(expected_total)}"
    )
