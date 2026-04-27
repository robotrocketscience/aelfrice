"""Verify the 10 slash-command markdown files match the CLI surface.

These files ship inside the aelfrice package so `setup.py` (v0.7.0) can
copy them into `~/.claude/commands/aelf/`. The tests check structural
properties — frontmatter, required fields, and that each command's
shell action references the matching CLI subcommand.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# The 10-tool surface — must match cli.py's subparsers exactly.
EXPECTED_COMMANDS = (
    "onboard",
    "search",
    "lock",
    "locked",
    "demote",
    "feedback",
    "stats",
    "health",
    "setup",
    "unsetup",
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


def test_slash_commands_match_cli_subcommands() -> None:
    """The 8 markdown files must correspond exactly to the CLI's
    subparser names. If the CLI grows or shrinks a command, this test
    forces the slash directory to follow."""
    from aelfrice.cli import build_parser
    parser = build_parser()
    # The first positional sub-action holds the subparsers.
    sub_actions = [a for a in parser._subparsers._actions  # type: ignore[union-attr]
                    if a.__class__.__name__ == "_SubParsersAction"]
    assert sub_actions, "cli has no subparsers"
    cli_names = sorted(sub_actions[0].choices.keys())  # type: ignore[attr-defined]
    md_names = sorted(EXPECTED_COMMANDS)
    assert cli_names == md_names, f"cli subcommands {cli_names} != slash {md_names}"
