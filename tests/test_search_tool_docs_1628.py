"""Every live description of the search-tool matcher names the shipped one (#1628).

#1626 widened the matcher from `Grep|Glob` to cover the web tools, and the
pages that state the matcher kept the old value: a docs-only sweep fixes them
once and rots again on the next widening. These tests read each site that
states the matcher to a reader -- the architecture hook table, the install
guide, the `/aelf:setup` command page, the README command table, the CLI help,
and the hook manifest -- and hold it to `SEARCH_TOOL_MATCHER`, so a change to
the tool set fails here until the prose follows.

Each site is located by a stable anchor (a hook name, an option, a manifest
row), not a line number, and a missing anchor fails rather than skips: a
locator that silently finds nothing would pass while the prose drifts.

Historical records are out of scope on purpose. The changelog, the roadmap,
and the v1.2.x design text in `docs/design/search_tool_hook.md` describe the
matcher as it shipped then.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aelfrice.cli import build_parser
from aelfrice.search_tool_names import SEARCH_TOOL_MATCHER, SEARCH_TOOL_NAMES

REPO = Path(__file__).resolve().parent.parent
_EVENT_MATCHER = f"PreToolUse:{SEARCH_TOOL_MATCHER}"


def _only_line(path: Path, *needles: str, exclude: str | None = None) -> str:
    """The single line of `path` containing every needle.

    Fails, rather than returning nothing, when the anchor matches zero lines
    or several: either means the locator no longer identifies the site.
    """
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if all(n in line for n in needles) and (exclude is None or exclude not in line)
    ]
    assert len(lines) == 1, (
        f"{path.relative_to(REPO)}: expected one line containing {needles!r}"
        f"{'' if exclude is None else f' and not {exclude!r}'}, found "
        f"{len(lines)}. If the page was reworded, update the anchor here."
    )
    return lines[0]


def _missing_names(text: str) -> list[str]:
    return [name for name in SEARCH_TOOL_NAMES if name not in text]


@pytest.mark.timeout(30)
def test_architecture_hook_table_states_the_shipped_matcher() -> None:
    row = _only_line(
        REPO / "docs" / "concepts" / "ARCHITECTURE.md",
        "| `aelf-search-tool-hook` |",
        exclude="PreToolUse:Bash",
    )
    # The table escapes `|` inside a cell; unescape before comparing.
    event_cell = row.split(" | ")[1].strip().strip("`").replace("\\|", "|")
    assert event_cell == _EVENT_MATCHER, (
        f"ARCHITECTURE.md gives the search-tool event as {event_cell!r}; "
        f"the shipped matcher is {_EVENT_MATCHER!r}"
    )


@pytest.mark.timeout(30)
def test_install_guide_opt_out_line_states_the_shipped_matcher() -> None:
    line = _only_line(
        REPO / "docs" / "user" / "INSTALL.md",
        "aelf setup --no-search-tool ",
    )
    assert _EVENT_MATCHER in line, (
        f"INSTALL.md's --no-search-tool line does not name {_EVENT_MATCHER!r}: "
        f"{line!r}"
    )


@pytest.mark.timeout(30)
def test_install_guide_hook_table_names_every_search_tool() -> None:
    row = _only_line(REPO / "docs" / "user" / "INSTALL.md", "| search-tool |")
    assert not _missing_names(row), (
        f"INSTALL.md's search-tool row omits {_missing_names(row)}: {row!r}"
    )


@pytest.mark.timeout(30)
def test_setup_command_page_states_the_shipped_matcher() -> None:
    line = _only_line(
        REPO / "src" / "aelfrice" / "slash_commands" / "setup.md",
        "`aelf-search-tool-hook`",
        exclude="PreToolUse:Bash",
    )
    assert f"`{_EVENT_MATCHER}`" in line, (
        f"/aelf:setup page does not name {_EVENT_MATCHER!r}: {line!r}"
    )


@pytest.mark.timeout(30)
def test_readme_search_command_row_names_every_search_tool() -> None:
    row = _only_line(REPO / "README.md", "| `/aelf:search <query>` |")
    assert not _missing_names(row), (
        f"README.md's /aelf:search row omits {_missing_names(row)}: {row!r}"
    )


def _search_tool_help(command: str) -> str:
    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    sub = subparsers.choices[command]
    for action in sub._actions:
        if "--search-tool" in action.option_strings:
            assert isinstance(action.help, str)
            return action.help
    raise AssertionError(f"`aelf {command}` has no --search-tool option")


@pytest.mark.timeout(30)
@pytest.mark.parametrize("command", ["setup", "unsetup"])
def test_cli_search_tool_help_states_the_shipped_matcher(command: str) -> None:
    text = _search_tool_help(command)
    assert _EVENT_MATCHER in text, (
        f"`aelf {command} --search-tool` help does not name "
        f"{_EVENT_MATCHER!r}: {text!r}"
    )


@pytest.mark.timeout(30)
def test_hook_manifest_description_states_the_shipped_matcher() -> None:
    manifest = json.loads(
        (REPO / "src" / "aelfrice" / "data" / "hook_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [r for r in manifest["hooks"] if r.get("name") == "search_tool"]
    assert len(rows) == 1, "hook_manifest.json has no single search_tool row"
    description = rows[0]["description"]
    assert _EVENT_MATCHER in description, (
        f"hook_manifest.json search_tool description does not name "
        f"{_EVENT_MATCHER!r}: {description!r}"
    )
