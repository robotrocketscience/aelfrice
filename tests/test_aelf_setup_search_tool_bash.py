"""Tests for `aelf setup --search-tool-bash` and
`aelf setup --no-search-tool-bash` (#155 AC7).

Coverage:
- Install writes a PreToolUse:Bash hook entry. Idempotent (second call
  is a no-op that prints "already installed").
- Uninstall (--no-search-tool-bash) removes it. Idempotent (no-op when
  absent).
- install + uninstall + install round-trip.
- Independent of --search-tool: can install both, either, or neither.
- aelf unsetup --search-tool-bash also removes the entry.
- install_search_tool_bash_hook / uninstall_search_tool_bash_hook
  low-level unit tests matching the commit-ingest + search-tool pattern.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.setup import (
    SEARCH_TOOL_BASH_MATCHER,
    SEARCH_TOOL_BASH_SCRIPT_NAME,
    SEARCH_TOOL_MATCHER,
    install_search_tool_bash_hook,
    install_search_tool_hook,
    uninstall_search_tool_bash_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pre_tool_use_entries(data: dict[str, object]) -> list[dict[str, object]]:
    hooks = data.get("hooks", {})
    assert isinstance(hooks, dict)
    entries = hooks.get("PreToolUse", [])
    assert isinstance(entries, list)
    return entries  # type: ignore[return-value]


def _bash_entries(data: dict[str, object]) -> list[dict[str, object]]:
    return [e for e in _pre_tool_use_entries(data) if e.get("matcher") == "Bash"]


def _grep_glob_entries(data: dict[str, object]) -> list[dict[str, object]]:
    return [e for e in _pre_tool_use_entries(data) if e.get("matcher") == "Grep|Glob"]


def _run_setup(settings_path: Path, *extra_args: str) -> tuple[int, str]:
    out = io.StringIO()
    rc = main(
        argv=[
            "setup",
            "--settings-path", str(settings_path),
            *extra_args,
        ],
        out=out,
    )
    return rc, out.getvalue()


def _run_unsetup(settings_path: Path, *extra_args: str) -> tuple[int, str]:
    out = io.StringIO()
    rc = main(
        argv=[
            "unsetup",
            "--settings-path", str(settings_path),
            "--command", "aelf-hook-stub",  # avoid touching the real hook
            *extra_args,
        ],
        out=out,
    )
    return rc, out.getvalue()


# ---------------------------------------------------------------------------
# Low-level: install_search_tool_bash_hook
# ---------------------------------------------------------------------------


def test_install_bash_hook_writes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    assert result.installed is True
    assert result.already_present is False
    data = _settings(p)
    bash = _bash_entries(data)
    assert len(bash) == 1
    assert bash[0]["matcher"] == SEARCH_TOOL_BASH_MATCHER
    inner = bash[0]["hooks"]
    assert isinstance(inner, list) and len(inner) == 1
    assert inner[0]["command"] == "aelf-search-tool-hook"


def test_install_bash_hook_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    result2 = install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    assert result2.already_present is True
    assert result2.installed is False
    # Still only one entry.
    assert len(_bash_entries(_settings(p))) == 1


def test_install_bash_hook_coexists_with_grep_glob(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_hook(p, command="aelf-search-tool-hook")
    install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    data = _settings(p)
    assert len(_grep_glob_entries(data)) == 1
    assert len(_bash_entries(data)) == 1


def test_uninstall_bash_hook_removes_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_bash_hook(p, command="/venv/bin/aelf-search-tool-hook")
    result = uninstall_search_tool_bash_hook(
        p, command_basename=SEARCH_TOOL_BASH_SCRIPT_NAME
    )
    assert result.removed == 1
    assert _bash_entries(_settings(p)) == []


def test_uninstall_bash_hook_idempotent_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = uninstall_search_tool_bash_hook(
        p, command_basename=SEARCH_TOOL_BASH_SCRIPT_NAME
    )
    assert result.removed == 0


def test_uninstall_bash_hook_leaves_grep_glob_intact(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_hook(p, command="aelf-search-tool-hook")
    install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    uninstall_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    data = _settings(p)
    assert len(_grep_glob_entries(data)) == 1
    assert _bash_entries(data) == []


def test_install_uninstall_install_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    uninstall_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    result3 = install_search_tool_bash_hook(p, command="aelf-search-tool-hook")
    assert result3.installed is True
    assert len(_bash_entries(_settings(p))) == 1


# ---------------------------------------------------------------------------
# CLI: aelf setup --search-tool-bash
# ---------------------------------------------------------------------------


def test_cli_setup_search_tool_bash_installs(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    rc, out = _run_setup(p, "--search-tool-bash")
    assert rc == 0
    assert "installed search-tool-bash" in out or "already installed" in out
    assert len(_bash_entries(_settings(p))) == 1


def test_cli_setup_search_tool_bash_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p, "--search-tool-bash")
    rc, out = _run_setup(p, "--search-tool-bash")
    assert rc == 0
    assert "already installed" in out
    assert len(_bash_entries(_settings(p))) == 1


def test_cli_setup_no_search_tool_bash_removes(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p, "--search-tool-bash")
    assert len(_bash_entries(_settings(p))) == 1
    rc, out = _run_setup(p, "--no-search-tool-bash")
    assert rc == 0
    assert "removed" in out
    assert _bash_entries(_settings(p)) == []


def test_cli_setup_no_search_tool_bash_idempotent_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    rc, out = _run_setup(p, "--no-search-tool-bash")
    assert rc == 0
    assert "no search-tool-bash" in out


def test_cli_setup_search_tool_bash_independent_of_search_tool(
    tmp_path: Path,
) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p, "--search-tool")
    _run_setup(p, "--search-tool-bash")
    data = _settings(p)
    assert len(_grep_glob_entries(data)) == 1
    assert len(_bash_entries(data)) == 1


def test_cli_setup_no_search_tool_bash_leaves_grep_glob_intact(
    tmp_path: Path,
) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p, "--search-tool")
    _run_setup(p, "--search-tool-bash")
    _run_setup(p, "--no-search-tool-bash")
    data = _settings(p)
    assert len(_grep_glob_entries(data)) == 1
    assert _bash_entries(data) == []


# ---------------------------------------------------------------------------
# CLI: aelf unsetup --search-tool-bash
# ---------------------------------------------------------------------------


def test_cli_unsetup_search_tool_bash_removes(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p, "--search-tool-bash")
    rc, out = _run_unsetup(p, "--search-tool-bash")
    assert rc == 0
    assert "removed" in out or "no search-tool-bash" in out
    assert _bash_entries(_settings(p)) == []


def test_cli_unsetup_search_tool_bash_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    rc, out = _run_unsetup(p, "--search-tool-bash")
    assert rc == 0
    assert "no search-tool-bash" in out
