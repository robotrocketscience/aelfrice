"""Tests for `aelf setup` pre-issue-guard flag behavior (#941).

Contract:
- Bare `aelf setup` wires the PreToolUse:Bash pre-issue-guard hook.
- `--no-pre-issue-guard` skips install AND persists the opt-out so the
  next auto-install reconcile does not re-add the hook.
- Low-level install_pre_issue_guard_hook / uninstall_pre_issue_guard_hook
  unit tests mirror the search-tool-bash pattern.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.setup import (
    PRE_ISSUE_GUARD_MATCHER,
    PRE_ISSUE_GUARD_SCRIPT_NAME,
    install_pre_issue_guard_hook,
    uninstall_pre_issue_guard_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CMD = f"/usr/bin/{PRE_ISSUE_GUARD_SCRIPT_NAME}"
_OPT_OUT_SETUP = (
    "--no-transcript-ingest",
    "--no-commit-ingest",
    "--no-session-start",
    "--no-stop-hook",
    "--no-search-tool",
    "--no-search-tool-bash",
)


def _read_settings(p: Path) -> dict[str, object]:
    import json as _json
    from typing import cast
    parsed = _json.loads(p.read_text())
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _pre_issue_entries(settings: dict[str, object]) -> list[dict[str, object]]:
    from typing import cast
    hooks = settings.get("hooks") or {}
    assert isinstance(hooks, dict)
    pre_tool_entries = cast(dict[str, object], hooks).get("PreToolUse") or []
    assert isinstance(pre_tool_entries, list)
    out = []
    for entry in cast(list[dict[str, object]], pre_tool_entries):
        # matcher is on the outer entry, not the inner hook
        if entry.get("matcher") != PRE_ISSUE_GUARD_MATCHER:
            continue
        inner = entry.get("hooks") or []
        assert isinstance(inner, list)
        for h in cast(list[dict[str, object]], inner):
            if PRE_ISSUE_GUARD_SCRIPT_NAME in str(h.get("command", "")):
                out.append(h)
    return out


# ---------------------------------------------------------------------------
# Unit: install_pre_issue_guard_hook
# ---------------------------------------------------------------------------


def test_install_writes_pre_tool_use_entry(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    result = install_pre_issue_guard_hook(p, command=_CMD)
    assert result.installed
    data = _read_settings(p)
    pre_tool = data["hooks"]
    assert isinstance(pre_tool, dict)
    entries = pre_tool["PreToolUse"]
    assert isinstance(entries, list)
    outer = entries[0]
    assert outer.get("matcher") == PRE_ISSUE_GUARD_MATCHER
    inner = outer["hooks"]
    assert isinstance(inner, list)
    assert inner[0]["command"] == _CMD


def test_install_idempotent(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    install_pre_issue_guard_hook(p, command=_CMD)
    result2 = install_pre_issue_guard_hook(p, command=_CMD)
    assert not result2.installed
    assert result2.already_present


def test_install_coexists_with_other_pre_tool_use(tmp_path: Path) -> None:
    """Installing the guard does not disturb an existing PreToolUse entry."""
    from aelfrice.setup import install_search_tool_hook
    p = tmp_path / ".claude" / "settings.json"
    install_search_tool_hook(p, command="aelf-search-tool-hook")
    install_pre_issue_guard_hook(p, command=_CMD)
    data = _read_settings(p)
    entries = data["hooks"]["PreToolUse"]
    assert isinstance(entries, list)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# Unit: uninstall_pre_issue_guard_hook
# ---------------------------------------------------------------------------


def test_uninstall_removes_entry(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    install_pre_issue_guard_hook(p, command=_CMD)
    result = uninstall_pre_issue_guard_hook(p, command=_CMD)
    assert result.removed == 1
    data = _read_settings(p)
    assert _pre_issue_entries(data) == []


def test_uninstall_by_basename(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    install_pre_issue_guard_hook(p, command=_CMD)
    result = uninstall_pre_issue_guard_hook(
        p, command_basename=PRE_ISSUE_GUARD_SCRIPT_NAME
    )
    assert result.removed == 1


def test_uninstall_noop_when_missing(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    result = uninstall_pre_issue_guard_hook(
        p, command_basename=PRE_ISSUE_GUARD_SCRIPT_NAME
    )
    assert result.removed == 0


def test_uninstall_leaves_other_pre_tool_use(tmp_path: Path) -> None:
    """Uninstalling the guard must not touch other PreToolUse entries."""
    from aelfrice.setup import install_search_tool_hook
    p = tmp_path / ".claude" / "settings.json"
    install_search_tool_hook(p, command="aelf-search-tool-hook")
    install_pre_issue_guard_hook(p, command=_CMD)
    uninstall_pre_issue_guard_hook(p, command=_CMD)
    data = _read_settings(p)
    entries = data["hooks"]["PreToolUse"]
    assert isinstance(entries, list)
    assert len(entries) == 1


def test_reinstall_after_uninstall(tmp_path: Path) -> None:
    p = tmp_path / ".claude" / "settings.json"
    install_pre_issue_guard_hook(p, command=_CMD)
    uninstall_pre_issue_guard_hook(p, command=_CMD)
    result3 = install_pre_issue_guard_hook(p, command=_CMD)
    assert result3.installed


# ---------------------------------------------------------------------------
# CLI: aelf setup wires the guard by default
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "aelf.db"))


@pytest.fixture(autouse=True)
def _no_uv_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import cli as cli_mod
    from aelfrice.lifecycle import MigrationResult
    monkeypatch.setattr(
        cli_mod,
        "maybe_migrate_to_uv",
        lambda: MigrationResult(False, False, "already on uv tool"),
    )


def test_setup_default_installs_pre_issue_guard(tmp_path: Path) -> None:
    """Bare aelf setup (without --no-pre-issue-guard) wires the guard hook."""
    buf = io.StringIO()
    code = main(
        argv=[
            "setup", "--scope", "project", "--project-root", str(tmp_path),
            *_OPT_OUT_SETUP,
        ],
        out=buf,
    )
    assert code == 0
    settings = tmp_path / ".claude" / "settings.json"
    data = _read_settings(settings)
    entries = _pre_issue_entries(data)
    assert len(entries) == 1
    assert PRE_ISSUE_GUARD_SCRIPT_NAME in entries[0]["command"]
    assert "pre-issue-guard" in buf.getvalue() or "pre_issue_guard" in buf.getvalue()


def test_setup_no_pre_issue_guard_skips_install(tmp_path: Path) -> None:
    """`aelf setup --no-pre-issue-guard` does not wire the hook."""
    buf = io.StringIO()
    code = main(
        argv=[
            "setup", "--scope", "project", "--project-root", str(tmp_path),
            "--no-pre-issue-guard",
            *_OPT_OUT_SETUP,
        ],
        out=buf,
    )
    assert code == 0
    settings = tmp_path / ".claude" / "settings.json"
    data = _read_settings(settings)
    entries = _pre_issue_entries(data)
    assert entries == []
