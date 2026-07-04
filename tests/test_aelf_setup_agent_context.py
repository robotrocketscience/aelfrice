"""Tests for the agent-context hook's setup / unsetup wiring (#1068).

Contract (same BooleanOptionalAction convention as every default-on hook):
- Bare `aelf setup` wires the PreToolUse ^(Agent|Task)$ entry.
- `--no-agent-context` skips install AND persists the opt-out so the
  next auto-install reconcile does not re-add the hook.
- `aelf unsetup` removes the entry by unique basename; other PreToolUse
  entries (Grep|Glob / Bash matchers) are untouched.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from aelfrice.cli import main
from aelfrice.setup import (
    AGENT_CONTEXT_MATCHER,
    AGENT_CONTEXT_SCRIPT_NAME,
    install_agent_context_hook,
    install_search_tool_hook,
    uninstall_agent_context_hook,
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


def _agent_entries(data: dict[str, object]) -> list[dict[str, object]]:
    return [
        e for e in _pre_tool_use_entries(data)
        if e.get("matcher") == AGENT_CONTEXT_MATCHER
    ]


def _run_setup(settings_path: Path, *extra_args: str) -> tuple[int, str]:
    out = io.StringIO()
    rc = main(
        argv=[
            "setup",
            "--settings-path", str(settings_path),
            "--no-pre-issue-guard",  # isolate: these tests cover agent-context only
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
            "--no-pre-issue-guard",  # isolate: these tests cover agent-context only
            *extra_args,
        ],
        out=out,
    )
    return rc, out.getvalue()


# ---------------------------------------------------------------------------
# Low-level: install_agent_context_hook / uninstall_agent_context_hook
# ---------------------------------------------------------------------------


def test_install_writes_anchored_matcher_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = install_agent_context_hook(p, command="/x/aelf-agent-context-hook")
    assert result.installed is True
    entries = _agent_entries(_settings(p))
    assert len(entries) == 1
    assert entries[0]["matcher"] == "^(Agent|Task)$"


def test_install_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_agent_context_hook(p, command="/x/aelf-agent-context-hook")
    result = install_agent_context_hook(p, command="/x/aelf-agent-context-hook")
    assert result.already_present is True
    assert len(_agent_entries(_settings(p))) == 1


def test_install_coexists_with_search_tool_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_hook(p, command="/x/aelf-search-tool-hook")
    install_agent_context_hook(p, command="/x/aelf-agent-context-hook")
    entries = _pre_tool_use_entries(_settings(p))
    assert len(entries) == 2


def test_uninstall_removes_only_agent_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    install_search_tool_hook(p, command="/x/aelf-search-tool-hook")
    install_agent_context_hook(p, command="/x/aelf-agent-context-hook")
    result = uninstall_agent_context_hook(
        p, command_basename=AGENT_CONTEXT_SCRIPT_NAME,
    )
    assert result.removed == 1
    entries = _pre_tool_use_entries(_settings(p))
    assert len(entries) == 1
    assert entries[0]["matcher"] == "Grep|Glob"


def test_uninstall_missing_is_zero(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    result = uninstall_agent_context_hook(
        p, command_basename=AGENT_CONTEXT_SCRIPT_NAME,
    )
    assert result.removed == 0


# ---------------------------------------------------------------------------
# CLI: aelf setup / unsetup flag behavior
# ---------------------------------------------------------------------------


def test_bare_setup_installs_agent_context(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    rc, out = _run_setup(p)
    assert rc == 0
    assert "agent-context" in out
    assert len(_agent_entries(_settings(p))) == 1


def test_no_agent_context_skips_install(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    rc, _ = _run_setup(p, "--no-agent-context")
    assert rc == 0
    assert _agent_entries(_settings(p)) == []


def test_setup_reruns_are_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p)
    rc, out = _run_setup(p)
    assert rc == 0
    assert "already installed" in out
    assert len(_agent_entries(_settings(p))) == 1


def test_unsetup_removes_agent_context(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p)
    assert len(_agent_entries(_settings(p))) == 1
    rc, out = _run_unsetup(p)
    assert rc == 0
    assert "agent-context" in out
    assert _agent_entries(_settings(p)) == []


def test_unsetup_no_agent_context_leaves_entry(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _run_setup(p)
    rc, _ = _run_unsetup(p, "--no-agent-context")
    assert rc == 0
    assert len(_agent_entries(_settings(p))) == 1
