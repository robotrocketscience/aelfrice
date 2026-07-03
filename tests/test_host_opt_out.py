"""Host-level auto-install opt-out (#1053).

A Codex-primary machine persists `{"opt_out_hosts": ["claude"]}` in the
existing opt-out ledger so `aelf` invocations stop rewriting the Claude
settings.json at CLI entry — no per-command env prefixes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.auto_install import (
    add_host_opt_out,
    read_host_opt_outs,
    read_opt_outs,
    remove_host_opt_out,
)
from aelfrice.host_codex import claude_host_has_aelfrice_hooks


# --- ledger round-trip -----------------------------------------------------


def test_missing_file_means_no_host_opt_outs(tmp_path: Path) -> None:
    assert read_host_opt_outs(tmp_path / "ledger.json") == frozenset()


def test_add_and_read_host_opt_out(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    add_host_opt_out("claude", ledger)
    assert read_host_opt_outs(ledger) == frozenset({"claude"})


def test_add_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    add_host_opt_out("claude", ledger)
    add_host_opt_out("claude", ledger)
    doc = json.loads(ledger.read_text(encoding="utf-8"))
    assert doc["opt_out_hosts"] == ["claude"]


def test_remove_host_opt_out_round_trip(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    add_host_opt_out("claude", ledger)
    assert remove_host_opt_out("claude", ledger) is True
    assert read_host_opt_outs(ledger) == frozenset()
    assert remove_host_opt_out("claude", ledger) is False


def test_host_key_coexists_with_per_hook_opt_outs(tmp_path: Path) -> None:
    """The new key must not clobber the existing per-hook ledger."""
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"opt_out": ["transcript_ingest"]}), encoding="utf-8",
    )
    add_host_opt_out("claude", ledger)
    assert read_opt_outs(ledger) == frozenset({"transcript_ingest"})
    assert read_host_opt_outs(ledger) == frozenset({"claude"})
    # And removal keeps the per-hook entries too.
    remove_host_opt_out("claude", ledger)
    assert read_opt_outs(ledger) == frozenset({"transcript_ingest"})


def test_malformed_ledger_fails_open(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"opt_out_hosts": "claude"}', encoding="utf-8")
    assert read_host_opt_outs(ledger) == frozenset()
    ledger.write_text("{not json", encoding="utf-8")
    assert read_host_opt_outs(ledger) == frozenset()


# --- CLI-entry gate --------------------------------------------------------


def test_cli_entry_skips_when_claude_opted_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_install_at_cli_entry returns before any settings merge."""
    import aelfrice.auto_install as ai

    ledger = tmp_path / "ledger.json"
    add_host_opt_out("claude", ledger)
    monkeypatch.setattr(ai, "OPT_OUT_PATH", ledger)
    called: list[str] = []
    monkeypatch.setattr(
        ai, "maybe_install_manifest",
        lambda installed_version: called.append(installed_version),
    )
    # Force the uv-tool check True so ONLY the opt-out can stop the merge.
    monkeypatch.setattr(ai, "is_running_from_uv_tool_install", lambda: True)
    monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)
    ai.auto_install_at_cli_entry("9.9.9")
    assert called == []


def test_cli_entry_proceeds_without_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aelfrice.auto_install as ai

    monkeypatch.setattr(ai, "OPT_OUT_PATH", tmp_path / "absent.json")
    called: list[str] = []

    class _R:
        message = ""

    monkeypatch.setattr(
        ai, "maybe_install_manifest",
        lambda installed_version: (called.append(installed_version), _R())[1],
    )
    monkeypatch.setattr(ai, "is_running_from_uv_tool_install", lambda: True)
    monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)
    ai.auto_install_at_cli_entry("9.9.9")
    assert called == ["9.9.9"]


# --- Codex-only detection --------------------------------------------------


def test_claude_host_detection_missing_file(tmp_path: Path) -> None:
    assert claude_host_has_aelfrice_hooks(tmp_path / "settings.json") is False


def test_claude_host_detection_with_aelf_hook(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command",
                            "command": "/Users/x/.local/bin/aelf-hook"}]},
            ],
        },
    }), encoding="utf-8")
    assert claude_host_has_aelfrice_hooks(settings) is True


def test_claude_host_detection_foreign_hooks_only(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "/opt/other"}]},
            ],
        },
    }), encoding="utf-8")
    assert claude_host_has_aelfrice_hooks(settings) is False


def test_claude_host_detection_broken_json_fails_closed(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{\n  "broken":', encoding="utf-8")
    assert claude_host_has_aelfrice_hooks(settings) is False
