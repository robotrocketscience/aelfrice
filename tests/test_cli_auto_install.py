"""Tests for the cli.main() <-> auto_install wiring (#623)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aelfrice import auto_install, cli


def _capture_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace auto_install_at_cli_entry with a recording stub."""
    calls: list[dict[str, Any]] = []

    def fake(installed_version: str) -> None:
        calls.append({"installed_version": installed_version})

    monkeypatch.setattr(cli, "auto_install_at_cli_entry", fake)
    return calls


def test_main_invokes_auto_install_for_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A run-of-the-mill command (status) triggers the auto-install hook."""
    calls = _capture_calls(monkeypatch)
    # Disable the update-check side effect so the test does not race on the
    # background fetcher.
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    # Run `aelf status` with a tmp HOME so no real store is touched.
    monkeypatch.setenv("HOME", str(tmp_path))
    cli.main(["status"])
    assert len(calls) == 1
    assert calls[0]["installed_version"] == cli._AELFRICE_VERSION


def test_main_skips_auto_install_for_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`aelf setup` mutates settings.json itself — auto-install must skip."""
    calls = _capture_calls(monkeypatch)
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    # Use an explicit project-scope settings path under tmp_path so the
    # real aelf setup does not touch the user's actual settings.json.
    settings = tmp_path / "settings.json"
    cli.main(["setup", "--settings", str(settings), "--scope", "project",
              "--no-statusline"])
    assert calls == []  # auto-install was NOT called


def test_skip_set_covers_destructive_commands() -> None:
    """Setup / unsetup / uninstall mutate settings.json themselves; doctor
    inspects on-disk state. None of them should trigger auto-install."""
    assert "setup" in cli._AUTO_INSTALL_SKIP_CMDS
    assert "unsetup" in cli._AUTO_INSTALL_SKIP_CMDS
    assert "uninstall" in cli._AUTO_INSTALL_SKIP_CMDS
    assert "doctor" in cli._AUTO_INSTALL_SKIP_CMDS


def test_auto_install_at_cli_entry_does_not_propagate_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A blown-up merge must NOT block the user's actual command."""
    monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)

    def boom(**_kw: object) -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(auto_install, "maybe_install_manifest", boom)
    # Must return normally (no exception).
    auto_install.auto_install_at_cli_entry(installed_version="2.2.0")
    captured = capsys.readouterr()
    # The defensive branch is allowed to log to stderr.
    assert "auto-install skipped" in captured.err
