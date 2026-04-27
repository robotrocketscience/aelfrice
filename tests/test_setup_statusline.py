"""Tests for setup.py's statusline composition logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.setup import (
    STATUSLINE_COMMAND,
    STATUSLINE_SUFFIX,
    install_statusline,
    uninstall_statusline,
)


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def test_install_into_empty_settings(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    r = install_statusline(p)
    assert r.mode == "installed"
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == STATUSLINE_COMMAND


def test_install_idempotent_when_already_ours(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": STATUSLINE_COMMAND}
    })
    r = install_statusline(p)
    assert r.mode == "already"
    # File content is unchanged:
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == STATUSLINE_COMMAND


def test_install_composes_with_simple_existing(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": "echo my-bar"}
    })
    r = install_statusline(p)
    assert r.mode == "composed"
    assert r.existing_command == "echo my-bar"
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == "echo my-bar" + STATUSLINE_SUFFIX


def test_install_skips_complex_command(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": "echo a | tr a A"}
    })
    r = install_statusline(p)
    assert r.mode == "skipped"
    data = json.loads(p.read_text())
    # Untouched:
    assert data["statusLine"]["command"] == "echo a | tr a A"


@pytest.mark.parametrize(
    "cmd",
    [
        "true && echo bar",
        "echo a || echo b",
        "echo `date`",
        "echo \\$HOME",
        "cat <<EOF\nfoo\nEOF",
    ],
)
def test_install_skips_various_complex_commands(
    tmp_path: Path, cmd: str
) -> None:
    p = tmp_path / "settings.json"
    _write(p, {"statusLine": {"type": "command", "command": cmd}})
    r = install_statusline(p)
    assert r.mode == "skipped"


def test_uninstall_drops_standalone_field(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": STATUSLINE_COMMAND},
        "other": 42,
    })
    r = uninstall_statusline(p)
    assert r.mode == "removed"
    data = json.loads(p.read_text())
    assert "statusLine" not in data
    assert data["other"] == 42


def test_uninstall_unwraps_composed(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {
            "type": "command",
            "command": "echo my-bar" + STATUSLINE_SUFFIX,
        }
    })
    r = uninstall_statusline(p)
    assert r.mode == "unwrapped"
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == "echo my-bar"


def test_uninstall_absent_when_unrelated_command(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": "echo other"}
    })
    r = uninstall_statusline(p)
    assert r.mode == "absent"
    # Untouched:
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == "echo other"


def test_uninstall_absent_when_file_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    r = uninstall_statusline(p)
    assert r.mode == "absent"


def test_install_then_uninstall_roundtrip_simple(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {})
    install_statusline(p)
    r = uninstall_statusline(p)
    assert r.mode == "removed"
    data = json.loads(p.read_text())
    assert "statusLine" not in data


def test_install_then_uninstall_roundtrip_compose(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    _write(p, {
        "statusLine": {"type": "command", "command": "echo my-bar"}
    })
    install_statusline(p)
    r = uninstall_statusline(p)
    assert r.mode == "unwrapped"
    data = json.loads(p.read_text())
    assert data["statusLine"]["command"] == "echo my-bar"
