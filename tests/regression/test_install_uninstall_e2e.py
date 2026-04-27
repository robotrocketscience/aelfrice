"""End-to-end regression: full lifecycle in a hermetic tmp HOME.

Pretends a fresh user runs:
  aelf setup --settings-path <tmp>
  aelf onboard <fixture>
  aelf search <query>          (sanity that the store works)
  aelf uninstall --keep-db --keep-hook    (preserves DB)
  aelf uninstall --purge --yes --keep-hook (deletes DB)

Goal: catch regressions where the lifecycle commands break the store
or the settings file. Uses the same in-process main(argv=...) entry
point as the unit tests so it runs fast and on the same DB shim.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=argv, out=buf)
    return code, buf.getvalue()


def _make_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# Project\nThis service uses Postgres for state.\n", encoding="utf-8"
    )
    (root / "main.py").write_text(
        "DATABASE_URL = 'postgres://...'\n", encoding="utf-8"
    )


def test_full_lifecycle_setup_onboard_search_uninstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    settings = tmp_path / "settings.json"
    fixture = tmp_path / "proj"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")  # silence notifier
    _make_fixture(fixture)

    # Setup wires hook + statusline:
    code, _out = _run(["setup", "--settings-path", str(settings)])
    assert code == 0
    data = json.loads(settings.read_text())
    assert data["hooks"]["UserPromptSubmit"]
    assert data["statusLine"]["command"] == "aelf statusline"

    # Onboard creates the DB:
    code, _ = _run(["onboard", str(fixture)])
    assert code == 0
    assert db.exists()

    # Search hits the onboarded content:
    code, out = _run(["search", "postgres"])
    assert code == 0
    assert "postgres" in out.lower() or "Postgres" in out

    # Uninstall --keep-db preserves the DB:
    code, _ = _run([
        "uninstall", "--keep-db", "--keep-hook",
        "--settings-path", str(settings),
    ])
    assert code == 0
    assert db.exists()

    # Uninstall --purge --yes deletes the DB:
    code, _ = _run([
        "uninstall", "--purge", "--yes", "--keep-hook",
        "--settings-path", str(settings),
    ])
    assert code == 0
    assert not db.exists()


def test_uninstall_default_also_removes_hook_and_statusline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")

    code, _ = _run(["setup", "--settings-path", str(settings)])
    assert code == 0
    db.touch()  # synthetic

    code, _ = _run([
        "uninstall", "--keep-db",
        "--settings-path", str(settings),
    ])
    assert code == 0
    data = json.loads(settings.read_text())
    # Hook removed:
    assert data.get("hooks", {}).get("UserPromptSubmit", []) == []
    # Statusline removed:
    assert "statusLine" not in data


def test_uninstall_purge_aborts_on_bad_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    db.write_bytes(b"sqlite-bytes")

    # Feed a wrong ack via stdin.
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "wrong")
    code, _ = _run(["uninstall", "--purge", "--keep-hook"])
    assert code == 1
    assert db.exists()
