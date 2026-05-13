"""Unit tests for lifecycle.py: update check, upgrade advice, uninstall.

Each test is hermetic: monkeypatched home dir, monkeypatched fetch
function, monkeypatched env. No real network calls.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from aelfrice import lifecycle


# --- is_newer ----------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("1.0.1", "1.0.0", True),
        ("1.1.0", "1.0.99", True),
        ("2.0.0", "1.99.99", True),
        ("1.0.0", "1.0.0", False),
        ("1.0.0", "1.0.1", False),
        ("0.0.0", "0.0.1", False),
        ("1.0.0-beta.1", "1.0.0", False),  # pre-release suffix stripped
        ("1.0.0", "1.0.0-beta.1", False),
        ("", "1.0.0", False),
        ("1.0.0", "", True),
        ("garbage", "1.0.0", False),
    ],
)
def test_is_newer(a: str, b: str, expected: bool) -> None:
    assert lifecycle.is_newer(a, b) is expected


# --- read_cache / write_cache -----------------------------------------


def test_read_cache_returns_empty_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    s = lifecycle.read_cache(p)
    assert s == lifecycle.UpdateStatus.empty()


def test_read_cache_handles_garbage_silently(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = lifecycle.read_cache(p)
    assert s == lifecycle.UpdateStatus.empty()


def test_write_then_read_cache_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    src = lifecycle.UpdateStatus(
        update_available=True,
        installed="1.0.0",
        latest="1.2.3",
        checked=12345.0,
        sha256="deadbeef",
    )
    lifecycle._write_cache(src, p)
    got = lifecycle.read_cache(p)
    assert got == src


# --- cache_is_fresh ----------------------------------------------------


def test_cache_is_fresh_within_ttl() -> None:
    s = lifecycle.UpdateStatus(False, "1", "1", time.time(), None)
    assert lifecycle.cache_is_fresh(s)


def test_cache_is_stale_after_ttl() -> None:
    s = lifecycle.UpdateStatus(
        False, "1", "1", time.time() - lifecycle.CACHE_TTL_SECONDS - 1, None
    )
    assert not lifecycle.cache_is_fresh(s)


def test_cache_is_stale_when_checked_zero() -> None:
    assert not lifecycle.cache_is_fresh(lifecycle.UpdateStatus.empty())


# --- is_disabled -------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "yes", "ON", "True"])
def test_is_disabled_truthy(val: str) -> None:
    assert lifecycle.is_disabled({"AELF_NO_UPDATE_CHECK": val})


@pytest.mark.parametrize("val", ["0", "false", "", "no"])
def test_is_disabled_falsy(val: str) -> None:
    assert not lifecycle.is_disabled({"AELF_NO_UPDATE_CHECK": val})


def test_is_disabled_unset() -> None:
    assert not lifecycle.is_disabled({})


# --- check_for_update --------------------------------------------------


def test_check_for_update_handles_network_failure(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    s = lifecycle.check_for_update(
        cache_path=p,
        fetch=lambda url: None,  # network failed
    )
    # Network failure: no cache to read, returns empty.
    assert s == lifecycle.UpdateStatus.empty()
    assert not p.exists()  # don't write garbage


def test_check_for_update_writes_cache_on_success(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    payload = {
        "info": {"version": "99.0.0"},
        "releases": {
            "99.0.0": [
                {
                    "packagetype": "bdist_wheel",
                    "filename": "aelfrice-99.0.0-py3-none-any.whl",
                    "digests": {"sha256": "abc123"},
                }
            ]
        },
    }
    s = lifecycle.check_for_update(
        cache_path=p, fetch=lambda url: payload, now=42.0,
    )
    assert s.update_available is True
    assert s.latest == "99.0.0"
    assert s.sha256 == "abc123"
    assert s.checked == 42.0
    # Cache written:
    assert json.loads(p.read_text())["latest"] == "99.0.0"


def test_check_for_update_no_update_when_at_latest(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    installed = lifecycle.installed_version()
    payload = {
        "info": {"version": installed},
        "releases": {
            installed: [{"digests": {"sha256": "x"}, "filename": "x.whl"}]
        },
    }
    s = lifecycle.check_for_update(cache_path=p, fetch=lambda url: payload)
    assert s.update_available is False


def test_check_for_update_disabled_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    called: list[str] = []

    def fetch_should_not_run(url: str) -> dict | None:
        called.append(url)
        return None

    s = lifecycle.check_for_update(
        cache_path=tmp_path / "c.json", fetch=fetch_should_not_run
    )
    assert s == lifecycle.UpdateStatus.empty()
    assert called == []


# --- _wheel_sha256 -----------------------------------------------------


def test_wheel_sha256_picks_wheel_over_sdist() -> None:
    files = [
        {"packagetype": "sdist", "filename": "x.tar.gz",
         "digests": {"sha256": "sdist-sha"}},
        {"packagetype": "bdist_wheel", "filename": "x.whl",
         "digests": {"sha256": "wheel-sha"}},
    ]
    assert lifecycle._wheel_sha256(files) == "wheel-sha"


def test_wheel_sha256_falls_back_to_sdist() -> None:
    files = [
        {"packagetype": "sdist", "filename": "x.tar.gz",
         "digests": {"sha256": "sdist-sha"}},
    ]
    assert lifecycle._wheel_sha256(files) == "sdist-sha"


def test_wheel_sha256_returns_none_when_empty() -> None:
    assert lifecycle._wheel_sha256([]) is None


# --- upgrade_advice ----------------------------------------------------


def test_upgrade_advice_in_venv() -> None:
    # The test runner is in some kind of managed environment.
    advice = lifecycle.upgrade_advice()
    assert advice.context in {"uv_tool", "non_uv"}
    assert "aelfrice" in advice.command


# --- clear_cache -------------------------------------------------------


def test_clear_cache_removes_existing(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("{}", encoding="utf-8")
    lifecycle.clear_cache(p)
    assert not p.exists()


def test_clear_cache_silent_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "missing.json"
    lifecycle.clear_cache(p)  # must not raise


# --- uninstall ---------------------------------------------------------


def test_uninstall_keep_db(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    sqlite3.connect(str(db)).executescript("CREATE TABLE x(y INT)").close()
    r = lifecycle.uninstall(db, keep_db=True)
    assert r.mode == "kept"
    assert db.exists()


def test_uninstall_purge(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    sqlite3.connect(str(db)).executescript("CREATE TABLE x(y INT)").close()
    r = lifecycle.uninstall(db, purge=True)
    assert r.mode == "purged"
    assert not db.exists()


def test_uninstall_purge_silent_when_db_absent(tmp_path: Path) -> None:
    db = tmp_path / "missing.db"
    r = lifecycle.uninstall(db, purge=True)
    assert r.mode == "purged"


def test_uninstall_archive_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    db = tmp_path / "memory.db"
    sqlite3.connect(str(db)).executescript(
        "CREATE TABLE x(y INT); INSERT INTO x VALUES(42);"
    ).close()
    archive = tmp_path / "out.aenc"
    r = lifecycle.uninstall(
        db, archive_path=archive, archive_password="hunter2"
    )
    assert r.mode == "archived"
    assert not db.exists()
    assert archive.exists()
    # Decrypt + verify:
    raw = lifecycle.decrypt_archive(archive, "hunter2")
    restored = tmp_path / "restored.db"
    restored.write_bytes(raw)
    rows = list(sqlite3.connect(str(restored)).execute("SELECT y FROM x"))
    assert rows == [(42,)]


def test_uninstall_archive_wrong_password_rejected(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from cryptography.fernet import InvalidToken

    db = tmp_path / "memory.db"
    sqlite3.connect(str(db)).executescript("CREATE TABLE x(y INT)").close()
    archive = tmp_path / "out.aenc"
    lifecycle.uninstall(db, archive_path=archive, archive_password="right")
    with pytest.raises(InvalidToken):
        lifecycle.decrypt_archive(archive, "wrong")


def test_uninstall_rejects_zero_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lifecycle.uninstall(tmp_path / "x.db")


def test_uninstall_rejects_multiple_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lifecycle.uninstall(tmp_path / "x.db", keep_db=True, purge=True)


def test_uninstall_archive_requires_password(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    db.write_bytes(b"data")
    with pytest.raises(ValueError):
        lifecycle.uninstall(
            db, archive_path=tmp_path / "out.aenc", archive_password=None,
        )
