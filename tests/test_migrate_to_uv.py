"""Tests for `lifecycle.maybe_migrate_to_uv` (#733).

The migration short-circuits on (1) sentinel existence, (2) uv_tool
context, (3) missing uv binary on PATH. The hot path runs
`uv tool install --force aelfrice` in a subprocess and writes a
sentinel on success only.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from aelfrice import lifecycle


def _make_advice(context: str) -> lifecycle.UpgradeAdvice:
    """Tiny helper to build an UpgradeAdvice for the test under test.

    The `command` field is irrelevant to the migration decision; only
    `context` is read.
    """
    return lifecycle.UpgradeAdvice(command="(test)", context=context)


def test_migrate_short_circuits_when_sentinel_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing sentinel must short-circuit before any other check.

    Specifically — we must NOT call upgrade_advice (which is the second
    cheapest check) when the sentinel is present. This guards against a
    silent re-migration if upgrade_advice momentarily returns non-uv
    (e.g., during a uv-tool reinstall window).
    """
    sentinel = tmp_path / "migrated-to-uv"
    sentinel.write_text("migrated from pipx at 1234\n")
    called = {"upgrade_advice": 0}

    def _boom() -> lifecycle.UpgradeAdvice:
        called["upgrade_advice"] += 1
        return _make_advice("pipx")

    monkeypatch.setattr(lifecycle, "upgrade_advice", _boom)
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.attempted is False
    assert result.succeeded is False
    assert "already migrated" in result.reason
    assert called["upgrade_advice"] == 0


def test_migrate_short_circuits_when_already_uv_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A uv-tool install needs no migration. Don't shell out."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("uv_tool")
    )
    # If subprocess.run is reached the test fails — set a tripwire.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("subprocess.run called on uv-tool install"),
    )
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result == lifecycle.MigrationResult(False, False, "already on uv tool")
    assert not sentinel.exists()


def test_migrate_skips_when_uv_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No `uv` on PATH → skip migration, point user at the installer."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: None)
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.attempted is False
    assert result.succeeded is False
    assert "uv not on PATH" in result.reason
    assert "docs.astral.sh/uv" in result.reason
    assert not sentinel.exists()


def _fake_proc(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["uv", "tool", "install", "--force", "aelfrice"],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


def test_migrate_happy_path_writes_sentinel_and_names_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Success path: subprocess returns 0 → sentinel written, succeeded
    True, reason names the orphan pipx venv path."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(lifecycle, "_is_pipx_install", lambda: True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_proc(returncode=0)
    )
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.attempted is True
    assert result.succeeded is True
    assert "pipx venv" in result.reason
    assert "pipx uninstall aelfrice" in result.reason
    assert sentinel.exists()
    assert "migrated from non_uv" in sentinel.read_text()


def test_migrate_pip_orphan_uses_pip_uninstall_verb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When _is_pipx_install() is False, the orphan note must use the
    `pip uninstall -y aelfrice` verb, not pipx."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(lifecycle, "_is_pipx_install", lambda: False)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_proc(returncode=0)
    )
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.succeeded is True
    assert "pip uninstall -y aelfrice" in result.reason
    assert "pipx" not in result.reason


def test_migrate_failure_does_not_write_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-zero exit → no sentinel, failed result, stderr excerpt in reason.

    The next /aelf:upgrade invocation must be free to retry.
    """
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _fake_proc(
            returncode=1, stderr="error: failed to resolve dependency\n"
        ),
    )
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.attempted is True
    assert result.succeeded is False
    assert "exited 1" in result.reason
    assert "failed to resolve dependency" in result.reason
    assert not sentinel.exists()


def test_migrate_timeout_does_not_write_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """subprocess.TimeoutExpired → failed, no sentinel, timeout in reason."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")

    def _raise_timeout(*a: Any, **k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="uv", timeout=120)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    result = lifecycle.maybe_migrate_to_uv(
        sentinel_path=sentinel, timeout=120
    )
    assert result.attempted is True
    assert result.succeeded is False
    assert "timed out" in result.reason
    assert "120" in result.reason
    assert not sentinel.exists()


def test_migrate_oserror_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If subprocess.run itself raises OSError (e.g. ENOENT on the uv
    binary path between which() and run()), we catch it and return a
    failed result rather than letting it propagate into the caller's
    `aelf setup` flow."""
    sentinel = tmp_path / "migrated-to-uv"
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")

    def _raise_oserror(*a: Any, **k: Any) -> Any:
        raise OSError("[Errno 2] No such file or directory: 'uv'")

    monkeypatch.setattr(subprocess, "run", _raise_oserror)
    result = lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel)
    assert result.attempted is True
    assert result.succeeded is False
    assert "failed to launch" in result.reason
    assert not sentinel.exists()


def test_migrate_force_bypasses_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """force=True must re-attempt migration even when the sentinel
    exists. This is the recovery path if a previous migration succeeded
    at the subprocess layer but the running install never picked up the
    new shim."""
    sentinel = tmp_path / "migrated-to-uv"
    sentinel.write_text("stale\n")
    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(lifecycle, "_is_pipx_install", lambda: True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_proc(returncode=0)
    )
    result = lifecycle.maybe_migrate_to_uv(
        sentinel_path=sentinel, force=True
    )
    assert result.attempted is True
    assert result.succeeded is True
    # Sentinel rewritten with fresh timestamp.
    assert "migrated from non_uv" in sentinel.read_text()


def test_migrate_subprocess_command_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The subprocess argv must be exactly
    `[uv_bin, "tool", "install", "--force", "aelfrice"]` — capture-output
    True, text True, timeout passed through. Guards against accidental
    flag drift."""
    sentinel = tmp_path / "migrated-to-uv"
    captured: dict[str, Any] = {}

    def _capture(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured["args"] = list(args)
        captured["kwargs"] = dict(kwargs)
        return _fake_proc(returncode=0)

    monkeypatch.setattr(
        lifecycle, "upgrade_advice", lambda: _make_advice("non_uv")
    )
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/opt/uv")
    monkeypatch.setattr(lifecycle, "_is_pipx_install", lambda: True)
    monkeypatch.setattr(subprocess, "run", _capture)
    lifecycle.maybe_migrate_to_uv(sentinel_path=sentinel, timeout=99)
    assert captured["args"] == [
        "/opt/uv", "tool", "install", "--force", "aelfrice",
    ]
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["timeout"] == 99
