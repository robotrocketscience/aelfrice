"""The Stop hook's core-gate paths stay on the claude host (#1812).

The Gemini adapter sets `AELFRICE_HOST=gemini` before it runs the Stop
handler. On that host the hook must neither write the session-end
continuation, which Gemini documents no `additionalContext` for on its
turn-end event, nor start the background labeler, which runs the `claude`
CLI. Each test names the mutation that kills it. Every store is under
`tmp_path`, and the belief text is synthetic.
"""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pytest

import aelfrice.core_gate_background as bg
import aelfrice.hook as hook
from aelfrice.hosts import HOST_ENV
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

SESSION = "widgetsession-host"
CLAUDE = "/opt/widget/bin/claude"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    monkeypatch.delenv(hook.CORE_GATE_SESSION_END_ENV, raising=False)
    monkeypatch.delenv(bg.BACKGROUND_ENV, raising=False)
    # The continuation is opt-in (#1740); turn it on so its absence on the
    # gemini host is the host check and not the default.
    (tmp_path / ".aelfrice.toml").write_text(
        "[core_gate]\nsession_end = true\n", encoding="utf-8",
    )
    store = MemoryStore(str(path))
    try:
        store.insert_belief(Belief(
            id="w1",
            content="The widgethost w1 setting is stable.",
            content_hash="hash-w1",
            alpha=3.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, created_at="2026-10-01T00:00:00Z",
            last_retrieved_at=None, session_id=SESSION,
            origin=ORIGIN_USER_TRANSCRIPT,
        ))
    finally:
        store.close()
    return path


def _stop(tmp_path: Path, env: dict[str, str]) -> str:
    payload = {"session_id": SESSION, "cwd": str(tmp_path)}
    out, err = io.StringIO(), io.StringIO()
    assert hook.stop(
        stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=err,
        env=env,
    ) == 0
    return out.getvalue()


def _batch_count(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM core_gate_batches",
        ).fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def no_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bg, "find_claude", lambda: None)


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(bg, "find_claude", lambda: CLAUDE)

    def fake_spawn(batch_id: str, *, db_path: str, claude: str) -> bool:
        calls.append(batch_id)
        return True

    monkeypatch.setattr(bg, "spawn", fake_spawn)
    return calls


def test_the_claude_host_writes_the_continuation(
    db: Path, tmp_path: Path, no_cli: None,
) -> None:
    """The control: with no marker the continuation fires, so the gemini
    case below fails on the host check and not on the fixture."""
    out = _stop(tmp_path, {})
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert _batch_count(db) == 1


def test_the_gemini_host_writes_no_continuation(
    db: Path, tmp_path: Path, no_cli: None,
) -> None:
    """Killed by: deleting the host check."""
    assert _stop(tmp_path, {HOST_ENV: "gemini"}) == ""
    assert _batch_count(db) == 0


def test_the_gemini_host_starts_no_labeler(
    db: Path, tmp_path: Path, spawned: list[str],
) -> None:
    """Killed by: moving the host check below the background route."""
    assert _stop(tmp_path, {HOST_ENV: "gemini"}) == ""
    assert spawned == []
    assert _batch_count(db) == 0


def test_the_claude_host_starts_the_labeler(
    db: Path, tmp_path: Path, spawned: list[str],
) -> None:
    """The control for the case above."""
    assert _stop(tmp_path, {}) == ""
    assert len(spawned) == 1


def test_the_marker_is_read_from_the_process_environment(
    db: Path, tmp_path: Path, no_cli: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter sets the marker in its process, and the installed Stop
    path passes no `env`. Killed by: reading only an explicit `env`."""
    monkeypatch.setenv(HOST_ENV, "gemini")
    payload = {"session_id": SESSION, "cwd": str(tmp_path)}
    out = io.StringIO()
    assert hook.stop(
        stdin=io.StringIO(json.dumps(payload)), stdout=out,
        stderr=io.StringIO(),
    ) == 0
    assert out.getvalue() == ""
    assert _batch_count(db) == 0
