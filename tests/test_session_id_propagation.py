"""#192 phantom-prereqs T3 — session_id propagation across ingest entry points.

Covers the three call sites that consume
:func:`aelfrice.session_resolution.resolve_session_id`:

  * ``ingest.ingest_turn`` (Q1.a, library-direct)
  * ``cli._cmd_lock`` (Q3.a, CLI ``aelf lock``)
  * ``mcp_server.tool_lock`` (Q4.a, MCP ``aelf_lock``)

Each surface must:

  * stamp an explicit ``session_id`` on the resulting belief and the
    ingest_log row;
  * fall back to ``$AELF_SESSION_ID`` when the caller omits it;
  * write NULL and warn-once when neither is present.

The producer-side surfaces (scanner.scan_repo, classification onboard
accept) already shipped via #357 and have their own tests; this file
only covers the helper-driven inference path.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

from aelfrice import ingest, mcp_server, session_resolution
from aelfrice.cli import _cmd_lock
from aelfrice.store import MemoryStore


@pytest.fixture
def db_path_fixture(tmp_path: Path) -> Path:
    return tmp_path / "aelfrice.db"


@pytest.fixture
def store(db_path_fixture: Path) -> MemoryStore:
    s = MemoryStore(str(db_path_fixture))
    yield s
    try:
        s.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test starts with a clean warn-dedupe set and no env var."""
    monkeypatch.delenv("AELF_SESSION_ID", raising=False)
    session_resolution._reset_warned_for_tests()


def _belief_session_ids(s: MemoryStore | Path) -> list[str | None]:
    s2 = MemoryStore(str(s)) if isinstance(s, Path) else s
    try:
        rows = s2._conn.execute(
            "SELECT session_id FROM beliefs ORDER BY id"
        ).fetchall()
    finally:
        if isinstance(s, Path):
            s2.close()
    return [r["session_id"] for r in rows]


def _ingest_log_session_ids(s: MemoryStore | Path) -> list[str | None]:
    s2 = MemoryStore(str(s)) if isinstance(s, Path) else s
    try:
        rows = s2._conn.execute(
            "SELECT session_id FROM ingest_log ORDER BY ts"
        ).fetchall()
    finally:
        if isinstance(s, Path):
            s2.close()
    return [r["session_id"] for r in rows]


# ---------------------------------------------------------------------------
# resolver helper

def test_resolve_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELF_SESSION_ID", "from-env")
    assert (
        session_resolution.resolve_session_id("explicit", surface_name="x")
        == "explicit"
    )


def test_resolve_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELF_SESSION_ID", "from-env")
    assert session_resolution.resolve_session_id(None, surface_name="x") == "from-env"


def test_resolve_null_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert session_resolution.resolve_session_id(None, surface_name="ingest_turn") is None
    captured = capsys.readouterr()
    assert "ingest_turn" in captured.err
    assert "AELF_SESSION_ID" in captured.err

    # Second call from same surface: silent.
    assert session_resolution.resolve_session_id(None, surface_name="ingest_turn") is None
    captured = capsys.readouterr()
    assert captured.err == ""

    # Different surface: warns again.
    assert session_resolution.resolve_session_id(None, surface_name="aelf lock") is None
    captured = capsys.readouterr()
    assert "aelf lock" in captured.err


# ---------------------------------------------------------------------------
# ingest_turn (Q1.a)

def test_ingest_turn_explicit_session_id_stamped(store: MemoryStore) -> None:
    ingest.ingest_turn(
        store,
        text="The system uses SQLite for storage.",
        source="t-ingest-explicit",
        session_id="sess-explicit",
    )
    assert _belief_session_ids(store) == ["sess-explicit"]


def test_ingest_turn_env_fallback(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELF_SESSION_ID", "sess-from-env")
    ingest.ingest_turn(
        store,
        text="The system uses SQLite for storage.",
        source="t-ingest-env",
    )
    assert _belief_session_ids(store) == ["sess-from-env"]


def test_ingest_turn_no_session_writes_null_and_warns(
    store: MemoryStore, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest.ingest_turn(
        store,
        text="The system uses SQLite for storage.",
        source="t-ingest-null",
    )
    assert _belief_session_ids(store) == [None]
    err = capsys.readouterr().err
    assert "ingest_turn" in err
    assert "AELF_SESSION_ID" in err


# ---------------------------------------------------------------------------
# CLI _cmd_lock (Q3.a)

def _make_lock_args(statement: str, session_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(statement=statement, session_id=session_id)


def test_cli_lock_explicit_session_id_stamped(
    db_path_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "aelfrice.cli._open_store", lambda: MemoryStore(str(db_path_fixture))
    )
    rc = _cmd_lock(_make_lock_args("locked statement A", "sess-cli"), io.StringIO())
    assert rc == 0
    assert _belief_session_ids(db_path_fixture) == ["sess-cli"]
    assert _ingest_log_session_ids(db_path_fixture) == ["sess-cli"]


def test_cli_lock_env_fallback(
    db_path_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELF_SESSION_ID", "sess-cli-env")
    monkeypatch.setattr(
        "aelfrice.cli._open_store", lambda: MemoryStore(str(db_path_fixture))
    )
    rc = _cmd_lock(_make_lock_args("locked statement B"), io.StringIO())
    assert rc == 0
    assert _belief_session_ids(db_path_fixture) == ["sess-cli-env"]
    assert _ingest_log_session_ids(db_path_fixture) == ["sess-cli-env"]


def test_cli_lock_null_and_warns(
    db_path_fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "aelfrice.cli._open_store", lambda: MemoryStore(str(db_path_fixture))
    )
    rc = _cmd_lock(_make_lock_args("locked statement C"), io.StringIO())
    assert rc == 0
    assert _belief_session_ids(db_path_fixture) == [None]
    assert _ingest_log_session_ids(db_path_fixture) == [None]
    assert "aelf lock" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# MCP tool_lock (Q4.a)

def test_mcp_lock_explicit_session_id_stamped(store: MemoryStore) -> None:
    out = mcp_server.tool_lock(
        store, statement="mcp locked A", session_id="sess-mcp"
    )
    assert out["action"] in ("locked", "corroborated")
    assert _belief_session_ids(store) == ["sess-mcp"]
    assert _ingest_log_session_ids(store) == ["sess-mcp"]


def test_mcp_lock_env_fallback(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELF_SESSION_ID", "sess-mcp-env")
    out = mcp_server.tool_lock(store, statement="mcp locked B")
    assert out["action"] in ("locked", "corroborated")
    assert _belief_session_ids(store) == ["sess-mcp-env"]


def test_mcp_lock_null_and_warns(
    store: MemoryStore, capsys: pytest.CaptureFixture[str]
) -> None:
    out = mcp_server.tool_lock(store, statement="mcp locked C")
    assert out["action"] in ("locked", "corroborated")
    assert _belief_session_ids(store) == [None]
    assert "mcp aelf_lock" in capsys.readouterr().err
