"""#554 integration guard — ≥80% of beliefs carry a non-NULL session_id
after a two-session synthetic ingest run across all three entry points.

This is a rate-level assertion on top of the per-surface unit tests in
:mod:`tests.test_session_id_propagation`.  It exercises `ingest_turn`,
`cli._cmd_lock`, and `mcp_server.tool_lock` in a single shared store,
interleaving explicit session_id kwargs, $AELF_SESSION_ID env fallbacks,
and a small remainder with neither, so the null rate never drops to zero
and the 80% threshold is meaningful.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from aelfrice import ingest, mcp_server, session_resolution
from aelfrice.cli import _cmd_lock
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "aelfrice_rate.db"


@pytest.fixture(autouse=True)
def _reset_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean warn-dedupe state and clear env before each test."""
    monkeypatch.delenv("AELF_SESSION_ID", raising=False)
    session_resolution._reset_warned_for_tests()


# ---------------------------------------------------------------------------
# helpers


def _make_args(statement: str, session_id: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(statement=statement, session_id=session_id)


def _belief_session_ids(store: MemoryStore) -> list[str | None]:
    rows = store._conn.execute("SELECT session_id FROM beliefs").fetchall()
    return [r["session_id"] for r in rows]


# ---------------------------------------------------------------------------
# two-session population-rate test


def test_session_id_population_rate_at_least_80_pct(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """≥80% of beliefs ingested across a two-session run have non-NULL session_id.

    Synthetic run: 12 ingest calls total.
    Session A (session-a): 4 ingest_turn (2 explicit, 2 env), 2 _cmd_lock (1 explicit, 1 env).
    Session B (session-b): 4 mcp tool_lock (2 explicit, 2 env).
    No-session (NULL): 2 ingest_turn calls with neither explicit nor env,
    keeping the NULL population small but non-zero.
    """
    # Patch _open_store so the CLI surface writes to the same shared DB.
    monkeypatch.setattr(
        "aelfrice.cli._open_store", lambda: MemoryStore(str(db_path))
    )

    store = MemoryStore(str(db_path))
    try:
        # --- Session A: ingest_turn — explicit session_id ---
        ingest.ingest_turn(
            store,
            text="The project uses a SQLite-backed belief store.",
            source="rate-test-a1",
            session_id="session-a",
        )
        ingest.ingest_turn(
            store,
            text="Beliefs are deduplicated by content hash.",
            source="rate-test-a2",
            session_id="session-a",
        )

        # --- Session A: ingest_turn — env fallback ---
        monkeypatch.setenv("AELF_SESSION_ID", "session-a")
        ingest.ingest_turn(
            store,
            text="Ingest runs are logged in the ingest_log table.",
            source="rate-test-a3",
        )
        ingest.ingest_turn(
            store,
            text="The store exposes a close method for cleanup.",
            source="rate-test-a4",
        )
        monkeypatch.delenv("AELF_SESSION_ID")

        # --- Session A: _cmd_lock — explicit session_id ---
        rc = _cmd_lock(_make_args("aelf lock preserves user intent.", "session-a"), io.StringIO())
        assert rc == 0

        # --- Session A: _cmd_lock — env fallback ---
        monkeypatch.setenv("AELF_SESSION_ID", "session-a")
        rc = _cmd_lock(_make_args("aelf lock is idempotent on the same statement."), io.StringIO())
        assert rc == 0
        monkeypatch.delenv("AELF_SESSION_ID")

        # --- Session B: tool_lock — explicit session_id ---
        result = mcp_server.tool_lock(
            store, statement="The MCP interface exposes aelf_lock.", session_id="session-b"
        )
        assert result["action"] in ("locked", "corroborated")
        result = mcp_server.tool_lock(
            store, statement="MCP tool responses include the belief id.", session_id="session-b"
        )
        assert result["action"] in ("locked", "corroborated")

        # --- Session B: tool_lock — env fallback ---
        monkeypatch.setenv("AELF_SESSION_ID", "session-b")
        result = mcp_server.tool_lock(
            store, statement="MCP session context flows through resolve_session_id."
        )
        assert result["action"] in ("locked", "corroborated")
        result = mcp_server.tool_lock(
            store, statement="Session ids are stamped on every ingest surface."
        )
        assert result["action"] in ("locked", "corroborated")
        monkeypatch.delenv("AELF_SESSION_ID")

        # --- NULL remainder: ingest_turn with neither explicit nor env ---
        ingest.ingest_turn(
            store,
            text="Belief graph edges are stored in the edges table.",
            source="rate-test-null-1",
        )
        ingest.ingest_turn(
            store,
            text="Graph traversal supports multi-hop BFS expansion.",
            source="rate-test-null-2",
        )

        # --- assert ≥80% population rate ---
        session_ids = _belief_session_ids(store)
        total = len(session_ids)
        non_null = sum(1 for sid in session_ids if sid is not None)
        assert total > 0, "no beliefs were written — ingest calls all no-oped"
        rate = non_null / total
        assert rate >= 0.80, (
            f"session_id population rate {rate:.2%} ({non_null}/{total}) is below 80%"
        )

    finally:
        store.close()
