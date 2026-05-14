"""Integration tests for the post-#264 mcp_server.tool_lock →
derivation_worker call shape.

Each test states a falsifiable hypothesis about the new contract:

    tool_lock writes one ingest_log row with `call_site=mcp_remember`,
    invokes run_worker once, and reads the stamped belief id back. The
    re-lock semantic still lives at the entry point (worker recorded a
    corroboration; entry point applies update_belief).

Anchors the new invariants for the MCP lock entry point.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import mcp_server
from aelfrice.derivation import DerivationOutput
from aelfrice.mcp_server import tool_lock
from aelfrice.models import LOCK_USER, ORIGIN_USER_STATED
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "mcp-lock-via-worker.db"))
    yield s
    s.close()


def test_lock_writes_log_row_with_call_site(store: MemoryStore) -> None:
    """Hypothesis: tool_lock writes exactly one log row with
    `raw_meta.call_site = mcp_remember`. Falsifiable if the row is
    missing, has a different call_site, or carries no raw_meta."""
    out = tool_lock(store, statement="atomic commits beat batched commits")
    assert out["action"] == "locked"
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta, source_kind FROM ingest_log"
    ).fetchall()
    assert len(rows) == 1
    meta = json.loads(rows[0]["raw_meta"]) if rows[0]["raw_meta"] else None
    assert isinstance(meta, dict)
    assert meta.get("call_site") == "mcp_remember"
    assert rows[0]["source_kind"] == "mcp_remember"


def test_log_row_is_stamped_after_lock(store: MemoryStore) -> None:
    """Hypothesis: the worker ran end-of-call and stamped the log row's
    derived_belief_ids. Falsifiable if any unstamped row remains."""
    tool_lock(store, statement="the sky is blue")
    unstamped = store.list_unstamped_ingest_log()
    assert unstamped == []


def test_relock_upgrades_existing_lock_id(store: MemoryStore) -> None:
    """Hypothesis: re-locking the same statement returns action=upgraded
    and the existing lock-id belief gets lock_level/locked_at refreshed
    via update_belief (worker would only record a corroboration).
    Falsifiable if the second call inserts a new belief, returns
    action!=upgraded, or leaves the prior belief's lock fields stale."""
    first = tool_lock(store, statement="we always sign commits")
    bid = first["id"]
    # demote then re-lock to force the upgrade path
    b = store.get_belief(bid)
    assert b is not None
    b.lock_level = 0  # LOCK_NONE
    b.locked_at = None
    store.update_belief(b)

    second = tool_lock(store, statement="we always sign commits")
    assert second["action"] == "upgraded"
    assert second["id"] == bid
    refreshed = store.get_belief(bid)
    assert refreshed is not None
    assert refreshed.lock_level == LOCK_USER
    assert refreshed.locked_at is not None
    assert refreshed.origin == ORIGIN_USER_STATED


def test_replay_full_equality_passes_after_lock(store: MemoryStore) -> None:
    """Hypothesis (CI gate): a tool_lock workload passes the full-equality
    replay probe with zero drift. Tripwire for any future regression that
    bypasses the worker or produces a divergent canonical belief.
    Falsifiable by any non-zero drift counter."""
    tool_lock(store, statement="we always sign commits with ssh")
    tool_lock(store, statement="aelfrice stores beliefs in sqlite")
    # Re-lock to exercise the upgrade path inside replay equality.
    tool_lock(store, statement="we always sign commits with ssh")

    report = replay_full_equality(store)
    assert report.total_log_rows > 0
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0


def test_lock_returns_structured_error_when_derivation_yields_no_belief(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: when `derive()` returns DerivationOutput(belief=None)
    (the classifier's persist=False path), tool_lock returns a
    well-formed lock.error dict with a non-empty `error` field — NOT a
    Python AssertionError that crashes the host's MCP tool surface.

    Falsifiable by an unhandled exception, an empty error string, or
    return shape diverging from the documented {kind, id, action, error}.
    """
    monkeypatch.setattr(
        mcp_server,
        "derive",
        lambda inp: DerivationOutput(belief=None, skip_reason="empty"),
    )
    out = tool_lock(store, statement="anything; derive will reject it")
    assert out["kind"] == "lock.error"
    assert out["action"] == "error"
    assert out["id"] == ""
    assert isinstance(out.get("error"), str) and out["error"], (
        "lock.error response missing populated `error` field"
    )


def test_lock_idempotent_on_canonical_state(store: MemoryStore) -> None:
    """Hypothesis: re-locking is idempotent on the canonical belief set
    even though it adds new log rows (one per call). Falsifiable by a
    duplicate belief or by a lock_id mismatch."""
    first = tool_lock(store, statement="atomic commits beat batched")
    n_before = store.count_beliefs()
    second = tool_lock(store, statement="atomic commits beat batched")
    assert second["id"] == first["id"]
    assert store.count_beliefs() == n_before
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log WHERE raw_text = ?",
        ("atomic commits beat batched",),
    ).fetchone()
    assert rows["n"] == 2
