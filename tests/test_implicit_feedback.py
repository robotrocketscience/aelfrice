"""Implicit retrieval-driven feedback sweeper (#191).

Covers acceptance criteria 1-8 from the issue:

  1. Schema landed (deferred_feedback_queue exists; smoke-checked at
     store construction).
  2. retrieve() post-hook enqueues one row per surfaced belief with
     event_type='retrieval_exposure'.
  3. CLI subcommand `aelf sweep-feedback` (covered separately in
     test_cli_sweep_feedback.py).
  4. Sweeper applies +epsilon to alpha exactly once per row in the
     no-contradiction path.
  5. Sweeper cancels (no alpha change) when an explicit signal lands
     in the grace window.
  6. audit_log records 'retrieval_driven_feedback' as the source so
     it is distinguishable from explicit user feedback.
  7. Idempotency: sweep x 2 = sweep x 1.
  8. Configurable T_grace + epsilon (env, kwarg, TOML, default).

All tests deterministic — clock injected, no real time, no real
sleep. In-memory store. Each test < 100 ms.
"""
from __future__ import annotations

import os
from pathlib import Path

from aelfrice.deferred_feedback import (
    DEFAULT_EPSILON,
    DEFAULT_T_GRACE_SECONDS,
    EVENT_RETRIEVAL_EXPOSURE,
    RETRIEVAL_DRIVEN_FEEDBACK_SOURCE,
    enqueue_retrieval_exposures,
    is_enqueue_on_retrieve_enabled,
    resolve_epsilon,
    resolve_grace_seconds,
    sweep_deferred_feedback,
)
from aelfrice.models import BELIEF_FACTUAL, EDGE_CONTRADICTS, LOCK_NONE, Belief, Edge
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

T0 = "2026-04-28T00:00:00Z"
T_BEFORE_GRACE = "2026-04-28T00:10:00Z"  # 10 min after T0
T_INSIDE_GRACE = "2026-04-28T00:25:00Z"  # 25 min after T0
T_AFTER_GRACE = "2026-04-28T01:00:00Z"   # 60 min after T0


def _mk(bid: str, content: str = "") -> Belief:
    return Belief(
        id=bid,
        content=content or f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin="unknown",
        corroboration_count=0,
    )


def _store(*beliefs: Belief) -> MemoryStore:
    s = MemoryStore(":memory:")
    for b in beliefs:
        s.insert_belief(b)
    return s


# --- AC1: schema sanity --------------------------------------------------


def test_schema_creates_deferred_feedback_queue_table() -> None:
    s = _store()
    cur = s._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='deferred_feedback_queue'"
    )
    assert cur.fetchone() is not None


def test_schema_creates_dfq_indexes() -> None:
    s = _store()
    cur = s._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name LIKE 'idx_dfq%'"
    )
    names = sorted(str(r["name"]) for r in cur.fetchall())
    assert "idx_dfq_belief" in names
    assert "idx_dfq_status_enq" in names


# --- AC2: retrieve() enqueues -------------------------------------------


def test_retrieve_enqueues_one_row_per_surfaced_belief() -> None:
    s = _store(_mk("b1", "apple banana"), _mk("b2", "cherry"))
    out = retrieve(s, "apple")
    assert {b.id for b in out} == {"b1"}
    rows = s.list_pending_deferred_feedback(cutoff_iso="2099-01-01T00:00:00Z")
    assert len(rows) == 1
    assert rows[0][1] == "b1"
    assert rows[0][3] == EVENT_RETRIEVAL_EXPOSURE


def test_retrieve_enqueue_can_be_disabled_via_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "0")
    s = _store(_mk("b1", "apple"))
    retrieve(s, "apple")
    assert s.count_deferred_feedback_by_status() == {}


def test_empty_query_does_not_enqueue() -> None:
    s = _store(_mk("b1", "apple"))
    retrieve(s, "")
    assert s.count_deferred_feedback_by_status() == {}


def test_enqueue_failure_does_not_break_retrieve(monkeypatch, capsys) -> None:
    s = _store(_mk("b1", "apple"))
    import aelfrice.deferred_feedback as df
    def boom(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(df, "enqueue_retrieval_exposures", boom)
    out = retrieve(s, "apple")
    assert {b.id for b in out} == {"b1"}
    assert "deferred-feedback enqueue failed" in capsys.readouterr().err


# --- AC4: applied path (+epsilon) ---------------------------------------


def test_sweep_applies_epsilon_after_grace() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 1
    assert r.cancelled == 0
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.05
    assert s.count_deferred_feedback_by_status() == {"applied": 1}


def test_sweep_skips_rows_inside_grace_window() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_INSIDE_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 0
    assert r.pending_unmet_grace == 1
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.0
    assert s.count_deferred_feedback_by_status() == {"enqueued": 1}


# --- AC5: cancellation path ---------------------------------------------


def test_explicit_feedback_in_grace_window_cancels_implicit() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    s.insert_feedback_event(
        "b1", valence=-1.0, source="user", created_at=T_BEFORE_GRACE
    )
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 0
    assert r.cancelled == 1
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.0


def test_contradiction_tiebreaker_event_in_grace_cancels() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    # Contradiction tiebreaker resolutions write to feedback_history
    # with a distinctive 'contradiction_tiebreaker:' source prefix.
    s.insert_feedback_event(
        "b1", valence=-1.0,
        source="contradiction_tiebreaker:lock_wins",
        created_at=T_BEFORE_GRACE,
    )
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 0
    assert r.cancelled == 1


def test_explicit_feedback_outside_grace_does_not_cancel() -> None:
    s = _store(_mk("b1"))
    # Explicit feedback BEFORE the enqueue → outside the row's window.
    s.insert_feedback_event(
        "b1", valence=-1.0, source="user",
        created_at="2026-04-27T23:00:00Z",
    )
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 1
    assert r.cancelled == 0


def test_belief_deleted_between_enqueue_and_sweep_cascades_queue_row() -> None:
    """ON DELETE CASCADE on the FK means a deleted belief takes its
    pending queue rows with it. The sweeper sees nothing for that
    belief — no apply, no cancel, queue stays consistent."""
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    s._conn.execute("DELETE FROM beliefs WHERE id='b1'")
    s._conn.commit()
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.applied == 0
    assert r.cancelled == 0
    assert s.count_deferred_feedback_by_status() == {}


# --- AC6: audit-log event type distinct ---------------------------------


def test_applied_row_writes_distinctive_audit_source() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    events = s.list_feedback_events(belief_id="b1")
    sources = [e.source for e in events]
    assert RETRIEVAL_DRIVEN_FEEDBACK_SOURCE in sources
    # And NOT the same as any explicit-feedback source.
    assert "user" not in sources


# --- AC7: idempotency + crash-safe ---------------------------------------


def test_sweep_twice_equals_sweep_once() -> None:
    s = _store(_mk("b1"), _mk("b2"))
    enqueue_retrieval_exposures(s, ["b1", "b2"], now=T0)
    s.insert_feedback_event(
        "b2", valence=-1.0, source="user", created_at=T_BEFORE_GRACE
    )
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    state_after_first = (
        s.get_belief("b1").alpha,  # type: ignore[union-attr]
        s.get_belief("b2").alpha,  # type: ignore[union-attr]
        s.count_deferred_feedback_by_status(),
    )
    r2 = sweep_deferred_feedback(
        s, now="2026-04-28T02:00:00Z", grace_seconds=1800, epsilon=0.05
    )
    assert r2.applied == 0 and r2.cancelled == 0
    state_after_second = (
        s.get_belief("b1").alpha,  # type: ignore[union-attr]
        s.get_belief("b2").alpha,  # type: ignore[union-attr]
        s.count_deferred_feedback_by_status(),
    )
    assert state_after_first == state_after_second


def test_already_applied_row_is_not_reapplied() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    # Manually re-enqueue the SAME row would be a new row; but an
    # already-applied row should be skipped on re-scan because
    # list_pending_deferred_feedback filters by status='enqueued'.
    pending = s.list_pending_deferred_feedback(
        cutoff_iso="2099-01-01T00:00:00Z"
    )
    assert pending == []


def test_partial_progress_resumes_correctly() -> None:
    """Three rows; sweep one, then sweep again — the remaining two land."""
    s = _store(_mk("b1"), _mk("b2"), _mk("b3"))
    enqueue_retrieval_exposures(s, ["b1", "b2", "b3"], now=T0)
    r1 = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05, limit=1
    )
    assert r1.applied == 1
    r2 = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r2.applied == 2
    # All three got exactly one increment.
    assert all(
        s.get_belief(b).alpha == 1.05  # type: ignore[union-attr]
        for b in ("b1", "b2", "b3")
    )


# --- AC8: configurable T_grace + epsilon -------------------------------


def test_resolve_grace_seconds_default() -> None:
    assert resolve_grace_seconds() == DEFAULT_T_GRACE_SECONDS


def test_resolve_grace_seconds_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS", "60")
    assert resolve_grace_seconds() == 60


def test_resolve_grace_seconds_kwarg_override() -> None:
    assert resolve_grace_seconds(120) == 120


def test_resolve_grace_seconds_toml_override(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[implicit_feedback]\ngrace_window_seconds = 90\n"
    )
    assert resolve_grace_seconds(start=tmp_path) == 90


def test_resolve_grace_seconds_env_invalid_falls_through(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_GRACE_SECONDS", "not-int")
    assert resolve_grace_seconds() == DEFAULT_T_GRACE_SECONDS


def test_resolve_epsilon_default() -> None:
    assert resolve_epsilon() == DEFAULT_EPSILON


def test_resolve_epsilon_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_EPSILON", "0.2")
    assert resolve_epsilon() == 0.2


def test_resolve_epsilon_negative_clamps_to_zero() -> None:
    assert resolve_epsilon(-1.0) == 0.0


def test_resolve_epsilon_toml_override(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[implicit_feedback]\nepsilon = 0.10\n"
    )
    assert resolve_epsilon(start=tmp_path) == 0.10


def test_is_enqueue_on_retrieve_default_true() -> None:
    assert is_enqueue_on_retrieve_enabled() is True


def test_is_enqueue_on_retrieve_env_off(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "false")
    assert is_enqueue_on_retrieve_enabled() is False


# --- Integration: epsilon respected end-to-end --------------------------


def test_custom_epsilon_landed_in_alpha_and_audit() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.25
    )
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.25
    events = s.list_feedback_events(belief_id="b1")
    retrieval_events = [
        e for e in events if e.source == RETRIEVAL_DRIVEN_FEEDBACK_SOURCE
    ]
    assert len(retrieval_events) == 1
    assert retrieval_events[0].valence == 0.25


def test_propagate_off_locked_neighbours_unchanged() -> None:
    """Implicit signal must not pressure user-locked contradictors —
    only explicit positive feedback fires the demotion-pressure walk."""
    s = MemoryStore(":memory:")
    src = _mk("X")
    locked = Belief(
        id="Y", content="locked",
        content_hash="hY", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level="user",
        locked_at="2026-04-26T00:00:00Z",
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin="unknown",
        corroboration_count=0,
    )
    s.insert_belief(src)
    s.insert_belief(locked)
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))

    enqueue_retrieval_exposures(s, ["X"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    # Y's pressure must be untouched: implicit signals do not propagate.
    y = s.get_belief("Y")
    assert y is not None
    assert y.lock_level == "user"
