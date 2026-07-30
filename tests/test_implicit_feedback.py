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

import pytest
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
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
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


def test_retrieve_does_not_enqueue_by_default() -> None:
    """#1162. The counterpart to the row below: with no opt-in, a
    retrieval writes nothing to the queue."""
    s = _store(_mk("b1", "apple banana"), _mk("b2", "cherry"))
    retrieve(s, "apple")
    assert s.count_deferred_feedback_by_status() == {}


def test_retrieve_enqueues_one_row_per_surfaced_belief(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "1")
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
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "1")
    s = _store(_mk("b1", "apple"))
    import aelfrice.deferred_feedback as df
    def boom(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(df, "enqueue_retrieval_exposures", boom)
    out = retrieve(s, "apple")
    assert {b.id for b in out} == {"b1"}
    assert "deferred-feedback enqueue failed" in capsys.readouterr().err


# --- AC4: eligible path, audit-only since #1162 -------------------------


def test_sweep_reports_the_eligible_row_and_moves_no_alpha() -> None:
    """The #1162 acceptance criterion in one test, both halves.

    A sweep over a store with pending rows must change no belief's
    alpha, and must still report a non-zero eligible count. Dropping
    either half leaves a passing test: an audit that reports zero
    satisfies the alpha assertion while hiding that the queue stopped
    being read, and a mutating sweeper satisfies the count.
    """
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.would_apply == 1
    assert r.would_cancel == 0
    assert r.alpha_withheld == pytest.approx(0.05)
    assert r.mutated is False
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.0
    # Nothing consumed: the row is still enqueued, not marked applied.
    assert s.count_deferred_feedback_by_status() == {"enqueued": 1}
    assert s.list_feedback_events(belief_id="b1") == []


def test_sweep_skips_rows_inside_grace_window() -> None:
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_INSIDE_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r.would_apply == 0
    assert r.pending_unmet_grace == 1
    assert r.alpha_withheld == 0.0
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
    assert r.would_apply == 0
    assert r.would_cancel == 1
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
    assert r.would_apply == 0
    assert r.would_cancel == 1


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
    assert r.would_apply == 1
    assert r.would_cancel == 0


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
    assert r.would_apply == 0
    assert r.would_cancel == 0
    assert s.count_deferred_feedback_by_status() == {}


# --- AC6: the sweep leaves no audit row at all --------------------------


def test_sweep_writes_no_feedback_history_row() -> None:
    """The inverse of the pre-#1162 assertion, and the more useful one.

    An audit-only sweep must not leave a `RETRIEVAL_DRIVEN_FEEDBACK_SOURCE`
    row behind, because such a row is what every downstream consumer
    reads as "implicit feedback was applied here". The source constant
    survives — it is still the exclusion key that decides which
    feedback events count as explicit for cancellation.
    """
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    events = s.list_feedback_events(belief_id="b1")
    assert events == []
    assert RETRIEVAL_DRIVEN_FEEDBACK_SOURCE not in [e.source for e in events]


# --- AC7: idempotency + crash-safe ---------------------------------------


def test_sweep_twice_reports_the_same_numbers() -> None:
    """Repeatability, which is a stronger property than the idempotency
    it replaces. The mutating sweeper was idempotent by consuming its
    input: run twice, the second run reported zero. That reads as
    "there is nothing here" rather than "this was already spent". The
    audit consumes nothing, so the count is a standing measurement.
    """
    s = _store(_mk("b1"), _mk("b2"))
    enqueue_retrieval_exposures(s, ["b1", "b2"], now=T0)
    s.insert_feedback_event(
        "b2", valence=-1.0, source="user", created_at=T_BEFORE_GRACE
    )
    r1 = sweep_deferred_feedback(
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
    assert (r2.would_apply, r2.would_cancel) == (r1.would_apply, r1.would_cancel)
    assert r1.would_apply == 1 and r1.would_cancel == 1
    state_after_second = (
        s.get_belief("b1").alpha,  # type: ignore[union-attr]
        s.get_belief("b2").alpha,  # type: ignore[union-attr]
        s.count_deferred_feedback_by_status(),
    )
    assert state_after_first == state_after_second


def test_sweep_leaves_every_row_enqueued() -> None:
    """The queue is not drained by being read. Pre-#1162 this row would
    have flipped to `applied` and dropped out of the pending scan."""
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    pending = s.list_pending_deferred_feedback(
        cutoff_iso="2099-01-01T00:00:00Z"
    )
    assert [row[1] for row in pending] == ["b1"]


def test_limit_bounds_the_audit_without_consuming_the_rest() -> None:
    """`--limit` still bounds one pass, but since nothing is consumed a
    subsequent unbounded pass sees all three rather than the remainder."""
    s = _store(_mk("b1"), _mk("b2"), _mk("b3"))
    enqueue_retrieval_exposures(s, ["b1", "b2", "b3"], now=T0)
    r1 = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05, limit=1
    )
    assert r1.would_apply == 1
    r2 = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.05
    )
    assert r2.would_apply == 3
    assert all(
        s.get_belief(b).alpha == 1.0  # type: ignore[union-attr]
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


def test_is_enqueue_on_retrieve_default_off() -> None:
    """#1162. Default-on wrote a queue row per surfaced belief on every
    `retrieve()`, on the argument that the queue is additive — true only
    while nothing schedules the sweeper. It is also a second route to
    the posterior bump #1086 turned off. Opt-in now."""
    assert is_enqueue_on_retrieve_enabled() is False


def test_is_enqueue_on_retrieve_env_on(monkeypatch) -> None:
    """The opt-in is still reachable, so the default is a default and
    not a removal."""
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "1")
    assert is_enqueue_on_retrieve_enabled() is True


def test_is_enqueue_on_retrieve_env_off(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE", "false")
    assert is_enqueue_on_retrieve_enabled() is False


# --- Integration: epsilon respected end-to-end --------------------------


def test_custom_epsilon_is_reported_as_withheld_not_applied() -> None:
    """epsilon still resolves and still sizes the projection — it just
    reaches `alpha_withheld` instead of `alpha`."""
    s = _store(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)
    r = sweep_deferred_feedback(
        s, now=T_AFTER_GRACE, grace_seconds=1800, epsilon=0.25
    )
    assert r.epsilon_used == 0.25
    assert r.alpha_withheld == pytest.approx(0.25)
    b = s.get_belief("b1")
    assert b is not None and b.alpha == 1.0
    assert s.list_feedback_events(belief_id="b1") == []


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


# --- #1168 AC4: the sweeper honours the lock floor ----------------------


def test_sweep_does_not_bump_a_locked_belief() -> None:
    """Hypothesis: a user lock is never bumped by the implicit lane.

    The sweeper writes alpha directly, bypassing apply_feedback, so before
    #1168 the retrieval-driven +epsilon landed on locks — which
    docs/user/LIMITATIONS.md and PRIVACY.md both promise cannot happen.
    Falsifiable by any alpha change, or by the row staying enqueued."""
    locked = _mk("b1", "apple banana")
    locked.lock_level = LOCK_USER
    locked.alpha = 9.0
    locked.beta = 0.5
    s = _store(locked)
    enqueue_retrieval_exposures(s, ["b1"], now=T0)

    result = sweep_deferred_feedback(s, now=T_AFTER_GRACE)

    assert result.would_apply == 0
    assert result.would_skip_locked == 1
    assert result.would_cancel == 1
    after = s.get_belief("b1")
    assert after is not None
    assert after.alpha == 9.0
    assert after.beta == 0.5
    assert s.list_feedback_events(belief_id="b1") == []


def test_sweep_still_bumps_an_unlocked_belief() -> None:
    """Control for the test above: the lock check must not suppress the
    ordinary path. Falsifiable by alpha not moving."""
    s = _store(_mk("b1", "apple banana"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)

    result = sweep_deferred_feedback(s, now=T_AFTER_GRACE)

    assert result.would_apply == 1
    assert result.would_skip_locked == 0
    after = s.get_belief("b1")
    assert after is not None
    assert after.alpha == 1.0


def test_sweep_does_not_bump_a_foreign_belief(monkeypatch) -> None:
    """Hypothesis: a federated peer's belief is never bumped locally.

    #655 makes foreign beliefs read-only through the local DB, but the
    sweeper wrote alpha directly and never checked. Patching the ownership
    probe is enough to exercise the branch without standing up a peer DB.
    Falsifiable by any alpha change, a row left enqueued, or a
    feedback_history row."""
    from aelfrice.federation import ForeignBeliefError

    s = _store(_mk("b1", "apple banana"))
    enqueue_retrieval_exposures(s, ["b1"], now=T0)

    def _foreign(belief_id: str) -> None:
        raise ForeignBeliefError(belief_id=belief_id, owning_scope="peer-a")

    monkeypatch.setattr(s, "assert_local_ownership", _foreign)

    result = sweep_deferred_feedback(s, now=T_AFTER_GRACE)

    assert result.would_apply == 0
    assert result.would_skip_foreign == 1
    assert result.would_cancel == 1
    after = s.get_belief("b1")
    assert after is not None
    assert after.alpha == 1.0
    assert s.list_feedback_events(belief_id="b1") == []


def test_sweep_issues_no_write_statement_at_all() -> None:
    """Successor to #1168's check-then-act ordering test.

    That test pinned `BEGIN IMMEDIATE` before the eligibility read, so
    a lock committed mid-row could not land +epsilon on a belief the
    checks had just rejected. #1162 closes that window by removing the
    write rather than ordering it, so the structural assertion becomes
    the stronger one: the sweep must issue no INSERT, UPDATE, DELETE or
    write transaction whatsoever.

    Statement-level rather than state-level on purpose — asserting that
    alpha did not move would still pass for a sweep that wrote and then
    happened to write the same value, or that mutated some other table.
    """
    s = _store(_mk("b1", "apple banana"))
    locked = _mk("b2", "apple cherry")
    locked.lock_level = LOCK_USER
    s.insert_belief(locked)
    enqueue_retrieval_exposures(s, ["b1", "b2"], now=T0)

    statements: list[str] = []
    s._conn.set_trace_callback(statements.append)
    try:
        result = sweep_deferred_feedback(s, now=T_AFTER_GRACE)
    finally:
        s._conn.set_trace_callback(None)

    writes = [
        stmt for stmt in statements
        if stmt.strip().lower().startswith(
            ("insert", "update", "delete", "begin immediate", "begin ")
        )
    ]
    assert writes == [], f"audit-only sweep issued writes: {writes}"
    # The sweep really did traverse the rows it declined to write to —
    # a no-op that read nothing would also issue no writes.
    assert result.would_apply == 1
    assert result.would_skip_locked == 1
