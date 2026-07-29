"""#1168: the posterior write is atomic, transactional, and lock-floored.

`apply_feedback` used to be an unsynchronised read-modify-write — a
`get_belief` in Python, an arithmetic update in Python, then a whole-row
`update_belief` that wrote the stale snapshot back. Concurrent hook
processes each added their delta to the same value they had all read, so
the last writer won: the issue reproduced 4 processes x 60 events landing
240 `feedback_history` rows but only 66 events' worth of alpha.

The race tests below run real OS threads on separate store handles, with
a barrier to maximise overlap. Each worker's `busy_timeout` is set
explicitly and the per-test wall-clock budget is set well above it, so a
lost-update regression fails on the conservation assertion rather than
being masked by — or confused with — the suite-wide 5 s timeout that
happens to equal SQLite's default `busy_timeout` (#1168 AC5).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from aelfrice.feedback import apply_feedback
from aelfrice.store import MemoryStore

_WORKERS = 8
_PER_WORKER = 15  # 120 events total — fast, but far more than enough to race

# Distinct from the 5 s suite default so a hang is attributable.
_RACE_BUDGET_SECONDS = 30
# Well under _RACE_BUDGET_SECONDS: contention must resolve as a wait, not
# as a test timeout.
_BUSY_TIMEOUT_MS = 2000

_SENTENCE = "The build cache lives under var and is pruned on every release."


def _seed(db: Path) -> tuple[str, float, float]:
    """Insert one belief; return (id, alpha, beta)."""
    store = MemoryStore(str(db))
    try:
        from aelfrice.ingest import ingest_turn

        ingest_turn(store, _SENTENCE, "test:atomicity")
        bid = store.list_belief_ids()[0]
        b = store.get_belief(bid)
        assert b is not None
        return (bid, b.alpha, b.beta)
    finally:
        store.close()


def _race(db: Path, belief_id: str, valence: float) -> list[Exception]:
    """Run _WORKERS threads x _PER_WORKER feedback events. Returns errors."""
    barrier = threading.Barrier(_WORKERS)
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        store = MemoryStore(str(db))
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}"
        )
        try:
            barrier.wait()
            for i in range(_PER_WORKER):
                apply_feedback(
                    store=store,
                    belief_id=belief_id,
                    valence=valence,
                    source=f"race-{wid}-{i}",
                    propagate=False,
                )
        except Exception as exc:  # noqa: BLE001 - surfaced by the caller
            with lock:
                errors.append(exc)
        finally:
            store.close()

    threads = [
        threading.Thread(target=worker, args=(w,)) for w in range(_WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


@pytest.mark.timeout(_RACE_BUDGET_SECONDS)
def test_concurrent_positive_feedback_loses_no_evidence(tmp_path: Path) -> None:
    """Hypothesis: alpha moves by exactly one unit per event under contention.

    Falsifiable by any shortfall — which is what the pre-#1168
    read-modify-write produced — or by a raised exception."""
    db = tmp_path / "race.db"
    bid, alpha0, beta0 = _seed(db)
    total = _WORKERS * _PER_WORKER

    errors = _race(db, bid, valence=1.0)
    assert not errors, f"workers raised: {errors!r}"

    store = MemoryStore(str(db))
    try:
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0 + total)
        assert b.beta == pytest.approx(beta0)
        # And the append-only log agrees with the projection.
        assert len(store.list_feedback_events(belief_id=bid, limit=10_000)) == (
            total
        )
    finally:
        store.close()


@pytest.mark.timeout(_RACE_BUDGET_SECONDS)
def test_concurrent_negative_feedback_loses_no_evidence(tmp_path: Path) -> None:
    """Hypothesis: the beta half is equally conserved. Falsifiable by a
    shortfall on beta or by alpha moving at all."""
    db = tmp_path / "race_neg.db"
    bid, alpha0, beta0 = _seed(db)
    total = _WORKERS * _PER_WORKER

    errors = _race(db, bid, valence=-1.0)
    assert not errors, f"workers raised: {errors!r}"

    store = MemoryStore(str(db))
    try:
        b = store.get_belief(bid)
        assert b is not None
        assert b.beta == pytest.approx(beta0 + total)
        assert b.alpha == pytest.approx(alpha0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The store primitive
# ---------------------------------------------------------------------------


def test_bump_posterior_returns_post_values(tmp_path: Path) -> None:
    """Hypothesis: bump_posterior adds both deltas and returns the result.
    Falsifiable by a wrong return value or an unpersisted change."""
    db = tmp_path / "bump.db"
    bid, alpha0, beta0 = _seed(db)
    store = MemoryStore(str(db))
    try:
        out = store.bump_posterior(bid, 2.5, 0.25)
        assert out == (pytest.approx(alpha0 + 2.5), pytest.approx(beta0 + 0.25))
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0 + 2.5)
        assert b.beta == pytest.approx(beta0 + 0.25)
    finally:
        store.close()


def test_bump_posterior_unknown_id_returns_none(tmp_path: Path) -> None:
    """Hypothesis: an id that matches no row returns None rather than
    raising or silently succeeding. Falsifiable by either."""
    store = MemoryStore(str(tmp_path / "bump_missing.db"))
    try:
        assert store.bump_posterior("nope", 1.0, 0.0) is None
    finally:
        store.close()


def test_bump_posterior_does_not_touch_other_columns(tmp_path: Path) -> None:
    """Hypothesis: unlike update_belief, bump_posterior writes only the
    posterior — so it cannot revert a lock a concurrent writer committed.

    Falsifiable by any non-(alpha, beta) column changing."""
    db = tmp_path / "bump_narrow.db"
    bid, _, _ = _seed(db)
    store = MemoryStore(str(db))
    try:
        before = store.get_belief(bid)
        assert before is not None
        store.bump_posterior(bid, 1.0, 0.0)
        after = store.get_belief(bid)
        assert after is not None
        for column in (
            "content", "content_hash", "type", "lock_level", "locked_at",
            "created_at", "session_id", "origin", "retention_class",
            "valid_to", "scope", "project_context", "lock_tier",
        ):
            assert getattr(after, column) == getattr(before, column), column
    finally:
        store.close()


def test_feedback_write_path_never_writes_lock_columns(
    tmp_path: Path,
) -> None:
    """Hypothesis: the feedback path issues no statement that assigns a lock
    or provenance column on `beliefs`.

    This closes the whole-row clobber from #1168 structurally rather than
    probabilistically. The old path read a `Belief`, then wrote every
    column back from that snapshot; a concurrent `aelf lock` committing
    inside that window was silently reverted. Asserting on the SQL the
    path actually emits proves the window cannot exist, without depending
    on winning a race. Falsifiable by any `beliefs` UPDATE that assigns a
    column other than alpha/beta."""
    db = tmp_path / "columns.db"
    bid, _, _ = _seed(db)

    store = MemoryStore(str(db))
    statements: list[str] = []
    try:
        conn = store._conn  # pyright: ignore[reportPrivateUsage]
        conn.set_trace_callback(statements.append)
        try:
            apply_feedback(
                store=store,
                belief_id=bid,
                valence=1.0,
                source="column-audit",
                propagate=False,
                respect_lock=False,
            )
        finally:
            conn.set_trace_callback(None)
    finally:
        store.close()

    belief_writes = [
        s for s in statements
        if "update" in s.lower() and "beliefs" in s.lower()
        and "beliefs_fts" not in s.lower()
    ]
    assert belief_writes, "expected at least one beliefs UPDATE"
    forbidden = (
        "lock_level", "locked_at", "origin", "lock_tier", "valid_to",
        "content", "content_hash", "scope", "project_context",
        "retention_class", "session_id", "type",
    )
    for stmt in belief_writes:
        lowered = stmt.lower()
        for column in forbidden:
            assert f"{column} =" not in lowered and f"{column}=" not in lowered, (
                f"feedback write assigns {column!r}: {stmt}"
            )


# ---------------------------------------------------------------------------
# transaction(immediate=True)
# ---------------------------------------------------------------------------


def test_immediate_transaction_commits_as_one_unit(tmp_path: Path) -> None:
    """Hypothesis: an immediate-mode block still commits its writes once,
    at outermost exit. Falsifiable by an unpersisted write."""
    db = tmp_path / "imm.db"
    bid, alpha0, _ = _seed(db)
    store = MemoryStore(str(db))
    try:
        with store.transaction(immediate=True):
            store.bump_posterior(bid, 1.0, 0.0)
            store.insert_feedback_event(
                belief_id=bid, valence=1.0, source="t", created_at="2026-01-01",
            )
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0 + 1.0)
        assert len(store.list_feedback_events(belief_id=bid)) == 1
    finally:
        store.close()


def test_immediate_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    """Hypothesis: the audit row and the posterior move roll back together,
    so the log can never claim evidence the projection never took.
    Falsifiable by either surviving."""
    db = tmp_path / "imm_rollback.db"
    bid, alpha0, _ = _seed(db)
    store = MemoryStore(str(db))
    try:
        with pytest.raises(RuntimeError):
            with store.transaction(immediate=True):
                store.bump_posterior(bid, 5.0, 0.0)
                store.insert_feedback_event(
                    belief_id=bid, valence=5.0, source="t",
                    created_at="2026-01-01",
                )
                raise RuntimeError("boom")
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0)
        assert store.list_feedback_events(belief_id=bid) == []
    finally:
        store.close()


def test_immediate_is_safe_when_nested(tmp_path: Path) -> None:
    """Hypothesis: passing immediate=True inside an open transaction joins
    the outer one instead of raising "cannot start a transaction within a
    transaction". Falsifiable by an OperationalError."""
    db = tmp_path / "imm_nested.db"
    bid, alpha0, _ = _seed(db)
    store = MemoryStore(str(db))
    try:
        with store.transaction():
            store.bump_posterior(bid, 1.0, 0.0)
            with store.transaction(immediate=True):
                store.bump_posterior(bid, 1.0, 0.0)
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0 + 2.0)
    finally:
        store.close()


def test_immediate_is_safe_when_a_transaction_is_already_open(
    tmp_path: Path,
) -> None:
    """Hypothesis: an implicit transaction opened by a prior write does not
    make immediate=True raise. Falsifiable by an OperationalError."""
    db = tmp_path / "imm_open.db"
    bid, alpha0, _ = _seed(db)
    store = MemoryStore(str(db))
    try:
        conn = store._conn  # pyright: ignore[reportPrivateUsage]
        conn.execute("BEGIN")
        assert conn.in_transaction
        raised: sqlite3.OperationalError | None = None
        try:
            with store.transaction(immediate=True):
                store.bump_posterior(bid, 1.0, 0.0)
        except sqlite3.OperationalError as exc:
            raised = exc
        assert raised is None, (
            f"immediate=True raised inside an open txn: {raised}"
        )
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(alpha0 + 1.0)
    finally:
        store.close()
