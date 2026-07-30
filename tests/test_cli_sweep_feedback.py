"""`aelf sweep-feedback` CLI subcommand (#191).

Round-trips the subcommand through the in-process parser to verify
flag plumbing, exit codes, and the cron-safe default behaviour.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import build_parser
from aelfrice.deferred_feedback import (
    RETRIEVAL_DRIVEN_FEEDBACK_SOURCE,
    enqueue_retrieval_exposures,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
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


@pytest.fixture
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at an isolated DB path under tmp_path."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _run(*args: str) -> tuple[int, str]:
    parser = build_parser()
    ns = parser.parse_args(["sweep-feedback", *args])
    out = io.StringIO()
    code: int = ns.func(ns, out)  # type: ignore[attr-defined]
    return code, out.getvalue()


def test_sweep_feedback_subcommand_registered() -> None:
    from aelfrice.cli import _known_cli_subcommands
    assert "sweep-feedback" in _known_cli_subcommands()


def test_sweep_feedback_empty_queue_exits_zero(store_path: Path) -> None:
    # Initialize empty store at the path.
    s = MemoryStore(str(store_path))
    s.close()
    code, output = _run()
    assert code == 0
    assert "would_apply=0 would_cancel=0" in output


def test_sweep_feedback_reports_but_changes_nothing(store_path: Path) -> None:
    """The #1162 acceptance criterion, both halves.

    Running the sweeper on a store with eligible rows must change no
    belief's alpha — and must still *report* a non-zero eligible count.
    The second half is the negative control: an audit-only sweeper that
    quietly reported zero would satisfy the first assertion while
    hiding that the queue had stopped being read at all.
    """
    s = MemoryStore(str(store_path))
    s.insert_belief(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now="2026-04-28T00:00:00Z")
    s.close()
    # grace=0 means everything is immediately eligible.
    code, output = _run("--grace-seconds", "0", "--epsilon", "0.10")
    assert code == 0
    assert "would_apply=1" in output
    assert "alpha_withheld=0.1000" in output
    assert "no alpha changed" in output
    s2 = MemoryStore(str(store_path))
    try:
        b = s2.get_belief("b1")
        assert b is not None and b.alpha == 1.0, "the sweeper moved alpha"
        assert s2.list_feedback_events(belief_id="b1") == []
        # The row is still enqueued: nothing was consumed, so a second
        # run reports the same number rather than draining to zero.
        assert s2.count_deferred_feedback_by_status() == {"enqueued": 1}
    finally:
        s2.close()
    code2, output2 = _run("--grace-seconds", "0", "--epsilon", "0.10")
    assert code2 == 0
    assert "would_apply=1" in output2


def test_sweep_feedback_gc_drops_only_the_banked_rows(
    store_path: Path,
) -> None:
    """--gc is the one destructive action, and it is never implicit."""
    s = MemoryStore(str(store_path))
    s.insert_belief(_mk("b1"))
    s.insert_belief(_mk("b2"))
    enqueue_retrieval_exposures(s, ["b1", "b2"], now="2026-04-28T00:00:00Z")
    # An 'applied' row from a sweep that really did run, back when the
    # sweeper mutated. That trail must survive the collector.
    s._conn.execute(  # noqa: SLF001 - fixture reaches for the trail directly
        "UPDATE deferred_feedback_queue SET status='applied' "
        "WHERE belief_id = 'b2'"
    )
    s._conn.commit()  # noqa: SLF001
    s.close()

    code, output = _run("--grace-seconds", "0")
    assert code == 0
    assert "--gc" not in output, "gc ran without being asked"
    s_mid = MemoryStore(str(store_path))
    assert s_mid.count_deferred_feedback_by_status() == {
        "enqueued": 1, "applied": 1,
    }
    s_mid.close()

    code, output = _run("--grace-seconds", "0", "--gc")
    assert code == 0
    assert "deleted 1 banked enqueued row(s)" in output
    s2 = MemoryStore(str(store_path))
    try:
        assert s2.count_deferred_feedback_by_status() == {"applied": 1}
    finally:
        s2.close()

    # Idempotent.
    code, output = _run("--grace-seconds", "0", "--gc")
    assert code == 0
    assert "deleted 0 banked enqueued row(s)" in output


def test_gc_deletes_only_what_the_run_reported_on(store_path: Path) -> None:
    """The destructive verb must not outscope the report that justifies
    it. With `--limit 3` over 8 eligible rows, the audit describes 3 and
    `--gc` removes those 3 — not all 8 — and the remainder is called out
    rather than silently dropped or silently kept.
    """
    s = MemoryStore(str(store_path))
    for i in range(8):
        s.insert_belief(_mk(f"b{i}"))
    enqueue_retrieval_exposures(
        s, [f"b{i}" for i in range(8)], now="2026-04-28T00:00:00Z"
    )
    s.close()

    code, output = _run("--grace-seconds", "0", "--limit", "3", "--gc")
    assert code == 0
    assert "would_apply=3" in output
    assert "deleted 3 banked enqueued row(s)" in output
    assert "pending_beyond_limit=5" in output
    assert "5 eligible row(s) past --limit (3)" in output

    s2 = MemoryStore(str(store_path))
    try:
        assert s2.count_deferred_feedback_by_status() == {"enqueued": 5}
    finally:
        s2.close()

    # Re-running reaches the next page rather than re-reporting the same
    # one — the property the audit-only design is supposed to buy.
    code, output = _run("--grace-seconds", "0", "--limit", "3", "--gc")
    assert code == 0
    assert "deleted 3 banked enqueued row(s)" in output
    s3 = MemoryStore(str(store_path))
    try:
        assert s3.count_deferred_feedback_by_status() == {"enqueued": 2}
    finally:
        s3.close()


def test_sweep_feedback_strict_flag_propagates_failure(
    store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the sweep to raise.
    import aelfrice.cli as cli_mod

    def boom(*a, **k):
        raise RuntimeError("simulated")

    monkeypatch.setattr(
        "aelfrice.deferred_feedback.sweep_deferred_feedback", boom
    )
    code, _ = _run("--strict")
    assert code == 1
    # Without --strict, exits 0 even on internal error.
    code2, _ = _run()
    assert code2 == 0
