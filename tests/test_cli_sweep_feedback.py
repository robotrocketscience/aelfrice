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
    assert "applied=0 cancelled=0" in output


def test_sweep_feedback_applies_with_grace_zero(store_path: Path) -> None:
    s = MemoryStore(str(store_path))
    s.insert_belief(_mk("b1"))
    enqueue_retrieval_exposures(s, ["b1"], now="2026-04-28T00:00:00Z")
    s.close()
    # grace=0 means everything is immediately eligible.
    code, output = _run("--grace-seconds", "0", "--epsilon", "0.10")
    assert code == 0
    assert "applied=1" in output
    s2 = MemoryStore(str(store_path))
    try:
        b = s2.get_belief("b1")
        assert b is not None and b.alpha == 1.10
        events = s2.list_feedback_events(belief_id="b1")
        assert any(
            e.source == RETRIEVAL_DRIVEN_FEEDBACK_SOURCE for e in events
        )
    finally:
        s2.close()


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
