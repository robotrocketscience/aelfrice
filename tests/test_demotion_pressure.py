"""Closes the v2.0 bug: demotion_pressure must be both written AND read.

In v2.0, the column existed and was updated, but no read path surfaced it.
v0.1.0 fixes this from day one by exposing demotion_pressure on Belief and
having get_belief return it. This test enforces that contract.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import Store


def _mk(dp: int) -> Belief:
    return Belief(
        id="b1",
        content="content",
        content_hash="h",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=dp,
        created_at="2026-04-25T00:00:00Z",
        last_retrieved_at=None,
    )


def test_demotion_pressure_initial_zero_persists() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk(0))
    got = s.get_belief("b1")
    assert got is not None
    assert got.demotion_pressure == 0


def test_demotion_pressure_update_to_three_reads_back() -> None:
    s = Store(":memory:")
    b = _mk(0)
    s.insert_belief(b)
    b.demotion_pressure = 3
    s.update_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.demotion_pressure == 3


def test_demotion_pressure_update_again_to_seven_reads_back() -> None:
    s = Store(":memory:")
    b = _mk(0)
    s.insert_belief(b)
    b.demotion_pressure = 3
    s.update_belief(b)
    b.demotion_pressure = 7
    s.update_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.demotion_pressure == 7


def test_demotion_pressure_inserted_nonzero_persists() -> None:
    """Belt-and-suspenders: insert path also persists nonzero values."""
    s = Store(":memory:")
    s.insert_belief(_mk(11))
    got = s.get_belief("b1")
    assert got is not None
    assert got.demotion_pressure == 11
