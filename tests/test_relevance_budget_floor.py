"""Reserved relevance-budget floor: locked beliefs (uncapped, never
trimmed per #379) must not starve query-relevant L2.5/L1 hits to zero.

Regression for the lock-saturation bug: a store whose locks alone meet or
exceed the token budget returned ONLY the locks for every prompt (observed
on a real 12-lock store: 2485 lock tokens vs a 2400 budget -> 0 relevance
tokens). The floor reserves a slice for relevance; it is a strict no-op
when the locks leave at least that much room.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    RELEVANCE_BUDGET_FLOOR_FRACTION,
    retrieve,
)
from aelfrice.store import MemoryStore

_PAD = " ".join(["context"] * 120)  # ~120 tokens -> realistic ~150-token beliefs


def _mk(bid: str, content: str, *, locked: bool, pad: bool = True) -> Belief:
    return Belief(
        id=bid,
        content=(content + " " + _PAD) if pad else content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-01-01T00:00:00Z" if locked else None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
    )


def test_saturating_locks_do_not_starve_relevance() -> None:
    """20 long locks overflow the default budget; a query-relevant
    non-lock belief must still surface (was 0 before the floor)."""
    s = MemoryStore(":memory:")
    for i in range(20):
        s.insert_belief(_mk(f"L{i}", f"unrelated locked fact about topic alpha {i}", locked=True))
    for i in range(3):
        s.insert_belief(_mk(f"T{i}", f"kubernetes deployment rollout pods replicas note {i}", locked=False))
    hits = retrieve(s, "kubernetes deployment rollout", token_budget=DEFAULT_TOKEN_BUDGET)
    relevant = [b for b in hits if b.id.startswith("T")]
    assert relevant, "relevance floor must surface >=1 query-relevant belief under lock saturation"


def test_floor_is_noop_when_locks_fit() -> None:
    """With locks that fit comfortably, every query-relevant belief
    surfaces exactly as before — the floor must not change the fit
    regime."""
    s = MemoryStore(":memory:")
    # 2 short locks (tiny token cost), 3 relevant non-lock beliefs.
    for i in range(2):
        s.insert_belief(_mk(f"L{i}", f"short lock {i}", locked=True, pad=False))
    for i in range(3):
        s.insert_belief(_mk(f"T{i}", f"kubernetes deployment rollout pods note {i}", locked=False, pad=False))
    hits = retrieve(s, "kubernetes deployment rollout", token_budget=DEFAULT_TOKEN_BUDGET)
    relevant = [b for b in hits if b.id.startswith("T")]
    assert len(relevant) == 3, "all relevant beliefs should surface when locks fit"


def test_floor_fraction_sane() -> None:
    assert 0.0 < RELEVANCE_BUDGET_FLOOR_FRACTION < 1.0
