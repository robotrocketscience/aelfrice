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
    _belief_tokens,
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


def test_floor_engages_at_moderate_lock_load() -> None:
    """#1023: with the 0.5 fraction the floor engages once locks exceed
    50% of the budget (not only at >75%). Locks at ~60% of budget plus
    abundant relevant content -> relevance is reserved (several hits) and
    total output exceeds the nominal budget by up to the floor."""
    assert RELEVANCE_BUDGET_FLOOR_FRACTION >= 0.5
    s = MemoryStore(":memory:")
    # Each padded belief ~150 tok. ~10 locks ~= 1500 tok ~= 62% of 2400.
    locks = [_mk(f"L{i}", f"unrelated locked fact topic alpha {i}", locked=True)
             for i in range(10)]
    for b in locks:
        s.insert_belief(b)
    for i in range(12):
        s.insert_belief(
            _mk(f"T{i}", f"kubernetes deployment rollout pods replicas note {i}", locked=False)
        )
    # Pin the fixture to the MODERATE regime: 50% < locked < 75% of budget,
    # the window where the floor newly engages at 0.5 but not at 0.25.
    locked_tokens = sum(_belief_tokens(b) for b in locks)
    assert 0.5 * DEFAULT_TOKEN_BUDGET < locked_tokens < 0.75 * DEFAULT_TOKEN_BUDGET
    hits = retrieve(s, "kubernetes deployment rollout pods", token_budget=DEFAULT_TOKEN_BUDGET)
    relevant = [b for b in hits if b.id.startswith("T")]
    assert len(relevant) >= 2
    # Distinguishes 0.5 from 0.25: at 0.25 the floor would NOT engage here
    # (locks ~62% < 75%) so the cap holds at the budget; at 0.5 it engages
    # and total overflows by up to floor(0.5 * budget).
    total = sum(_belief_tokens(b) for b in hits)
    assert total > DEFAULT_TOKEN_BUDGET


def test_floor_fraction_sane() -> None:
    assert 0.0 < RELEVANCE_BUDGET_FLOOR_FRACTION < 1.0
