"""L0 locked beliefs are always positioned ahead of L1 BM25 results.

Property: in the retrieve() output list, every locked belief's index is
strictly less than every non-locked belief's index. Order within each
layer is independently determined (locked: locked_at DESC; L1: BM25
relevance) but the layer boundary is hard.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _populate_mixed_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    # 3 locked beliefs (L0), 5 unlocked beliefs (L1 candidates).
    s.insert_belief(_mk("L_a", "the user pinned a strong opinion about coffee",
                        lock_level=LOCK_USER, locked_at="2026-04-26T03:00:00Z"))
    s.insert_belief(_mk("L_b", "another locked truth about coffee beans",
                        lock_level=LOCK_USER, locked_at="2026-04-26T02:00:00Z"))
    s.insert_belief(_mk("L_c", "a locked belief about something unrelated",
                        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("F_1", "morning coffee is critical to productivity"))
    s.insert_belief(_mk("F_2", "afternoon coffee is overrated"))
    s.insert_belief(_mk("F_3", "tea is also a valid morning beverage"))
    s.insert_belief(_mk("F_4", "water hydration matters more than coffee"))
    s.insert_belief(_mk("F_5", "the kitchen has a coffee maker"))
    return s


def test_locked_beliefs_appear_before_any_unlocked_match() -> None:
    s = _populate_mixed_store()
    hits = retrieve(s, query="coffee", token_budget=10_000)

    # All three locked beliefs should be present and contiguous at the head.
    locked_ids = {"L_a", "L_b", "L_c"}
    indices_locked: list[int] = [
        i for i, b in enumerate(hits) if b.id in locked_ids
    ]
    indices_unlocked: list[int] = [
        i for i, b in enumerate(hits) if b.id not in locked_ids
    ]

    assert len(indices_locked) == 3, f"expected all 3 locked, got {indices_locked}"
    # Every locked index must precede every unlocked index.
    if indices_unlocked:
        assert max(indices_locked) < min(indices_unlocked), (
            f"layer boundary violated: locked at {indices_locked}, "
            f"unlocked at {indices_unlocked}"
        )


def test_l0_ordered_by_locked_at_desc() -> None:
    """Within L0 the most recently locked belief comes first."""
    s = _populate_mixed_store()
    hits = retrieve(s, query="coffee", token_budget=10_000)
    locked_in_order = [b.id for b in hits if b.lock_level == LOCK_USER]
    # locked_at: L_a 03:00, L_b 02:00, L_c 01:00 -> DESC means L_a, L_b, L_c.
    assert locked_in_order == ["L_a", "L_b", "L_c"]


def test_locked_first_holds_when_query_matches_no_unlocked() -> None:
    """Even with zero L1 results, L0 layer order is preserved."""
    s = _populate_mixed_store()
    hits = retrieve(s, query="zzznomatchquery", token_budget=10_000)
    ids = [b.id for b in hits]
    assert ids == ["L_a", "L_b", "L_c"]
