"""Token-budget property: L1 trims to fit; L0 always survives.

The retrieve() function returns L0 in full regardless of token budget
(locked beliefs are user-asserted ground truth) and trims L1 from the
tail until the cumulative ~4-chars-per-token estimate is at or below
the budget.

Three properties asserted:
1. Output total tokens never exceed budget when L0 alone fits under it
2. L0 always returned in full even when L0 alone exceeds the budget
   (and L1 is then empty in the output)
3. Tighter budget never returns more L1 results than a looser budget
   (monotonicity)
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

_CHARS_PER_TOKEN = 4.0


def _estimate(b: Belief) -> int:
    n = len(b.content)
    if n == 0:
        return 0
    return int((n + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _total_tokens(beliefs: list[Belief]) -> int:
    return sum(_estimate(b) for b in beliefs)


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


def _store_with_n_facts(n: int, content_len: int = 200) -> MemoryStore:
    """N unlocked beliefs, each with content_len chars matching 'fact'."""
    s = MemoryStore(":memory:")
    body = "fact " * (content_len // 5)
    for i in range(n):
        s.insert_belief(_mk(f"F{i}", body[:content_len]))
    return s


def test_output_tokens_at_or_below_budget_when_l0_fits() -> None:
    """20 unlocked facts, no locks, budget 100 tokens: output stays under."""
    s = _store_with_n_facts(20, content_len=80)  # ~20 tokens each
    hits = retrieve(s, query="fact", token_budget=100, l1_limit=20)
    assert _total_tokens(hits) <= 100
    # Should include several L1 results (each ~20 tokens, budget 100 -> ~5).
    assert 1 <= len(hits) <= 6


def test_l0_returned_in_full_even_when_l0_alone_exceeds_budget() -> None:
    """Three locked beliefs, each ~50 tokens; budget 10. All three survive,
    L1 is empty in the output."""
    s = MemoryStore(":memory:")
    big_content = "x" * 200  # ~50 tokens
    s.insert_belief(_mk("L1", big_content, lock_level=LOCK_USER,
                        locked_at="2026-04-26T03:00:00Z"))
    s.insert_belief(_mk("L2", big_content, lock_level=LOCK_USER,
                        locked_at="2026-04-26T02:00:00Z"))
    s.insert_belief(_mk("L3", big_content, lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("F1", big_content))  # would be L1 hit but no budget left
    s.insert_belief(_mk("F2", big_content))

    hits = retrieve(s, query="x", token_budget=10, l1_limit=20)
    locked_ids = {h.id for h in hits if h.lock_level == LOCK_USER}
    unlocked_ids = {h.id for h in hits if h.lock_level == LOCK_NONE}
    assert locked_ids == {"L1", "L2", "L3"}, "L0 must survive in full"
    assert unlocked_ids == set(), "L1 must be empty when L0 exhausts budget"


def test_tighter_budget_returns_no_more_results_than_looser() -> None:
    """Monotonicity: as budget shrinks, output length is non-increasing."""
    s = _store_with_n_facts(30, content_len=80)
    counts: list[int] = []
    for budget in [1000, 500, 200, 50, 20]:
        hits = retrieve(s, query="fact", token_budget=budget, l1_limit=30)
        counts.append(len(hits))
    # Each subsequent (tighter) budget must yield <= prior count.
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], (
            f"non-monotonic: counts={counts}"
        )


def test_zero_budget_with_no_locks_returns_empty() -> None:
    """Zero budget, no locked beliefs: L0 empty + L1 trimmed entirely."""
    s = _store_with_n_facts(5, content_len=80)
    hits = retrieve(s, query="fact", token_budget=0, l1_limit=5)
    assert hits == []
