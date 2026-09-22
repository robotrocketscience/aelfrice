"""Token-budget property: L1 trims to fit; L0 always survives.

The retrieve() function returns L0 in full regardless of token budget
(locked beliefs are user-asserted ground truth) and trims L1 from the
tail until the cumulative ~4-chars-per-token estimate is at or below
the budget.

Three properties asserted:
1. Output total tokens never exceed budget when L0 alone fits under it
2. L0 always returned in full even when L0 alone exceeds the budget
   (and L1 is then empty in the output)
3. On an EQUAL-PRICED fixture, a tighter budget returns no more results.

Property 3 used to be stated without the qualifier, as general budget
monotonicity. It is false in general (#1574): `pack_with_clusters` skips
an over-budget belief and keeps filling, so the budget selects rather
than truncating, and one extra token can admit a dear belief that evicts
several cheap ones. It held here only because every belief in
`_store_with_n_facts` is `body[:content_len]` and therefore identically
priced, which makes the greedy fill a prefix — the same equal-cost shape
that hid the defect in the census fixture. Both arms are now pinned.
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


def test_equal_priced_beliefs_make_the_count_monotone_in_the_budget() -> None:
    """A property of THIS fixture, not of the packer (#1574).

    Every belief here is `body[:content_len]`, so all of them cost the
    same. Under equal prices the greedy fill is a prefix, and a prefix
    cannot evict: a tighter budget really does return no more results.

    This was previously asserted as a general monotonicity property of
    the budget. It is not one — see
    `test_a_tighter_budget_can_return_more_results_when_prices_differ`.
    The equal-cost fixture is exactly what kept the false version green,
    the same shape #1574 found in the census fixture.
    """
    s = _store_with_n_facts(30, content_len=80)
    prices = {
        b.id: _estimate(b)
        for b in retrieve(s, query="fact", token_budget=10**9, l1_limit=30)
    }
    assert len(set(prices.values())) == 1, (
        f"this test only holds while the fixture is equal-priced: {prices}"
    )

    counts: list[int] = []
    for budget in [1000, 500, 200, 50, 20]:
        hits = retrieve(s, query="fact", token_budget=budget, l1_limit=30)
        counts.append(len(hits))
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], f"non-monotonic: counts={counts}"


def test_a_tighter_budget_can_return_more_results_when_prices_differ() -> None:
    """The general claim, and it is false (#1574).

    `clustering.pack_with_clusters` skips an over-budget belief and keeps
    filling, so the budget selects rather than truncating. A budget too
    small for a dear top-ranked belief spends itself on several cheaper
    ones; one more token admits the dear belief and evicts them all, and
    the output gets SHORTER as the budget GROWS.
    """
    store = MemoryStore(":memory:")
    try:
        dear = _mk("dear", "tomato staked " * 12 + "gardening notes season")
        cheap = [
            _mk("c0", "tomato staked SFO"),
            _mk("c1", "tomato staked JFK"),
        ]
        for b in [dear, *cheap]:
            store.insert_belief(b)

        pool = [
            b.id
            for b in retrieve(store, query="tomato staked", token_budget=10**9)
        ]
        assert pool[0] == "dear", (
            f"the dear belief must rank first for this to bite: {pool}"
        )

        drops: list[tuple[int, int, int]] = []
        previous: tuple[int, int] | None = None
        for budget in range(1, 120):
            n = len(retrieve(store, query="tomato staked", token_budget=budget))
            if previous is not None and n < previous[1]:
                drops.append((previous[0], budget, previous[1] - n))
            previous = (budget, n)

        assert drops, (
            "raising the budget never shortened the output, so either the "
            "packer became monotone or this fixture stopped pricing its "
            "beliefs unequally"
        )
    finally:
        store.close()


def test_zero_budget_with_no_locks_returns_empty() -> None:
    """Zero budget, no locked beliefs: L0 empty + L1 trimmed entirely."""
    s = _store_with_n_facts(5, content_len=80)
    hits = retrieve(s, query="fact", token_budget=0, l1_limit=5)
    assert hits == []
