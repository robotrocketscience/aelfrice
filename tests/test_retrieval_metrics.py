"""Tests for benchmarks.retrieval_metrics and benchmarks.metric_status.

These cover the reader-independent half of the #1160 metric separation:
rank-based scores over the retrieved list, and the sentinel an adapter
writes when a metric cannot be computed at all.

Issue: #1160.
"""
from __future__ import annotations

import pytest

from aelfrice import retrieval
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.store import MemoryStore
from benchmarks import metric_status, retrieval_metrics as rm

# A three-item ranking whose only gold-bearing item sits at rank 2.
RANKING: list[str] = [
    "Discussed the quarterly roadmap with the team.",
    "The user's home airport is SFO.",
    "Weather was clear all week.",
]
GOLD: list[str] = ["SFO"]


def _budget_belief(bid: str, content: str) -> Belief:
    """A belief whose retrieval cost follows from its content length."""
    return Belief(
        id=bid,
        content=content,
        content_hash=f"t_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


def test_gold_ranks_are_one_indexed():
    assert rm.gold_ranks(RANKING, GOLD) == [2]


def test_gold_ranks_reports_every_hit_in_order():
    ranking = ["SFO terminal 2", "unrelated", "flew out of SFO"]
    assert rm.gold_ranks(ranking, GOLD) == [1, 3]


def test_gold_ranks_empty_when_answer_absent():
    assert rm.gold_ranks(RANKING, ["JFK"]) == []


def test_reciprocal_rank_is_one_over_first_hit():
    assert rm.reciprocal_rank(RANKING, GOLD) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_answer_absent():
    assert rm.reciprocal_rank(RANKING, ["JFK"]) == 0.0


def test_recall_at_k_boundary_is_inclusive():
    """The hit is at rank 2: k=2 finds it, k=1 does not."""
    assert rm.recall_at_k(RANKING, GOLD, 1) == 0.0
    assert rm.recall_at_k(RANKING, GOLD, 2) == 1.0


def test_recall_at_k_below_one_inspects_nothing():
    assert rm.recall_at_k(RANKING, GOLD, 0) == 0.0


def test_any_gold_surface_counts_as_a_hit():
    """Multi-answer gold lists alternative surfaces, not separate facts."""
    assert rm.recall_at_k(RANKING, ["San Francisco", "SFO"], 5) == 1.0


def test_gold_that_normalises_to_empty_does_not_score():
    """Without the empty-gold guard, `"" in anything` awards a free 1.0.

    `normalize_answer` drops articles and punctuation, so a gold surface
    of "the" or "." normalises away entirely. Same defect
    `qa_scoring.score_substring_exact_match` guards at its own entry.
    """
    assert rm.reciprocal_rank(RANKING, ["the"]) == 0.0
    assert rm.recall_at_k(RANKING, ["."], 20) == 0.0
    # And the guard is per-surface, not all-or-nothing: a real surface
    # alongside a degenerate one still scores.
    assert rm.recall_at_k(RANKING, ["the", "SFO"], 5) == 1.0


def test_keeping_fewer_items_never_raises_a_metric():
    """The property that makes these metrics readable where token-F1 is not.

    Token-F1 over the joined blob *rises* when the list is truncated,
    because precision improves as the denominator shrinks. Every metric
    here is monotone non-decreasing in the number of items kept.

    Renamed from `test_shrinking_the_budget_never_raises_a_metric`
    (#1574). It truncates a list; it never sets a budget, prices a
    belief, or calls the packer. The old name imported a claim about the
    *budget* into a test of a claim about *items kept*, and the two come
    apart — see
    `test_raising_the_budget_can_lower_a_metric` below.
    """
    ranking = ["noise"] * 9 + ["the answer is SFO"] + ["more noise"] * 5
    full = rm.retrieval_metrics(ranking, GOLD)
    for cut in range(len(ranking), 0, -1):
        truncated = rm.retrieval_metrics(ranking[:cut], GOLD)
        for key, value in truncated.items():
            assert value <= full[key], f"{key} rose when items were dropped"


def test_raising_the_budget_can_lower_a_metric():
    """Monotone in items kept does NOT give monotone in budget (#1574).

    `benchmarks/retrieval_metrics.py` claimed it did, reasoning that
    "retrieval fills the budget in rank order, so cutting the budget
    truncates the tail". `clustering.pack_with_clusters` does not
    truncate: stage 1 abandons on the first unaffordable representative
    and stage 2 skips an over-budget belief and keeps filling. The budget
    therefore **selects**, and a budget too small for a dear irrelevant
    belief spends itself on a cheap relevant one.

    Here the gold-bearing belief is the cheap one and ranks second. One
    extra token lets the dear irrelevant belief in, which evicts it and
    takes every metric to zero. Driven through production `retrieve()`,
    not through the packer directly, because the claim being corrected is
    about the shipped path.
    """
    query = "tomato staked"
    store = MemoryStore(":memory:")
    try:
        dear = _budget_belief(
            "dear", "tomato staked " * 12 + "gardening notes for the season"
        )
        lean = _budget_belief("lean", "tomato staked SFO")
        for b in (dear, lean):
            store.insert_belief(b)

        costs = {b.id: retrieval._belief_tokens(b) for b in (dear, lean)}
        assert len(set(costs.values())) > 1, (
            f"the fixture must price its beliefs unequally: {costs}"
        )
        pool = [b.id for b in retrieval.retrieve(store, query, token_budget=10**9)]
        assert pool == ["dear", "lean"], pool

        def metrics_at(budget: int) -> dict[str, float]:
            hits = retrieval.retrieve(store, query, token_budget=budget)
            return rm.retrieval_metrics([b.content for b in hits], GOLD)

        inversions = []
        previous = None
        for budget in range(1, sum(costs.values()) + 3):
            current = metrics_at(budget)
            if previous is not None:
                for key, value in current.items():
                    if value < previous[1][key]:
                        inversions.append((previous[0], budget, key))
            previous = (budget, current)

        assert inversions, (
            "no budget increase lowered any metric, so either the packer "
            "became monotone in the budget or this fixture stopped pricing "
            f"its beliefs unequally: {costs}"
        )
    finally:
        store.close()


def test_retrieval_metrics_reports_every_default_cutoff():
    out = rm.retrieval_metrics(RANKING, GOLD)
    assert set(out) == {"reciprocal_rank"} | {
        f"recall_at_{k}" for k in rm.DEFAULT_KS
    }


def test_mean_metrics_renames_reciprocal_rank_to_mrr():
    per_query = [
        rm.retrieval_metrics(RANKING, GOLD),      # rr = 0.5
        rm.retrieval_metrics(RANKING, ["JFK"]),   # rr = 0.0
    ]
    out = rm.mean_metrics(per_query)
    assert out["mrr"] == pytest.approx(0.25)
    assert "reciprocal_rank" not in out


def test_mean_metrics_keeps_its_shape_on_an_empty_run():
    """A leaf that appears or vanishes between runs breaks the band-check."""
    out = rm.mean_metrics([])
    assert set(out) == {"mrr"} | {f"recall_at_{k}" for k in rm.DEFAULT_KS}
    assert all(v == 0.0 for v in out.values())


def test_metrics_are_deterministic():
    assert rm.retrieval_metrics(RANKING, GOLD) == rm.retrieval_metrics(
        RANKING, GOLD,
    )


def test_not_applicable_sentinel_round_trips():
    assert metric_status.is_not_applicable(metric_status.NOT_APPLICABLE)
    assert metric_status.is_not_applicable("N/A")
    assert metric_status.is_not_applicable("  n/a  ")


def test_not_applicable_rejects_numbers_and_other_strings():
    """0.0 is the value the sentinel exists to stop being written."""
    assert not metric_status.is_not_applicable(0.0)
    assert not metric_status.is_not_applicable(0)
    assert not metric_status.is_not_applicable("")
    assert not metric_status.is_not_applicable("na")
    assert not metric_status.is_not_applicable(None)
