"""Tests for benchmarks.retrieval_metrics and benchmarks.metric_status.

These cover the reader-independent half of the #1160 metric separation:
rank-based scores over the retrieved list, and the sentinel an adapter
writes when a metric cannot be computed at all.

Issue: #1160.
"""
from __future__ import annotations

import pytest

from benchmarks import metric_status, retrieval_metrics as rm

# A three-item ranking whose only gold-bearing item sits at rank 2.
RANKING: list[str] = [
    "Discussed the quarterly roadmap with the team.",
    "The user's home airport is SFO.",
    "Weather was clear all week.",
]
GOLD: list[str] = ["SFO"]


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


def test_shrinking_the_budget_never_raises_a_metric():
    """The property that makes these metrics readable where token-F1 is not.

    Retrieval fills the budget in rank order, so a smaller budget
    truncates the tail. Token-F1 over the joined blob *rises* under that
    truncation (precision improves as the denominator shrinks); every
    metric here is monotone non-decreasing in the number of items kept.
    """
    ranking = ["noise"] * 9 + ["the answer is SFO"] + ["more noise"] * 5
    full = rm.retrieval_metrics(ranking, GOLD)
    for cut in range(len(ranking), 0, -1):
        truncated = rm.retrieval_metrics(ranking[:cut], GOLD)
        for key, value in truncated.items():
            assert value <= full[key], f"{key} rose when the budget shrank"


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
