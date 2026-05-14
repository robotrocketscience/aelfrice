"""Tests for #796 R4 rank-overlap metrics — ordered_top_k_overlap and
rank_biased_overlap.

Properties under test (per the operator-mandated acceptance):

* ``ordered_top_k_overlap`` is 1.0 on identical prefixes and 0.0 on a
  full reverse of an even-length prefix. Linear in the count of
  position-wise matches.
* ``rank_biased_overlap`` (RBO_EXT) is 1.0 on identical equal-length
  lists, 0.0 on disjoint lists, and monotone non-decreasing as the
  shared prefix length grows.
* Both are deterministic across calls.
* Empty-list edge cases hold (both empty → 1.0 for RBO, 0.0 for
  ordered_top_k; one empty → 0.0 for RBO).

These two metrics widen the eval panel so the γ-vs-log-additive
bench (and any future A/B retrieval comparison) can discriminate top-
of-list churn from middle-of-list reorderings — the R4 finding that
PR@K + Spearman ρ alone cannot.
"""
from __future__ import annotations

import pytest

from aelfrice.calibration_metrics import (
    ordered_top_k_overlap,
    rank_biased_overlap,
)
from aelfrice.eval_harness import (
    DEFAULT_RBO_PERSISTENCE,
    compare_ranking_panel,
    format_ranking_comparison,
)


# ---------------------------------------------------------------------------
# ordered_top_k_overlap
# ---------------------------------------------------------------------------

def test_otk_identical_prefix_is_one() -> None:
    assert ordered_top_k_overlap([1, 2, 3], [1, 2, 3], 3) == 1.0


def test_otk_reversed_even_prefix_is_zero() -> None:
    """An even-length full reverse has no position-wise match."""
    assert ordered_top_k_overlap([1, 2, 3, 4], [4, 3, 2, 1], 4) == 0.0


def test_otk_reversed_odd_prefix_keeps_middle() -> None:
    """An odd-length reverse fixes the middle element; score = 1/k."""
    assert ordered_top_k_overlap([1, 2, 3], [3, 2, 1], 3) == 1.0 / 3.0


def test_otk_disjoint_prefix_is_zero() -> None:
    assert ordered_top_k_overlap([1, 2, 3], [4, 5, 6], 3) == 0.0


def test_otk_shorter_input_counts_missing_as_mismatch() -> None:
    assert ordered_top_k_overlap([1, 2, 3], [], 3) == 0.0
    assert ordered_top_k_overlap([1, 2, 3], [1], 3) == pytest.approx(1.0 / 3.0)


def test_otk_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        ordered_top_k_overlap([1, 2, 3], [1, 2, 3], 0)
    with pytest.raises(ValueError):
        ordered_top_k_overlap([1, 2, 3], [1, 2, 3], -1)


def test_otk_linear_in_matches() -> None:
    """k=4, three matching positions → 3/4."""
    assert ordered_top_k_overlap(
        [1, 2, 3, 4], [1, 2, 3, 99], 4,
    ) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# rank_biased_overlap (RBO_EXT)
# ---------------------------------------------------------------------------

def test_rbo_identical_equal_length_is_one() -> None:
    assert rank_biased_overlap([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Single-element identical also caps at 1.0
    assert rank_biased_overlap([1], [1]) == pytest.approx(1.0)


def test_rbo_disjoint_is_zero() -> None:
    assert rank_biased_overlap([1, 2, 3], [4, 5, 6]) == 0.0


def test_rbo_both_empty_is_one() -> None:
    assert rank_biased_overlap([], []) == 1.0


def test_rbo_one_empty_is_zero() -> None:
    assert rank_biased_overlap([], [1, 2, 3]) == 0.0
    assert rank_biased_overlap([1, 2, 3], []) == 0.0


def test_rbo_monotone_in_prefix_agreement() -> None:
    """Extending a shared prefix never lowers the RBO score."""
    no_overlap = rank_biased_overlap([1, 2, 3], [9, 9, 9])
    one_overlap = rank_biased_overlap([1, 2, 3], [1, 9, 9])
    two_overlap = rank_biased_overlap([1, 2, 3], [1, 2, 9])
    three_overlap = rank_biased_overlap([1, 2, 3], [1, 2, 3])
    assert no_overlap <= one_overlap <= two_overlap <= three_overlap
    assert three_overlap == pytest.approx(1.0)


def test_rbo_top_swap_costs_more_than_tail_swap() -> None:
    """A swap at rank 1 hurts RBO more than the same swap deep in the list.

    This is the load-bearing property — γ vs log-additive can have
    identical PR@5 + ρ but different top-K ordering, and RBO at
    p=0.9 picks that up.
    """
    base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    top_swap = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    tail_swap = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]
    top_score = rank_biased_overlap(base, top_swap, p=0.9)
    tail_score = rank_biased_overlap(base, tail_swap, p=0.9)
    assert top_score < tail_score < 1.0


def test_rbo_invalid_p_raises() -> None:
    with pytest.raises(ValueError):
        rank_biased_overlap([1], [1], p=0.0)
    with pytest.raises(ValueError):
        rank_biased_overlap([1], [1], p=1.0)


def test_rbo_deterministic_across_calls() -> None:
    a = [1, 2, 3, 4, 5]
    b = [1, 3, 2, 5, 4]
    first = rank_biased_overlap(a, b, p=0.9)
    second = rank_biased_overlap(a, b, p=0.9)
    assert first == second


# ---------------------------------------------------------------------------
# Eval-harness panel
# ---------------------------------------------------------------------------

def test_compare_ranking_panel_averages_correctly() -> None:
    pairs = [
        ([1, 2, 3], [1, 2, 3]),  # identical
        ([1, 2, 3], [3, 2, 1]),  # reversed
    ]
    r = compare_ranking_panel(pairs, k=3, p=DEFAULT_RBO_PERSISTENCE)
    # ordered_top_k mean: (1.0 + 1/3) / 2 = 2/3
    assert r.ordered_top_k == pytest.approx(2.0 / 3.0)
    # RBO mean: (1.0 + rbo([1,2,3],[3,2,1])) / 2; just bound-check.
    assert 0.0 <= r.rbo <= 1.0
    assert r.n_queries == 2
    assert r.k == 3
    assert r.p == DEFAULT_RBO_PERSISTENCE


def test_compare_ranking_panel_empty_pairs_raises() -> None:
    with pytest.raises(ValueError):
        compare_ranking_panel([], k=3)


def test_format_ranking_comparison_is_stable_text() -> None:
    pairs = [([1, 2, 3], [1, 2, 3])]
    r = compare_ranking_panel(pairs, k=3, p=0.9)
    text = format_ranking_comparison(r)
    assert "rank-comparison panel" in text
    assert "n_queries:    1" in text
    assert "ordered_top_k@3:  1.0000" in text
