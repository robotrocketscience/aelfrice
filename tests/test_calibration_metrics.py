"""Tests for `aelfrice.calibration_metrics` (#365 R1).

Cover:
- Precision@K basic / clamped / over-K input.
- ROC-AUC: monotonic separation, perfect/inverted ranking, ties,
  single-class undefined → None.
- Spearman ρ: monotonic, ties, zero-variance undefined → None,
  short inputs.
- Determinism: bytes-identical return across reruns of the same input
  (the ship gate per #365).
"""
from __future__ import annotations

import math

import pytest

from aelfrice.calibration_metrics import (
    precision_at_k,
    roc_auc,
    spearman_rho,
)


# ---- precision_at_k ----------------------------------------------------


def test_precision_at_k_basic() -> None:
    assert precision_at_k([True, False, True, False, True], 5) == 0.6


def test_precision_at_k_truncates_to_k() -> None:
    # Only top-3 counted even if input has more.
    assert precision_at_k([True, True, True, False, False], 3) == 1.0


def test_precision_at_k_pads_short_input_with_not_relevant() -> None:
    # Input shorter than k -> missing slots count as 0.
    assert precision_at_k([True, True], 5) == 2 / 5


def test_precision_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k([True], 0)
    with pytest.raises(ValueError):
        precision_at_k([True], -1)


# ---- roc_auc -----------------------------------------------------------


def test_roc_auc_perfect_ranking() -> None:
    # Higher score -> higher label probability, no overlap.
    scores = [3.0, 2.0, 1.0]
    labels = [True, False, False]
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_inverted_ranking() -> None:
    # All positives at the bottom of the score order.
    scores = [3.0, 2.0, 1.0]
    labels = [False, False, True]
    assert roc_auc(scores, labels) == 0.0


def test_roc_auc_chance() -> None:
    # Symmetric around the median (positives at extremes) -> 0.5.
    # pairs >: (4,2)+(4,3)=2; pairs <: (1,2)+(1,3)=2 -> AUC=0.5.
    scores = [4.0, 3.0, 2.0, 1.0]
    labels = [True, False, False, True]
    assert roc_auc(scores, labels) == 0.5


def test_roc_auc_three_quarter_separation() -> None:
    # Two positives at score 4 and 2, two negatives at 3 and 1.
    # Comparable pairs: (4,3),(4,1),(2,3),(2,1) -> 3 of 4 -> 0.75.
    scores = [4.0, 3.0, 2.0, 1.0]
    labels = [True, False, True, False]
    assert roc_auc(scores, labels) == 0.75


def test_roc_auc_handles_ties_via_average_rank() -> None:
    # All scores tied -> Mann-Whitney with average ranks gives 0.5.
    scores = [1.0, 1.0, 1.0, 1.0]
    labels = [True, False, True, False]
    assert roc_auc(scores, labels) == 0.5


def test_roc_auc_single_class_returns_none() -> None:
    assert roc_auc([1.0, 2.0, 3.0], [True, True, True]) is None
    assert roc_auc([1.0, 2.0, 3.0], [False, False, False]) is None


def test_roc_auc_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        roc_auc([1.0, 2.0], [True])


# ---- spearman_rho ------------------------------------------------------


def test_spearman_rho_monotonic_perfect() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 20.0, 30.0, 40.0]
    assert spearman_rho(xs, ys) == pytest.approx(1.0)


def test_spearman_rho_monotonic_inverted() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [40.0, 30.0, 20.0, 10.0]
    assert spearman_rho(xs, ys) == pytest.approx(-1.0)


def test_spearman_rho_handles_ties() -> None:
    # Tied ranks on the y side; ρ is well-defined and nonzero.
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 1.0, 2.0, 2.0]
    rho = spearman_rho(xs, ys)
    assert rho is not None
    # Pearson on rx=[1,2,3,4], ry=[1.5,1.5,3.5,3.5]:
    # numerator = 4*29 - 10*10 = 16
    # denom    = sqrt((4*30-100) * (4*29-100)) = sqrt(320) = 17.888...
    # rho = 16 / sqrt(320) = 4/sqrt(20) = 2/sqrt(5).
    expected = 2.0 / math.sqrt(5.0)
    assert rho == pytest.approx(expected, rel=1e-12)


def test_spearman_rho_zero_variance_returns_none() -> None:
    # All-equal xs -> rank vector is constant -> denominator zero.
    assert spearman_rho([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def test_spearman_rho_short_input_returns_none() -> None:
    assert spearman_rho([1.0], [2.0]) is None
    assert spearman_rho([], []) is None


def test_spearman_rho_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        spearman_rho([1.0, 2.0], [3.0])


# ---- determinism (ship-gate per #365) ----------------------------------


def test_metrics_are_deterministic_across_reruns() -> None:
    """Same inputs -> bytes-identical floats. Required by the spec
    ("Determinism: same `(store_snapshot, query_set)` -> bytes-identical
    metrics across reruns")."""
    scores = [0.9, 0.8, 0.4, 0.3, 0.1]
    labels = [True, False, True, False, False]

    p_a = precision_at_k(labels, 3)
    p_b = precision_at_k(labels, 3)
    assert p_a == p_b
    # Bit-exact float comparison via bit pattern.
    assert math.copysign(1.0, p_a) == math.copysign(1.0, p_b)

    a_a = roc_auc(scores, labels)
    a_b = roc_auc(scores, labels)
    assert a_a is not None and a_b is not None
    assert a_a == a_b

    r_a = spearman_rho(scores, [1.0 if x else 0.0 for x in labels])
    r_b = spearman_rho(scores, [1.0 if x else 0.0 for x in labels])
    assert r_a is not None and r_b is not None
    assert r_a == r_b
