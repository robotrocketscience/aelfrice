"""#899 StructMemEval per-sub-task scorer fix.

`check_state_correctness` is a single word-overlap heuristic over the
whole reference and is invalid for scoring reader predictions: for
location it counts the "should NOT mention <distractors>" exclusion
words (so raw retrieval containing every stale city scores high while a
concise correct answer scores 0); for tree the binary Yes/No polarity is
drowned by boilerplate. These tests lock the task-aware replacements.
"""
from __future__ import annotations

from benchmarks.structmemeval_adapter import (
    _polarity,
    _split_location_reference,
    score_location_prediction,
    score_prediction,
    score_tree_prediction,
)

_LOC_REF = (
    "Since you currently live in Sydney (Bondi), your Saturday activity "
    "is sunrise yoga on Bondi Beach followed by brunch at Bills. The "
    "answer should NOT mention activities from your previous cities "
    "(Lisbon Feira da Ladra, Tokyo Tsukiji Market, London Columbia Road, "
    "or Barcelona La Boqueria)."
)


# --- location ------------------------------------------------------------


def test_location_split_separates_expected_from_exclusion() -> None:
    expected, exclusion = _split_location_reference(_LOC_REF)
    assert "Sydney" in expected and "Bondi Beach" in expected
    assert "should not mention" in exclusion.lower()
    assert "Lisbon" in exclusion and "Tokyo" in exclusion


def test_location_correct_current_state_passes() -> None:
    pred = "You currently live in Sydney (Bondi); Saturday is sunrise yoga on Bondi Beach then brunch at Bills."
    assert score_location_prediction(pred, _LOC_REF) is True


def test_location_stale_state_fails() -> None:
    # Names a previous city (Lisbon / Feira da Ladra) — the #909 reader's
    # original error. Must score False.
    pred = "Visit Feira da Ladra flea market; you live in Alfama, Lisbon."
    assert score_location_prediction(pred, _LOC_REF) is False


def test_location_raw_retrieval_containing_all_cities_fails() -> None:
    # The old metric scored this HIGH (it matches the exclusion words).
    # The fix must reject it: it leaks excluded distractors.
    raw = (
        "Life in Sydney Bondi sunrise yoga Bills. Life in Lisbon Feira "
        "da Ladra. Life in Tokyo Tsukiji. Life in London Columbia Road. "
        "Life in Barcelona La Boqueria."
    )
    assert score_location_prediction(raw, _LOC_REF) is False


def test_location_correct_state_but_leaks_one_distractor_fails() -> None:
    pred = "You live in Sydney (Bondi), sunrise yoga at Bondi Beach and Bills — unlike Tokyo Tsukiji."
    assert score_location_prediction(pred, _LOC_REF) is False


# --- tree ----------------------------------------------------------------


def test_polarity_extraction() -> None:
    assert _polarity("Yes, they are indirect colleagues.") == "yes"
    assert _polarity("No, there is no path between them.") == "no"
    assert _polarity("They might be related somehow.") is None


def test_tree_matching_polarity_passes() -> None:
    ref = "Yes, they are indirect colleagues according to their graph relations."
    assert score_tree_prediction("Yes, a path connects them.", ref) is True


def test_tree_opposite_polarity_fails() -> None:
    ref = "Yes, they are indirect colleagues according to their graph relations."
    assert score_tree_prediction("No, they are in disconnected components.", ref) is False


def test_tree_missing_polarity_fails() -> None:
    ref = "Yes, they are indirect colleagues according to their graph relations."
    assert score_tree_prediction("They share a manager.", ref) is False


# --- dispatch ------------------------------------------------------------


def test_score_prediction_dispatches_by_task() -> None:
    good_loc = "Sydney Bondi sunrise yoga Bondi Beach Bills."
    assert score_prediction(good_loc, _LOC_REF, "location") is True
    tree_ref = "Yes, they are indirect colleagues according to their graph relations."
    assert score_prediction("Yes, connected.", tree_ref, "tree") is True
    assert score_prediction("No, disconnected.", tree_ref, "tree") is False
