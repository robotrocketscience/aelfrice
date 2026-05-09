"""Tests for benchmarks.qa_scoring — deterministic QA correctness leaf.

Spec: #507. The leaf is pure stdlib; tests cover normalization, the
three score functions, and the multi-answer best-of wrapper.
"""
from __future__ import annotations

from benchmarks import qa_scoring


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_lowercases():
    assert qa_scoring.normalize_answer("Hello World") == "hello world"


def test_normalize_strips_punctuation():
    assert qa_scoring.normalize_answer("Hello, world!") == "hello world"


def test_normalize_drops_articles():
    assert qa_scoring.normalize_answer("the quick a fox") == "quick fox"


def test_normalize_collapses_whitespace():
    assert qa_scoring.normalize_answer("foo   bar\tbaz\n") == "foo bar baz"


def test_normalize_empty_string():
    assert qa_scoring.normalize_answer("") == ""


def test_normalize_only_articles_and_punctuation():
    # Articles → " ", punctuation removed; result is empty after collapse.
    assert qa_scoring.normalize_answer("the, a! an?") == ""


# ---------------------------------------------------------------------------
# score_exact_match
# ---------------------------------------------------------------------------


def test_em_identical():
    assert qa_scoring.score_exact_match("Paris", "paris") == 1.0


def test_em_punctuation_insensitive():
    assert qa_scoring.score_exact_match("Paris.", "paris") == 1.0


def test_em_substring_does_not_match():
    assert qa_scoring.score_exact_match("Paris is the capital", "paris") == 0.0


def test_em_empty_inputs():
    assert qa_scoring.score_exact_match("", "") == 1.0


# ---------------------------------------------------------------------------
# score_substring_exact_match
# ---------------------------------------------------------------------------


def test_sem_substring_match():
    assert qa_scoring.score_substring_exact_match(
        "The capital of France is Paris.", "paris",
    ) == 1.0


def test_sem_no_match():
    assert qa_scoring.score_substring_exact_match(
        "The capital of Germany is Berlin.", "paris",
    ) == 0.0


def test_sem_case_insensitive():
    assert qa_scoring.score_substring_exact_match("PARIS", "paris") == 1.0


def test_sem_empty_ground_truth_is_zero():
    # Vacuous match would inflate scores; empty GT must not score.
    assert qa_scoring.score_substring_exact_match("anything", "") == 0.0


def test_sem_articles_dropped_in_both():
    # "the moon" -> "moon"; pred "...moon..." matches.
    assert qa_scoring.score_substring_exact_match(
        "We saw the moon last night", "the moon",
    ) == 1.0


# ---------------------------------------------------------------------------
# score_f1
# ---------------------------------------------------------------------------


def test_f1_identical_tokens_is_one():
    assert qa_scoring.score_f1("paris france", "paris france") == 1.0


def test_f1_no_overlap_is_zero():
    assert qa_scoring.score_f1("paris", "berlin") == 0.0


def test_f1_partial_overlap():
    # pred=["paris","france","capital"] gt=["paris","france"]
    # common=2 ; p=2/3 r=2/2=1 ; f1=2*0.6667*1 / (0.6667+1) = 0.8
    f1 = qa_scoring.score_f1("paris france capital", "paris france")
    assert abs(f1 - 0.8) < 1e-9


def test_f1_empty_prediction_is_zero():
    assert qa_scoring.score_f1("", "paris") == 0.0


def test_f1_empty_ground_truth_is_zero():
    assert qa_scoring.score_f1("paris", "") == 0.0


# ---------------------------------------------------------------------------
# score_multi_answer
# ---------------------------------------------------------------------------


def test_multi_picks_best_match():
    result = qa_scoring.score_multi_answer(
        "The capital is Paris.",
        ["london", "paris", "berlin"],
    )
    assert result["substring_exact_match"] == 1.0
    assert result["exact_match"] == 0.0  # full string isn't equal to "paris"
    assert result["f1"] > 0.0


def test_multi_empty_list_is_zero():
    result = qa_scoring.score_multi_answer("anything", [])
    assert result == {"exact_match": 0.0, "substring_exact_match": 0.0, "f1": 0.0}


def test_multi_em_dominates_when_one_matches_exactly():
    result = qa_scoring.score_multi_answer(
        "paris",
        ["london", "paris", "berlin"],
    )
    assert result["exact_match"] == 1.0
    assert result["substring_exact_match"] == 1.0
    assert result["f1"] == 1.0


# ---------------------------------------------------------------------------
# determinism: same inputs → bit-identical outputs (acceptance #3)
# ---------------------------------------------------------------------------


def test_determinism_repeated_calls():
    pred = "The Eiffel Tower is in Paris, France."
    gts = ["paris", "france"]
    a = qa_scoring.score_multi_answer(pred, gts)
    b = qa_scoring.score_multi_answer(pred, gts)
    assert a == b
