"""sentiment_feedback module — detection, escalator, application, config.

Unit tests only. The bench-gate accuracy test against the labeled
corpus lives at tests/bench_gate/test_sentiment.py.
"""
from __future__ import annotations

import os
from typing import Iterator

import pytest

from aelfrice.feedback import apply_feedback
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.sentiment_feedback import (
    AMPLIFIED_VALENCE,
    BASE_VALENCE,
    CORRECTION_FREQ_THRESHOLD,
    ENV_SENTIMENT,
    ESCALATED_NEGATIVE_VALENCE,
    MAX_PROMPT_CHARS,
    NEGATIVE,
    POSITIVE,
    SENTIMENT_INFERRED_SOURCE,
    SentimentSignal,
    apply_sentiment_to_pending,
    classify,
    detect_correction_frequency,
    detect_sentiment,
    is_enabled,
)
from aelfrice.store import MemoryStore


# --- Fixtures ------------------------------------------------------------


def _mk(bid: str, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    s.insert_belief(_mk("b2"))
    s.insert_belief(_mk("b3"))
    return s


@pytest.fixture
def clean_env() -> Iterator[None]:
    """Snapshot + restore the sentiment env var around each test."""
    saved = os.environ.pop(ENV_SENTIMENT, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV_SENTIMENT, None)
        else:
            os.environ[ENV_SENTIMENT] = saved


# --- detect_sentiment: positive base patterns ----------------------------


def test_detect_yes_returns_positive() -> None:
    s = detect_sentiment("yes")
    assert s is not None and s.sentiment == POSITIVE


def test_detect_thanks_returns_positive() -> None:
    s = detect_sentiment("thanks for the fix")
    assert s is not None and s.sentiment == POSITIVE


def test_detect_looks_good_returns_positive() -> None:
    s = detect_sentiment("looks good")
    assert s is not None and s.sentiment == POSITIVE


# --- detect_sentiment: negative base patterns ----------------------------


def test_detect_no_returns_negative() -> None:
    s = detect_sentiment("no")
    assert s is not None and s.sentiment == NEGATIVE


def test_detect_doesnt_work_returns_negative() -> None:
    s = detect_sentiment("that doesnt work")
    assert s is not None and s.sentiment == NEGATIVE


def test_detect_undo_returns_negative() -> None:
    s = detect_sentiment("undo that")
    assert s is not None and s.sentiment == NEGATIVE


# --- detect_sentiment: strong patterns elevate confidence ----------------


def test_strong_positive_uses_amplified_valence() -> None:
    s = detect_sentiment("perfect")
    assert s is not None
    assert s.sentiment == POSITIVE
    assert s.valence == AMPLIFIED_VALENCE


def test_strong_negative_uses_amplified_valence() -> None:
    s = detect_sentiment("that is wrong")
    assert s is not None
    assert s.sentiment == NEGATIVE
    assert s.valence == -AMPLIFIED_VALENCE


def test_base_positive_uses_base_valence() -> None:
    s = detect_sentiment("yes")
    assert s is not None
    assert s.valence == BASE_VALENCE


def test_strong_pattern_higher_confidence_than_base() -> None:
    base = detect_sentiment("yes")
    strong = detect_sentiment("perfect")
    assert base is not None and strong is not None
    assert strong.confidence > base.confidence


# --- detect_sentiment: precedence + length guard -------------------------


def test_strong_negative_wins_over_base_positive_in_same_prompt() -> None:
    # "yes thats wrong" contains both a base-positive and a strong-negative.
    s = detect_sentiment("yes thats wrong")
    assert s is not None
    assert s.sentiment == NEGATIVE


def test_long_prompt_returns_none() -> None:
    long_prompt = "yes " + ("x" * (MAX_PROMPT_CHARS + 1))
    assert detect_sentiment(long_prompt) is None


def test_empty_prompt_returns_none() -> None:
    assert detect_sentiment("") is None


def test_no_match_returns_none() -> None:
    assert detect_sentiment("please continue with the next item") is None


def test_matched_text_records_substring() -> None:
    s = detect_sentiment("perfect, thanks")
    assert s is not None
    assert s.matched_text.lower() == "perfect"


def test_pattern_id_records_pattern_name() -> None:
    s = detect_sentiment("perfect")
    assert s is not None
    assert s.pattern == "perfect"


# --- classify: three-way label adapter -----------------------------------


def test_classify_positive() -> None:
    assert classify("yes") == "positive"


def test_classify_negative() -> None:
    assert classify("no") == "negative"


def test_classify_neutral() -> None:
    assert classify("please continue with the next item") == "neutral"


def test_classify_long_prompt_neutral() -> None:
    assert classify("yes " + ("x" * (MAX_PROMPT_CHARS + 1))) == "neutral"


# --- detect_correction_frequency -----------------------------------------


def _neg() -> SentimentSignal:
    return SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")


def _pos() -> SentimentSignal:
    return SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")


def test_correction_frequency_below_min_turns_returns_false() -> None:
    # 4 turns is below default min_turns=5 even at 100% negatives.
    assert detect_correction_frequency([_neg(), _neg(), _neg(), _neg()]) is False


def test_correction_frequency_at_threshold_fires() -> None:
    # 2 of 5 = 0.4, exactly at default threshold.
    window = [_neg(), _neg(), _pos(), _pos(), _pos()]
    assert detect_correction_frequency(window) is True


def test_correction_frequency_below_threshold_does_not_fire() -> None:
    # 1 of 5 = 0.2.
    window = [_neg(), _pos(), _pos(), _pos(), _pos()]
    assert detect_correction_frequency(window) is False


def test_correction_frequency_none_entries_count_in_denominator() -> None:
    # 2 negatives in 6 entries = 0.33 < 0.4. Two None entries do not
    # raise the rate even though they fill the window.
    window = [_neg(), _neg(), None, None, _pos(), _pos()]
    assert detect_correction_frequency(window) is False


def test_correction_frequency_threshold_override() -> None:
    window = [_neg(), _pos(), _pos(), _pos(), _pos()]  # 0.2
    assert detect_correction_frequency(window, threshold=0.1) is True


# --- apply_sentiment_to_pending ------------------------------------------


def test_apply_distributes_positive_signal_across_all_pending(
    store: MemoryStore,
) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    results = apply_sentiment_to_pending(store, sig, ["b1", "b2", "b3"])
    assert len(results) == 3
    for bid in ("b1", "b2", "b3"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == 2.0  # was 1.0, +1.0 valence


def test_apply_uses_sentiment_inferred_source(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    apply_sentiment_to_pending(store, sig, ["b1"])
    rows = store._conn.execute(
        "SELECT source FROM feedback_history WHERE belief_id = ?", ("b1",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == SENTIMENT_INFERRED_SOURCE


def test_apply_skips_missing_belief_silently(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    results = apply_sentiment_to_pending(store, sig, ["b1", "ghost", "b2"])
    assert len(results) == 2
    assert {r.belief_id for r in results} == {"b1", "b2"}


def test_apply_negative_signal_increments_beta(store: MemoryStore) -> None:
    sig = SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")
    apply_sentiment_to_pending(store, sig, ["b1"])
    b = store.get_belief("b1")
    assert b is not None
    assert b.beta == 2.0


def test_apply_escalated_negative_uses_doubled_magnitude(
    store: MemoryStore,
) -> None:
    sig = SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")
    apply_sentiment_to_pending(store, sig, ["b1"], escalated=True)
    b = store.get_belief("b1")
    assert b is not None
    # base beta 1.0 + ESCALATED_NEGATIVE_VALENCE
    assert b.beta == 1.0 + ESCALATED_NEGATIVE_VALENCE


def test_apply_escalated_does_not_affect_positive_signal(
    store: MemoryStore,
) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    apply_sentiment_to_pending(store, sig, ["b1"], escalated=True)
    b = store.get_belief("b1")
    assert b is not None
    assert b.alpha == 2.0  # base, NOT escalated


def test_apply_empty_pending_returns_empty(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    results = apply_sentiment_to_pending(store, sig, [])
    assert results == []


# --- is_enabled: config + env --------------------------------------------


def test_is_enabled_default_off(clean_env: None) -> None:
    assert is_enabled() is False


def test_is_enabled_off_when_config_false(clean_env: None) -> None:
    cfg = {"feedback": {"sentiment_from_prose": False}}
    assert is_enabled(cfg) is False


def test_is_enabled_on_when_config_true(clean_env: None) -> None:
    cfg = {"feedback": {"sentiment_from_prose": True}}
    assert is_enabled(cfg) is True


def test_is_enabled_env_truthy_wins(clean_env: None) -> None:
    os.environ[ENV_SENTIMENT] = "1"
    cfg = {"feedback": {"sentiment_from_prose": False}}
    assert is_enabled(cfg) is True


def test_is_enabled_env_falsy_wins(clean_env: None) -> None:
    os.environ[ENV_SENTIMENT] = "0"
    cfg = {"feedback": {"sentiment_from_prose": True}}
    assert is_enabled(cfg) is False


def test_is_enabled_env_unrecognized_falls_through_to_config(
    clean_env: None,
) -> None:
    os.environ[ENV_SENTIMENT] = "maybe"
    cfg = {"feedback": {"sentiment_from_prose": True}}
    assert is_enabled(cfg) is True


def test_is_enabled_missing_section(clean_env: None) -> None:
    cfg = {"other_section": {"x": 1}}
    assert is_enabled(cfg) is False


def test_is_enabled_non_bool_value_treated_as_off(clean_env: None) -> None:
    cfg = {"feedback": {"sentiment_from_prose": "yes"}}
    assert is_enabled(cfg) is False
