"""sentiment_feedback module — detection, escalator, application, config.

Unit tests only. The bench-gate accuracy test against the labeled
corpus lives at tests/bench_gate/test_sentiment.py.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Iterator

import pytest

from aelfrice.feedback import apply_feedback
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
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


# --- imperative "correct" must not read as praise (#1372 §11) ------------

# Real corrective phrasings a user types when asking for a repair. Every
# one of these contains the literal word "correct"; under the bare
# word-boundary pattern every one scored +1.5 strong-positive — the
# largest magnitude the lane emits, with the sign inverted.
CORRECTIVE_PHRASINGS: tuple[str, ...] = (
    "Correct the import path in the hook.",
    "correct that typo",
    "Please correct the docstring.",
    "Can you correct the config?",
    "correct it",
    "You need to correct the test name.",
    "correct this before committing",
    "Let's correct the ordering.",
    "I want you to correct the regex.",
    "correct the spelling of the module",
    "go back and correct the earlier edit",
    "correct these paths",
    "Correct the formatting, then run the suite.",
    "please correct my earlier statement",
    "we should correct the doc",
    "correct the failing assertion",
    "Correct me if I am off here.",
)


@pytest.mark.parametrize("prompt", CORRECTIVE_PHRASINGS)
def test_corrective_phrasing_never_scores_positive(prompt: str) -> None:
    assert classify(prompt) != POSITIVE, (
        f"corrective phrasing scored as praise: {prompt!r}"
    )


# Evaluative frames that genuinely are affirmations. Guards the fix
# against over-correcting into "never score `correct` at all".
AFFIRMATIVE_CORRECT_PHRASINGS: tuple[str, ...] = (
    "correct",
    "Correct.",
    "Correct!",
    "Correct, that's the one.",
    "that's correct",
    "thats correct",
    "That is correct.",
    "you're correct",
    "you are correct",
    "its correct",
    "yes, correct",
    "looks correct",
    "the path is correct",
    "absolutely correct",
)


@pytest.mark.parametrize("prompt", AFFIRMATIVE_CORRECT_PHRASINGS)
def test_evaluative_correct_frame_still_scores_positive(prompt: str) -> None:
    s = detect_sentiment(prompt)
    assert s is not None, f"affirmation went unmatched: {prompt!r}"
    assert s.sentiment == POSITIVE
    assert s.pattern == "correct"
    assert s.valence == AMPLIFIED_VALENCE


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


def test_apply_reaches_every_pending_belief(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    results = apply_sentiment_to_pending(store, sig, ["b1", "b2", "b3"])
    assert len(results) == 3
    assert {r.belief_id for r in results} == {"b1", "b2", "b3"}
    for bid in ("b1", "b2", "b3"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha > 1.0


# --- one utterance is one unit of evidence (#1372 §13) -------------------
#
# The lane used to write the *whole* valence onto every belief in the
# prior turn's retrieval pack, so a wide pack manufactured evidence in
# proportion to its own width. It is split into equal shares now. This
# is the magnitude half of §13; the credit-assignment half (only the
# referenced belief should move) needs an instrument this module lacks.


def test_apply_splits_valence_into_equal_shares(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    apply_sentiment_to_pending(store, sig, ["b1", "b2", "b3"])
    for bid in ("b1", "b2", "b3"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(1.0 + BASE_VALENCE / 3.0)


def test_apply_total_evidence_is_one_unit_regardless_of_pack_width(
    store: MemoryStore,
) -> None:
    """Two packs of different width must move the same total mass."""
    narrow = MemoryStore(":memory:")
    narrow.insert_belief(_mk("only"))
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")

    apply_sentiment_to_pending(narrow, sig, ["only"])
    apply_sentiment_to_pending(store, sig, ["b1", "b2", "b3"])

    narrow_total = sum(
        b.alpha - 1.0
        for b in (narrow.get_belief("only"),)
        if b is not None
    )
    wide_total = sum(
        b.alpha - 1.0
        for b in (store.get_belief(x) for x in ("b1", "b2", "b3"))
        if b is not None
    )
    assert wide_total == pytest.approx(narrow_total)
    assert wide_total == pytest.approx(BASE_VALENCE)


def test_apply_negative_signal_splits_the_same_way(store: MemoryStore) -> None:
    sig = SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")
    apply_sentiment_to_pending(store, sig, ["b1", "b2"])
    for bid in ("b1", "b2"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.beta == pytest.approx(1.0 + BASE_VALENCE / 2.0)


def test_apply_escalated_negative_is_split_too(store: MemoryStore) -> None:
    sig = SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")
    apply_sentiment_to_pending(store, sig, ["b1", "b2"], escalated=True)
    for bid in ("b1", "b2"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.beta == pytest.approx(1.0 + ESCALATED_NEGATIVE_VALENCE / 2.0)


def test_apply_share_divides_by_resolvable_ids_only(store: MemoryStore) -> None:
    """Stale ids in the pack must not shrink the evidence delivered.

    Two of the four ids no longer exist; the divisor is 2, not 4, so the
    live pair still absorbs one whole unit between them.
    """
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    apply_sentiment_to_pending(store, sig, ["b1", "ghost1", "b2", "ghost2"])
    delivered = 0.0
    for bid in ("b1", "b2"):
        b = store.get_belief(bid)
        assert b is not None
        assert b.alpha == pytest.approx(1.0 + BASE_VALENCE / 2.0)
        delivered += b.alpha - 1.0
    assert delivered == pytest.approx(BASE_VALENCE)


def test_apply_records_the_share_not_the_whole_signal_in_the_audit_row(
    store: MemoryStore,
) -> None:
    """feedback_history must record what was actually applied."""
    sig = SentimentSignal(NEGATIVE, -BASE_VALENCE, 0.6, "no", "no")
    apply_sentiment_to_pending(store, sig, ["b1", "b2", "b3", "b1"])
    rows = store._conn.execute(
        "SELECT valence FROM feedback_history WHERE source = ?",
        (SENTIMENT_INFERRED_SOURCE,),
    ).fetchall()
    assert rows, "no feedback_history rows written"
    for (valence,) in rows:
        assert valence == pytest.approx(-BASE_VALENCE / 4.0)


def test_apply_locked_belief_consumes_a_share_without_moving(
    store: MemoryStore,
) -> None:
    """The divisor counts locked beliefs, so the movable set gets <= 1 unit.

    Rescaling over only the movable subset was rejected: on a pack that
    is nearly all locks it would hand one unlocked belief the entire
    signal. Under-delivering is the conservative direction for an
    inferred lane, so it is what this pins.
    """
    store.insert_belief(replace(_mk("locked1"), lock_level=LOCK_USER))
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    apply_sentiment_to_pending(store, sig, ["b1", "locked1"])

    unlocked = store.get_belief("b1")
    held = store.get_belief("locked1")
    assert unlocked is not None and held is not None
    assert unlocked.alpha == pytest.approx(1.0 + BASE_VALENCE / 2.0)
    assert held.alpha == 1.0, "the lock floor must hold"
    delivered = (unlocked.alpha - 1.0) + (held.alpha - 1.0)
    assert delivered == pytest.approx(BASE_VALENCE / 2.0)
    assert delivered < BASE_VALENCE


def test_apply_all_ids_stale_returns_empty(store: MemoryStore) -> None:
    sig = SentimentSignal(POSITIVE, BASE_VALENCE, 0.6, "yes", "yes")
    assert apply_sentiment_to_pending(store, sig, ["ghost1", "ghost2"]) == []


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


# ---------------------------------------------------------------------------
# #1291 — the inferred lane does not propagate
# ---------------------------------------------------------------------------


def _propagation_chain() -> "MemoryStore":
    """A -SUPPORTS-> B, both neutral. Mirrors test_feedback_propagation."""
    from aelfrice.models import EDGE_SUPPORTS, Edge
    from aelfrice.store import MemoryStore

    store = MemoryStore(":memory:")
    for bid in ("A", "B"):
        store.insert_belief(
            Belief(
                id=bid,
                content=f"belief {bid}",
                content_hash=f"h_{bid}",
                alpha=5.0,
                beta=5.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    store.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
    return store


def test_explicit_feedback_still_propagates_on_this_fixture() -> None:
    """Control arm — without this the suppression test proves nothing.

    If the fixture had no live propagation path, the assertion below
    would pass whether or not `propagate=False` were passed.
    """
    store = _propagation_chain()
    result = apply_feedback(store, "A", valence=-1.0, source="user")
    downstream = store.get_belief("B")
    assert result.propagated, "fixture must actually propagate"
    assert downstream is not None
    assert (downstream.alpha, downstream.beta) != (5.0, 5.0)


def test_sentiment_signal_does_not_propagate() -> None:
    """One prose signal must not walk the edge graph (#1291).

    It is already credited uniformly to every belief on the prior turn,
    which is not a set of exchangeable trials about any one of them.
    Propagating it multiplies an attribution the signal never had.
    """
    store = _propagation_chain()
    signal = detect_sentiment("no that's wrong")
    assert signal is not None

    results = apply_sentiment_to_pending(
        store=store, signal=signal, pending_belief_ids=["A"],
    )

    assert len(results) == 1
    assert results[0].propagated == []
    # The directly-addressed belief still moves...
    direct = store.get_belief("A")
    assert direct is not None
    assert direct.beta > 5.0
    # ...and the neighbour is untouched.
    downstream = store.get_belief("B")
    assert downstream is not None
    assert (downstream.alpha, downstream.beta) == (5.0, 5.0)
    assert store.count_feedback_events(belief_id="B") == 0
