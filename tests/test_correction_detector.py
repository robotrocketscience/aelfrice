"""correction.detect_correction: per-signal-class atomic tests.

The corpus-level 92% accuracy claim is contingent on the original
labeled corrections corpus, which is not part of v1.0. These tests
cover each signal class in isolation, the threshold transition,
confidence scaling, and the deterministic-output contract.
"""
from __future__ import annotations

from aelfrice.correction import (
    CORRECTION_SIGNAL_THRESHOLD,
    CorrectionResult,
    detect_correction,
)


# --- Signal-class isolation: each fires exactly when expected ------------


def test_imperative_signal_fires_on_use_verb_start() -> None:
    r = detect_correction("use snake_case for variable names")
    assert "imperative" in r.signals


def test_imperative_signal_does_not_fire_mid_sentence() -> None:
    r = detect_correction("the team prefers to use snake_case")
    assert "imperative" not in r.signals


def test_always_never_signal_fires_on_always() -> None:
    r = detect_correction("always run tests before pushing")
    assert "always_never" in r.signals


def test_always_never_signal_fires_on_never() -> None:
    r = detect_correction("never commit secrets to the repo")
    assert "always_never" in r.signals


def test_always_never_signal_fires_on_from_now_on() -> None:
    r = detect_correction("from now on we use main not master")
    assert "always_never" in r.signals


def test_negation_signal_fires_on_do_not() -> None:
    r = detect_correction("do not amend commits after pushing")
    assert "negation" in r.signals


def test_negation_signal_fires_on_contraction() -> None:
    r = detect_correction("don't squash main")
    assert "negation" in r.signals


def test_emphasis_signal_fires_on_exclamation() -> None:
    r = detect_correction("we are switching to uv!")
    assert "emphasis" in r.signals


def test_emphasis_signal_fires_on_ever_again() -> None:
    r = detect_correction("we are not using npm ever again")
    assert "emphasis" in r.signals


def test_prior_ref_signal_fires_on_we_discussed() -> None:
    r = detect_correction("we discussed this last week")
    assert "prior_ref" in r.signals


def test_prior_ref_signal_fires_on_already() -> None:
    r = detect_correction("we already agreed on the schema")
    assert "prior_ref" in r.signals


def test_declarative_signal_fires_on_is_the() -> None:
    r = detect_correction("the source of truth is the manifest")
    assert "declarative" in r.signals


def test_declarative_signal_fires_on_should_be_only() -> None:
    r = detect_correction("the export should be only json")
    assert "declarative" in r.signals


def test_directive_signal_fires_on_must() -> None:
    r = detect_correction("commits must be signed")
    assert "directive" in r.signals


def test_directive_signal_fires_on_hard_rule() -> None:
    r = detect_correction("hard rule: every PR has a test")
    assert "directive" in r.signals


# --- Threshold (default 2): is_correction transition --------------------


def test_zero_signals_is_not_correction() -> None:
    r = detect_correction("the cat sat on the mat quietly")
    assert r.is_correction is False
    assert r.signals == []


def test_one_signal_is_not_correction() -> None:
    r = detect_correction("commits must be signed")
    # "must" -> directive only (one signal); below threshold of 2.
    assert r.signals == ["directive"]
    assert r.is_correction is False


def test_two_signals_is_correction() -> None:
    r = detect_correction("always use signed commits")
    # "always" -> always_never; "use" at start -> imperative; 2 signals.
    assert len(r.signals) >= CORRECTION_SIGNAL_THRESHOLD
    assert r.is_correction is True


def test_three_signals_is_correction() -> None:
    r = detect_correction("never commit secrets! we discussed this")
    # always_never + emphasis + prior_ref + (negation? "no" within "secrets"? no) -> 3
    assert len(r.signals) >= 3
    assert r.is_correction is True


# --- Confidence scaling --------------------------------------------------


def test_confidence_zero_when_no_signals() -> None:
    r = detect_correction("the cat sat on the mat quietly")
    assert r.confidence == 0.0


def test_confidence_scales_linearly_below_cap() -> None:
    r = detect_correction("commits must be signed")  # 1 signal
    assert abs(r.confidence - 0.3) < 1e-9


def test_confidence_caps_at_one() -> None:
    # Construct text that fires at least 4 signals -> confidence >= 1.2 capped.
    text = (
        "always do not commit secrets! we already agreed it must be the rule"
    )
    r = detect_correction(text)
    assert r.confidence == min(1.0, len(r.signals) * 0.3)
    assert r.confidence <= 1.0


# --- Determinism + output contract --------------------------------------


def test_repeated_call_returns_same_signals() -> None:
    text = "always run tests! we discussed this"
    r1 = detect_correction(text)
    r2 = detect_correction(text)
    assert r1.signals == r2.signals
    assert r1.is_correction == r2.is_correction
    assert r1.confidence == r2.confidence


def test_signals_list_has_no_duplicates() -> None:
    r = detect_correction("always always always run tests")
    assert len(r.signals) == len(set(r.signals))


def test_result_object_is_correction_result_typed() -> None:
    r = detect_correction("anything")
    assert isinstance(r, CorrectionResult)


def test_empty_string_input_returns_no_signals() -> None:
    r = detect_correction("")
    assert r.signals == []
    assert r.is_correction is False


def test_whitespace_only_input_returns_no_signals() -> None:
    r = detect_correction("   \n\t  ")
    assert r.signals == []


# --- Case insensitivity --------------------------------------------------


def test_uppercase_text_fires_same_signals_as_lowercase() -> None:
    upper = detect_correction("ALWAYS RUN TESTS BEFORE PUSHING")
    lower = detect_correction("always run tests before pushing")
    assert upper.signals == lower.signals


# --- correction_anchor signal --------------------------------------------


def test_correction_anchor_fires_on_supersedes() -> None:
    r = detect_correction("the previous instruction supersedes the README")
    assert "correction_anchor" in r.signals


def test_correction_anchor_fires_on_no_longer() -> None:
    r = detect_correction("the old approach is no longer valid")
    assert "correction_anchor" in r.signals


def test_correction_anchor_fires_on_the_previous() -> None:
    r = detect_correction("the previous decision was incorrect")
    assert "correction_anchor" in r.signals


def test_correction_anchor_fires_on_instead() -> None:
    r = detect_correction("use pytest instead of unittest")
    assert "correction_anchor" in r.signals


def test_correction_anchor_fires_on_actually() -> None:
    r = detect_correction("actually the API returns a list not a dict")
    assert "correction_anchor" in r.signals


def test_correction_anchor_fires_case_insensitive() -> None:
    r = detect_correction("This SUPERSEDES the earlier spec")
    assert "correction_anchor" in r.signals


def test_correction_anchor_does_not_fire_on_unrelated_text() -> None:
    r = detect_correction("the cat sat on the mat quietly")
    assert "correction_anchor" not in r.signals


def test_correction_anchor_counted_as_distinct_signal_from_imperative() -> None:
    # "instead" fires correction_anchor; "use" at start fires imperative — distinct.
    r = detect_correction("use pytest instead of unittest")
    assert "imperative" in r.signals
    assert "correction_anchor" in r.signals
    assert r.signals.count("correction_anchor") == 1
    assert r.signals.count("imperative") == 1


# --- requirement_anchor signal -------------------------------------------


def test_requirement_anchor_fires_on_do_not() -> None:
    r = detect_correction("do not commit secrets")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_fires_on_never() -> None:
    r = detect_correction("never push to main without a review")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_fires_on_must_not() -> None:
    r = detect_correction("you must not merge without CI passing")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_fires_on_cannot() -> None:
    r = detect_correction("the agent cannot modify files outside src/")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_fires_on_fails_if() -> None:
    r = detect_correction("the build fails if pyright reports errors")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_fires_case_insensitive() -> None:
    r = detect_correction("NEVER push to main")
    assert "requirement_anchor" in r.signals


def test_requirement_anchor_does_not_fire_on_unrelated_text() -> None:
    r = detect_correction("the cat sat on the mat quietly")
    assert "requirement_anchor" not in r.signals


def test_requirement_anchor_counted_as_distinct_signal_from_always_never() -> None:
    # "never" fires both always_never (term list) and requirement_anchor (regex) —
    # they are separate categories and both must appear.
    r = detect_correction("never push to main")
    assert "always_never" in r.signals
    assert "requirement_anchor" in r.signals
    assert len(set(r.signals)) == len(r.signals)  # no duplicates within a category


def test_requirement_anchor_counted_as_distinct_signal_from_negation() -> None:
    # "do not" fires both negation (term list) and requirement_anchor (regex).
    r = detect_correction("do not commit secrets")
    assert "negation" in r.signals
    assert "requirement_anchor" in r.signals


# --- Three categories are distinct signal names --------------------------


def test_three_anchor_categories_are_distinct() -> None:
    # A sentence that fires all three anchor categories plus others must
    # list each category at most once and never collapse them.
    r = detect_correction(
        "the previous rule supersedes the README: do not push to main"
    )
    assert "correction_anchor" in r.signals
    assert "requirement_anchor" in r.signals
    assert len(r.signals) == len(set(r.signals))
    assert r.is_correction is True


def test_correction_utterance_from_issue_fires_anchor_signal() -> None:
    # Canonical example from issue #225: correction_anchor fires.
    r = detect_correction("the previous instruction supersedes the README")
    assert "correction_anchor" in r.signals


def test_correction_utterance_two_signals_is_correction() -> None:
    # With an additional signal the threshold is crossed.
    r = detect_correction("the previous rule must supersede the README")
    assert "correction_anchor" in r.signals
    assert r.is_correction is True


def test_requirement_utterance_do_not_is_detected() -> None:
    # "do not" triggers negation + requirement_anchor -> 2 signals -> correction.
    r = detect_correction("do not commit secrets")
    assert r.is_correction is True


def test_requirement_utterance_never_push_is_detected() -> None:
    # "never" triggers always_never + requirement_anchor -> 2 signals -> correction.
    r = detect_correction("never push to main")
    assert r.is_correction is True
