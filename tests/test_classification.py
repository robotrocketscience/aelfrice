"""classify_sentence: per-rule atomic tests.

Each test asserts one rule of the synchronous classifier in isolation.
The host-LLM handshake path (polymorphic onboard contract) does not
exist yet at v0.5.0; it lands in v0.6.0 with the MCP server.
"""
from __future__ import annotations

from aelfrice.classification import (
    TYPE_PRIORS,
    ClassificationResult,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
)


# --- TYPE_PRIORS table integrity -----------------------------------------


def test_type_priors_keyed_on_v1_belief_types() -> None:
    assert set(TYPE_PRIORS.keys()) == {
        BELIEF_REQUIREMENT,
        BELIEF_CORRECTION,
        BELIEF_PREFERENCE,
        BELIEF_FACTUAL,
    }


def test_requirement_prior_is_high_confidence() -> None:
    a, b = TYPE_PRIORS[BELIEF_REQUIREMENT]
    assert (a, b) == (9.0, 0.5)


def test_correction_prior_matches_requirement() -> None:
    assert TYPE_PRIORS[BELIEF_CORRECTION] == TYPE_PRIORS[BELIEF_REQUIREMENT]


def test_preference_prior_below_correction() -> None:
    pa, _ = TYPE_PRIORS[BELIEF_PREFERENCE]
    ca, _ = TYPE_PRIORS[BELIEF_CORRECTION]
    assert pa < ca


def test_factual_prior_is_lowest_alpha() -> None:
    fa, _ = TYPE_PRIORS[BELIEF_FACTUAL]
    others = [TYPE_PRIORS[t][0] for t in TYPE_PRIORS if t != BELIEF_FACTUAL]
    assert fa <= min(others)


# --- get_source_adjusted_prior ------------------------------------------


def test_user_source_returns_full_prior() -> None:
    assert get_source_adjusted_prior(BELIEF_REQUIREMENT, "user") == (9.0, 0.5)


def test_non_user_source_deflates_alpha() -> None:
    a, b = get_source_adjusted_prior(BELIEF_REQUIREMENT, "scanner")
    assert a < 9.0
    assert b == 0.5  # beta unchanged


def test_non_user_source_alpha_at_or_above_floor() -> None:
    """Deflated alpha must never drop below 0.5 to keep posterior math sane."""
    for t in TYPE_PRIORS:
        a, _ = get_source_adjusted_prior(t, "scanner")
        assert a >= 0.5


def test_unknown_type_falls_back_to_factual_prior() -> None:
    a, b = get_source_adjusted_prior("nonexistent", "user")
    assert (a, b) == TYPE_PRIORS[BELIEF_FACTUAL]


# --- Empty / whitespace --------------------------------------------------


def test_empty_string_classifies_factual_no_persist() -> None:
    r = classify_sentence("", "user")
    assert r.belief_type == BELIEF_FACTUAL
    assert r.persist is False


def test_whitespace_only_classifies_factual_no_persist() -> None:
    r = classify_sentence("   \n\t  ", "user")
    assert r.belief_type == BELIEF_FACTUAL
    assert r.persist is False


# --- Question form -------------------------------------------------------


def test_question_starts_what_classifies_factual_no_persist() -> None:
    r = classify_sentence("what is the deploy command?", "user")
    assert r.belief_type == BELIEF_FACTUAL
    assert r.persist is False


def test_question_starts_how_no_persist() -> None:
    r = classify_sentence("how do I run the tests?", "user")
    assert r.persist is False


def test_question_starts_why_no_persist() -> None:
    r = classify_sentence("why did we pick uv?", "user")
    assert r.persist is False


def test_statement_ending_with_period_persists() -> None:
    r = classify_sentence("we deploy via the staging gate workflow.", "user")
    assert r.persist is True


def test_what_in_middle_of_sentence_is_not_a_question() -> None:
    """Question heuristic checks prefix only; mid-sentence interrogatives
    don't trip the filter."""
    r = classify_sentence("the team agreed on what we ship.", "user")
    assert r.persist is True


# --- Requirement classification (any source; deflated for non-user) -----


def test_user_must_keyword_classifies_requirement() -> None:
    r = classify_sentence("commits must be signed", "user")
    assert r.belief_type == BELIEF_REQUIREMENT


def test_user_mandatory_classifies_requirement() -> None:
    r = classify_sentence("CI on main is mandatory", "user")
    assert r.belief_type == BELIEF_REQUIREMENT


def test_non_user_must_classifies_requirement_with_deflated_prior() -> None:
    """#226 fix: doc/ast/git-sourced requirement keywords classify as
    BELIEF_REQUIREMENT (not factual). False-positive risk is handled by
    the source-prior deflation lowering alpha, not by suppressing
    classification entirely. Pre-fix the scanner produced zero
    requirement counts because every onboard candidate carried a
    non-user source."""
    r = classify_sentence("you must restart the daemon", "doc")
    assert r.belief_type == BELIEF_REQUIREMENT
    # Deflation kicks in: alpha < user-source alpha
    user_r = classify_sentence("you must restart the daemon", "user")
    assert r.alpha < user_r.alpha


def test_requirement_uses_high_alpha_prior_for_user() -> None:
    r = classify_sentence("commits must be signed", "user")
    assert r.alpha == TYPE_PRIORS[BELIEF_REQUIREMENT][0]


# --- Correction classification (any source; deflated for non-user) ------


def test_user_correction_phrasing_classifies_correction() -> None:
    r = classify_sentence("we discussed this; do not amend commits", "user")
    assert r.belief_type == BELIEF_CORRECTION


def test_non_user_correction_phrasing_classifies_correction_with_deflated_prior() -> None:
    """#226 fix: doc-sourced correction phrasings classify as
    BELIEF_CORRECTION. The deflated prior at non-user sources handles
    the false-positive risk that the previous user-only gate was trying
    to address."""
    r = classify_sentence("we discussed this; do not amend commits", "doc")
    assert r.belief_type == BELIEF_CORRECTION
    user_r = classify_sentence("we discussed this; do not amend commits", "user")
    assert r.alpha < user_r.alpha


# --- Preference classification (any source) ------------------------------


def test_prefer_keyword_classifies_preference_user_source() -> None:
    r = classify_sentence("we prefer uv over pip", "user")
    assert r.belief_type == BELIEF_PREFERENCE


def test_prefer_keyword_classifies_preference_doc_source() -> None:
    """Preference keywords classify regardless of source — preference
    statements in documents are still preferences."""
    r = classify_sentence("we prefer uv over pip", "doc")
    assert r.belief_type == BELIEF_PREFERENCE


def test_doc_source_preference_uses_deflated_prior() -> None:
    user_r = classify_sentence("we prefer uv over pip", "user")
    doc_r = classify_sentence("we prefer uv over pip", "doc")
    assert doc_r.alpha < user_r.alpha


# --- Factual default -----------------------------------------------------


def test_neutral_statement_classifies_factual() -> None:
    r = classify_sentence("the project uses Python 3.12", "user")
    assert r.belief_type == BELIEF_FACTUAL


def test_factual_statement_persists() -> None:
    r = classify_sentence("the project uses Python 3.12", "user")
    assert r.persist is True


def test_factual_default_uses_factual_prior() -> None:
    r = classify_sentence("the project uses Python 3.12", "user")
    a, b = TYPE_PRIORS[BELIEF_FACTUAL]
    assert (r.alpha, r.beta) == (a, b)


# --- Result object surface -----------------------------------------------


def test_result_object_is_typed() -> None:
    r = classify_sentence("anything", "user")
    assert isinstance(r, ClassificationResult)


def test_pending_classification_always_true_in_v05() -> None:
    """v0.5.0 ships only the regex fallback. Every result is flagged
    pending so a future v0.6.0 host-LLM pass can refine."""
    r = classify_sentence("commits must be signed", "user")
    assert r.pending_classification is True


# --- Determinism --------------------------------------------------------


def test_repeated_call_returns_same_result() -> None:
    r1 = classify_sentence("we must always use signed commits", "user")
    r2 = classify_sentence("we must always use signed commits", "user")
    assert r1.belief_type == r2.belief_type
    assert (r1.alpha, r1.beta) == (r2.alpha, r2.beta)
    assert r1.persist == r2.persist
    assert r1.pending_classification == r2.pending_classification


def test_case_insensitive_keyword_match() -> None:
    upper = classify_sentence("COMMITS MUST BE SIGNED", "user")
    lower = classify_sentence("commits must be signed", "user")
    assert upper.belief_type == lower.belief_type
