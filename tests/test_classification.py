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


def test_conversational_question_without_wh_prefix_no_persist() -> None:
    """#1027: a single-sentence interrogative the wh-prefix list misses
    ("want me to …?", "which way?") is still a question — no belief."""
    for q in ("Want me to run it?", "Which way?", "ok so whats next?",
              "are there any PRs that need review?"):
        assert classify_sentence(q, "user").persist is False, q


def test_multi_sentence_ending_in_question_persists() -> None:
    """A belief with declarative content that merely closes with a
    question must persist — only the WHOLE-sentence question is dropped."""
    r = classify_sentence(
        "Status: shipped. Phase 1 landed at v1.5. Want me to continue?",
        "user",
    )
    assert r.persist is True


def test_midsentence_question_mark_persists() -> None:
    r = classify_sentence(
        'Distinctive: turns "am I doing this right?" into a number.',
        "user",
    )
    assert r.persist is True


def test_prior_sentence_ending_in_closer_then_question_persists() -> None:
    """#1027 review: a prior sentence ending in `."` / `.)` is still a
    sentence boundary, so a belief with declarative content that closes
    with a question keeps that content."""
    for c in (
        'He said "done." Want me to continue?',
        "The flag is on (default). Should we flip it?",
    ):
        assert classify_sentence(c, "user").persist is True, c


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


# --- Speculative-float filter (#1081) ------------------------------------


def test_float_leading_maybe_no_persist() -> None:
    """A statement-form hedge that opens with 'maybe' is a float, not a
    belief — even though it has no trailing '?' so `_is_question` misses it.
    """
    r = classify_sentence("maybe we need one more layer before targets.", "user")
    assert r.belief_type == BELIEF_FACTUAL
    assert r.persist is False


def test_float_leading_hedges_no_persist() -> None:
    """Each high-precision leading hedge marks the sentence as a float."""
    for t in (
        "Perhaps the retry count is off.",
        "What if we tried a calendar concept instead.",
        "Should we flip the default.",
        "Could we batch these writes.",
        "Not sure this handles unicode.",
        "I wonder about a dedicated lane here.",
        "I guess we could split the module.",
        "Or maybe there should be a separate schedule.",
        "How about a second index.",
    ):
        assert classify_sentence(t, "user").persist is False, t


def test_float_internal_proposal_hedge_no_persist() -> None:
    """Unambiguous proposal-hedge phrases float the sentence wherever they
    appear, not only at the start."""
    for t in (
        "We probably need a new concept of a calendar.",
        "Honestly we might want to revisit the budget.",
        "For onboarding we should probably add a banner.",
        "Not sure if the lane survives the trim.",
    ):
        assert classify_sentence(t, "user").persist is False, t


def test_float_dropped_result_stays_factual_and_pending() -> None:
    """A dropped float keeps the factual type/prior and the pending flag —
    only `persist` flips, mirroring the question path."""
    r = classify_sentence("maybe we should cache the tree.", "user")
    assert r.belief_type == BELIEF_FACTUAL
    assert (r.alpha, r.beta) == TYPE_PRIORS[BELIEF_FACTUAL]
    assert r.pending_classification is True


def test_midsentence_hedge_word_persists() -> None:
    """A genuine assertion that merely CONTAINS a hedge word mid-sentence
    (not a leading hedge, not a proposal-hedge phrase) is kept."""
    for t in (
        "The test probably fails on slow hardware.",
        "This maybe-flag controls the lane.",
        "Users perhaps expect the older default.",
    ):
        assert classify_sentence(t, "user").persist is True, t


def test_leading_should_be_is_not_a_float() -> None:
    """'should we' hedges a proposal; 'should be' asserts one. Only the
    former is filtered."""
    r = classify_sentence("There should be a lock on this belief.", "user")
    assert r.persist is True


def test_bald_musing_without_hedge_persists_out_of_scope() -> None:
    """Hedge-free declarative musings have no deterministic signal and are
    intentionally out of scope for this ingest gate (handled downstream)."""
    r = classify_sentence("All averages are not useful.", "user")
    assert r.persist is True


def test_typed_belief_kept_even_when_hedged() -> None:
    """The float check runs last, so a requirement / correction / preference
    is never dropped for being hedged — the type signal wins."""
    req = classify_sentence("maybe the commits must be signed.", "user")
    assert req.belief_type == BELIEF_REQUIREMENT
    assert req.persist is True
    pref = classify_sentence("maybe I prefer tabs over spaces.", "user")
    assert pref.belief_type == BELIEF_PREFERENCE
    assert pref.persist is True


def test_float_filter_case_insensitive() -> None:
    assert classify_sentence("MAYBE WE NEED A CALENDAR.", "user").persist is False


def test_float_filter_deterministic() -> None:
    r1 = classify_sentence("what if we added a spine lane.", "user")
    r2 = classify_sentence("what if we added a spine lane.", "user")
    assert r1.persist == r2.persist is False


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
