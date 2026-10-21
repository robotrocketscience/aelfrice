"""Tests for the reworded-restatement matcher (#1650).

Each veto has a test whose pair is contained in the target and rejected
by that veto alone, so deleting or inverting the veto turns the pair into
a match and fails the test. Each veto test also checks a control pair
that differs only in the vetoed feature and does match, so the veto test
cannot pass because the pair was never matchable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.reworded_restatement import (
    REJECT_LOOKALIKE,
    REJECT_MIXED_SCRIPT,
    REJECT_NEGATION_MISMATCH,
    REJECT_NEGATION_SCOPE,
    REJECT_NO_TARGET_SENTENCES,
    REJECT_NOT_CONTAINED,
    REJECT_ORDER_MISMATCH,
    REJECT_POLARITY_MISMATCH,
    REJECT_QUANTIFIER_MISMATCH,
    REJECT_TOKEN_STUFFING,
    REJECT_TOO_SHORT,
    REJECT_VALUE_CONFLICT,
    REJECTION_REASONS,
    REWORDED_MIN_RESIDUAL_TOKENS,
    RULE_STEMMED_RESIDUAL_CONTAINMENT,
    RewordedMatch,
    explain_rejection,
    match_reworded,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "promotion_adversarial.json"


def _assert_vetoed(user: str, target: str, reason: str) -> None:
    assert match_reworded(user, target) is None
    assert explain_rejection(user, target) == reason


def _assert_matches(user: str, target: str) -> RewordedMatch:
    match = match_reworded(user, target)
    assert match is not None, explain_rejection(user, target)
    assert explain_rejection(user, target) is None
    return match


# --- The rule -------------------------------------------------------------


def test_reworded_sentence_matches_with_index_tokens_and_rule() -> None:
    target = (
        "Our release checklist covers the changelog. "
        "The scanner reliably detects leaked secrets in committed source code."
    )
    match = _assert_matches("the scanner detects leaked secrets in source code", target)
    assert match.target_sentence_index == 1
    assert match.target_sentence.startswith("The scanner reliably detects")
    assert match.matched_tokens == (
        "scanner", "detect", "leak", "secret", "in", "sourc", "code",
    )
    assert match.rule == RULE_STEMMED_RESIDUAL_CONTAINMENT


def test_tokens_split_across_two_target_sentences_do_not_match() -> None:
    target = (
        "The scanner reads committed source code nightly. "
        "A separate job reports leaked secrets to the team."
    )
    _assert_vetoed(
        "the scanner reports leaked secrets", target, REJECT_NOT_CONTAINED,
    )


@pytest.mark.parametrize(("user", "target"), [
    ("the scanner calls the lexer",
     "The scanner calls the parser and the parser calls the lexer."),
    ("the billing service writes to the audit log",
     "The billing service writes to the cache and the search service "
     "writes to the audit log."),
    ("the billing service writes to the audit log",
     "The billing service writes to the cache; the search service "
     "writes to the audit log."),
], ids=["and", "and-subject", "semicolon"])
def test_tokens_split_across_two_clauses_of_one_sentence_do_not_match(
    user: str, target: str,
) -> None:
    _assert_vetoed(user, target, REJECT_NOT_CONTAINED)


def test_each_clause_of_a_compound_target_matches_on_its_own() -> None:
    target = "The scanner calls the parser and the parser calls the lexer."
    _assert_matches("the scanner calls the parser", target)
    match = _assert_matches("the parser calls the lexer", target)
    assert match.target_sentence_index == 0


def test_a_user_sentence_with_more_breaks_than_the_target_still_matches() -> None:
    """Two user clauses may share one target clause."""
    _assert_matches(
        "config: section with parameters?", "The config section with parameters.",
    )
    _assert_matches(
        "the scanner calls the parser, and the parser calls the lexer",
        "The scanner calls the parser and the parser calls the lexer.",
    )


def test_target_with_no_stored_sentence_is_rejected() -> None:
    _assert_vetoed("the scanner detects leaked secrets", "   ", REJECT_NO_TARGET_SENTENCES)


def test_minimum_residual_tokens_is_enforced_at_the_boundary() -> None:
    target = "The cache layer stores parsed results for reuse."
    assert REWORDED_MIN_RESIDUAL_TOKENS == 3
    _assert_vetoed("cache layer", target, REJECT_TOO_SHORT)
    _assert_matches("cache layer stores", target)


def test_direction_words_do_not_count_toward_the_minimum() -> None:
    _assert_vetoed("up to over", "Move up to over the line.", REJECT_TOO_SHORT)
    _assert_vetoed("cache layer to", "Move the cache layer to disk.", REJECT_TOO_SHORT)
    _assert_matches("cache layer moves to disk", "The cache layer moves to disk.")


def test_direction_words_are_not_dropped() -> None:
    """over vs under, into vs out, up vs down: never a restatement."""
    over = "The update is over the version threshold for rollout."
    _assert_vetoed("the update is under the version threshold", over, REJECT_NOT_CONTAINED)
    _assert_matches("the update is over the version threshold", over)

    into = "Deploy the billing service into production after review."
    _assert_vetoed("deploy the billing service out of production", into, REJECT_NOT_CONTAINED)

    up = "Scroll up in the panel to see newer messages."
    _assert_vetoed("scroll down in the panel to see newer messages", up, REJECT_NOT_CONTAINED)


def test_both_words_of_each_direction_pair_are_kept() -> None:
    """on vs off and in vs out: the detector drops on and in as stopwords."""
    off = "Turn the cache off for the test suite."
    _assert_vetoed("turn the cache on for the test suite", off, REJECT_NOT_CONTAINED)
    _assert_matches("turn the cache off for the test suite", off)

    out = "Copy the logs out of the archive folder."
    _assert_vetoed("copy the logs in the archive folder", out, REJECT_NOT_CONTAINED)


def test_passive_by_is_kept_so_roles_cannot_swap() -> None:
    target = "The scanner calls the parser."
    _assert_vetoed("the scanner is called by the parser", target, REJECT_NOT_CONTAINED)
    _assert_matches("the scanner calls the parser", target)


_DEPLOY = "deploy the billing service into production"


@pytest.mark.parametrize("negated", [
    "We don\u02bct deploy the billing service into production.",
    "We don`t deploy the billing service into production.",
    "We do n\u200bot deploy the billing service into production.",
    "We do \uff4e\uff4f\uff54 deploy the billing service into production.",
    "We nev\u00ader deploy the billing service into production.",
], ids=["modifier-apostrophe", "backtick", "zero-width-space", "fullwidth", "soft-hyphen"])
def test_negation_hidden_by_unicode_is_seen_after_normalizing(negated: str) -> None:
    _assert_vetoed(_DEPLOY, negated, REJECT_NEGATION_MISMATCH)


def test_fullwidth_user_text_matches_its_plain_form() -> None:
    _assert_matches(
        "the scanner detects leaked \uff53\uff45\uff43\uff52\uff45\uff54\uff53",
        "The scanner detects leaked secrets.",
    )


def test_veto_mixed_script() -> None:
    """A Cyrillic o inside "not" survives NFKC and hides the negation."""
    spoofed = "We do n\u043et deploy the billing service into production."
    _assert_vetoed(_DEPLOY, spoofed, REJECT_MIXED_SCRIPT)
    _assert_vetoed(
        "deploy the billing service into pr\u043eduction",
        "Deploy the billing service into production.",
        REJECT_MIXED_SCRIPT,
    )
    _assert_matches(_DEPLOY, "Deploy the billing service into production.")


@pytest.mark.parametrize("word", [
    "n\u1d0ft", "n\u2c9ft", "n\u00f3t", "\ua4e0\ua4b3\ua4d4", "n\u034fot", "n0t",
    "d0n't",
], ids=["small-capital-o", "coptic-o", "acute-o", "lisu", "grapheme-joiner",
        "digit-zero", "digit-contraction"])
def test_veto_lookalike_negation_in_the_target(word: str) -> None:
    """Each spelling of not survives NFKC and the mixed-script check."""
    _assert_vetoed(
        _DEPLOY, f"We do {word} deploy the billing service into production.",
        REJECT_LOOKALIKE,
    )


@pytest.mark.parametrize(("user", "target"), [
    ("integration tests are flaky on windows",
     "0nly integration tests are flaky on windows."),
    ("deploys to staging fail", "Deploys to staging fail 1f the cache is cold."),
], ids=["digit-led-only", "one-as-i"])
def test_veto_digit_spelled_polarity_word_in_the_target(user: str, target: str) -> None:
    _assert_vetoed(user, target, REJECT_LOOKALIKE)


def test_lookalike_control_characters_and_words_the_user_also_typed() -> None:
    _assert_matches("the caf\u00e9 menu updates daily", "The caf\u00e9 menu updates daily.")
    _assert_matches(
        "the worker runs on x86 hosts", "The worker runs on x86 hosts with 4 cores.",
    )
    _assert_matches("we do n0t deploy the billing service", "We do n0t deploy the billing service.")


_REQUEST = "the parser handles each request in 5 seconds"


@pytest.mark.parametrize(("user", "target"), [
    (_REQUEST, "The parser handles each request in 5 seconds with 20\u00b5s jitter."),
    (_REQUEST, "The parser handles each request in 5 seconds with 20\u03bcs jitter."),
    (_REQUEST, "The parser handles each request in 5 seconds with 20 \u03bcs jitter."),
    ("the pull resistor value is 10 kiloohm",
     "The pull resistor value is 10 kiloohm (10k\u2126)."),
    ("the server room stays cool", "The server room stays cool at 18 \u00b0C."),
], ids=["micro-sign", "greek-mu", "spaced-mu", "ohm-sign", "degree"])
def test_a_unit_sign_after_a_number_is_not_a_lookalike(user: str, target: str) -> None:
    _assert_matches(user, target)


def test_a_lookalike_letter_after_a_digit_is_still_vetoed() -> None:
    """Only unit signs are exempt: a small-capital l in 0nly is not one."""
    _assert_vetoed(
        "integration tests are flaky on windows",
        "0n\u029fy integration tests are flaky on windows.",
        REJECT_LOOKALIKE,
    )


# --- One test per veto ----------------------------------------------------


def test_veto_token_stuffing() -> None:
    target = "The system stays fast and reliable under heavy load."
    _assert_vetoed("fast fast fast system reliable", target, REJECT_TOKEN_STUFFING)
    _assert_matches("the system is fast and reliable", target)


def test_token_stuffing_counts_typed_words_not_stems_or_directions() -> None:
    """iterate, iterative and iteratively share one stem; to repeats in a route."""
    iterate = "the team will iterate on the iterative parser design iteratively"
    _assert_matches(iterate, iterate.capitalize() + ".")
    route = "move the file from staging to review to release to archive"
    _assert_matches(route, route.capitalize() + ".")


def test_veto_negation_mismatch() -> None:
    target = "We do not deploy the billing service into production on fridays."
    _assert_vetoed(
        "deploy the billing service into production", target, REJECT_NEGATION_MISMATCH,
    )
    _assert_matches("do not deploy the billing service into production", target)


def test_veto_negation_mismatch_when_only_the_user_sentence_is_negated() -> None:
    """A negated user sentence against an affirmative belief contradicts it."""
    target = "Deploy the billing service into production on fridays."
    _assert_vetoed(
        "do not deploy the billing service into production", target,
        REJECT_NEGATION_MISMATCH,
    )
    _assert_matches("deploy the billing service into production", target)


def test_veto_negation_scope() -> None:
    """Both sentences are negated, but the target negates another clause."""
    _assert_vetoed(
        "the parser module is not fast",
        "The parser module that is not cached runs fast.",
        REJECT_NEGATION_SCOPE,
    )
    _assert_vetoed(
        "the cache layer is not enabled",
        "The cache layer is enabled and the debug logger is not.",
        REJECT_NEGATION_SCOPE,
    )
    _assert_vetoed(
        "the parser module doesn't run fast",
        "The parser module runs fast and the scanner doesn't run fast.",
        REJECT_NEGATION_SCOPE,
    )
    _assert_matches(
        "the parser module is not fast",
        "The parser module is not fast and the scanner is slow.",
    )
    _assert_matches(
        "the parser module doesn't run fast",
        "The parser module does not run fast on large inputs.",
    )


@pytest.mark.parametrize(("user", "target"), [
    ("do not deploy the billing service",
     "Deploy the billing service and do not deploy the search service."),
    ("never restart the billing service",
     "Restart the billing service and never restart the search service."),
], ids=["not", "never"])
def test_veto_negation_scope_reads_only_the_matched_clauses(user: str, target: str) -> None:
    """The target negates its second clause; the user sentence fits the first."""
    _assert_vetoed(user, target, REJECT_NEGATION_SCOPE)


def test_negation_with_no_word_before_it_matches_a_target_that_names_a_subject() -> None:
    _assert_matches(
        "do not deploy the billing service into production",
        "The team does not deploy the billing service into production.",
    )


def test_negation_with_no_word_after_it_in_the_target_leaves_the_pair_to_other_vetoes() -> None:
    assert explain_rejection(
        "the debug logger is not enabled", "The enabled debug logger is not.",
    ) == REJECT_ORDER_MISMATCH


def test_any_negation_of_the_matched_clauses_can_hold_the_user_negation() -> None:
    _assert_matches(
        "the parser module is not fast",
        "The parser module is not fast with no cache.",
    )


def test_negation_scope_compares_the_word_after_the_negation() -> None:
    _assert_vetoed(
        "the parser module is not fast",
        "The parser module is not slow yet fast.",
        REJECT_NEGATION_SCOPE,
    )


def test_veto_quantifier_mismatch() -> None:
    target = "The nightly job always rebuilds the search index."
    _assert_vetoed(
        "the nightly job rebuilds the search index", target, REJECT_QUANTIFIER_MISMATCH,
    )
    _assert_vetoed(
        "the nightly job sometimes rebuilds the search index",
        target,
        REJECT_QUANTIFIER_MISMATCH,
    )
    _assert_matches("the nightly job always rebuilds the search index", target)


def test_veto_quantifier_mismatch_when_only_the_user_sentence_has_one() -> None:
    target = "The nightly job rebuilds the search index."
    _assert_vetoed(
        "the nightly job always rebuilds the search index", target,
        REJECT_QUANTIFIER_MISMATCH,
    )
    _assert_matches("the nightly job rebuilds the search index", target)


def test_veto_quantifier_mismatch_compares_every_quantifier() -> None:
    """Both sides open with a universal, so a first-quantifier check passes it."""
    target = "All developers occasionally write tests for the parser."
    _assert_vetoed(
        "all developers write tests for the parser", target, REJECT_QUANTIFIER_MISMATCH,
    )


@pytest.mark.parametrize(("user", "target"), [
    ("the worker can reach the database host",
     "The worker cannot reach the database host."),
    ("the worker can reach the database host",
     "The worker is unable to reach the database host."),
    ("integration tests are flaky on windows",
     "Few integration tests are flaky on windows."),
    ("integration tests are flaky on windows",
     "Only integration tests are flaky on windows."),
    ("the cache is stale after deploy",
     "The cache might be stale after deploy."),
    ("all tests pass on linux",
     "All tests pass on linux except the network tests."),
    ("deploys to staging fail",
     "Deploys to staging fail unless the cache is warm."),
    ("feature flags are enabled in staging",
     "Feature flags enabled in prod are disabled in staging."),
    ("the timeout is 30 seconds",
     "The timeout went from 30 seconds to 30 minutes."),
    ("the cache layer is enabled",
     "The cache layer is enabled, but the debug logger is slow."),
    ("the cache was stale after deploy",
     "The cache is stale after deploy."),
    ("the limit is 10 requests per minute",
     "The limit is 10 requests per second or 100 requests per minute."),
], ids=["cannot", "unable", "few", "only", "might", "except", "unless",
        "disabled", "went", "but", "user-past-tense", "or"])
def test_veto_polarity_mismatch(user: str, target: str) -> None:
    """A polarity word on one side only, where containment cannot see it."""
    _assert_vetoed(user, target, REJECT_POLARITY_MISMATCH)


_REACH = "the worker reaches the database host"
_REBUILD = "the nightly job rebuilds the search index"
_FIX = "the cache layer fixes the latency spike"


@pytest.mark.parametrize(("user", "target"), [
    (_REACH, "The worker used to reach the database host."),
    (_REACH, "The worker stopped reaching the database host."),
    (_REACH, "The worker hardly reaches the database host."),
    (_REACH, "The worker barely reaches the database host."),
    (_REBUILD, "The nightly job hardly ever rebuilds the search index."),
    (_REBUILD, "The nightly job fails to rebuild the search index."),
    ("the cache has stale entries after deploy",
     "The cache had stale entries after deploy."),
    ("the parser supports yaml input files",
     "The parser removed support for yaml input files."),
    ("run the scanner for tokenization",
     "Run the parser instead of the scanner for tokenization."),
    ("run the scanner for tokenization",
     "Run the parser rather than the scanner for tokenization."),
    ("all tests pass on linux",
     "All tests pass on linux excluding the network tests."),
    ("the upload call is blocking", "The upload call is non-blocking."),
    ("integration tests pass on linux", "0 integration tests pass on linux."),
    ("integration tests pass on linux", "Zero integration tests pass on linux."),
    ("all integration tests pass on windows",
     "Almost all integration tests pass on windows."),
    ("all integration tests pass on windows",
     "Nearly all integration tests pass on windows."),
    (_FIX, "The cache layer allegedly fixes the latency spike."),
    (_FIX, "The cache layer supposedly fixes the latency spike."),
    (_FIX, "The cache layer scarcely fixes the latency spike."),
], ids=["used-to", "stopped", "hardly", "barely", "hardly-ever", "fails-to",
        "had", "removed", "instead-of", "rather-than", "excluding", "non",
        "digit-zero", "word-zero", "almost", "nearly", "allegedly",
        "supposedly", "scarcely"])
def test_veto_polarity_mismatch_on_near_negation_hedges_and_past_state(
    user: str, target: str,
) -> None:
    _assert_vetoed(user, target, REJECT_POLARITY_MISMATCH)


_STREAMING = "the parser module supports streaming input"


@pytest.mark.parametrize("target", [
    "Does the parser module support streaming input?",
    "The claim that the parser module supports streaming input is wrong.",
    "It is a myth that the parser module supports streaming input.",
], ids=["question", "claim-wrong", "myth"])
def test_veto_a_target_that_asks_or_disputes_the_claim(target: str) -> None:
    _assert_vetoed(_STREAMING, target, REJECT_POLARITY_MISMATCH)


def test_a_user_question_may_match_a_target_question() -> None:
    _assert_matches(
        "does the parser module support streaming input?",
        "Does the parser module support streaming input?",
    )


def test_a_zero_on_both_sides_and_a_nonzero_number_are_not_polarity() -> None:
    _assert_matches(
        "0 integration tests pass on linux", "0 integration tests pass on linux.",
    )
    _assert_matches(
        "integration tests pass on linux", "Integration tests pass on linux 10 times.",
    )


def test_veto_polarity_mismatch_control() -> None:
    _assert_matches(
        "the worker cannot reach the database host",
        "The worker cannot reach the database host.",
    )
    _assert_matches(
        "deploys to staging fail unless the cache is warm",
        "Deploys to staging fail unless the cache is warm.",
    )


def test_veto_value_conflict() -> None:
    target = "Set retries = 3 at startup with retries = 5 for the uploader."
    _assert_vetoed("set retries = 5 for the uploader", target, REJECT_VALUE_CONFLICT)
    control = "Set retries = 5 at startup with timeout = 3 for the uploader."
    _assert_matches("set retries = 5 for the uploader", control)


def test_veto_value_conflict_reads_only_the_containing_sentence() -> None:
    """A conflicting value in another sentence of the belief does not veto."""
    target = "Set retries = 5 for the uploader. The timeout = 3 and retries = 7 on the backup."
    match = _assert_matches("set retries = 5 for the uploader", target)
    assert match.target_sentence_index == 0


def test_a_sentence_naming_two_values_of_one_key_matches_itself() -> None:
    sentence = "upgrade the agent from v2.0 to v2.1 before the freeze"
    _assert_matches(sentence, "Upgrade the agent from v2.0 to v2.1 before the freeze.")
    _assert_matches(
        "v1.2 supersedes v1.1 due to critical fixes",
        "v1.2 supersedes v1.1 due to critical fixes.",
    )


def test_veto_value_conflict_on_a_dotted_number_the_target_extends() -> None:
    """3.12 tokenizes to 3 and 12, which 3.12.1 contains."""
    target = "Requires python 3.12.1 on linux with docker."
    _assert_vetoed(
        "requires python 3.12 on linux with docker", target, REJECT_VALUE_CONFLICT,
    )
    _assert_matches("requires python 3.12.1 on linux with docker", target)


def test_veto_value_conflict_on_a_comma_grouped_number_the_target_extends() -> None:
    """1,000 tokenizes to 1 and 000, which 1,000,000 contains."""
    _assert_vetoed(
        "the rate limit is 1,000", "The rate limit is 1,000,000.", REJECT_VALUE_CONFLICT,
    )
    _assert_matches("the rate limit is 1,000,000", "The rate limit is 1,000,000.")
    assert match_reworded(
        "the rate limit is 1,000 requests per minute",
        "The rate limit is 1,000,000 requests per minute.",
    ) is None


def test_veto_order_mismatch() -> None:
    target = "The scanner depends on the parser for tokenization in the build."
    _assert_vetoed(
        "the parser depends on the scanner for tokenization", target, REJECT_ORDER_MISMATCH,
    )
    _assert_matches("the scanner depends on the parser for tokenization", target)


def test_first_matching_sentence_wins_over_an_earlier_vetoed_one() -> None:
    target = (
        "The parser depends on the scanner for tokenization at startup. "
        "The scanner depends on the parser for tokenization in the build."
    )
    match = _assert_matches("the scanner depends on the parser for tokenization", target)
    assert match.target_sentence_index == 1


def test_rejection_reasons_are_the_published_set() -> None:
    assert set(REJECTION_REASONS) == {
        REJECT_TOO_SHORT, REJECT_TOKEN_STUFFING, REJECT_NO_TARGET_SENTENCES,
        REJECT_NOT_CONTAINED, REJECT_NEGATION_MISMATCH,
        REJECT_QUANTIFIER_MISMATCH, REJECT_VALUE_CONFLICT,
        REJECT_ORDER_MISMATCH, REJECT_MIXED_SCRIPT, REJECT_POLARITY_MISMATCH,
        REJECT_NEGATION_SCOPE, REJECT_LOOKALIKE,
    }
    assert len(REJECTION_REASONS) == len(set(REJECTION_REASONS))


# --- Determinism ----------------------------------------------------------

_DETERMINISM_SCRIPT = """
import json, sys
from aelfrice.reworded_restatement import explain_rejection, match_reworded
d = json.load(open(sys.argv[1]))
out = []
for c in d["regression_cases"] + d["edge_cases"]:
    m = match_reworded(c["lock_text"], c["phantom_text"])
    out.append([c["id"], None if m is None else [m.target_sentence_index,
                list(m.matched_tokens), m.rule],
                explain_rejection(c["lock_text"], c["phantom_text"])])
print(json.dumps(out))
"""


def _fixture_verdicts(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SCRIPT, str(_FIXTURE)],
        capture_output=True, text=True, env=env, timeout=120, check=True,
    )
    return result.stdout


@pytest.mark.timeout(120)
def test_verdicts_are_identical_across_runs_and_hash_seeds() -> None:
    first = _fixture_verdicts("0")
    assert first == _fixture_verdicts("1")
    assert first == _fixture_verdicts("12345")
    verdicts = json.loads(first)
    assert any(v[1] is not None for v in verdicts)
    for case_id, match, reason in verdicts:
        assert (match is None) == (reason is not None), case_id


def test_repeated_calls_return_equal_results() -> None:
    target = "The scanner reliably detects leaked secrets in committed source code."
    user = "the scanner detects leaked secrets in source code"
    assert match_reworded(user, target) == match_reworded(user, target)


# --- Fixture precision ----------------------------------------------------


def test_precision_on_the_adversarial_fixture() -> None:
    """The user restatement is ``lock_text``; the target is ``phantom_text``.

    At the time of writing: 18 fire, 18 correct, precision 1.000. D4-01
    and D4-02 hide a format character inside a word; normalizing deletes
    it, so the joined word is not contained in the target.
    """
    data = json.loads(_FIXTURE.read_text())
    cases = data["regression_cases"] + data["edge_cases"]
    true_pos = false_pos = 0
    for case in cases:
        fired = match_reworded(case["lock_text"], case["phantom_text"]) is not None
        if fired and case["expected_should_promote"]:
            true_pos += 1
        elif fired:
            false_pos += 1
    assert true_pos >= 15, "matcher went near-silent; precision would be vacuous"
    precision = true_pos / (true_pos + false_pos)
    assert precision >= 0.78, (true_pos, false_pos, precision)


@pytest.mark.parametrize("case_id", ["A1-01", "A5-01", "A5-03", "D4-01", "D4-02",
                                     "D5-01", "D5-02", "D5-03", "D5-04", "E2-01",
                                     "F4-01", "F4-03"])
def test_named_adversarial_negatives_do_not_match(case_id: str) -> None:
    data = json.loads(_FIXTURE.read_text())
    case = next(
        c for c in data["regression_cases"] + data["edge_cases"] if c["id"] == case_id
    )
    assert case["expected_should_promote"] is False
    assert match_reworded(case["lock_text"], case["phantom_text"]) is None
