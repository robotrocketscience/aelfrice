"""Write-side classifier word-boundary regression corpus (#1368).

Discharges #1159 §4/§5/§6/§13 and #1162 §5 — five findings that are all the
same bug: a classifier testing membership with unbounded substring
containment (`kw in text_lower`) where it means a *word*.

The centrepiece is `MUST_SURVIVE_CORPUS`: ordinary declarative sentences
that must survive classification unchanged — every one of them was typed as
a `correction` or a `requirement` at the 94.7% prior (alpha=9.0, beta=0.5)
before the fix, purely because a keyword happened to be a substring of an
unrelated word ("pia*no*", "*Must*ard", "*requirement*s.txt", SQL
`constraint`).

Each corpus row is mutation-sensitive by construction: reverting any single
boundary fix to substring containment turns rows of this corpus red. See the
per-defect tests below for the narrow assertions.
"""
from __future__ import annotations

import re

import pytest

from aelfrice import correction
from aelfrice.classification_core import (
    TYPE_PRIORS,
    classify_sentence,
)
from aelfrice.correction import (
    _EMPHASIS_TERMS,
    _IMPERATIVE_RE,
    _NEGATION_TERMS,
    detect_correction,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
)
from aelfrice.value_compare import extract_values, find_conflicts

# --- The must-survive corpus ------------------------------------------


# (text, expected_belief_type). Source "user" throughout: it is the
# undeflated prior, so a mis-type here is a belief stored at mu=0.947.
MUST_SURVIVE_CORPUS: tuple[tuple[str, str], ...] = (
    # #1159 §5 — `"no "` matched inside "pia*no* is", and the extremely
    # common `_DECLARATIVE_RE` ("is the") supplied the second signal.
    ("The piano is the default instrument.", BELIEF_FACTUAL),
    ("The casino is the only house that wins.", BELIEF_FACTUAL),
    ("The mono repo is the default layout.", BELIEF_FACTUAL),
    # `"period"` inside "periodic".
    ("The periodic sweep is the only cleanup path.", BELIEF_FACTUAL),
    # #1159 §6 — `"require"` inside "requirements.txt", `"must"` inside
    # "Mustard".
    ("See requirements.txt for the pinned dependency list.", BELIEF_FACTUAL),
    ("Mustard is not a dependency.", BELIEF_FACTUAL),
    ("Read requirement.md before filing.", BELIEF_FACTUAL),
    # NOT here, deliberately: "The FOREIGN KEY constraint on edges cascades
    # on delete." #1368's acceptance lists it, and it still types as a
    # requirement — see test_bare_constraint_still_types_as_a_requirement
    # for why that is a deferral rather than a miss.
    # #1159 §4 — one token satisfying two "independent" signal categories.
    # "cannot" fired `_REQUIREMENT_ANCHOR_RE` *and* negation's `"not "`;
    # "stop" fired the imperative bank, negation and emphasis at once.
    ("The store cannot be opened read-only.", BELIEF_FACTUAL),
    ("Stop the server before running migrations.", BELIEF_FACTUAL),
    # Real onboarded rows named in the #1159 §4 evidence as mis-typed
    # `correction`.
    (
        "Resolve the installed aelfrice version from package metadata.",
        BELIEF_FACTUAL,
    ),
    ("Newest ts in the canonical turns.jsonl, or None.", BELIEF_FACTUAL),
    # The other direction: genuine requirements must NOT be lost to an
    # over-tightened keyword set.
    ("Commits must be signed.", BELIEF_REQUIREMENT),
    ("Signed commits are mandatory.", BELIEF_REQUIREMENT),
    ("Hard rule: every PR has a test.", BELIEF_REQUIREMENT),
    ("The gate requires two approvals.", BELIEF_REQUIREMENT),
    ("Review is required before merge.", BELIEF_REQUIREMENT),
    ("The hard cap is 50 beliefs.", BELIEF_REQUIREMENT),
    ("That is a hard constraint on the budget.", BELIEF_REQUIREMENT),
    # Requirement still precedes preference in `classify_sentence`'s
    # pipeline, so a sentence carrying both signals types as the former.
    # That ordering is unchanged by #1368 (its acceptance asks for word
    # boundaries, not a re-ordered pipeline); the row is pinned here so a
    # later re-ordering is a deliberate, visible change.
    ("I prefer uv, but the CI must use pip.", BELIEF_REQUIREMENT),
    # Preference keywords with no requirement keyword are untouched.
    ("I prefer uv over pip.", BELIEF_PREFERENCE),
)


@pytest.mark.parametrize(("text", "expected_type"), MUST_SURVIVE_CORPUS)
def test_must_survive_corpus_types(text: str, expected_type: str) -> None:
    """Every corpus row classifies as its stated type, at that type's prior."""
    result = classify_sentence(text, "user")
    assert result.belief_type == expected_type, (
        f"{text!r} typed {result.belief_type!r}, expected {expected_type!r}"
    )
    assert result.persist is True
    assert (result.alpha, result.beta) == TYPE_PRIORS[expected_type]


@pytest.mark.parametrize(("text", "expected_type"), MUST_SURVIVE_CORPUS)
def test_must_survive_corpus_is_deterministic(text: str, expected_type: str) -> None:
    """Re-running the classifier on a corpus row is byte-stable.

    The write path is replayed by `aelf rebuild`; a corpus row that types
    differently on the second call would desync the store from its log.
    """
    first = classify_sentence(text, "user")
    second = classify_sentence(text, "user")
    assert first == second


# --- #1159 §5: `_NEGATION_TERMS` word boundaries ----------------------


@pytest.mark.parametrize(
    "text",
    [
        "The piano is the default instrument.",
        "The casino is the only house that wins.",
        "The mono repo is the default layout.",
    ],
)
def test_negation_does_not_fire_inside_an_unrelated_word(text: str) -> None:
    """`"no"` inside "piano" / "casino" / "mono" is not a negation."""
    assert "negation" not in detect_correction(text).signals


def test_negation_still_fires_on_the_standalone_word() -> None:
    """The boundary fix must not cost the signal its real matches."""
    for text in (
        "do not amend commits after pushing",
        "don't squash main",
        "there is no fallback path",
        "that is not the agreed schema",
        "no more force pushes",
    ):
        assert "negation" in detect_correction(text).signals, text


# --- #1159 §6: `_REQUIREMENT_KEYWORDS` word boundaries ----------------


@pytest.mark.parametrize(
    "text",
    [
        "See requirements.txt for the pinned dependency list.",
        "Mustard is not a dependency.",
        "Read requirement.md before filing.",
        "The mustache template is checked in.",
    ],
)
def test_requirement_keywords_do_not_substring_match(text: str) -> None:
    assert classify_sentence(text, "user").belief_type != BELIEF_REQUIREMENT


def test_bare_constraint_still_types_as_a_requirement() -> None:
    """Pins the ONE #1368 acceptance row this PR deliberately does not fix.

    #1368 asks for `constraint` to be "dropped or tightened". It is
    neither here. The operator ruling of 2026-08-06 split this issue:
    word-boundary matching is mechanical and ships now, but *removing* a
    keyword changes what the write path admits, and the write path is
    irreversible for beliefs it discards — so keyword removals wait for a
    funded must-survive corpus.

    The boundary fix does apply to `constraint`; it just narrows it to the
    whole word rather than deleting it. Both rows below therefore still
    type as requirements, including the FOREIGN KEY sentence #1368's
    acceptance names.

    This test exists to fail loudly when the deferred half lands: whoever
    drops the keyword must delete this test and move its rows into the
    must-survive corpus above, rather than discovering the acceptance row
    was quietly left unfixed.
    """
    assert (
        classify_sentence("Add a UNIQUE constraint on the id column.", "user").belief_type
        == BELIEF_REQUIREMENT
    )
    assert (
        classify_sentence(
            "The FOREIGN KEY constraint on edges cascades on delete.", "user"
        ).belief_type
        == BELIEF_REQUIREMENT
    )


# --- #1159 §4: signal categories are token-disjoint -------------------


def _imperative_verb_bank() -> frozenset[str]:
    """The alternation members of `_IMPERATIVE_RE`, as a token set."""
    body = _IMPERATIVE_RE.pattern
    match = re.fullmatch(r"\^\((?P<alts>.+)\)\\b", body)
    assert match is not None, f"unexpected _IMPERATIVE_RE shape: {body!r}"
    return frozenset(match.group("alts").split("|"))


def test_signal_category_token_sets_are_pairwise_disjoint() -> None:
    """No token may satisfy two of the three overlapping categories.

    `CORRECTION_SIGNAL_THRESHOLD = 2` exists to require two *independent*
    signals. "stop" used to sit in all three of these lists, so one token
    cleared the threshold on its own.
    """
    imperative = _imperative_verb_bank()
    negation = frozenset(_NEGATION_TERMS)
    emphasis = frozenset(_EMPHASIS_TERMS)
    assert imperative & negation == frozenset()
    assert imperative & emphasis == frozenset()
    assert negation & emphasis == frozenset()


def test_stop_fires_exactly_one_signal_category() -> None:
    r = detect_correction("Stop the server before running migrations.")
    assert r.signals == ["imperative"]
    assert r.is_correction is False


def test_cannot_fires_exactly_one_signal_category() -> None:
    """`\\bnot\\b` does not match inside "cannot", so the requirement
    anchor no longer double-counts as a negation."""
    r = detect_correction("The store cannot be opened read-only.")
    assert r.signals == ["imperative"]
    assert r.is_correction is False


# --- #1162 §5: the `directive` category is deleted --------------------


def test_directive_terms_are_deleted() -> None:
    assert not hasattr(correction, "_DIRECTIVE_TERMS")


def test_directive_is_never_emitted_as_a_signal() -> None:
    for text in (
        "commits must be signed",
        "hard rule: every PR has a test",
        "two approvals are mandatory",
        "the hard cap is 50",
    ):
        assert "directive" not in detect_correction(text).signals, text


def test_both_docstrings_say_six_categories() -> None:
    for doc in (correction.__doc__, detect_correction.__doc__):
        assert doc is not None
        assert "six categories" in doc or "six correction-signal categories" in doc
        assert "seven" not in doc


# --- #1159 §13: value_compare's member boundary excludes `-` ----------


def test_non_deterministic_tags_exactly_one_slot() -> None:
    slots = extract_values("retrieval is non-deterministic")
    members = sorted(s.member for s in slots.enum)
    assert members == ["non-deterministic"]


def test_determinism_category_can_produce_a_conflict() -> None:
    """The flagship contradiction case. Before the fix `deterministic`
    matched inside `non-deterministic`, both groups tagged, and
    `find_conflicts` short-circuited on the group-disjointness test — the
    category could never report a conflict."""
    a = extract_values("retrieval is non-deterministic")
    b = extract_values("retrieval is deterministic")
    conflicts = find_conflicts(a, b)
    assert [c.key for c in conflicts] == ["determinism"]
    assert {conflicts[0].value_a, conflicts[0].value_b} == {
        "non-deterministic",
        "deterministic",
    }


def test_hyphenated_member_does_not_leak_into_a_second_category() -> None:
    """`full` inside `full-scan` was cross-category noise between
    `completeness` and `storage_mode`."""
    slots = extract_values("the index falls back to a full-scan")
    assert sorted(s.category for s in slots.enum) == ["storage_mode"]


def test_control_conflict_still_fires() -> None:
    """Named in #1368 as the control: the boundary change must not cost
    an already-working category its conflict."""
    conflicts = find_conflicts(
        extract_values("the store is read-only"),
        extract_values("the store is writable"),
    )
    assert [c.key for c in conflicts] == ["access_mode"]


def test_hyphenated_members_still_match_themselves() -> None:
    for text, member in (
        ("the flag is default-on", "default-on"),
        ("the flag is default-off", "default-off"),
        ("the store is read-only", "read-only"),
        ("the store is read-write", "read-write"),
    ):
        assert member in {s.member for s in extract_values(text).enum}, text


@pytest.mark.parametrize(
    "text,member",
    [
        # Every row is a real string from the live repo-local store.
        ("status: shipped-default-on", "default-on"),
        ("flag: shipped-default-off", "default-off"),
        ("merged-but-default-off for now", "default-off"),
        ("the secrets-scan gate is required", "scan"),
        ("pattern-scan and history-scan both run", "scan"),
    ],
)
def test_member_inside_a_longer_hyphenated_compound_still_tags(
    text: str, member: str
) -> None:
    """The regression that the obvious §13 fix would have shipped.

    Adding `-` to the boundary class kills the defect, and also kills
    every member that legitimately sits inside a longer hyphenated
    phrase. Measured against the live repo-local store (44,683 active
    beliefs): the both-sides variant changes 588 beliefs and destroys
    568 whole-category tags; left-side-only changes 310 and destroys
    290. Roughly 40% of the losses are `default_state` assertions of
    exactly the shape below.

    Longest-match ordering distinguishes the two cases where a boundary
    class cannot: `deterministic` is dropped from "non-deterministic"
    because a *longer member* claimed that span, while `default-on`
    survives inside "shipped-default-on" because "shipped-" is not a
    member of anything. Same store, that variant changes 10 beliefs,
    all of them the real fix, and loses nothing.

    These rows fail under either boundary-class variant and pass under
    longest-match, so they are what stops the simpler fix coming back.
    """
    assert member in {s.member for s in extract_values(text).enum}, text


def test_shorter_member_kept_when_it_also_occurs_outside_the_longer_span() -> None:
    """Longest-match claims spans, not members.

    A member suppressed inside one occurrence must still tag the belief
    if it appears somewhere else in the same text — otherwise the fix
    would silently drop real signal from mixed sentences.
    """
    members = {s.member for s in extract_values(
        "the run is non-deterministic, but the replay is deterministic"
    ).enum}
    assert {"non-deterministic", "deterministic"} <= members
