"""#1440 — a window outside the count/unit vocabulary is still *stated*.

`_stated_windows_with_positions` recorded a window only when one of the
two resolving patterns matched it. A window spelled outside that
vocabulary — a sub-day unit, a count above ten, a quantifier or a range —
was not recorded at all, so a sentence stating two windows reported one
and resolved to the one stated **second**, contradicting both exported
docstrings ("only the first window stated is returned", "True when the
text states more than one distinct window").

The load-bearing assertion here is the ambiguity one. A test asserting
only that the unusable window yields no spec passes unchanged on the
broken module, because a window that is invisible also proposes nothing.
What distinguishes the two designs is what happens when an unusable
window sits beside a usable one: invisible resolves to the survivor,
recorded refuses.

Every fixture is memory-anchored and asserts it cleared `detect_directive`
and `stated_window_attaches_to_memory` first, so it reaches the ambiguity
arm rather than being refused one gate earlier and passing for the wrong
reason — the trap `test(1315): reach the gates these fixtures claim to
pin` had to repair.
"""
from __future__ import annotations

import itertools

import pytest

from aelfrice.directive_detector import detect_directive
from aelfrice.hook import _directive_window_spec
from aelfrice.lock_expiry import (
    _NEXT_UNIT_RE,
    _STATED_WINDOW_RE,
    _UNUSABLE_WINDOW_RE,
    _stated_windows_with_positions,
    extract_stated_window,
    stated_window_attaches_to_memory,
    stated_window_is_ambiguous,
)

# The three classes #1440 names, each in the shape the issue measured:
# an unusable window juxtaposed with a usable one, no connective between
# them (a connective is refused earlier by `_gap_opens_a_new_predicate`).
#
# The sub-day class carries all three of its spellings, because a class
# is not covered by one of them: `for the next hour` states its count
# the way `for the next week` does — implied by "next" — and that
# spelling was still invisible after the first pass at this fix. Same
# for the count class, whose scale words ("two hundred days", "a dozen
# days") sit in a slot no leading-word list reaches.
_CLASSES = [
    pytest.param("for 30 minutes", id="sub-day-digit"),
    pytest.param("for two hours", id="sub-day-word"),
    pytest.param("for the next hour", id="sub-day-count-implied"),
    pytest.param("for the next minute", id="sub-day-count-implied-minute"),
    pytest.param("for twenty days", id="count-above-ten"),
    pytest.param("for twenty-five days", id="count-compound"),
    pytest.param("for two hundred days", id="count-scaled"),
    pytest.param("for a dozen days", id="count-dozen"),
    pytest.param("for three thousand days", id="count-thousand"),
    pytest.param("for a few days", id="quantifier-few"),
    pytest.param("for several days", id="quantifier-several"),
    pytest.param("for a couple of days", id="quantifier-couple"),
    pytest.param("for 2-3 days", id="range"),
]


@pytest.mark.parametrize("phrase", _CLASSES)
def test_an_unusable_window_alone_proposes_nothing(phrase: str) -> None:
    """Hypothesis: recording an unusable window costs no recall — alone,
    it still yields no spec, so no `--for` is proposed from it.

    This is the half of #1440 that must NOT change behaviour. Falsifiable
    by any pattern that guesses a spec for "a few days": the extractor
    would return one and the caller would render a `--for` the user never
    stated, which is the failure mode `extract_stated_window`'s docstring
    exists to forbid.
    """
    text = f"Always remember this {phrase}."
    assert detect_directive(text) is True, "fixture must reach the window arm"
    assert extract_stated_window(text) is None
    assert stated_window_is_ambiguous(text) is False, "one window is not two"
    assert _directive_window_spec(text) is None


@pytest.mark.parametrize("phrase", _CLASSES)
def test_an_unusable_window_beside_a_usable_one_refuses(phrase: str) -> None:
    """Hypothesis: an unusable window counts as a stated window, so a
    sentence naming one beside "for a week" is ambiguous and refused.

    Falsifiable by dropping `_UNUSABLE_WINDOW_RE`: the unusable half goes
    invisible again, `stated_window_is_ambiguous` reports False, and the
    caller proposes `1w` — the window stated *second* — for every param.
    Asserted on the refusal itself (`_directive_window_spec is None`), not
    on the absence of a write, so an implementation that resolves the
    survivor and merely declines to write it still fails here.
    """
    text = f"Always remember this {phrase} for a week."
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the ambiguity arm"
    )
    assert stated_window_is_ambiguous(text) is True
    assert _directive_window_spec(text) is None, (
        "two stated windows must refuse, not resolve to the second"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Always remember this for the next week.", "1w", id="next-week",
        ),
        pytest.param(
            "Always remember this for two months.", "2mo", id="two-months",
        ),
        pytest.param(
            "Always remember this for 3 days.", "3d", id="digit-days",
        ),
        pytest.param(
            "Always remember this for a year.", "1y", id="a-year",
        ),
        pytest.param(
            "Always remember this for a week; the migration takes some "
            "planning.",
            "1w",
            id="quantifier-without-a-unit",
        ),
    ],
)
def test_the_supported_vocabulary_still_resolves(text: str, expected: str) -> None:
    """Hypothesis: the new pattern is unit-anchored, so it does not
    invent a window out of a quantifier with no unit after it, and these
    five sentences still resolve to the spec they state.

    That is how this fix would silently break the shipped path: a
    `for <any noun>` reading makes "some planning" a second window, the
    directive reads as ambiguous, and the whole suffix turns off — which
    no assertion on the refusal arms would notice. The other way to turn
    it off is a double-count, and these five texts would catch it too,
    but only for themselves; the general form is
    `test_no_two_patterns_claim_the_same_window` below.
    """
    assert stated_window_is_ambiguous(text) is False
    assert extract_stated_window(text) == expected
    assert _directive_window_spec(text) == expected


# One `for` clause, spelled every way the three patterns can read one.
# The cross product is the point: a collision between two patterns is a
# property of their *slots*, so it has to be searched for rather than
# spot-checked, and the slots only differ under a count/unit sweep.
_PROBE_PREFIXES = ("", "the ", "next ", "the next ")
_PROBE_COUNTS = (
    "", "1 ", "7 ", "30 ", "a ", "an ", "one ", "two ", "ten ", "twenty ",
    "twenty-five ", "twenty five ", "ninety ", "hundred ", "a hundred ",
    "two hundred ", "thousand ", "three thousand ", "twenty thousand ",
    "dozen ", "a dozen ", "two dozen ", "dozens of ", "a few ", "several ",
    "a couple ", "a couple of ", "a number of ", "many ", "some ", "few ",
    "numerous ", "2-3 ", "2 - 3 ", "2–3 ", "two hundred and fifty ",
)
_PROBE_UNITS = (
    "second", "seconds", "minute", "minutes", "hour", "hours", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "trip",
    "sprint", "of", "planning",
)

_PATTERNS = (
    ("_STATED_WINDOW_RE", _STATED_WINDOW_RE),
    ("_NEXT_UNIT_RE", _NEXT_UNIT_RE),
    ("_UNUSABLE_WINDOW_RE", _UNUSABLE_WINDOW_RE),
)


def test_no_two_patterns_claim_the_same_window() -> None:
    """Hypothesis: one stated window is recorded once, whatever it is
    spelled like — the three patterns' count and unit slots are disjoint,
    so a text with one `for` clause yields at most one entry and can
    never read as ambiguous.

    This is the invariant `_stated_windows_with_positions` used to defend
    with a dedupe on match start. The dedupe was unreachable — no input
    produced the collision — so it was removed and the property it stood
    for is asserted here instead, over a 2,592-phrase sweep of the count
    and unit slots rather than over a handful of examples. A guard no
    input reaches proves nothing; this sweep does.

    Falsifiable by widening any pattern's count slot into another's: add
    `_NUMBER_WORDS` to `_UNUSABLE_WINDOW_RE`'s unreadable-count
    alternation and "for two days" is claimed by two patterns at the same
    offset, so it records twice and every such directive refuses.
    """
    duplicated: list[tuple[str, list[str]]] = []
    ambiguous: list[str] = []
    probes = 0
    for prefix, count, unit in itertools.product(
        _PROBE_PREFIXES, _PROBE_COUNTS, _PROBE_UNITS,
    ):
        text = f"Always remember this for {prefix}{count}{unit}."
        probes += 1
        by_start: dict[int, list[str]] = {}
        for name, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                by_start.setdefault(match.start(), []).append(name)
        duplicated.extend(
            (text, names) for names in by_start.values() if len(names) > 1
        )
        if len(_stated_windows_with_positions(text)) > 1:
            duplicated.append((text, ["merged twice"]))
        if stated_window_is_ambiguous(text):
            ambiguous.append(text)
    assert probes == 2592, "the sweep lost a slot"
    assert duplicated == []
    assert ambiguous == [], "a single stated window must not read as two"


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param("for two hundred and fifty days", id="and-joined-tail"),
        pytest.param("for dozens of days", id="partitive"),
        pytest.param("for a couple hundred days", id="quantified-scale"),
    ],
)
def test_the_named_residual_count_forms_are_still_invisible(phrase: str) -> None:
    """Hypothesis: the count vocabulary's stated gaps are exactly these,
    and they still behave the pre-#1440 way — invisible, so the sentence
    resolves to the window stated second.

    Pinned rather than left to the comment, so the partial fix cannot be
    read as a complete one. These are the forms `_SCALE_NUMBER_WORDS`
    names as uncovered; widening the count slot to reach them is what
    starts colliding with the resolving patterns, which is why it was not
    done here. A future pass that does cover them turns this red, which
    is the intended signal, not a regression.
    """
    text = f"Always remember this {phrase} for a week."
    assert stated_window_is_ambiguous(text) is False
    assert extract_stated_window(text) == "1w"
