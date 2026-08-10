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
# days") sit in a slot no leading-word list reaches, and same for the
# quantifier class, whose articled entries ("a couple of", "a number
# of") were unreachable under `_UNUSABLE_WINDOW_RE`'s own `the next`
# prefix — the prefix eats the article — so only the entries that
# happened to be listed unarticled ("few", "several") worked there.
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
    pytest.param("for a number of days", id="quantifier-number-of"),
    pytest.param(
        "for the next couple of days", id="quantifier-couple-under-prefix",
    ),
    pytest.param(
        "for the next number of hours", id="quantifier-number-of-under-prefix",
    ),
    pytest.param("for the next few hours", id="quantifier-few-under-prefix"),
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
    "dozen ", "a dozen ", "two dozen ", "dozens ", "dozens of ", "a few ",
    "few ", "several ", "a couple ", "couple ", "a couple of ", "couple of ",
    "a number of ", "number of ", "many ", "some ", "numerous ", "2-3 ",
    "2 - 3 ", "2–3 ", "two hundred and fifty ", "two hundred and fifty-five ",
    "a couple hundred ", "a couple of hundred ", "few thousand ",
    "hundreds of ", "twenty-five hundred ",
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
    for is asserted here instead, over a 3,312-phrase sweep of the count
    and unit slots rather than over a handful of examples. A guard no
    input reaches proves nothing; this sweep does.

    The sweep is also the evidence that widening the count slot over the
    partitive, the `and`-joined tail and a quantified scale is safe: all
    three spellings are probes here, and they collide with nothing.

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
    assert probes == 3312, "the sweep lost a slot"
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
def test_the_named_residual_count_forms_are_read(phrase: str) -> None:
    """Hypothesis: the three count spellings this fix once named as
    residuals are stated windows like any other — alone they propose
    nothing, beside a usable window they refuse.

    They were left uncovered on the claim that widening the count slot
    far enough to read them "starts colliding with the resolving
    patterns". That claim was false and nothing backed it: widening the
    slot produces zero same-offset collisions over
    `test_no_two_patterns_claim_the_same_window`'s sweep — which now
    probes all three spellings — and none of the five resolving fixtures
    changes. This test is the inverse of the pin that published the
    claim, so reverting the widening turns it red rather than green.
    """
    alone = f"Always remember this {phrase}."
    assert detect_directive(alone) is True, "fixture must reach the window arm"
    assert extract_stated_window(alone) is None
    assert stated_window_is_ambiguous(alone) is False, "one window is not two"

    text = f"Always remember this {phrase} for a week."
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the ambiguity arm"
    )
    assert stated_window_is_ambiguous(text) is True
    assert _directive_window_spec(text) is None, (
        "two stated windows must refuse, not resolve to the second"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Always remember this for a week, and thanks for a second opinion.",
        "Always remember this for a week; I looked for a minute detail.",
    ],
)
def test_an_adjectival_unit_beside_a_real_window_does_not_refuse(text: str) -> None:
    """`second` and `minute` are adjectives too, and refusing on them is a
    recall cost this pattern's own contract forbids.

    `_UNUSABLE_WINDOW_RE` is new in this change, so this is not inherited
    behaviour: without the clause-boundary guard, "for a second opinion"
    reads as a second stated window and the sentence pairing it with a
    real `for a week` resolves to nothing instead of to `1w`. The guard
    is narrow on purpose — it applies only to `a`/`an` + `second`/`minute`,
    the one form where the adjective reading collides with a duration.
    """
    assert [w for _, w in _stated_windows_with_positions(text)] == ["1w"]
    assert stated_window_is_ambiguous(text) is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Remember this for a second.", [None]),
        ("Remember this for a second, then drop it.", [None]),
        ("Remember this for a minute!", [None]),
        ("Remember this for an hour.", [None]),
        ("Remember this for 30 minutes.", [None]),
        ("Remember this for two hours.", [None]),
    ],
)
def test_a_genuine_sub_day_duration_is_still_stated(
    text: str, expected: list[str | None]
) -> None:
    """The guard must not cost the durations it sits next to.

    Each of these states an unresolvable window and must keep doing so —
    a `None` here is what makes `aelf lock --for` refuse rather than
    invent an expiry. If the guard ever swallowed one, the directive
    would silently resolve some *other* window in the sentence instead.
    """
    assert [w for _, w in _stated_windows_with_positions(text)] == expected
