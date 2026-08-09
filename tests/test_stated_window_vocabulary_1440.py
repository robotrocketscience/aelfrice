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

import pytest

from aelfrice.directive_detector import detect_directive
from aelfrice.hook import _directive_window_spec
from aelfrice.lock_expiry import (
    extract_stated_window,
    stated_window_attaches_to_memory,
    stated_window_is_ambiguous,
)

# The three classes #1440 names, each in the shape the issue measured:
# an unusable window juxtaposed with a usable one, no connective between
# them (a connective is refused earlier by `_gap_opens_a_new_predicate`).
_CLASSES = [
    pytest.param("for 30 minutes", id="sub-day-digit"),
    pytest.param("for two hours", id="sub-day-word"),
    pytest.param("for twenty days", id="count-above-ten"),
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
    """Hypothesis: the new pattern is unit-anchored, so it neither
    double-counts a window the resolving patterns already saw nor invents
    one out of a quantifier with no unit after it.

    Both are how this fix would silently break the shipped path: a
    double-count makes every single-window directive "ambiguous", and a
    `for <any noun>` reading makes "some planning" a second window. Either
    turns the whole suffix off, which no assertion on the refusal arms
    would notice.
    """
    assert stated_window_is_ambiguous(text) is False
    assert extract_stated_window(text) == expected
    assert _directive_window_spec(text) == expected
