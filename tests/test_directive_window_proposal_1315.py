"""#1315 — a stated window is proposed, never written.

Operator ruling 2026-08-06 rescoped this lane to **confirmation-gated**:
the detector proposes a pre-filled `aelf lock … --for <spec>` command and
the user runs it. Nothing reaches the store until they do, which is why
the H1 precision bar (P=0.665 against a 0.80 bar) stopped being the
blocker — a false positive costs a declined suggestion rather than a
silently-wrong expiring lock.

The load-bearing test here is the one that DISTINGUISHES proposing from
writing. A test asserting only that a directive produces a lock command
would pass just as happily on a design that writes first and asks after.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.directive_detector import detect_directive
from aelfrice.hook import (
    _belief_is_lock_candidate,
    _directive_window_spec,
    _format_stop_prompt,
)
from aelfrice.lock_expiry import (
    extract_stated_window,
    parse_for,
    stated_window_is_ambiguous,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore

_SESSION = "s-1315"
_TS = "2026-08-06T00:00:00+00:00"

_DIRECTIVE = "Always use tabs in this repo for the next week."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "prop.db"))
    yield s
    s.close()


def _belief(content: str, *, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id="b" + str(abs(hash(content)) % 10**12),
        content=content,
        content_hash="h" + str(abs(hash(content)) % 10**12),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=None,
        created_at=_TS,
        last_retrieved_at=None,
        session_id=_SESSION,
        origin=ORIGIN_USER_TRANSCRIPT,
    )


# --- the window extractor -----------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("prioritise this for the next week", "1w", id="next-week"),
        pytest.param("keep this for two months", "2mo", id="two-months"),
        pytest.param("use tabs for the next 3 days", "3d", id="digit-days"),
        pytest.param("lock this for a year", "1y", id="a-year"),
    ],
)
def test_a_stated_window_resolves_to_a_for_spec(text: str, expected: str) -> None:
    """Hypothesis: an explicitly-stated window maps to the `--for` spec
    `parse_for` accepts. Falsifiable by any mapping that `parse_for`
    then rejects, which the second assertion catches directly."""
    spec = extract_stated_window(text)
    assert spec == expected
    # The spec must be one `parse_for` actually accepts, or the proposal
    # renders a command that fails when the user runs it.
    from datetime import datetime, timezone

    assert parse_for(spec, now=datetime(2026, 8, 6, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("remember this", id="no-window"),
        pytest.param("keep this for the trip", id="unmeasured-window"),
        pytest.param("hold onto this until I'm back", id="relative-event"),
        pytest.param("keep this for 0 days", id="zero-window"),
        pytest.param("", id="empty"),
    ],
)
def test_an_unstated_or_unmeasurable_window_returns_none(text: str) -> None:
    """Hypothesis: anything short of an explicit, countable window
    returns None.

    None means *no window was stated*, not *use a default*. Inferring an
    expiry the user did not state is an explicit non-goal — a guessed
    window expires their lock on a date they never agreed to. Pairs with
    the test above: those yield a spec, these yield None, so a helper
    that substitutes a default fails here while passing there.
    """
    assert extract_stated_window(text) is None


def test_two_different_windows_are_ambiguous_and_refused() -> None:
    """Hypothesis: a sentence naming two distinct windows is refused, not
    resolved to the first.

    Falsifiable by `_directive_window_spec` returning `2d` here — which
    `extract_stated_window` alone does, deliberately, so the refusal has
    to live in the caller and be asserted there.
    """
    # The text must ALSO pass `detect_directive`, or this test passes for
    # the wrong reason: a non-directive is refused by the detector arm and
    # never reaches the ambiguity check, so removing the ambiguity check
    # leaves it green. Asserted explicitly so the trap cannot reopen.
    text = "Always use tabs for two days, then for a week."
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_is_ambiguous(text) is True
    assert extract_stated_window(text) == "2d"      # first match, alone
    assert _directive_window_spec(text) is None      # but refused overall


@pytest.mark.parametrize(
    ("text", "first"),
    [
        pytest.param(
            "Always use tabs for the next week, then for two days.",
            "1w",
            id="next-form-first",
        ),
        pytest.param(
            "Always use tabs for two days, then for the next week.",
            "2d",
            id="counted-form-first",
        ),
        pytest.param(
            "Always use tabs for the next week, and for the next month.",
            "1w",
            id="both-in-the-next-form",
        ),
    ],
)
def test_ambiguity_sees_both_spellings_of_a_window(text: str, first: str) -> None:
    """Hypothesis: ambiguity is judged over *every* window the text
    states, not only the ones the counted-form pattern can see.

    `"for the next week"` has no count word, so it is matched by
    `_NEXT_UNIT_RE` and is invisible to `_STATED_WINDOW_RE`. An ambiguity
    check that scans only the latter reports a single window on all three
    of these and resolves to whichever one it *can* see — for
    `next-form-first` that is `2d`, the window stated **second**, which
    also contradicts the documented "first stated wins" rule.

    Falsifiable by narrowing `_stated_windows` back to one pattern: each
    param then proposes a concrete spec instead of refusing. Pairs with
    the `first` assertion so that a fix which refuses *everything*
    (making the module useless) fails here too.
    """
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert extract_stated_window(text) == first, "first stated window wins"
    assert stated_window_is_ambiguous(text) is True
    assert _directive_window_spec(text) is None


def test_a_zero_window_beside_a_real_one_still_refuses() -> None:
    """Hypothesis: a zero-length window is *stated*, so a sentence naming
    it alongside a usable window is ambiguous rather than resolvable.

    Falsifiable by dropping zero-length windows from the scan instead of
    keeping them as `None`: the sentence would then look like it names
    exactly one window and would propose `1w`, silently discarding the
    half of the sentence that could not be parsed.
    """
    text = "Always use tabs for 0 days, then for a week."
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_is_ambiguous(text) is True
    assert _directive_window_spec(text) is None


def test_a_non_directive_with_a_window_is_not_proposed() -> None:
    """Hypothesis: both halves are required — a window alone is not a
    directive.

    "the outage lasted for three days" states a window and asks for
    nothing. Falsifiable by dropping the `detect_directive` arm, which
    would propose a lock on ordinary narration.
    """
    text = "The outage lasted for three days."
    assert extract_stated_window(text) == "3d"
    assert _directive_window_spec(text) is None


# --- the proposal, and that it is only a proposal ------------------------


def test_the_rendered_command_carries_the_window(store: MemoryStore) -> None:
    """Hypothesis: a directive stating a window renders a `--for` on its
    pre-filled command. Falsifiable by dropping the suffix."""
    text = _format_stop_prompt([_belief(_DIRECTIVE)])
    assert "aelf lock " in text
    assert "--for 1w" in text


def test_a_directive_without_a_window_renders_no_for_flag(
    store: MemoryStore,
) -> None:
    """Hypothesis: the flag appears only when a window was stated.

    Pairs with the test above — that one renders `--for 1w`, this one
    renders no `--for` at all — so a renderer that always appends a
    default window fails here.
    """
    text = _format_stop_prompt([_belief("Always use tabs in this repo.")])
    assert "aelf lock " in text
    assert "--for" not in text


def test_proposing_writes_nothing_to_the_store(store: MemoryStore) -> None:
    """The distinguishing test: rendering a proposal must not touch the
    store.

    This is the whole of the confirmation gate. Asserting only that a
    directive yields a lock command would pass on a design that writes
    the lock first and shows the command afterwards — and that design is
    exactly what the H1 precision bar existed to prevent. So the
    assertion is on the store being untouched, before and after.
    """
    candidate = _belief(_DIRECTIVE)
    store.insert_belief(candidate)
    before = store.get_belief(candidate.id)
    assert before is not None and before.lock_level == LOCK_NONE
    assert before.lock_expires_at is None

    rendered = _format_stop_prompt([candidate])
    assert "--for 1w" in rendered, "fixture must actually produce a proposal"

    after = store.get_belief(candidate.id)
    assert after is not None
    assert after.lock_level == LOCK_NONE, "proposing locked the belief"
    assert after.lock_expires_at is None, "proposing wrote an expiry"
    assert store.count_feedback_events() == 0, "proposing wrote an audit row"


def test_a_windowed_directive_is_a_candidate_whatever_its_type() -> None:
    """Hypothesis: the candidate predicate admits a windowed directive on
    its own merits, not only via the correction-class arms.

    The belief here is `factual` with `user_transcript` origin — neither
    of the pre-#1315 arms matches — so this fails if the new clause is
    dropped. An already-locked one is still excluded, since locking it
    again is a no-op.
    """
    assert _belief_is_lock_candidate(_belief(_DIRECTIVE), _SESSION) is True
    assert _belief_is_lock_candidate(
        _belief(_DIRECTIVE, lock=LOCK_USER), _SESSION
    ) is False
    assert _belief_is_lock_candidate(
        _belief("The build finished."), _SESSION
    ) is False
