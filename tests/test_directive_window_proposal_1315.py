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

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.directive_detector import detect_directive
from aelfrice.hook import (
    _belief_is_lock_candidate,
    _directive_window_spec,
    _format_stop_prompt,
)
from aelfrice.lock_expiry import (
    extract_stated_window,
    parse_for,
    stated_window_attaches_to_memory,
    stated_window_is_ambiguous,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore

_SESSION = "s-1315"
_TS = "2026-08-06T00:00:00+00:00"

# The window has to be governed by a memory verb (operator ruling
# 2026-08-06). This fixture originally read "Always use tabs in this repo
# for the next week." — which states how long to use tabs, not how long to
# remember the rule, and is exactly the shape the attachment gate now
# refuses. That the PR's own canonical example was a subject-matter window
# is the clearest evidence available that the two are easy to conflate.
_DIRECTIVE = "Always use tabs in this repo. Remember this for the next week."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(_db_path(tmp_path)))


def _db_path(tmp_path: Path) -> Path:
    """The one database both the test and the production path resolve to.

    The store-untouched test is only meaningful if it inspects the store
    a write would actually land in. `_open_store()` resolves `$AELFRICE_DB`,
    so a fixture on any other path would leave the load-bearing assertion
    watching an empty file while the write went elsewhere — and a
    write-first implementation would pass.
    """
    return tmp_path / "pinned.db"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(_db_path(tmp_path)))
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
    # The text must clear EVERY gate upstream of the ambiguity check, or
    # this test passes for the wrong reason. It originally guarded only
    # `detect_directive`; the attachment gate then landed upstream of the
    # ambiguity arm and reopened the trap in a new place, because
    # "use tabs" is not a memory anchor and the fixture was refused one
    # gate earlier. Both are asserted now.
    text = "Always remember this for two days, then for a week."
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the ambiguity arm"
    )
    assert stated_window_is_ambiguous(text) is True
    assert extract_stated_window(text) == "2d"      # first match, alone
    assert _directive_window_spec(text) is None      # but refused overall


@pytest.mark.parametrize(
    ("text", "first"),
    [
        pytest.param(
            "Always remember this for the next week, then for two days.",
            "1w",
            id="next-form-first",
        ),
        pytest.param(
            "Always remember this for two days, then for the next week.",
            "2d",
            id="counted-form-first",
        ),
        pytest.param(
            "Always remember this for the next week, and for the next month.",
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

    Falsifiable by narrowing `_stated_windows` back to one pattern: the
    two counted-form params then propose a concrete spec instead of
    refusing, and `both-in-the-next-form` proposes nothing at all, because
    neither of its windows is visible to the counted pattern. Both are
    caught — the first two by the ambiguity assertion, the third by the
    `first` assertion, which also means a fix that refuses *everything*
    (making the module useless) fails here too.
    """
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the ambiguity arm"
    )
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
    text = "Always remember this for 0 days, then for a week."
    assert detect_directive(text) is True, "fixture must reach the ambiguity arm"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the ambiguity arm"
    )
    assert stated_window_is_ambiguous(text) is True
    assert _directive_window_spec(text) is None


def test_a_non_directive_with_a_window_is_not_proposed() -> None:
    """Hypothesis: both halves are required — a window alone is not a
    directive.

    Falsifiable by dropping the `detect_directive` arm, which would
    propose a lock on ordinary narration. That requires a fixture which
    clears the *attachment* gate and fails only the detector: a question
    about retaining "this" is a memory-anchored window that asks for
    nothing. "The outage lasted for three days." is kept as a second case
    but does not discriminate on its own — it has no memory anchor, so it
    is refused one gate later whether the detector arm exists or not.
    """
    text = "Why would anyone retain this for two years?"
    assert detect_directive(text) is False, "fixture must fail only the detector arm"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must reach the detector arm"
    )
    assert extract_stated_window(text) == "2y"
    assert _directive_window_spec(text) is None
    # Since the decoupling the detector is the *whole* of candidacy, so
    # dropping that arm no longer merely adds a `--for` to narration — it
    # admits narration into the proposal list outright.
    assert _belief_is_lock_candidate(_belief(text), _SESSION) is False

    narration = "The outage lasted for three days."
    assert extract_stated_window(narration) == "3d"
    assert _directive_window_spec(narration) is None
    assert _belief_is_lock_candidate(_belief(narration), _SESSION) is False


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


def test_autolock_does_not_write_a_windowed_directive(
    store: MemoryStore, tmp_path: Path
) -> None:
    """The confirmation gate has to hold on the one path that writes
    without asking.

    `AELF_AUTOLOCK_CORRECTIONS=1` locks candidates unprompted, and it
    grants a *permanent* lock — `_autolock_candidates` sets
    `lock_expires_at = None` and rewrites origin to `user_stated`. With
    the #1315 arm feeding it, a detector measured at P=0.665 mints
    permanent ground truth on a false positive, with the stated window
    discarded: strictly worse than the wrong-expiry case the precision
    bar existed to prevent, and the direct contradiction of "nothing
    reaches the store until they do".

    Driven through `stop()` rather than `_autolock_candidates` directly,
    because the defect was in which candidates reach that call — a
    unit-level test of the helper cannot see it.

    Falsifiable by dropping the `_belief_is_correction_class` filter at
    the `stop()` call site: the directive is then locked and this fails
    on `lock_level`. The correction alongside it is the control — it must
    still be locked, so a filter that disables autolock outright fails
    here too.

    `bare` carries the decoupling: since candidacy no longer requires a
    stated window, the population autolock has to withhold is now every
    directive, not just windowed ones — 3,003 beliefs on this repo's
    store rather than 0. A filter keyed on the window rather than on
    correction-class would pass the `directive` assertions and launder
    this one to `user_stated`.
    """
    directive = _belief(_DIRECTIVE)
    bare = _belief("Always use tabs in this repo.")
    correction = _belief("Actually the flag is --foo, not --bar.")
    correction.type = BELIEF_CORRECTION
    store.insert_belief(directive)
    store.insert_belief(bare)
    store.insert_belief(correction)
    store.close()

    err = io.StringIO()
    hook.stop(
        stdin=io.StringIO(json.dumps({"session_id": _SESSION})),
        stdout=io.StringIO(),
        stderr=err,
        env={"AELF_AUTOLOCK_CORRECTIONS": "1"},
    )

    reopened = MemoryStore(str(_db_path(tmp_path)))
    try:
        after_directive = reopened.get_belief(directive.id)
        assert after_directive is not None
        assert after_directive.lock_level == LOCK_NONE, "autolock wrote a proposal"
        assert after_directive.origin == ORIGIN_USER_TRANSCRIPT, "origin laundered"

        after_bare = reopened.get_belief(bare.id)
        assert after_bare is not None
        assert after_bare.lock_level == LOCK_NONE, (
            "autolock wrote a windowless directive"
        )
        assert after_bare.origin == ORIGIN_USER_TRANSCRIPT, "origin laundered"

        after_correction = reopened.get_belief(correction.id)
        assert after_correction is not None
        assert after_correction.lock_level == LOCK_USER, (
            "control: corrections must still auto-lock"
        )
    finally:
        reopened.close()


def test_autolock_still_proposes_what_it_may_not_write(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Excluding a windowed directive from autolock must not discard it.

    Withholding these from `_autolock_candidates` is correct, but the
    Stop prompt is the only surface a #1315 proposal has: if the
    exclusion also skips the prompt, then `AELF_AUTOLOCK_CORRECTIONS=1`
    is an off-switch for the feature rather than an automation of its
    locking step. The user is left with a belief that is neither locked
    nor suggested — and the block itself recommends setting this very
    flag, so the advice would be advertising its own suppression.

    Falsifiable by restoring the `if/else` at the `stop()` call site:
    the directive is then silently dropped and the `--for` assertion
    fails on empty stderr. The correction is the control in the other
    direction — it is written, so it must NOT also be proposed, which is
    what fails if the exclusion is dropped and everything is prompted.
    """
    directive = _belief(_DIRECTIVE)
    correction = _belief("Actually the flag is --foo, not --bar.")
    correction.type = BELIEF_CORRECTION
    store.insert_belief(directive)
    store.insert_belief(correction)
    store.close()

    err = io.StringIO()
    hook.stop(
        stdin=io.StringIO(json.dumps({"session_id": _SESSION})),
        stdout=io.StringIO(),
        stderr=err,
        env={"AELF_AUTOLOCK_CORRECTIONS": "1"},
    )
    out = err.getvalue()

    assert "--for 1w" in out, "autolock dropped the proposal instead of showing it"
    assert directive.content in out
    assert correction.content not in out, (
        "control: an auto-locked correction is written, so it must not "
        "also be proposed"
    )


def test_a_directive_is_a_candidate_whatever_its_type() -> None:
    """Hypothesis: the candidate predicate admits a directive on its own
    merits, not only via the correction-class arms.

    The belief here is `factual` with `user_transcript` origin — neither
    of the pre-#1315 arms matches — so this fails if the new clause is
    dropped. An already-locked one is still excluded, since locking it
    again is a no-op.

    The negative controls are the two halves of `detect_directive` that
    candidacy now rests on entirely: a statement with no imperative verb,
    and narration that states a duration without being a rule. Without
    the second, the clause could be weakened to "mentions any duration"
    and stay green.
    """
    assert _belief_is_lock_candidate(_belief(_DIRECTIVE), _SESSION) is True
    assert _belief_is_lock_candidate(
        _belief(_DIRECTIVE, lock=LOCK_USER), _SESSION
    ) is False
    assert _belief_is_lock_candidate(
        _belief("The build finished."), _SESSION
    ) is False
    # states a window, but is not a directive
    assert _belief_is_lock_candidate(
        _belief("The outage lasted for three days."), _SESSION
    ) is False


def test_a_directive_whose_window_is_refused_is_still_a_candidate() -> None:
    """Candidacy is decoupled from the `--for` suffix (operator ruling
    2026-08-06). This is the test that pins the decoupling.

    Every fixture is a directive whose window `_directive_window_spec`
    refuses — for a *different* reason each time, so no single gate
    moving back upstream can satisfy all four. All four must still be
    proposed, because the gates exist to prevent a wrong expiry literal,
    not to withhold the proposal.

    Falsifiable by restoring `return _directive_window_spec(b.content) is
    not None` as the candidacy arm, which is the exact shape the ruling
    supersedes: all four then report False. That coupling is what made
    the feature unreachable — 0 candidates against 44,683 active beliefs
    on this repo's store, 3,003 of which pass `detect_directive`.
    """
    refused = [
        # no window stated at all — the overwhelmingly common shape
        "Always use tabs in this repo.",
        # a window, but it is the subject matter's, not the memory's
        "Always keep CI logs for 30 days.",
        # two windows, so the spec refuses rather than resolving one
        "Always remember this for two days, then for a week.",
        # a memory clause, then a new predicate that takes the window
        "Always remember this and cache the index for two weeks.",
    ]
    for text in refused:
        assert detect_directive(text) is True, f"fixture must be a directive: {text}"
        assert _directive_window_spec(text) is None, f"fixture must be refused: {text}"
        assert _belief_is_lock_candidate(_belief(text), _SESSION) is True, (
            f"candidacy re-coupled to the --for suffix: {text}"
        )


def test_a_refused_window_renders_the_command_without_a_for_flag() -> None:
    """The other half of the decoupling: wide candidacy must not widen
    the suffix.

    A subject-matter duration reaches the prompt now, so the attachment
    gate has to hold at the render site rather than being enforced by the
    belief never getting there. Falsifiable by dropping
    `stated_window_attaches_to_memory` from `_directive_window_spec`:
    this renders `--for 30d`, telling the user to forget a retention
    policy in 30 days.
    """
    out = _format_stop_prompt([_belief("Always keep CI logs for 30 days.")])
    assert "aelf lock 'Always keep CI logs for 30 days.'" in out, (
        "the belief must be proposed"
    )
    assert "--for" not in out, "a subject-matter duration became a lock window"


def test_a_correction_class_candidate_never_passed_the_detector() -> None:
    """Why `_directive_window_spec` keeps its own `detect_directive`
    guard now that candidacy applies one upstream.

    `_format_stop_prompt` renders a suffix for *every* candidate, and a
    correction-class belief becomes a candidate by type or origin without
    the directive arm ever running. So candidacy does not imply
    `detect_directive` at the render site — not even after the
    decoupling — and the guard is the only thing standing between a
    correction whose content happens to be a memory-anchored question and
    a `--for 2y` on it.

    Falsifiable by deleting the `detect_directive` arm from
    `_directive_window_spec` as now-redundant, which is the obvious
    cleanup the decoupling invites: this then renders `--for 2y`.
    """
    text = "Why would anyone retain this for two years?"
    correction = _belief(text)
    correction.type = BELIEF_CORRECTION

    assert detect_directive(text) is False, "fixture must fail only the detector"
    assert stated_window_attaches_to_memory(text) is True, (
        "fixture must clear every gate except the detector"
    )
    assert extract_stated_window(text) == "2y"
    assert _belief_is_lock_candidate(correction, _SESSION) is True, (
        "fixture must be a candidate by its correction type, not as a directive"
    )

    out = _format_stop_prompt([correction])
    assert "aelf lock " in out
    assert "--for" not in out, "a question rendered an expiry literal"


# --- the window must be the memory's, not the subject matter's ----------
#
# Operator ruling 2026-08-06. Before this gate the arm fired 9 times on a
# 44,679-belief live store and 0 of the 9 stated a retention window, so
# realized attachment precision was 0. The alternatives considered and
# rejected were shipping the suffix as-is and dropping it entirely.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Always remember this for two weeks.", "2w", id="remember-this"),
        pytest.param("Always keep this for the next month.", "1mo", id="keep-this"),
        pytest.param("Always prioritise this for 30 days.", "30d", id="prioritise"),
        pytest.param("Always retain this rule for two years.", "2y", id="object-noun"),
        pytest.param(
            "Never forget: keep this preference for a week.", "1w", id="after-colon",
        ),
        pytest.param("Always hold on to this for three days.", "3d", id="hold-on-to"),
    ],
)
def test_a_window_governed_by_a_memory_verb_still_proposes(
    text: str, expected: str,
) -> None:
    """The shape the feature exists for must survive the gate.

    Paired with the rejection cases below: without this half, a gate that
    refused everything would pass them all, and refusing everything is a
    real risk here — the live firing rate after this change is 0.
    """
    assert stated_window_attaches_to_memory(text) is True
    assert _directive_window_spec(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Always retain build artifacts for 90 days.", id="artifacts"),
        pytest.param("Always keep CI logs for 30 days.", id="ci-logs"),
        pytest.param("Never cache the index for two weeks.", id="cache"),
        pytest.param("Always support each release for two years.", id="release"),
        pytest.param(
            "Always keep the branch blocked for 9 days after review.", id="blocked",
        ),
        pytest.param(
            "Always keep results available for 29 days after creation.",
            id="results-available",
        ),
    ],
)
def test_a_subject_matter_duration_gets_no_for_suffix(text: str) -> None:
    """`for 90 days` is a property of the artifacts, not of the memory.

    Every string here is a directive and states a countable window, so
    both pre-gate halves pass and only attachment separates them. The
    first four are the shapes review measured on live data; running the
    suggestion they used to produce would forget the retention policy on
    a date the user never chose.

    Named for the *suffix*, not the proposal: since the 2026-08-06
    decoupling ruling these are all still proposed — as a permanent lock,
    which is what the user would get by typing `aelf lock` themselves.
    Only the window is withheld. `test_a_directive_whose_window_is_
    refused_is_still_a_candidate` pins that other half.
    """
    assert detect_directive(text) is True
    assert extract_stated_window(text) is not None
    assert stated_window_attaches_to_memory(text) is False
    assert _directive_window_spec(text) is None


def test_the_memory_verb_alone_does_not_attach_a_window() -> None:
    """The self-referential object is what carries the gate.

    An anchor of just the verb admits `keep CI logs for 30 days` — the
    shape the gate exists to reject, since the 30 days belongs to the
    logs. This pins that the object is required, so a future
    simplification to a keyword test fails here. (It is an illustrative
    shape, not a measured one: retention-of-CI-logs appears in none of the
    nine live hits, and the only beliefs in that store mentioning CI logs
    state no duration at all.)
    """
    assert stated_window_attaches_to_memory("Always keep CI logs for 30 days.") is False
    assert stated_window_attaches_to_memory("Always keep this for 30 days.") is True


def test_an_anchor_in_a_different_clause_does_not_govern() -> None:
    """A memory verb earlier in the text is not a licence for any window.

    Without the clause-break and gap rules, any belief that says
    "remember this" anywhere would attach the next duration it mentions,
    which reintroduces the defect a sentence later.
    """
    assert stated_window_attaches_to_memory(
        "Always remember this. Unrelatedly, the retention policy is for 90 days."
    ) is False
    assert stated_window_attaches_to_memory(
        "Always remember to rotate the key. Builds are kept for 30 days."
    ) is False
    assert stated_window_attaches_to_memory(
        "Always remember this rule, which the team agreed after a long "
        "discussion last Thursday, for 30 days."
    ) is False


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "Always remember this, and cache the index for two weeks.", id="and-comma"
        ),
        pytest.param(
            "Always remember this and cache the index for two weeks.", id="and"
        ),
        pytest.param(
            "Always remember this but cache the index for two weeks.", id="but"
        ),
        pytest.param(
            "Always remember this then cache the index for two weeks.", id="then"
        ),
        pytest.param(
            "Always remember this while caching the index for two weeks.", id="while"
        ),
        pytest.param(
            "Always remember this so cache the index for two weeks.", id="so"
        ),
        pytest.param(
            "Always remember this - cache the index for two weeks.", id="hyphen"
        ),
        pytest.param(
            "Always remember this — cache the index for two weeks.", id="em-dash"
        ),
    ],
)
def test_a_new_predicate_after_the_anchor_takes_the_window_with_it(
    text: str,
) -> None:
    """Prefixing a memory clause must not license a subject-matter window.

    `Always cache the index for two weeks.` is refused, and it has to stay
    refused when a memory clause is put in front of it: the two weeks is
    still how long to cache, not how long to remember. This is the gate's
    cheapest bypass — the attacker is ordinary English, not adversarial
    input — and punctuation cannot see it, because none of these spellings
    needs a comma. Adding `,` to `_CLAUSE_BREAKS` closes `and-comma` only
    and leaves the other seven open.

    Falsifiable by dropping the connective half of
    `_gap_opens_a_new_predicate`: all eight then propose `2w`.
    """
    assert detect_directive(text) is True, "fixture must reach the gate"
    assert extract_stated_window(text) == "2w", "fixture must state a window"
    assert stated_window_attaches_to_memory(text) is False
    assert _directive_window_spec(text) is None


def test_the_bare_connective_does_not_swallow_a_hyphenated_word() -> None:
    """`-` is in the connective set; `build-time` must survive it.

    Tokens are stripped of surrounding punctuation rather than split on
    it, so the clause-joining bare `-` and the compound-word character are
    distinguishable. Falsifiable by splitting the gap on punctuation
    instead of stripping it: this proposes nothing.
    """
    text = "Always remember this build-time rule for a week."
    assert _directive_window_spec(text) == "1w"
