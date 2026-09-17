"""Bounds on the injected block: per-belief cap and block ceiling (#1551).

The drop policy these tests pin, in one sentence: **both bounds stop at
`lock="user"`.** A lock is never capped and never dropped, and a block
that cannot fit without dropping one is emitted over the ceiling with a
note on stderr. That is the #379 / #1016-B contract — locks are the
always-injected pool — and a ceiling that deleted them would have made
this module's own bound the thing that broke it.

Sizes here are written as literals, not derived from
`BELIEF_CONTENT_CHAR_CAP` or `HOOK_BLOCK_TOKEN_CEILING`. A fixture built
from the constant under test moves with it, so it cannot detect the
constant moving; `test_shipped_constants_are_pinned` is the other half of
that and asserts the two values outright.

Nothing here reads the ambient environment: every ceiling resolution
passes an explicit mapping, or `monkeypatch` pins the variable. An
exported `AELFRICE_HOOK_BLOCK_CEILING` used to red two of these.
"""
from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

import aelfrice.hook as hook_mod
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)

# Bound off the one module object rather than imported a second way. Two
# tests reach for attributes through `hook_mod`, so that name has to exist;
# adding `from aelfrice.hook import ...` beside it imports the same module
# twice and trips CodeQL's py/import-and-import-from.
BELIEF_CONTENT_CHAR_CAP = hook_mod.BELIEF_CONTENT_CHAR_CAP
HOOK_BLOCK_TOKEN_CEILING = hook_mod.HOOK_BLOCK_TOKEN_CEILING
_audit_tokens_from_block = hook_mod._audit_tokens_from_block
_belief_element_line = hook_mod._belief_element_line
_cap_belief_content = hook_mod._cap_belief_content
_escape_for_hook_block = hook_mod._escape_for_hook_block
_format_hits = hook_mod._format_hits
_split_belief_lines = hook_mod._split_belief_lines
_ups_belief_line_cost = hook_mod._ups_belief_line_cost
_write_memory_block = hook_mod._write_memory_block
enforce_block_ceiling = hook_mod.enforce_block_ceiling
resolve_block_ceiling = hook_mod.resolve_block_ceiling

_CEILING_ENV = "AELFRICE_HOOK_BLOCK_CEILING"

# Literal sizes, held here so a reader can see the arithmetic. The
# estimator is 4 characters per token, so 24,000 characters is exactly the
# shipped 6,000-token ceiling and an element of 4,000 characters is 1,000
# tokens of it.
_CEILING_CHARS = 24_000
_ELEMENT_CHARS = 4_000


def _element(bid: str, chars: int, *, locked: bool = False) -> str:
    attr = 'lock="user"' if locked else 'lock="none"'
    return f'<belief id="{bid}" {attr}>{"x" * chars}</belief>\n'


def _seen(bid: str) -> str:
    """One `seen` manifest line, as `hook._split_belief_lines` emits it.

    Two spaces of indent and a quoted topic: the shape
    `hook._SEEN_MANIFEST_RE` matches.
    """
    return f'  seen {bid}: "a topic"\n'


def _block(*elements: str) -> str:
    return "<aelfrice-memory>\n" + "".join(elements) + "</aelfrice-memory>"


def _ids(body: str) -> list[str]:
    return [m.group("id") for m in hook_mod._BELIEF_ELEMENT_RE.finditer(body)]


# ---------------------------------------------------------------------------
# The shipped values
# ---------------------------------------------------------------------------


def test_shipped_constants_are_pinned() -> None:
    """Both constants, asserted as literals, in the unsafe direction.

    **What each raise costs is published as the tests it reds, by file,
    and not as a pass total.** Three earlier versions of this docstring
    quoted totals and all three went stale at once when the branch was
    rebased; a name survives a rebase and "eight" does not. Measured with
    `uv run pytest tests -q -p no:randomly` on `fe90b455`, the last commit
    on this branch that changes anything a test can see.

    `BELIEF_CONTENT_CHAR_CAP = 30000` — the value that reinstates the
    24k-35k character rows #1551 exists to bound — reds **ten across four
    files**:

    * here, three: `test_shipped_constants_are_pinned`,
      `test_content_one_character_over_the_cap_is_truncated`,
      `test_oversized_content_is_capped_and_says_so`;
    * `test_hook_injection_ceiling_wiring.py`, three:
      `test_ups_total_chars_stays_in_one_unit_across_the_ceiling`,
      `test_ups_caps_one_oversized_belief_instead_of_dropping_the_block`,
      `test_core_section_caps_an_oversized_belief`;
    * `test_exploration_slot_1279.py`, two:
      `test_a_drawn_belief_is_priced_at_the_element_the_lane_renders`,
      `test_a_displaced_belief_frees_only_what_the_lane_would_have_shipped`,
      whose prices are quoted at the shipped cap;
    * `test_render_cost_1526.py`, two:
      `test_session_start_lane_never_trims_its_l0_pool`, both parametrised
      cases, on `assert shortest > largest_cap` — reported as
      `assert 1225 > 30000`.

    **That fourth file is the counter-example, and the cause is worth
    stating rather than just counting.** `_shipped_content_caps()` there
    discovers every per-belief cap constant in `aelfrice.hook` and
    `aelfrice.hook_search_tool` by name shape — `CHAR_CAP`,
    `MAX_CONTENT`, `_CHARS` — and sizes its SessionStart fixture above the
    largest one it finds. `BELIEF_CONTENT_CHAR_CAP` joined that discovered
    set the moment #1551 named it, and at 1200 it is already the largest
    member (the others are 1000, 200, 80 and 60), so the fixture's 1225-
    character locks clear it by 25. Raise it and they do not. That is
    #1526's discovery mechanism working exactly as designed, and it is the
    same shape of counter-example as the ceiling raise's.

    `HOOK_BLOCK_TOKEN_CEILING = 40 * DEFAULT_HOOK_TOKEN_BUDGET` reds
    **twenty-six across three files**. Seven here:
    `test_shipped_constants_are_pinned`, the four that resolve the value
    (`test_unset_env_resolves_the_shipped_ceiling`,
    `test_non_integer_falls_back_to_the_default_with_a_note`,
    `test_negative_falls_back_to_the_default_with_a_note`,
    `test_resolve_reads_os_environ_when_no_mapping_is_given`, the last
    three because the note quotes the default in its text), and the two
    that drive `_write_memory_block`
    (`test_write_memory_block_trims_notes_and_writes`,
    `test_write_memory_block_notes_an_unavoidable_overrun`). Eighteen in
    the wiring file, which asserts a trim or an overrun on nearly every
    fixture it has. And one in `test_derived_figures_1469.py`:
    `test_the_repo_passes_the_producer_checks`, which reports
    `producer scripts/measure_block_ceiling.py exited 1 under
    --emit-figures: the ceiling dropped nothing: the comparison would be
    vacuous`. The figures this branch publishes are bound to their
    producer, so a constant that moves them fails CI whether or not a
    #1551 test names it.
    """
    assert BELIEF_CONTENT_CHAR_CAP == 1200
    assert HOOK_BLOCK_TOKEN_CEILING == 6000
    # The arithmetic the fixtures in this file are sized by, asserted
    # rather than left in a comment: the audit estimator charges four
    # characters per token, so `_CEILING_CHARS` is the shipped ceiling
    # expressed in the unit the fixtures build bodies in. A test that
    # states the relation in prose and sizes its bodies by hand cannot
    # notice the estimator changing underneath it.
    assert _audit_tokens_from_block("x" * _CEILING_CHARS) == (
        HOOK_BLOCK_TOKEN_CEILING
    )
    assert _audit_tokens_from_block("x" * (_CEILING_CHARS + 4)) > (
        HOOK_BLOCK_TOKEN_CEILING
    )


# ---------------------------------------------------------------------------
# _cap_belief_content
# ---------------------------------------------------------------------------


def test_content_below_the_cap_is_returned_unchanged() -> None:
    content = "a" * 1199
    assert _cap_belief_content(content) == content


def test_content_at_exactly_the_cap_is_returned_unchanged() -> None:
    """The boundary the `<=` in the shipped code decides.

    Without this case, flipping `<=` to `<` leaves the suite green: every
    other input is either comfortably short or comfortably long.
    """
    content = "a" * 1200
    assert _cap_belief_content(content) == content


def test_content_one_character_over_the_cap_is_truncated() -> None:
    capped = _cap_belief_content("a" * 1201)
    assert capped == "a" * 1200 + " […truncated]"


def test_oversized_content_is_capped_and_says_so() -> None:
    capped = _cap_belief_content("a" * 35_164)
    assert capped == "a" * 1200 + " […truncated]"


def test_user_locked_content_is_never_capped() -> None:
    """The drop policy, at the cap end.

    A lock cut mid-clause can assert the opposite of what the operator
    locked, and unlike a retrieval hit nothing ranked it here for the
    model to discount. So an oversized lock is emitted whole; the
    bounded alternative is the reference tier, which since #1558 reaches
    the first-prompt `<locked>` render as well as the per-turn one.
    """
    content = "a" * 35_164
    assert _cap_belief_content(content, locked=True) == content


# ---------------------------------------------------------------------------
# resolve_block_ceiling
# ---------------------------------------------------------------------------


def test_unset_env_resolves_the_shipped_ceiling() -> None:
    assert resolve_block_ceiling({}) == 6000


def test_explicit_value_wins() -> None:
    assert resolve_block_ceiling({_CEILING_ENV: "250"}) == 250


def test_literal_zero_disables_the_ceiling() -> None:
    assert resolve_block_ceiling({_CEILING_ENV: "0"}) == 0


def test_non_integer_falls_back_to_the_default_with_a_note() -> None:
    serr = io.StringIO()
    assert resolve_block_ceiling({_CEILING_ENV: "not-a-number"}, stderr=serr) == 6000
    assert "not an integer" in serr.getvalue()
    assert "only 0 disables it" in serr.getvalue()


def test_negative_falls_back_to_the_default_with_a_note() -> None:
    """`-1` used to disable the only bound on the injected block.

    `max(value, 0)` mapped every negative to 0, and 0 is the documented
    disable value — so a typo in the override silently removed the
    backstop, with nothing said. Only a literal 0 disables.
    """
    serr = io.StringIO()
    assert resolve_block_ceiling({_CEILING_ENV: "-1"}, stderr=serr) == 6000
    assert "negative" in serr.getvalue()


def test_resolve_reads_os_environ_when_no_mapping_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `env=None` default, pinned without depending on the shell."""
    monkeypatch.setenv(_CEILING_ENV, "123")
    assert resolve_block_ceiling() == 123
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    assert resolve_block_ceiling() == 6000


# ---------------------------------------------------------------------------
# enforce_block_ceiling
# ---------------------------------------------------------------------------


def test_block_under_the_ceiling_is_returned_byte_identical() -> None:
    body = _block(_element("a" * 16, 100))
    out = enforce_block_ceiling(body, 6000)
    assert out.body == body
    assert out.dropped_ids == ()
    assert out.over_ceiling is False


def test_a_disabled_ceiling_returns_the_body_untouched() -> None:
    body = _block(_element("a" * 16, 400_000))
    out = enforce_block_ceiling(body, 0)
    assert out.body == body
    assert out.dropped_ids == ()
    assert out.over_ceiling is False


def test_over_ceiling_drops_whole_unlocked_elements_from_the_end() -> None:
    body = _block(*(_element(str(i) * 16, _ELEMENT_CHARS) for i in range(10)))
    assert _audit_tokens_from_block(body) > 6000
    out = enforce_block_ceiling(body, 6000)
    assert 0 < out.n_dropped < 10
    assert _audit_tokens_from_block(out.body) <= 6000
    assert out.over_ceiling is False
    # Framing survives and no element is left half-removed.
    assert out.body.startswith("<aelfrice-memory>")
    assert out.body.endswith("</aelfrice-memory>")
    assert out.body.count("<belief ") == out.body.count("</belief>")
    # Dropping is from the end: the highest-ranked element is kept.
    assert "0" * 16 in _ids(out.body)
    assert "9" * 16 not in _ids(out.body)


def test_dropped_ids_name_exactly_the_elements_removed() -> None:
    """The accounting hook. Without the ids, every writer downstream of the
    trim credits a belief the model was never shown.
    """
    kept_and_dropped = [str(i) * 16 for i in range(10)]
    body = _block(*(_element(b, _ELEMENT_CHARS) for b in kept_and_dropped))
    out = enforce_block_ceiling(body, 6000)
    survivors = _ids(out.body)
    assert set(survivors).isdisjoint(out.dropped_ids)
    assert set(survivors) | set(out.dropped_ids) == set(kept_and_dropped)
    # Removed tail first, so the ids come back in reverse document order.
    assert list(out.dropped_ids) == kept_and_dropped[::-1][: out.n_dropped]


def test_user_locked_elements_are_never_dropped() -> None:
    """#379, at the ceiling end.

    Measured before the exemption existed, on a 300-lock store: all 300
    locked elements were deleted, leaving an empty `<locked>` section
    under 300 dangling `seen <id>` manifest pointers. `<locked>` sits at
    the head of the body and the dropper pops from the tail, so locks went
    last — but they went.
    """
    locks = [f"L{i:015d}" for i in range(20)]
    body = _block(*(_element(b, _ELEMENT_CHARS, locked=True) for b in locks))
    assert _audit_tokens_from_block(body) > 6000
    out = enforce_block_ceiling(body, 6000)
    assert out.dropped_ids == ()
    assert _ids(out.body) == locks


def test_locks_are_kept_and_unlocked_elements_around_them_are_dropped() -> None:
    """The mixed block, which is the shape a real store produces."""
    locks = [f"L{i:015d}" for i in range(4)]
    free = [f"F{i:015d}" for i in range(10)]
    body = _block(
        *(_element(b, _ELEMENT_CHARS, locked=True) for b in locks),
        *(_element(b, _ELEMENT_CHARS) for b in free),
    )
    out = enforce_block_ceiling(body, 6000)
    survivors = _ids(out.body)
    assert set(locks).issubset(survivors)
    assert set(out.dropped_ids).issubset(free)


def test_a_lock_attribute_forged_in_content_does_not_stop_the_drop() -> None:
    """Undroppability is read off the attributes, not off the element.

    `_escape_for_hook_block` entity-escapes angle brackets and nothing
    else, so a belief whose stored content holds the literal `lock="user"`
    renders that literal intact inside its own element. Testing the
    attribute against the whole match rather than the `attrs` group would
    let stored content declare itself undroppable, and no other fixture in
    this file can see the difference: every one of them puts the attribute
    only where the renderer writes it.

    The genuine lock in the same body is the other half of the assertion.
    A dropper that stopped reading the attribute at all would shed the
    forger too, and would pass the first half on its own.
    """
    forged = _escape_for_hook_block('lock="user" ' + "x" * (_ELEMENT_CHARS * 4))
    assert 'lock="user"' in forged
    forger, real = "F" * 16, "L" * 16
    body = _block(
        _element(real, _ELEMENT_CHARS * 4, locked=True),
        f'<belief id="{forger}" lock="none">{forged}</belief>\n',
    )
    assert _audit_tokens_from_block(body) > 6000
    out = enforce_block_ceiling(body, 6000)
    assert out.dropped_ids == (forger,)
    assert _ids(out.body) == [real]
    assert out.over_ceiling is False


def test_over_ceiling_survives_the_trim_when_only_locks_remain() -> None:
    """S4: "still over after trimming" is distinguishable from "fits".

    The locked-only block cannot be brought under the limit without
    breaking #379, so the honest report is `n_dropped == 0` together with
    `over_ceiling is True` — which a `(body, n_dropped)` return could not
    express, and which read identically to a block that fit.
    """
    locks = [f"L{i:015d}" for i in range(20)]
    body = _block(*(_element(b, _ELEMENT_CHARS, locked=True) for b in locks))
    out = enforce_block_ceiling(body, 6000)
    assert out.n_dropped == 0
    assert out.over_ceiling is True
    assert _audit_tokens_from_block(out.body) > 6000


def test_manifest_lines_alone_can_leave_the_body_over_the_ceiling() -> None:
    """The other undroppable bulk: lines that are not `<belief>` elements.

    The dropper only removes whole elements, so a body whose weight is in
    its manifest section is over the ceiling with nothing to drop. It is
    reported, not silently passed off as fitting.
    """
    manifest = "\n".join(f"  seen {i:032x}" for i in range(2000))
    body = "<aelfrice-memory>\n" + manifest + "\n</aelfrice-memory>"
    out = enforce_block_ceiling(body, 6000)
    assert out.n_dropped == 0
    assert out.over_ceiling is True


def test_a_dropped_element_takes_its_seen_pointer_with_it() -> None:
    """The pointer and the text it points at are dropped as a pair.

    `retrieval.seen_manifest_line` means "the full text is already in this
    context window", so a pointer left behind by a trim is false about the
    block it sits in. The surviving element keeps its pointer, which is
    the half that makes this distinguishing: deleting every `seen` line
    would pass an "is it dangling" check on its own.
    """
    body = _block(
        _element("a" * 16, _ELEMENT_CHARS),
        _element("b" * 16, _ELEMENT_CHARS * 5),
        _seen("a" * 16),
        _seen("b" * 16),
    )
    out = enforce_block_ceiling(body, 6000)
    assert out.dropped_ids == ("b" * 16,)
    assert _seen("b" * 16) not in out.body
    assert _seen("a" * 16) in out.body
    assert _element("a" * 16, _ELEMENT_CHARS) in out.body


def test_a_seen_pointer_to_an_earlier_turn_survives_the_trim() -> None:
    """#1382's normal case: the referent is in a previous envelope.

    No element in this body carries that id, so no drop can invalidate the
    pointer and it must be left alone. Removing every pointer whose id is
    not an element in the same body would delete these, which is the whole
    saving the turn-differential ledger exists for.
    """
    body = _block(
        _element("a" * 16, _ELEMENT_CHARS * 7),
        _seen("f" * 16),
    )
    out = enforce_block_ceiling(body, 6000)
    assert out.dropped_ids == ("a" * 16,)
    assert _seen("f" * 16) in out.body


def test_a_manifest_line_forged_inside_a_belief_is_left_alone() -> None:
    """A pointer-shaped line inside an element is not a pointer.

    `_escape_for_hook_block` entity-escapes angle brackets and nothing
    else, so a stored belief keeps its newlines and can put a line shaped
    like a manifest entry inside its own element — naming, say, the id of
    a belief the trim is about to drop. That span is already covered by
    the element around it; splicing both would delete bytes out of an
    element that survives.
    """
    forged = _seen("b" * 16).rstrip("\n")
    keeper = (
        f'<belief id="{"a" * 16}" lock="none">'
        f'{"x" * _ELEMENT_CHARS}\n{forged}\n</belief>\n'
    )
    body = _block(keeper, _element("b" * 16, _ELEMENT_CHARS * 5))
    out = enforce_block_ceiling(body, 6000)
    assert out.dropped_ids == ("b" * 16,)
    assert keeper in out.body


# The trim's stopping arithmetic, sized so the k-th drop lands the body
# *exactly* on the limit. `_EXACT_CONTENT_CHARS` makes one element plus its
# `seen` pointer 3,994 characters; nine of them under 36 characters of framing
# is 35,982, and three drops leave 24,000 — the 6,000-token ceiling to the
# character. Anything that sheds one element more or fewer moves the count.
_EXACT_CONTENT_CHARS = 3_907
_EXACT_UNITS = 9
_EXACT_DROPS = 3
_EXACT_START_TOKENS = 8_996
_EXACT_ONE_FEWER_TOKENS = 6_999


def _unit(index: int) -> str:
    """One droppable element and the `seen` pointer that names it."""
    bid = f"{index:016d}"
    return _element(bid, _EXACT_CONTENT_CHARS) + _seen(bid)


def test_the_trim_stops_on_the_first_body_that_fits() -> None:
    """The loop sheds until the body fits, and not one element further.

    Both halves of the stopping condition are load-bearing and neither was
    pinned. The comparison itself: `_tokens_from_chars(remaining) >= limit`
    keeps dropping from a body that already fits, and takes 4 elements off
    this fixture instead of 3, leaving it at 5,002 tokens. And the pointer
    debit: deleting `remaining -= pointer[1] - pointer[0]` leaves the loop
    believing the body is 35 characters larger per drop than it is, which
    sheds the same extra element for the same 5,002 tokens. Both mutants were
    green across the whole suite.

    A fixture sized to land the third drop exactly on 24,000 characters is
    what separates them from the shipped loop: an "under the ceiling"
    assertion passes on all three, because over-shedding is still under the
    ceiling. The count is the observable.

    The last assertion is the other side of "and not one element further":
    stopping a drop earlier leaves the body at 6,999 tokens, so 3 is the
    fewest drops that fit rather than merely a number the loop reached.
    """
    units = [_unit(i) for i in range(_EXACT_UNITS)]
    body = _block(*units)
    assert _audit_tokens_from_block(body) == _EXACT_START_TOKENS

    out = enforce_block_ceiling(body, 6000)

    assert out.n_dropped == _EXACT_DROPS
    assert _audit_tokens_from_block(out.body) == 6000
    assert out.over_ceiling is False
    # Tail-first within the lane, so the survivors are the leading units.
    assert out.body == _block(*units[: _EXACT_UNITS - _EXACT_DROPS])
    # One fewer drop does not fit.
    assert _audit_tokens_from_block(
        _block(*units[: _EXACT_UNITS - _EXACT_DROPS + 1])
    ) == _EXACT_ONE_FEWER_TOKENS


# ---------------------------------------------------------------------------
# _write_memory_block — the single emit path
# ---------------------------------------------------------------------------


def test_write_memory_block_trims_notes_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    body = _block(*(_element(str(i) * 16, _ELEMENT_CHARS) for i in range(10)))
    sout, serr = io.StringIO(), io.StringIO()
    out = _write_memory_block(body, stdout=sout, stderr=serr)
    assert sout.getvalue() == out.body
    assert _audit_tokens_from_block(sout.getvalue()) <= 6000
    assert "dropped" in serr.getvalue()
    assert "still over" not in serr.getvalue()


def test_write_memory_block_notes_an_unavoidable_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    locks = [f"L{i:015d}" for i in range(20)]
    body = _block(*(_element(b, _ELEMENT_CHARS, locked=True) for b in locks))
    sout, serr = io.StringIO(), io.StringIO()
    out = _write_memory_block(body, stdout=sout, stderr=serr)
    assert sout.getvalue() == body
    assert out.dropped_ids == ()
    err = serr.getvalue()
    assert "still over the 6000-token ceiling" in err
    assert "never happens (#379)" in err
    # The note names the remedy again. It was withdrawn while the `<locked>`
    # loop of `_build_session_start_subblock` rendered a reference lock
    # verbatim, because on a session's first prompt — the fire a lock-only
    # store overruns on — demoting a lock changed nothing. #1558 gave that
    # loop the diversion `_split_belief_lines` had, so the advice is now
    # true on every write this function bounds and it no longer has to tell
    # them apart from a body. Re-derived at 273 / 273 / 256 / 233 reference
    # against 7700 / 7796 / 7683 / 7660 frozen by
    # `scripts/measure_block_ceiling.py --reference-tier`, whose own
    # per-write vacuity guard reds if any one of those pairs is equal.
    assert "`aelf lock --reference`" in err
    assert "#1558" not in err


def test_write_memory_block_is_silent_on_a_block_that_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    body = _block(_element("a" * 16, 100))
    sout, serr = io.StringIO(), io.StringIO()
    _write_memory_block(body, stdout=sout, stderr=serr)
    assert sout.getvalue() == body
    assert serr.getvalue() == ""


# ---------------------------------------------------------------------------
# _ups_belief_line_cost — the reference-lock arm
# ---------------------------------------------------------------------------

# One belief text, rendered two ways by the tier alone. 20,019 characters
# is long enough that the two costs are three orders of magnitude apart,
# so neither assertion below can be satisfied by the other's value.
_REF_CONTENT = "reference material " + "a" * 20_000

# Produced by `uv run python -c` over `hook._ups_belief_line_cost` on the
# two beliefs `_ref_lock` builds; see the docstring below for why they are
# literals rather than a formula re-derived here.
_REF_MANIFEST_TOKENS = 31
_REF_ELEMENT_TOKENS = 5022


def _ref_lock(bid: str, tier: str) -> Belief:
    return Belief(
        id=bid,
        content=_REF_CONTENT,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        lock_tier=tier,
        locked_at="2026-04-26T00:00:00Z",
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def test_reference_lock_costs_its_manifest_line_not_its_element() -> None:
    """A reference lock is charged what it renders: one manifest line.

    `belief_cost_fn` overrides `retrieval.lock_injection_tokens`
    entirely (`retrieval.retrieve_with_tiers`), so this lane has to make
    the #1016-B bound itself — a lane that renders its own shape renders
    locks in that shape as well. Replacing the branch with the
    unconditional element cost is revert-green and measurably halves the
    hits that reach the model: this 20,019-character lock goes from 31
    tokens to 5,022, `locked_used` then swamps `DEFAULT_HOOK_TOKEN_BUDGET`
    (1,500), and the relevance budget collapses to its floor.

    The two costs are literals rather than arithmetic re-derived here,
    because a formula transcribed from `_ups_belief_line_cost` moves with
    it and could not detect it changing. They are pinned in the same
    assertion, so the shipped value is named and the wrong value is named
    with it.

    The second assertion ties both to the renderer: `_split_belief_lines`
    must actually emit this belief as a manifest entry and no element, or
    the cost being charged is not the cost of the line that ships.
    """
    ref = _ref_lock("R" + "0" * 31, LOCK_TIER_REFERENCE)
    frozen = _ref_lock("F" + "0" * 31, LOCK_TIER_FROZEN)
    assert {
        "reference": _ups_belief_line_cost(ref),
        "frozen": _ups_belief_line_cost(frozen),
    } == {
        "reference": _REF_MANIFEST_TOKENS,
        "frozen": _REF_ELEMENT_TOKENS,
    }

    belief_lines, manifest_lines = _split_belief_lines([ref])
    assert (belief_lines, len(manifest_lines)) == ([], 1)
    assert manifest_lines[0].startswith(f'  ref {ref.id}: "')


# ---------------------------------------------------------------------------
# _ups_belief_line_cost — the newline the join adds
# ---------------------------------------------------------------------------

# Two ordinary beliefs one content character apart, sized so the newline
# charge discriminates on one of them and not the other. The element
# around 1,001 characters of content is 1,068 characters, an exact
# multiple of the 4-character estimator, so the joining newline is the
# character that crosses into a 268th token; 1,000 characters render
# 1,067 and cost 267 whether or not the newline is charged.
_CROSSING_CONTENT_CHARS = 1_001
_CROSSING_ELEMENT_CHARS = 1_068
_CROSSING_TOKENS = 268
_FLAT_CONTENT_CHARS = 1_000
_FLAT_TOKENS = 267


def _plain_belief(bid: str, chars: int) -> Belief:
    """An unlocked, non-speculative belief of `chars` content characters."""
    return Belief(
        id=bid,
        content="x" * chars,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def test_element_cost_charges_the_newline_the_block_ships() -> None:
    """The cost includes the newline `_format_hits` joins the lines with.

    `_belief_element_line` returns the element without its separator, and
    `_format_hits` emits the lines through `"\\n".join`, so every element
    in the block costs one character more than the renderer returns.
    Deleting the `+ 1` from `_ups_belief_line_cost` is not a no-op: it
    under-charges by one token every belief whose element length is an
    exact multiple of the estimator's 4 characters, which moves admission
    at the budget boundary.

    The reference-lock test above cannot see that deletion. Its element is
    20,086 characters, so 20,086 and 20,087 both round up to 5,022, and
    its manifest line lands the same way at 31 tokens — both literals are
    identical with and without the term. The pair here is chosen to
    straddle a boundary instead: the crossing belief is the assertion the
    deletion reds, and the flat belief holds the value the deletion would
    charge for both, so the wrong answer is named beside the right one.

    The second assertion ties the charge to what ships. The block has to
    contain this element followed by a newline, or the byte being charged
    is not a byte the lane emits.
    """
    crossing = _plain_belief("E" + "0" * 31, _CROSSING_CONTENT_CHARS)
    flat = _plain_belief("F" + "1" * 31, _FLAT_CONTENT_CHARS)
    element = _belief_element_line(crossing)
    assert {
        "element_chars": len(element),
        "crossing": _ups_belief_line_cost(crossing),
        "flat": _ups_belief_line_cost(flat),
    } == {
        "element_chars": _CROSSING_ELEMENT_CHARS,
        "crossing": _CROSSING_TOKENS,
        "flat": _FLAT_TOKENS,
    }

    assert element + "\n" in _format_hits([crossing])


# ---------------------------------------------------------------------------
# One emit path, structurally
# ---------------------------------------------------------------------------


def test_enforce_block_ceiling_has_exactly_one_caller_in_hook_py() -> None:
    """A fourth emit site cannot be added unbounded.

    The behavioural tests in `test_hook_injection_ceiling_wiring.py` pin
    the three sites that exist. This pins the property that made those
    three fixable at all: the trim is reachable from one function, so a
    new `sout.write` of a memory block either goes through
    `_write_memory_block` or does not get a ceiling — and the reviewer of
    that change is looking at this assertion.
    """
    src = Path(hook_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    callers: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "enforce_block_ceiling"
            ):
                callers.append(func.name)
    assert callers == ["_write_memory_block"], callers
