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
from aelfrice.hook import (
    BELIEF_CONTENT_CHAR_CAP,
    HOOK_BLOCK_TOKEN_CEILING,
    _audit_tokens_from_block,
    _cap_belief_content,
    _write_memory_block,
    enforce_block_ceiling,
    resolve_block_ceiling,
)

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

    Raising `BELIEF_CONTENT_CHAR_CAP` from 1200 to 30000 reinstates the
    24k-35k character rows #1551 exists to bound, and every other test in
    this file still passes: they assert that the cap is applied, not what
    it is. The ceiling is the same shape — a larger number is a weaker
    bound and nothing else notices.
    """
    assert BELIEF_CONTENT_CHAR_CAP == 1200
    assert HOOK_BLOCK_TOKEN_CEILING == 6000


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
    model to discount. `aelf lock --reference` is the bounded form.
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
    assert "never dropped" in err
    assert "--reference" in err


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
