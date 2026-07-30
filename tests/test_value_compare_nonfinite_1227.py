"""Non-finite numeric slots must never reach the comparator (#1227).

`float()` does not raise on overflow, it saturates to `inf`. The exponent
branch of `_NUMERIC_RE` matches abbreviated git SHAs — `592e701` is a hex
string that happens to hold one `e` between digits — so parsing it as
scientific notation yields `inf`. `_format_number` then narrowed with a bare
`int(x)` and raised `OverflowError`.

Both halves are pinned here: the extractor must not admit the slot, and the
formatter must survive a non-finite input regardless of how it got one. Either
fix alone leaves the other path live, so both are asserted.

The literals below are real values found on a production store, where the
crash surfaced through `aelf search` under `AELF_SHOW_CONFLICTS=1`.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.contradiction import _slot_conflict_preextracted, extract_values
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.value_compare import _format_number, find_conflicts

# Overflow to +/-inf when read as scientific notation. The first two are
# abbreviated commit SHAs.
OVERFLOWING = ("592e701", "1e124732", "4e999", "-3e400")


def _mk(bid: str, content: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-07-30T00:00:00Z" if locked else None,
        created_at="2026-07-30T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.mark.parametrize("literal", OVERFLOWING)
def test_the_literal_really_does_overflow(literal: str) -> None:
    """Guard the premise. If `float()` ever stopped saturating, every other
    test in this file would pass vacuously against a finite value."""
    assert not math.isfinite(float(literal))


@pytest.mark.parametrize("literal", OVERFLOWING)
def test_extractor_drops_the_non_finite_slot(literal: str) -> None:
    """An overflowing literal is a parse artifact, not a measurement, so it
    must not become a comparable slot at all."""
    slots = extract_values(f"commit {literal} landed the fix")
    assert all(math.isfinite(s.value) for s in slots.numeric)


def test_a_finite_neighbour_is_still_extracted() -> None:
    """Control. Dropping non-finite values must not take ordinary numerics
    with it — otherwise the fix could pass by extracting nothing."""
    slots = extract_values("commit 592e701 bumped timeout to 30")
    assert any(s.value == 30.0 for s in slots.numeric)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_format_number_survives_a_non_finite_input(value: float) -> None:
    """Defence in depth: even handed a non-finite value directly, the
    formatter must not raise. `int(inf)` raises OverflowError and `int(nan)`
    raises ValueError, so a bare `int()` narrowing fails both."""
    assert _format_number(value) in {"inf", "-inf", "nan"}


def test_find_conflicts_does_not_raise_on_a_sha_bearing_belief() -> None:
    """The end-to-end path that actually broke: a belief carrying a
    SHA-shaped literal, compared against a lock holding a numeric slot."""
    lock = _mk("lock1", "timeout is 30 seconds", locked=True)
    sha = _mk("b1", "reverted in 51ebe7a commit 1e124732 timeout is 45")
    lock_slots = extract_values(lock.content)
    # Direct comparator call — this raised OverflowError before #1227.
    find_conflicts(extract_values(sha.content), lock_slots)
    # And through the wiring `aelf search` uses.
    _slot_conflict_preextracted(sha, [(lock, lock_slots)])


def test_a_real_conflict_is_still_detected_alongside_a_sha() -> None:
    """Control for the above: the SHA must not suppress detection of the
    genuine numeric disagreement sharing the same belief."""
    lock = _mk("lock1", "timeout is 30", locked=True)
    lock_slots = extract_values(lock.content)
    sha = _mk("b1", "commit 1e124732 changed timeout to 45")
    assert _slot_conflict_preextracted(sha, [(lock, lock_slots)]) == "lock1"
