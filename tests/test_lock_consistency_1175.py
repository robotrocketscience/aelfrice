"""Lock-consistency annotation (#1175 build-first item).

The suppression rules are the design, so each is pinned with a case that fails
if that rule alone is removed, plus a control proving the annotation still
fires on a genuine conflict.
"""
from __future__ import annotations

import pytest

from aelfrice.lock_consistency import (
    CALENDAR_YEAR_MAX,
    CALENDAR_YEAR_MIN,
    VERSION_KEYS,
    annotation_slots,
    lock_conflict_annotations,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.value_compare import extract_values


def _b(bid: str, content: str, *, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0 if lock == LOCK_USER else 1.0,
        beta=0.5 if lock == LOCK_USER else 1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-07-31T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


def _slots(text: str):
    return extract_values(text)


def _keys(slots) -> set[str]:
    return {s.key for s in slots.numeric}


# --- the three suppressions ------------------------------------------------


def test_multi_valued_key_is_dropped() -> None:
    """A key with two values in one belief is not a functional dependency.

    This is the rule that needs no vocabulary, and the live-store lock that
    motivated it carried key `v` with four distinct values in one sentence.
    """
    slots = _slots("timeout is 30 and timeout is 45")
    assert "timeout" in _keys(slots), "fixture did not produce the slot"
    assert "timeout" not in _keys(annotation_slots(slots))


def test_single_valued_key_survives() -> None:
    """Control for the rule above: one value per key is exactly the case the
    annotation exists to compare. Without this, a filter that dropped every
    numeric slot would pass the multi-valued test."""
    slots = _slots("timeout is 30")
    assert "timeout" in _keys(annotation_slots(slots))


def test_repeating_the_same_value_is_not_multi_valued() -> None:
    """Restating a value must not suppress it — only disagreement does."""
    slots = _slots("timeout is 30 and the timeout is 30")
    assert "timeout" in _keys(annotation_slots(slots))


@pytest.mark.parametrize("key", sorted(VERSION_KEYS))
def test_version_keys_are_dropped(key: str) -> None:
    """`v3.6.0` yields key `v` value 3.6, so any other version 'conflicts'."""
    slots = _slots(f"{key} 3.6 is current")
    assert key in _keys(slots), "fixture did not produce the slot"
    assert key not in _keys(annotation_slots(slots))


def test_calendar_year_is_dropped() -> None:
    slots = _slots("shipped 2026 with the release")
    assert "shipped" in _keys(slots)
    assert "shipped" not in _keys(annotation_slots(slots))


@pytest.mark.parametrize(
    "value", [CALENDAR_YEAR_MIN - 1, CALENDAR_YEAR_MAX + 1, 42, 1899, 2101]
)
def test_values_outside_the_calendar_band_survive(value: int) -> None:
    """The band is a band, not 'drop every integer'."""
    slots = annotation_slots(_slots(f"count is {value}"))
    assert "count" in _keys(slots), f"{value} was suppressed"


def test_a_non_integral_value_in_the_year_range_survives() -> None:
    """2026.5 is a measurement, not a year."""
    assert "ratio" in _keys(annotation_slots(_slots("ratio is 2026.5")))


def test_enum_slots_are_never_suppressed() -> None:
    """Enums come from a fixed vocabulary and carry none of these failures."""
    slots = _slots("the mode is synchronous")
    if not slots.enum:
        pytest.skip("fixture produced no enum slot on this vocabulary")
    assert annotation_slots(slots).enum == slots.enum


# --- the annotation itself -------------------------------------------------


def test_a_genuine_conflict_is_annotated() -> None:
    """The control that keeps every suppression test honest.

    If the suppressions were over-broad enough to annotate nothing, every test
    above would still pass. This one would not.
    """
    lock = _b("LK", "the retry limit is 3", lock=LOCK_USER)
    cand = _b("b0", "the retry limit is 9")
    out = lock_conflict_annotations(
        [(cand.id, _slots(cand.content))], [(lock, _slots(lock.content))]
    )
    assert out == {"b0": "LK"}


def test_agreement_is_not_annotated() -> None:
    lock = _b("LK", "the retry limit is 3", lock=LOCK_USER)
    cand = _b("b0", "the retry limit is 3")
    out = lock_conflict_annotations(
        [(cand.id, _slots(cand.content))], [(lock, _slots(lock.content))]
    )
    assert out == {}


def test_the_version_lock_no_longer_annotates_everything() -> None:
    """The measured failure, reproduced in miniature.

    A version-and-date lock against a belief naming a different version is the
    case that produced 63.7% of all conflicts on the live store. It must not
    annotate.
    """
    lock = _b("LK", "aelfrice v3.6.0 shipped 2026-06-19", lock=LOCK_USER)
    cand = _b("b0", "aelfrice v4.1.0 shipped 2026-07-30")
    out = lock_conflict_annotations(
        [(cand.id, _slots(cand.content))], [(lock, _slots(lock.content))]
    )
    assert out == {}, "the version/date lock still annotates"


def test_a_real_conflict_survives_alongside_a_version_literal() -> None:
    """Suppression must be slot-scoped, not belief-scoped.

    A belief carrying both a version literal and a genuine disagreement must
    still be annotated on the genuine one. A belief-level 'contains a version,
    skip it' shortcut would pass every other test here and fail this.
    """
    lock = _b("LK", "aelfrice v3.6.0 and the retry limit is 3", lock=LOCK_USER)
    cand = _b("b0", "aelfrice v4.1.0 and the retry limit is 9")
    out = lock_conflict_annotations(
        [(cand.id, _slots(cand.content))], [(lock, _slots(lock.content))]
    )
    assert out == {"b0": "LK"}


def test_no_locks_means_no_work() -> None:
    assert lock_conflict_annotations([("b0", _slots("retry limit is 9"))], []) == {}


def test_beliefs_without_conflicts_are_absent_not_none() -> None:
    lock = _b("LK", "the retry limit is 3", lock=LOCK_USER)
    out = lock_conflict_annotations(
        [("b0", _slots("nothing numeric here"))], [(lock, _slots(lock.content))]
    )
    assert out == {}


def test_first_lock_in_order_wins() -> None:
    """Deterministic: the caller's ordering decides, not dict iteration."""
    l1 = _b("LK1", "the retry limit is 3", lock=LOCK_USER)
    l2 = _b("LK2", "the retry limit is 5", lock=LOCK_USER)
    cand_slots = _slots("the retry limit is 9")
    pairs = [(l1, _slots(l1.content)), (l2, _slots(l2.content))]
    assert lock_conflict_annotations([("b0", cand_slots)], pairs) == {"b0": "LK1"}
    assert lock_conflict_annotations(
        [("b0", cand_slots)], list(reversed(pairs))
    ) == {"b0": "LK2"}


def test_annotation_is_pure_and_repeatable() -> None:
    """Same inputs, same output, across repeated calls in one process."""
    lock = _b("LK", "the retry limit is 3", lock=LOCK_USER)
    pairs = [(lock, _slots(lock.content))]
    cands = [(f"b{i}", _slots(f"the retry limit is {i + 4}")) for i in range(6)]
    first = lock_conflict_annotations(cands, pairs)
    for _ in range(5):
        assert lock_conflict_annotations(cands, pairs) == first
    assert first, "fixture annotated nothing — the test would be vacuous"


def test_suppression_is_symmetric_not_lock_side_only() -> None:
    """An ambiguous key is meaningless on the candidate side too.

    The lock states one retry limit; the candidate states two. The candidate's
    key is multi-valued and therefore not a functional dependency, so there is
    nothing to contradict and it must not be annotated.

    Verified by mutation: passing the candidate's raw slots through instead of
    `annotation_slots(slots)` fails this test and nothing else in the file --
    every other case here is suppressed on the lock side first, so the
    asymmetry is invisible to them.
    """
    lock = _b("LK", "the retry limit is 3", lock=LOCK_USER)
    ambiguous = _slots("the retry limit is 45 and the retry limit is 60")
    assert "retry" in _keys(ambiguous) or "limit" in _keys(ambiguous), (
        "fixture produced no numeric slot — the test would be vacuous"
    )
    out = lock_conflict_annotations(
        [("b0", ambiguous)], [(lock, _slots(lock.content))]
    )
    assert out == {}, "candidate-side suppression did not run"
