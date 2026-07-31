"""Lock-consistency annotation (#1175 build-first item).

The L0 locked tier is injected first and no consistency check runs between it
and the L1 / L2.5 tiers, so the model reads the user's explicit ground truth
and then reads a contradiction of it in the same prompt. Measured on the live
store, that happens to a real fraction of injected beliefs.

**ANNOTATE, never DROP.** The slot keys come from a preceding-alphabetic-token
heuristic that emits junk, so a filter would silently delete beliefs on a bad
key. Handing the model the defeat instead of the deletion also preserves the
#605 posture: aelfrice supplies context, the agent adjudicates.

Suppression is the whole design problem, and it was measured rather than
asserted. Unsuppressed, 6.12% of retrieved unlocked beliefs conflict with a
lock and **63.7% of those trace to a single version-and-date lock** — an
annotation that noisy is one the user learns to ignore. Two candidate rules
were tried and rejected before the shipped one:

  * *Cap the share of one pack a single lock may flag.* Refuted by the data:
    the dominant lock flags a p50 of 2.9% of any one pack. Its noise is spread
    across queries, not concentrated within one, so a per-retrieval threshold
    cannot see it.
  * *Drop keys that are multi-valued within one belief.* Helps (6.12% -> 4.21%)
    and evicts the worst lock, but concentration got **worse** — the next
    version-and-date lock took over at 72.4%. It treated a symptom.

What the extractor is actually doing shows up in that lock's slots: key ``v``
with four distinct values in one belief, ``shipped=2026.0``, a SHA fragment as
a key (``b88fd4=9.0``), and a CI run id. Suppressing all three sources takes
the rate to **1.38% with the top lock at 27.2%** and the residual spread across
five locks rather than dominated by one.

This filter lives here rather than in `value_compare.extract_values` on
purpose. Narrowing the shipped extractor would change what counts as a numeric
slot for every consumer — the wider blast radius #1228 explicitly declined to
take. The annotation gets its own filtered view; the extractor is untouched.

Pure and deterministic: local predicates over a belief's own slots, no store
read, no corpus scan, no per-store threshold, no clock.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Final, Iterable, Sequence

from .models import Belief
from .value_compare import ValueSlots, find_conflicts

__all__ = [
    "CALENDAR_YEAR_MAX",
    "CALENDAR_YEAR_MIN",
    "VERSION_KEYS",
    "annotation_slots",
    "lock_conflict_annotations",
]

# Keys whose numeric value is a version component rather than a measurement.
# `v3.6.0` yields `NumericSlot(key='v', value=3.6)`, so a belief naming any
# other version "conflicts". Deliberately a short, closed set of tokens that
# introduce a version literal — not a general stopword list, which is the
# thing the fan-effect work refuted for the entity lane.
VERSION_KEYS: Final[frozenset[str]] = frozenset(
    {"v", "version", "rev", "release"}
)

# A bare 4-digit integer in this range is a calendar year, not a quantity.
# Bounds are wide on purpose: the cost of missing a year is a false conflict,
# the cost of an over-wide range is losing a genuine measurement that happens
# to be ~2000, which is rarer in this corpus than dates are.
CALENDAR_YEAR_MIN: Final[int] = 1900
CALENDAR_YEAR_MAX: Final[int] = 2100


def _is_calendar_year(value: float) -> bool:
    """True for an integral value in the calendar-year band."""
    if value != int(value):
        return False
    return CALENDAR_YEAR_MIN <= int(value) <= CALENDAR_YEAR_MAX


def annotation_slots(slots: ValueSlots) -> ValueSlots:
    """Return `slots` with the numeric slots that cannot support a conflict removed.

    Three suppressions, each measured (see the module docstring):

      1. **Multi-valued key.** A key taking more than one distinct value inside
         a single belief is not a functional-dependency key, so a mismatch
         against it says nothing. This is the only rule that needs no
         vocabulary at all.
      2. **Version key.** :data:`VERSION_KEYS`.
      3. **Calendar value.** A 4-digit year.

    Enum slots pass through untouched — they are drawn from a fixed
    ``ENUM_VOCAB`` and carry none of these failure modes.
    """
    by_key: dict[str, set[float]] = defaultdict(set)
    for slot in slots.numeric:
        # Round before comparing so float noise cannot make one key look
        # multi-valued and silently suppress a real slot.
        by_key[slot.key].add(round(slot.value, 12))
    kept = tuple(
        slot for slot in slots.numeric
        if len(by_key[slot.key]) == 1
        and slot.key.lower() not in VERSION_KEYS
        and not _is_calendar_year(slot.value)
    )
    return ValueSlots(numeric=kept, enum=slots.enum)


def lock_conflict_annotations(
    candidates: Iterable[tuple[str, ValueSlots]],
    locked_pairs: Sequence[tuple[Belief, ValueSlots]],
) -> dict[str, str]:
    """Map belief id -> the id of the first lock it slot-conflicts with.

    `candidates` is `(belief_id, slots)`; `locked_pairs` is the caller's
    once-per-retrieval `[(lock, extract_values(lock.content))]`. Both sides are
    passed through :func:`annotation_slots` here rather than by the caller, so
    a caller cannot accidentally annotate on unsuppressed slots.

    "First" is by the order of `locked_pairs`, which the caller controls; there
    is no attempt to pick a *best* conflicting lock. A belief conflicting with
    two locks is already an unusual enough signal that naming one is sufficient
    to make the agent look.

    Beliefs with no conflict are absent from the result rather than mapped to
    None, so the common case allocates nothing.
    """
    if not locked_pairs:
        return {}
    filtered_locks = [
        (lock, annotation_slots(slots)) for lock, slots in locked_pairs
    ]
    out: dict[str, str] = {}
    for belief_id, slots in candidates:
        candidate_slots = annotation_slots(slots)
        if not candidate_slots.numeric and not candidate_slots.enum:
            continue
        for lock, lock_slots in filtered_locks:
            if find_conflicts(candidate_slots, lock_slots):
                out[belief_id] = lock.id
                break
    return out
