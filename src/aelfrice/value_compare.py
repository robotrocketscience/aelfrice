"""Typed-slot value-comparison gate for contradiction detection (#422).

Stdlib-only successor to the residual-overlap relatedness gate in
``relationship_detector``. The R2 regex shape from #201 missed real
natural-language contradictions because adversarial paraphrase
collapses Jaccard token overlap below the floor; this module
sidesteps that by extracting **typed slots** (numerics + enumerated
vocabulary) and firing ``contradicts`` directly on mutual-exclusion
across slot values. No token overlap required.

Design:

  * Extraction is regex / vocabulary lookup — deterministic, no
    embeddings, no learned classifiers. Same run produces same
    slots byte-for-byte.
  * Numeric slots: ``(key_token, value)``. The key is the
    nearest alphabetic token preceding the number; the value is
    parsed as float. Unit-aware extraction/comparison is explicitly
    out of scope (see ``NumericSlot``).
  * Enum slots: ``(category, member)``. The category is the name
    of a curated mutual-exclusion group; member is the matching
    vocabulary token. Adding a category extends the gate to a new
    contradiction surface.
  * The comparator fires ``contradicts`` when two beliefs share a
    slot key (numeric ``key_token`` or enum ``category``) with
    materially different values (numeric: outside relative
    tolerance; enum: different members).

Out of scope:

  * Boolean / negation slots — already covered by the modality
    pass in ``relationship_detector``; do not duplicate.
  * Subject disambiguation — two beliefs that mention "alpha = 0.5"
    and "alpha = 1.0" but refer to different alphas will produce a
    false positive. The acceptable surface for the v3 detector is
    audit + ``aelf resolve``-style human-in-loop, not auto-emit;
    auto-emit policy is decided per #422 acceptance #3.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final


# --- Numeric slot extraction ------------------------------------------

# Capture an alphabetic key token immediately before a number, with
# optional ``=``, ``:``, ``of``, or whitespace separator. The number
# admits a leading sign, decimal, exponent. Optional alphabetic unit
# token may follow.
#
# Examples that match (key, value, unit?):
#   ``alpha = 0.5``           → (alpha, 0.5, None)
#   ``timeout: 30s``          → (timeout, 30, s)
#   ``max_depth = 2``         → (max_depth, 2, None)
#   ``set retries to 3``      → (retries, 3, None)
#   ``budget of 100 nodes``   → (budget, 100, nodes)
#
# Excluded by the key requirement: bare numerics like "0.5" with no
# preceding alphabetic token (insufficient subject anchor).
_NUMERIC_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<key>[A-Za-z][A-Za-z0-9_]{0,31})        # key token (≤32 chars)
    \s*
    (?:
        (?:=|:|\bis\b|\bof\b|\bto\b|\bequals?\b)
        \s*
    )?
    (?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)   # number
    \b
    """,
    re.VERBOSE,
)

# Default relative tolerance for numeric comparison. Two values that
# differ by less than this fraction of max(|a|, |b|) are treated as
# the same — guards against float-format fuzz, not against semantic
# equivalence.
DEFAULT_NUMERIC_REL_TOL: Final[float] = 0.01


# --- Enum vocabulary --------------------------------------------------

# Each entry: ``category`` → tuple of mutually-exclusive alias groups,
# where each group is a frozenset of synonymous member tokens. Two
# beliefs contradict on this category when they tag different groups
# within the same category. Members within a single group are
# *aliases* (e.g. ``sync`` ≡ ``synchronous``) and do NOT contradict.
#
# Members must be lowercase; hyphens preserved. Adding a category
# extends the contradiction surface. The taxonomy below was chosen
# for engineering / spec contradiction patterns surfaced in the #201
# adversarial corpus and SHOULD grow as bench evidence flags new
# patterns. Source-of-truth maintenance: this dict; review on each
# bench-gate failure.
ENUM_VOCAB: Final[dict[str, tuple[frozenset[str], ...]]] = {
    "execution_mode": (
        frozenset({"synchronous", "sync"}),
        frozenset({"asynchronous", "async"}),
    ),
    "default_state": (
        frozenset({"default-on", "enabled"}),
        frozenset({"default-off", "disabled"}),
    ),
    "storage_mode": (
        frozenset({"indexed"}),
        frozenset({"scan", "full-scan", "table-scan"}),
    ),
    "completeness": (
        frozenset({"full"}),
        frozenset({"incremental"}),
        frozenset({"partial"}),
    ),
    "strictness": (
        frozenset({"strict"}),
        frozenset({"lax", "permissive"}),
    ),
    "necessity": (
        frozenset({"required"}),
        frozenset({"optional"}),
    ),
    "visibility": (
        frozenset({"public"}),
        frozenset({"private"}),
    ),
    "access_mode": (
        frozenset({"readonly", "read-only"}),
        frozenset({"writable", "read-write"}),
    ),
    "determinism": (
        frozenset({"deterministic"}),
        frozenset({"non-deterministic", "nondeterministic", "stochastic"}),
    ),
}

# Reverse lookup: member token → (category, group_id). The group_id
# is the alphabetically-first member of its group, used as a stable
# identifier in conflict reporting. Built once at import.
#
# `sorted(group)` is load-bearing, not cosmetic (#1370 §8, #1157): the
# groups are frozensets, so iterating them raw keys this dict's insertion
# order on string hash randomisation — a different order every process.
# `_extract_enums` walks this dict and appends in the order it gets, so
# `extract_values` would violate its own "same input → byte-identical
# output" contract. Same bug class as the `MUTABLE_FIELDS` iteration in
# `replay._mutable_field_diff`.
#
# The sort is hoisted into its own clause rather than repeated: written
# as `sorted(group)[0]` in the value position it re-sorts once per
# *member*, not once per group.
_ENUM_MEMBER_INDEX: Final[dict[str, tuple[str, str]]] = {
    member: (category, members[0])
    for category, groups in ENUM_VOCAB.items()
    for members in (sorted(group) for group in groups)
    for member in members
}

# Characters that may not sit immediately either side of a member match.
# NOTE the hyphen is deliberately NOT here — see `_ENUM_MEMBER_ORDER`.
_ENUM_BOUNDARY_CLASS: Final[str] = r"[A-Za-z0-9_]"

# One compiled pattern per member, built once at import rather than
# re-resolved from the `re` module cache on every extraction.
_ENUM_MEMBER_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    member: re.compile(
        rf"(?<!{_ENUM_BOUNDARY_CLASS}){re.escape(member)}"
        rf"(?!{_ENUM_BOUNDARY_CLASS})"
    )
    for member in _ENUM_MEMBER_INDEX
}

# #1159 §13: "deterministic" matched *inside* "non-deterministic", so one
# belief tagged both groups of the `determinism` category and
# `find_conflicts` short-circuited on its group-disjointness test — the
# category could never report a conflict at all.
#
# The obvious fix is to add `-` to the boundary class above. Measured on the
# live repo-local store (44,683 active beliefs) that is much worse than the
# defect: adding `-` on both sides changes 588 beliefs and destroys **568**
# whole-category tags, and adding it only on the left still destroys 290.
# The losses are ordinary hyphenated English that the vocabulary is supposed
# to match — `shipped-default-on`, `secrets-scan`, `session-private` — and a
# boundary class cannot tell those from `non-deterministic`.
#
# What actually distinguishes them is length, not punctuation: the problem is
# a SHORT member matching inside a LONGER one. So members are offered
# longest-first and a span already claimed by a longer member is not offered
# again. Same principle as `directive_detector.py:62-66`, which #1368 names
# as the in-tree shape, expressed as claim-ordering rather than as one
# alternation because each member here carries its own category.
# On the same store this changes 10 beliefs, all of them the real fix, and
# loses nothing.
#
# Sorted by (-length, member) rather than by length alone: a plain
# length sort leaves equal-length members in dict order, and #1157's
# determinism contract requires the tie-break be part of the key.
_ENUM_MEMBER_ORDER: Final[tuple[str, ...]] = tuple(
    sorted(_ENUM_MEMBER_INDEX, key=lambda m: (-len(m), m))
)


# --- Slot dataclasses -------------------------------------------------


@dataclass(frozen=True)
class NumericSlot:
    """A ``key = value`` pair extracted from prose.

    ``key`` is the alphabetic token preceding the number (lowercased);
    ``value`` is parsed as float. Unit-aware comparison is out of
    scope — the regex's greedy capture of trailing tokens as units
    introduced false negatives (e.g. ``alpha = 0.5 prior`` vs
    ``alpha = 1.0 in config`` produced different units and silently
    skipped the conflict). If unit-aware comparison becomes needed,
    file a separate issue with a curated unit vocabulary.
    """

    key: str
    value: float


@dataclass(frozen=True)
class EnumSlot:
    """A ``(category, group_id, member)`` triple from the vocabulary.

    ``category`` is the bucket name in ``ENUM_VOCAB``. ``group_id``
    is the alphabetically-first member of the alias group the token
    belongs to (stable identifier across alias swaps). ``member``
    is the actual matched token (lowercased, hyphenation preserved).
    """

    category: str
    group_id: str
    member: str


@dataclass(frozen=True)
class ValueSlots:
    """All typed slots extracted from a single belief."""

    numeric: tuple[NumericSlot, ...]
    enum: tuple[EnumSlot, ...]


# --- Extraction -------------------------------------------------------


def extract_values(text: str) -> ValueSlots:
    """Extract numeric + enum slots from a single belief's text.

    Pure function. Same input → byte-identical output.
    """
    numerics = _extract_numerics(text)
    enums = _extract_enums(text)
    return ValueSlots(numeric=numerics, enum=enums)


def _extract_numerics(text: str) -> tuple[NumericSlot, ...]:
    out: list[NumericSlot] = []
    seen: set[tuple[str, float]] = set()
    for m in _NUMERIC_RE.finditer(text):
        key = m.group("key").lower()
        if key in _NUMERIC_KEY_DROP:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        # `float()` does not raise on overflow — it saturates to
        # +/-inf. The exponent branch of `_NUMERIC_RE` matches
        # abbreviated git SHAs like `592e701`, which are hex strings
        # that happen to hold one `e` between digits; parsed as
        # scientific notation they become `inf`. That is not a
        # measurement, so admitting it as a slot manufactures a
        # comparison against a value no belief actually asserts.
        # Dropping it here also keeps every downstream consumer off
        # the non-finite path (#1227).
        if not math.isfinite(value):
            continue
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(NumericSlot(key=key, value=value))
    return tuple(out)


def _extract_enums(text: str) -> tuple[EnumSlot, ...]:
    lowered = text.lower()

    # Pass 1 — claim spans longest-member-first (#1159 §13). A shorter
    # member that only occurs inside a longer member's span is dropped;
    # one that also occurs somewhere else keeps that other occurrence.
    matched: set[str] = set()
    claimed: list[tuple[int, int]] = []
    for member in _ENUM_MEMBER_ORDER:
        for hit in _ENUM_MEMBER_PATTERNS[member].finditer(lowered):
            start, end = hit.span()
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            matched.add(member)

    # Pass 2 — emit in `_ENUM_MEMBER_INDEX` order, which is `ENUM_VOCAB`
    # declaration order. Emission order is part of this function's output
    # contract, so it stays independent of the length ordering above.
    out: list[EnumSlot] = []
    seen: set[tuple[str, str]] = set()
    for member, (category, group_id) in _ENUM_MEMBER_INDEX.items():
        if member not in matched:
            continue
        pair = (category, member)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(EnumSlot(category=category, group_id=group_id, member=member))
    return tuple(out)


# Filler / stop-keys that the numeric regex captures spuriously as
# the "key" but which carry no subject information. Keep narrow: only
# tokens that the regex's optional separator words also match.
_NUMERIC_KEY_DROP: Final[frozenset[str]] = frozenset({
    "is", "are", "was", "were", "be", "of", "to", "at", "on", "in",
    "as", "by", "for", "with", "the", "a", "an", "this", "that",
    "and", "or", "but", "equals", "equal",
})



# --- Mutual-exclusion comparator --------------------------------------


@dataclass(frozen=True)
class SlotConflict:
    """One contradicting slot match across two beliefs.

    ``kind`` is ``"numeric"`` or ``"enum"``; ``key`` identifies the
    slot key (numeric key token or enum category); ``value_a`` and
    ``value_b`` are the conflicting values stringified for
    diagnostics.
    """

    kind: str
    key: str
    value_a: str
    value_b: str


def find_conflicts(
    slots_a: ValueSlots,
    slots_b: ValueSlots,
    *,
    numeric_rel_tol: float = DEFAULT_NUMERIC_REL_TOL,
) -> tuple[SlotConflict, ...]:
    """Return all mutual-exclusion conflicts between two beliefs' slots.

    Numeric conflict: same ``key`` with values outside the
    relative-tolerance band. Units are not extracted or compared —
    two values under the same key expressed in different units are
    compared as raw numbers (out of scope; see ``NumericSlot``).

    Enum conflict: same ``category`` with different ``member`` values.

    Empty tuple means no conflict found, NOT that the pair is
    related — the caller decides what no-conflict means.
    """
    conflicts: list[SlotConflict] = []
    a_num_by_key: dict[str, list[NumericSlot]] = {}
    for s in slots_a.numeric:
        a_num_by_key.setdefault(s.key, []).append(s)
    for sb in slots_b.numeric:
        for sa in a_num_by_key.get(sb.key, ()):
            if not _numeric_close(sa.value, sb.value, numeric_rel_tol):
                conflicts.append(
                    SlotConflict(
                        kind="numeric",
                        key=sa.key,
                        value_a=_format_number(sa.value),
                        value_b=_format_number(sb.value),
                    )
                )

    # Conflict on enum is by group_id, not member: ``sync`` and
    # ``synchronous`` are aliases (same group_id) and do not conflict
    # with each other. Conflict fires only when A and B tag DIFFERENT
    # groups within the same category (group_id sets disjoint).
    a_groups_by_cat: dict[str, set[str]] = {}
    for s in slots_a.enum:
        a_groups_by_cat.setdefault(s.category, set()).add(s.group_id)
    b_groups_by_cat: dict[str, set[str]] = {}
    for s in slots_b.enum:
        b_groups_by_cat.setdefault(s.category, set()).add(s.group_id)
    for category, a_groups in a_groups_by_cat.items():
        b_groups = b_groups_by_cat.get(category)
        if not b_groups:
            continue
        if a_groups & b_groups:
            continue
        for ag in sorted(a_groups):
            for bg in sorted(b_groups):
                conflicts.append(
                    SlotConflict(
                        kind="enum",
                        key=category,
                        value_a=ag,
                        value_b=bg,
                    )
                )
    return tuple(conflicts)


def _numeric_close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return True
    return abs(a - b) / denom <= rel_tol


def _format_number(x: float) -> str:
    # `int()` raises on non-finite input — OverflowError for +/-inf,
    # ValueError for nan — so the narrowing below cannot be reached
    # unguarded. `_extract_numerics` already refuses to admit such a
    # value as a slot (#1227); this is the second line of defence, for
    # any caller that reaches the comparator by another route.
    if not math.isfinite(x):
        return f"{x:g}"
    if x == int(x):
        return str(int(x))
    return f"{x:g}"
