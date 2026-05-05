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
  * Numeric slots: ``(key_token, value, unit?)``. The key is the
    nearest alphabetic token preceding the number; the value is
    parsed as float; the unit is the alphabetic token immediately
    after the number, when present.
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
_ENUM_MEMBER_INDEX: Final[dict[str, tuple[str, str]]] = {
    member: (category, sorted(group)[0])
    for category, groups in ENUM_VOCAB.items()
    for group in groups
    for member in group
}


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
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(NumericSlot(key=key, value=value))
    return tuple(out)


def _extract_enums(text: str) -> tuple[EnumSlot, ...]:
    lowered = text.lower()
    out: list[EnumSlot] = []
    seen: set[tuple[str, str]] = set()
    for member, (category, group_id) in _ENUM_MEMBER_INDEX.items():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(member)}(?![A-Za-z0-9_])", lowered):
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

    Numeric conflict: same ``(key, unit)`` with values outside the
    relative-tolerance band. Unit-mismatch is silent — different
    units mean different scales and the comparator cannot adjudicate
    without a unit-conversion table (out of scope).

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
    if x == int(x):
        return str(int(x))
    return f"{x:g}"
