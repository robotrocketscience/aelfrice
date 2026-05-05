"""Unit tests for `aelfrice.value_compare` (#422)."""
from __future__ import annotations

import pytest

from aelfrice.value_compare import (
    DEFAULT_NUMERIC_REL_TOL,
    ENUM_VOCAB,
    EnumSlot,
    NumericSlot,
    SlotConflict,
    ValueSlots,
    extract_values,
    find_conflicts,
)


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------


def test_numeric_simple_assignment() -> None:
    s = extract_values("alpha = 0.5 prior")
    assert s.numeric == (NumericSlot(key="alpha", value=0.5),)


def test_numeric_separator_words_picked_up() -> None:
    """``is``, ``of``, ``to``, ``equals`` between key and number all work."""
    s = extract_values("set timeout to 30")
    keys = {n.key for n in s.numeric}
    assert "timeout" in keys
    assert any(n.value == 30 for n in s.numeric)


def test_numeric_filler_keys_dropped() -> None:
    """Bare ``is 5`` keyed on ``is`` would be a noise slot — filtered."""
    s = extract_values("is 5 the answer")
    assert all(n.key != "is" for n in s.numeric)


def test_numeric_negative_and_decimal() -> None:
    s = extract_values("offset = -1.25e-3 baseline")
    assert NumericSlot(key="offset", value=-0.00125) in s.numeric


def test_numeric_dedup_within_belief() -> None:
    s = extract_values("alpha=0.5 alpha=0.5 alpha=0.5")
    assert len([n for n in s.numeric if n.key == "alpha"]) == 1


def test_numeric_multiple_kv_pairs_extracted() -> None:
    s = extract_values("depth=2 max_depth=4")
    pairs = {(n.key, n.value) for n in s.numeric}
    assert ("depth", 2.0) in pairs
    assert ("max_depth", 4.0) in pairs


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------


def test_enum_simple_match() -> None:
    s = extract_values("synchronous on hot path")
    assert any(
        e.category == "execution_mode" and e.member == "synchronous"
        for e in s.enum
    )


def test_enum_alias_collapse_to_group_id() -> None:
    """``sync`` and ``synchronous`` both belong to the same group_id."""
    a = extract_values("use sync mode")
    b = extract_values("synchronous everywhere")
    a_gid = next(e.group_id for e in a.enum if e.category == "execution_mode")
    b_gid = next(e.group_id for e in b.enum if e.category == "execution_mode")
    assert a_gid == b_gid


def test_enum_hyphenated_member_match() -> None:
    s = extract_values("default-on flag here")
    assert any(
        e.member == "default-on" for e in s.enum
    )


def test_enum_word_boundary_no_substring_match() -> None:
    """``async`` should not match inside ``asynchrony`` (a token we don't
    enumerate). ``\\b`` boundary handles this; the test pins the contract."""
    s = extract_values("the asynchronous behavior matters")
    # ``asynchronous`` IS in the vocab, so it should match. The point is
    # we don't ALSO match ``async`` as a substring. Verify by counting
    # group_ids for the category — a single group should be tagged.
    gids = {e.group_id for e in s.enum if e.category == "execution_mode"}
    assert len(gids) == 1


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def test_no_conflict_on_empty_slots() -> None:
    a = extract_values("totally unrelated text here")
    b = extract_values("something else entirely")
    assert find_conflicts(a, b) == ()


def test_numeric_conflict_fires_on_value_mismatch() -> None:
    a = extract_values("alpha = 0.5 prior")
    b = extract_values("alpha = 1.0 in config")
    conflicts = find_conflicts(a, b)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.kind == "numeric"
    assert c.key == "alpha"
    assert {c.value_a, c.value_b} == {"0.5", "1"}


def test_numeric_within_relative_tolerance_no_conflict() -> None:
    a = extract_values("alpha is 0.5")
    b = extract_values("alpha is 0.502")
    # Within DEFAULT_NUMERIC_REL_TOL (~1%) → silent
    assert find_conflicts(a, b) == ()


def test_numeric_outside_tolerance_conflict() -> None:
    a = extract_values("alpha is 0.5")
    b = extract_values("alpha is 0.9")
    assert any(c.kind == "numeric" for c in find_conflicts(a, b))


def test_numeric_zero_zero_no_conflict() -> None:
    """``rel_tol`` denominator guard: 0 vs 0 is not a conflict."""
    a = extract_values("count = 0 items")
    b = extract_values("count = 0 entries")
    assert find_conflicts(a, b) == ()


def test_numeric_custom_tolerance_overrides_default() -> None:
    a = extract_values("alpha is 0.5")
    b = extract_values("alpha is 0.6")
    # 20% diff. With default 1% tol this conflicts; with 50% tol it doesn't.
    assert find_conflicts(a, b, numeric_rel_tol=0.5) == ()
    assert find_conflicts(a, b, numeric_rel_tol=0.01) != ()


def test_enum_conflict_on_distinct_groups() -> None:
    a = extract_values("synchronous on hot path")
    b = extract_values("async execution model")
    conflicts = find_conflicts(a, b)
    assert any(
        c.kind == "enum" and c.key == "execution_mode" for c in conflicts
    )


def test_enum_alias_pair_does_not_conflict() -> None:
    a = extract_values("use sync mode")
    b = extract_values("synchronous everywhere")
    assert find_conflicts(a, b) == ()


def test_enum_default_state_conflict() -> None:
    a = extract_values("default-on flag")
    b = extract_values("default-off flag")
    conflicts = find_conflicts(a, b)
    assert any(c.key == "default_state" for c in conflicts)


def test_enum_enabled_disabled_conflict_via_aliases() -> None:
    """``enabled``/``disabled`` share groups with default-on/default-off."""
    a = extract_values("enabled by default")
    b = extract_values("disabled by default")
    conflicts = find_conflicts(a, b)
    assert any(c.kind == "enum" for c in conflicts)


def test_enum_completeness_full_vs_incremental() -> None:
    a = extract_values("full backup nightly")
    b = extract_values("incremental backup nightly")
    assert any(c.key == "completeness" for c in find_conflicts(a, b))


def test_enum_access_mode_aliases_collapse() -> None:
    """``readonly`` and ``read-only`` are the same group."""
    a = extract_values("readonly mode")
    b = extract_values("read-only mode")
    assert find_conflicts(a, b) == ()


def test_mixed_numeric_and_enum_conflicts_combine() -> None:
    a = extract_values("alpha = 0.5 in synchronous mode")
    b = extract_values("alpha = 1.0 in async mode")
    conflicts = find_conflicts(a, b)
    kinds = {c.kind for c in conflicts}
    assert kinds == {"numeric", "enum"}


# ---------------------------------------------------------------------------
# Vocab integrity + determinism
# ---------------------------------------------------------------------------


def test_enum_vocab_groups_pairwise_disjoint_within_category() -> None:
    """A member can belong to AT MOST one group within a category;
    cross-category collisions are allowed (a member can mean different
    things in different categories) but within-category aliasing must
    be resolvable to one group_id."""
    for category, groups in ENUM_VOCAB.items():
        seen: set[str] = set()
        for group in groups:
            assert not (seen & group), (
                f"category {category!r} has overlapping groups; member "
                f"belongs to multiple groups: {seen & group}"
            )
            seen |= group


def test_default_numeric_rel_tol_pinned() -> None:
    """Drift on this value silently changes the bench gate; pin it."""
    assert DEFAULT_NUMERIC_REL_TOL == 0.01


def test_determinism_byte_identical_repeat() -> None:
    text_a = "alpha = 0.5 in synchronous full mode"
    text_b = "alpha = 1.0 in async incremental mode"
    sa = extract_values(text_a)
    sb = extract_values(text_b)
    a_again = extract_values(text_a)
    b_again = extract_values(text_b)
    assert sa == a_again
    assert sb == b_again
    assert find_conflicts(sa, sb) == find_conflicts(a_again, b_again)


def test_value_slots_dataclass_is_hashable() -> None:
    """Frozen dataclass = hashable + comparable. Future caching layers
    rely on this — pin the contract."""
    slots = extract_values("alpha = 1.0")
    assert hash(slots) == hash(extract_values("alpha = 1.0"))


def test_slot_conflict_dataclass_round_trip() -> None:
    c = SlotConflict(kind="numeric", key="alpha", value_a="0.5", value_b="1")
    assert c.kind == "numeric"
    assert c.key == "alpha"
    assert c.value_a == "0.5"
    assert c.value_b == "1"


def test_extract_returns_value_slots_type() -> None:
    s = extract_values("alpha = 0.5")
    assert isinstance(s, ValueSlots)
    assert isinstance(s.numeric, tuple)
    assert isinstance(s.enum, tuple)


def test_enum_slot_dataclass_round_trip() -> None:
    e = EnumSlot(category="execution_mode", group_id="async", member="async")
    assert (e.category, e.group_id, e.member) == ("execution_mode", "async", "async")
