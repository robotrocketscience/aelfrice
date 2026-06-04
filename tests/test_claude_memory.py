"""Unit tests for the claude_memory module — parser and slot-equality (#935).

All fixtures are synthetic.  No real ~/.claude/ content is referenced.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.claude_memory import (
    ComparisonResult,
    MemoryBullet,
    SlotRow,
    compare_slots,
    derive_memory_dir,
    extract_slot,
    parse_memory_bullets,
    slot_row_from_belief,
    slot_row_from_bullet,
)


# ---------------------------------------------------------------------------
# derive_memory_dir
# ---------------------------------------------------------------------------


def test_derive_memory_dir_basic() -> None:
    result = derive_memory_dir("/projects/myapp")
    home = Path.home()
    expected = home / ".claude" / "projects" / "-projects-myapp" / "memory"
    assert result == expected


def test_derive_memory_dir_deep_path() -> None:
    result = derive_memory_dir("/a/b/c")
    home = Path.home()
    expected = home / ".claude" / "projects" / "-a-b-c" / "memory"
    assert result == expected


def test_derive_memory_dir_single_segment() -> None:
    result = derive_memory_dir("/workspace")
    home = Path.home()
    expected = home / ".claude" / "projects" / "-workspace" / "memory"
    assert result == expected


def test_derive_memory_dir_accepts_path_object() -> None:
    result = derive_memory_dir(Path("/projects/myapp"))
    home = Path.home()
    expected = home / ".claude" / "projects" / "-projects-myapp" / "memory"
    assert result == expected


# ---------------------------------------------------------------------------
# parse_memory_bullets — synthetic content only
# ---------------------------------------------------------------------------

_SYNTHETIC_MEMORY_MD = """\
# Memory

Some introductory text that is not a bullet.

- [Fake topic](fake_topic.md) — synthetic memory line about topic A
- [Another topic](another.md) — another synthetic bullet
- [No description](no_desc.md)
- plain line without list marker
- [Missing close paren](broken.md — not a bullet

## Section heading

- [Final item](final.md) — last synthetic bullet
"""


def test_parse_memory_bullets_count() -> None:
    bullets = parse_memory_bullets(_SYNTHETIC_MEMORY_MD, source_path="MEMORY.md")
    # Three correctly-formed bullets: Fake topic, Another topic, No description, Final item
    assert len(bullets) == 4


def test_parse_memory_bullets_link_text() -> None:
    bullets = parse_memory_bullets(_SYNTHETIC_MEMORY_MD)
    texts = [b.link_text for b in bullets]
    assert "Fake topic" in texts
    assert "Another topic" in texts
    assert "No description" in texts
    assert "Final item" in texts


def test_parse_memory_bullets_description_present() -> None:
    bullets = parse_memory_bullets(_SYNTHETIC_MEMORY_MD)
    fake = next(b for b in bullets if b.link_text == "Fake topic")
    assert fake.description == "synthetic memory line about topic A"


def test_parse_memory_bullets_description_absent() -> None:
    bullets = parse_memory_bullets(_SYNTHETIC_MEMORY_MD)
    no_desc = next(b for b in bullets if b.link_text == "No description")
    assert no_desc.description == ""


def test_parse_memory_bullets_source_path_propagated() -> None:
    bullets = parse_memory_bullets("- [X](x.md) — desc\n", source_path="/path/MEMORY.md")
    assert bullets[0].source_path == "/path/MEMORY.md"


def test_parse_memory_bullets_empty_content() -> None:
    assert parse_memory_bullets("") == []


def test_parse_memory_bullets_no_bullets() -> None:
    assert parse_memory_bullets("# Heading\n\nSome text.") == []


def test_parse_memory_bullets_single_line() -> None:
    line = "- [Widget cache](widget_cache.md) — cache invalidation policy"
    bullets = parse_memory_bullets(line)
    assert len(bullets) == 1
    assert bullets[0].link_text == "Widget cache"
    assert bullets[0].description == "cache invalidation policy"


def test_parse_memory_bullets_returns_namedtuples() -> None:
    bullets = parse_memory_bullets("- [X](x.md) — text\n")
    assert isinstance(bullets[0], MemoryBullet)


# ---------------------------------------------------------------------------
# extract_slot
# ---------------------------------------------------------------------------


def test_extract_slot_simple_two_word() -> None:
    # Regex may capture a one- or two-word subject depending on the input.
    # "Database schema uses normalised form" → subject="Database schema",
    # predicate="uses", value="normalised form".
    result = extract_slot("Database schema uses normalised form")
    assert result is not None
    subject, predicate, value = result
    assert "Database" in subject
    assert predicate in ("schema", "uses")
    assert value  # something captured after the predicate


def test_extract_slot_three_word_subject() -> None:
    # Two-word subject + predicate
    result = extract_slot("Widget cache policy stores LRU entries")
    assert result is not None
    # Subject gets at most two tokens; predicate is the next token
    subject, predicate, value = result
    assert "Widget" in subject


def test_extract_slot_single_word_returns_none() -> None:
    assert extract_slot("Singleword") is None


def test_extract_slot_empty_returns_none() -> None:
    assert extract_slot("") is None


def test_extract_slot_whitespace_only_returns_none() -> None:
    assert extract_slot("   ") is None


def test_extract_slot_returns_tuple_of_three() -> None:
    result = extract_slot("Foo bar baz")
    assert result is not None
    assert len(result) == 3


def test_extract_slot_value_may_be_empty() -> None:
    # Exactly two tokens: subject and predicate only, no value
    result = extract_slot("Auth disabled")
    assert result is not None
    subject, predicate, value = result
    assert subject == "Auth"
    assert predicate == "disabled"
    assert value == ""


# ---------------------------------------------------------------------------
# slot_row_from_belief
# ---------------------------------------------------------------------------


def test_slot_row_from_belief_success() -> None:
    row = slot_row_from_belief("abc123", "Cache policy uses LRU", "/data/aelf.db")
    assert row is not None
    assert isinstance(row, SlotRow)
    assert row.source == "abc123"
    assert row.source_path == "/data/aelf.db"
    assert row.raw_text == "Cache policy uses LRU"


def test_slot_row_from_belief_no_slot_returns_none() -> None:
    assert slot_row_from_belief("abc123", "singleword", "/data/aelf.db") is None


# ---------------------------------------------------------------------------
# slot_row_from_bullet
# ---------------------------------------------------------------------------


def test_slot_row_from_bullet_uses_description() -> None:
    bullet = MemoryBullet(
        link_text="Fake topic",
        link_target="fake.md",
        description="Auth token expires after 24 hours",
        source_path="MEMORY.md",
    )
    row = slot_row_from_bullet(bullet)
    assert row is not None
    assert row.raw_text == "Auth token expires after 24 hours"
    assert row.source == "Fake topic"


def test_slot_row_from_bullet_falls_back_to_link_text() -> None:
    bullet = MemoryBullet(
        link_text="Cache policy invalidated",
        link_target="cache.md",
        description="",
        source_path="MEMORY.md",
    )
    row = slot_row_from_bullet(bullet)
    assert row is not None
    assert row.raw_text == "Cache policy invalidated"


def test_slot_row_from_bullet_unextractable_returns_none() -> None:
    bullet = MemoryBullet(
        link_text="x",
        link_target="x.md",
        description="singleword",
        source_path="MEMORY.md",
    )
    assert slot_row_from_bullet(bullet) is None


# ---------------------------------------------------------------------------
# compare_slots — four-bucket logic
# ---------------------------------------------------------------------------


def _make_slot(
    subject: str,
    predicate: str,
    value: str,
    source: str = "src",
    source_path: str = "path",
) -> SlotRow:
    return SlotRow(
        slot_subject=subject,
        slot_predicate=predicate,
        slot_value=value,
        raw_text=f"{subject} {predicate} {value}",
        source=source,
        source_path=source_path,
    )


def test_compare_slots_empty_inputs() -> None:
    result = compare_slots([], [])
    assert isinstance(result, ComparisonResult)
    assert result.duplicates == []
    assert result.contradictions == []
    assert result.aelfrice_only == []
    assert result.claude_only == []


def test_compare_slots_duplicate() -> None:
    arow = _make_slot("Auth", "token", "lasts 24 hours", source="belief-1")
    crow = _make_slot("Auth", "token", "lasts 24 hours", source="Fake bullet")
    result = compare_slots([arow], [crow])
    assert len(result.duplicates) == 1
    assert result.contradictions == []
    assert result.aelfrice_only == []
    assert result.claude_only == []
    a, c = result.duplicates[0]
    assert a.source == "belief-1"
    assert c.source == "Fake bullet"


def test_compare_slots_contradiction() -> None:
    arow = _make_slot("Auth", "token", "lasts 24 hours")
    crow = _make_slot("Auth", "token", "lasts 48 hours")
    result = compare_slots([arow], [crow])
    assert result.duplicates == []
    assert len(result.contradictions) == 1
    assert result.aelfrice_only == []
    assert result.claude_only == []


def test_compare_slots_aelfrice_only() -> None:
    arow = _make_slot("DB", "schema", "normalised")
    result = compare_slots([arow], [])
    assert result.aelfrice_only == [arow]
    assert result.duplicates == []
    assert result.contradictions == []
    assert result.claude_only == []


def test_compare_slots_claude_only() -> None:
    crow = _make_slot("Cache", "policy", "LRU")
    result = compare_slots([], [crow])
    assert result.claude_only == [crow]
    assert result.duplicates == []
    assert result.contradictions == []
    assert result.aelfrice_only == []


def test_compare_slots_case_insensitive_key_match() -> None:
    arow = _make_slot("auth", "Token", "lasts 24 hours")
    crow = _make_slot("Auth", "token", "lasts 24 hours")
    result = compare_slots([arow], [crow])
    assert len(result.duplicates) == 1
    assert result.contradictions == []


def test_compare_slots_case_insensitive_value_match() -> None:
    arow = _make_slot("Auth", "token", "Lasts 24 Hours")
    crow = _make_slot("Auth", "token", "lasts 24 hours")
    result = compare_slots([arow], [crow])
    assert len(result.duplicates) == 1


def test_compare_slots_mixed_buckets() -> None:
    a_dup = _make_slot("Auth", "token", "lasts 24 hours", source="a-dup")
    a_cont = _make_slot("DB", "schema", "normalised", source="a-cont")
    a_only = _make_slot("Cache", "policy", "LRU", source="a-only")

    c_dup = _make_slot("Auth", "token", "lasts 24 hours", source="c-dup")
    c_cont = _make_slot("DB", "schema", "non-normalised", source="c-cont")
    c_only = _make_slot("Retry", "logic", "exponential backoff", source="c-only")

    result = compare_slots(
        [a_dup, a_cont, a_only],
        [c_dup, c_cont, c_only],
    )
    assert len(result.duplicates) == 1
    assert result.duplicates[0][0].source == "a-dup"
    assert result.duplicates[0][1].source == "c-dup"

    assert len(result.contradictions) == 1
    assert result.contradictions[0][0].source == "a-cont"
    assert result.contradictions[0][1].source == "c-cont"

    assert len(result.aelfrice_only) == 1
    assert result.aelfrice_only[0].source == "a-only"

    assert len(result.claude_only) == 1
    assert result.claude_only[0].source == "c-only"


def test_compare_slots_empty_slot_goes_to_respective_only() -> None:
    # Rows with empty subject/predicate are unmatched but still land in
    # the correct bucket.
    arow = SlotRow("", "", "value", "raw", "src", "path")
    result = compare_slots([arow], [])
    assert arow in result.aelfrice_only

    crow = SlotRow("", "", "value", "raw", "src", "path")
    result2 = compare_slots([], [crow])
    assert crow in result2.claude_only
