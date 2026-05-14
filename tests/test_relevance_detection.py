"""Tests for #779 Layer 2 — reference detection (exact-substring)."""
from __future__ import annotations

import pytest

from aelfrice.relevance_detection import (
    STRATEGY_EXACT_SUBSTRING,
    STRATEGY_NGRAM_OVERLAP,
    is_referenced,
    normalize_text,
    score_references,
)


# --- normalize_text --------------------------------------------------

def test_normalize_idempotent() -> None:
    """Re-running normalize on its own output is a fixed point."""
    for sample in [
        "Hello World",
        "café  resume",
        "MIXED Case TEXT",
        "  leading and trailing  ",
        "tabs\tand\nnewlines\r\nmixed",
    ]:
        once = normalize_text(sample)
        twice = normalize_text(once)
        assert once == twice, (sample, once, twice)


def test_normalize_casefold_mixed_case() -> None:
    assert normalize_text("Hello WORLD") == "hello world"


def test_normalize_nfc_compose_decomposed_diacritics() -> None:
    """NFC: pre-composed and decomposed café match after normalize."""
    composed = "café"  # 'é' as single codepoint
    decomposed = "café"  # 'e' + combining acute
    assert normalize_text(composed) == normalize_text(decomposed)


def test_normalize_collapses_whitespace_variants() -> None:
    """CRLF, tabs, multiple spaces all collapse to single spaces."""
    raw = "line one\r\n\tline two   line three"
    assert normalize_text(raw) == "line one line two line three"


def test_normalize_strips_leading_trailing() -> None:
    assert normalize_text("   x   ") == "x"


def test_normalize_handles_german_sharp_s() -> None:
    """casefold beats lower() on German ß: it folds to 'ss' so that
    a belief written with ß matches a response that spells it out."""
    assert normalize_text("Straße") == normalize_text("STRASSE")


def test_normalize_empty_string() -> None:
    assert normalize_text("") == ""


# --- is_referenced ---------------------------------------------------

def test_referenced_exact_substring_hit() -> None:
    belief = "the dedup_key migration shipped in PR #784"
    response = "yes the dedup_key migration shipped in pr #784 yesterday"
    assert is_referenced(belief, response) is True


def test_referenced_case_insensitive() -> None:
    assert is_referenced(
        "TYPE-aware Compression", "the type-aware compression flip is bench-gated",
    ) is True


def test_referenced_whitespace_normalised() -> None:
    """Belief has single spaces; response has CRLF + tabs in between."""
    belief = "load-bearing claim"
    response = "the\tload-bearing\r\n  claim was contested"
    assert is_referenced(belief, response) is True


def test_referenced_not_present() -> None:
    assert is_referenced(
        "completely unrelated content xyz",
        "the response talks about something else entirely",
    ) is False


def test_referenced_too_short_belief_returns_false() -> None:
    """Beliefs that normalise to less than 8 chars never match —
    precision bias prevents 'abc' from matching almost any response."""
    assert is_referenced("xyz", "the response includes xyz a lot xyz xyz") is False
    assert is_referenced("a b c", "a b c is in here") is False  # only 5 chars normalised


def test_referenced_at_threshold_boundary() -> None:
    """Exactly at 8 chars is allowed (>= threshold)."""
    belief = "abcdefgh"  # 8 chars, normalised to "abcdefgh"
    assert is_referenced(belief, "prefix abcdefgh suffix") is True


def test_referenced_empty_belief_false() -> None:
    assert is_referenced("", "anything here") is False


def test_referenced_empty_response_false() -> None:
    assert is_referenced("some belief content here", "") is False


def test_referenced_unicode_diacritic_match() -> None:
    """Composed belief content matches decomposed response form."""
    belief = "the café opens at noon"  # composed é
    response = "the café opens at noon every weekday"  # decomposed
    assert is_referenced(belief, response) is True


# --- score_references (bulk) -----------------------------------------

def test_score_references_returns_one_pair_per_input() -> None:
    pairs = [
        (1, "the load-bearing claim was overruled"),
        (2, "type-aware compression flip is bench-gated"),
        (3, "totally unrelated foobar widget"),
    ]
    response = (
        "the load-bearing claim was overruled by the type-aware "
        "compression flip is bench-gated discussion"
    )
    scored = score_references(pairs, response)
    assert scored == [(1, 1), (2, 1), (3, 0)]


def test_score_references_preserves_input_order() -> None:
    pairs = [
        (10, "needle one in the haystack"),
        (3, "needle two in the haystack"),
        (7, "needle three in the haystack"),
    ]
    response = "the response only mentions needle two in the haystack"
    out = score_references(pairs, response)
    assert [eid for eid, _ in out] == [10, 3, 7]


def test_score_references_empty_input() -> None:
    assert score_references([], "any response text") == []


def test_score_references_short_belief_zero_not_filtered() -> None:
    """Short beliefs return (event_id, 0) — not filtered, just scored 0
    — so the sweeper's idempotent update_injection_referenced gets to
    stamp them as scored rather than leaving them pending forever."""
    pairs = [(42, "xy")]
    out = score_references(pairs, "xy is everywhere in this response")
    assert out == [(42, 0)]


def test_score_references_rejects_unsupported_strategy() -> None:
    with pytest.raises(ValueError):
        score_references(
            [(1, "anything here")], "response",
            strategy=STRATEGY_NGRAM_OVERLAP,
        )


def test_score_references_explicit_substring_strategy() -> None:
    """Passing the v1 strategy name explicitly works."""
    out = score_references(
        [(1, "explicit match content")],
        "the response contains explicit match content here",
        strategy=STRATEGY_EXACT_SUBSTRING,
    )
    assert out == [(1, 1)]


# --- determinism -----------------------------------------------------

def test_score_references_deterministic_across_runs() -> None:
    """Same (events, response) → byte-identical output."""
    pairs = [
        (1, "deterministic substrate content here"),
        (2, "another belief about the same"),
        (3, "yet a third unrelated thing"),
    ]
    response = (
        "the deterministic substrate content here is what we want; "
        "yet a third unrelated thing not so much"
    )
    a = score_references(pairs, response)
    b = score_references(list(pairs), response)
    assert a == b
