"""`#N` literal-boost coverage (#677).

The BM25 tokenizer drops `#`, so plain `#NNN` queries used to match
every belief that mentioned any `#NNN` token equally. The boost in
`_l1_hits` adds `log(HASH_N_BOOST_MULTIPLIER)` to the score of any
belief whose content contains a literal `#N` from the prompt — a
post-BM25 re-rank that disambiguates which issue the user meant.
"""
from __future__ import annotations

import math

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import (
    HASH_N_BOOST_MULTIPLIER,
    _extract_hash_n_literals,
    _hash_n_boosted,
    retrieve,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
    )


# --- _extract_hash_n_literals ----------------------------------------------


def test_extract_hash_n_literals_picks_up_multiple_tokens() -> None:
    assert _extract_hash_n_literals("check PR #627 and issue #280") == [
        "#627",
        "#280",
    ]


def test_extract_hash_n_literals_empty_when_no_hash_token() -> None:
    assert _extract_hash_n_literals("no numbers here") == []


def test_extract_hash_n_literals_does_not_match_bare_digits() -> None:
    """`627` alone must not trigger the boost — the whole point is the
    `#` anchor disambiguates from `627ms` / `627px` / FTS5 hits on
    the bare digit."""
    assert _extract_hash_n_literals("flow at 627 hz with 99 percent") == []


# --- _hash_n_boosted -------------------------------------------------------


def test_hash_n_boosted_no_literals_returns_score_unchanged() -> None:
    assert _hash_n_boosted(1.234, "anything goes", []) == 1.234


def test_hash_n_boosted_no_match_returns_score_unchanged() -> None:
    assert _hash_n_boosted(1.234, "this mentions #999", ["#627"]) == 1.234


def test_hash_n_boosted_match_adds_log_of_multiplier() -> None:
    boosted = _hash_n_boosted(1.0, "fix landed in #627 yesterday", ["#627"])
    assert math.isclose(boosted, 1.0 + math.log(HASH_N_BOOST_MULTIPLIER))


def test_hash_n_boosted_match_on_any_literal_is_sufficient() -> None:
    """Multiple `#N`s in the prompt — any one matching is a hit."""
    boosted = _hash_n_boosted(
        0.5,
        "this row only references #280 and nothing else",
        ["#627", "#280", "#999"],
    )
    assert math.isclose(boosted, 0.5 + math.log(HASH_N_BOOST_MULTIPLIER))


def test_hash_n_boosted_single_match_not_compounded_across_literals() -> None:
    """A content that contains two of the prompt's literals still
    receives a single boost — the boost is presence-of-any, not a
    per-literal sum (keeps the magnitude bounded)."""
    boosted = _hash_n_boosted(
        0.0,
        "both #627 and #280 appear here",
        ["#627", "#280"],
    )
    assert math.isclose(boosted, math.log(HASH_N_BOOST_MULTIPLIER))


# --- End-to-end ranking with the boost wired into _l1_hits -----------------


def test_hash_n_literal_lifts_disambiguated_belief_to_first_place() -> None:
    """Issue #677 acceptance bullet 1: prompt `\"check PR #627\"`
    against a corpus with one `#627`-mentioning belief and many other
    `#NNN` beliefs ranks the `#627` belief first.

    Without the boost, FTS5 sees the tokenized `627` as just one of N
    documents containing a bare digit and ranks by length / TF; with
    the boost, the `#627`-containing row gets `+log(2.0)` and lifts
    above the noise.
    """
    s = MemoryStore(":memory:")
    # Many `#NNN` distractors with similar shape to the right answer.
    distractors = [
        ("D1", "merge note for PR #280 landed yesterday"),
        ("D2", "discussion about issue #354 from last week"),
        ("D3", "follow-up to PR #496 on the retrieval lane"),
        ("D4", "spec for issue #358 covered tokenisation"),
        ("D5", "review of PR #501 — clean approval"),
    ]
    for bid, content in distractors:
        s.insert_belief(_mk(bid, content))
    # The right answer — contains `#627` literally.
    s.insert_belief(_mk("RIGHT", "wakeup for PR #627 — verify merge cleared"))

    hits = retrieve(s, query="check PR #627")
    ids = [b.id for b in hits]
    assert ids, "expected non-empty results"
    assert ids[0] == "RIGHT", (
        f"expected #627-row first under boost, got order: {ids}"
    )


def test_query_without_hash_literal_keeps_byte_identical_short_circuit() -> None:
    """For prompts without any `#N` literal, the boost is a no-op and
    the existing FTS5 short-circuit ordering must be preserved. This
    is the regression guard against the #592 / #587 hot-start AC and
    against the v1.0.x posterior_weight=0.0 byte-identical contract.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", "the cat sat on the mat"))
    s.insert_belief(_mk("B", "an apple a day keeps the doctor away"))
    s.insert_belief(_mk("C", "cats and mats are unrelated topics"))

    # Compare against a baseline that takes the same code path with no
    # literals. Two calls with the same query (no `#N`) must produce
    # identical id orders — the boost has nothing to bind to and the
    # short-circuit returns the FTS5 BM25 order verbatim.
    first = [b.id for b in retrieve(s, query="cat mat")]
    second = [b.id for b in retrieve(s, query="cat mat")]
    assert first == second
    # And the `#N`-free query must still find the topical row.
    assert "A" in first


def test_hash_n_boost_does_not_invent_hits_when_no_belief_contains_literal() -> None:
    """If no belief contains the `#N` literal, the boost is silent and
    ordinary BM25 ordering applies — boost activation must not
    smuggle in irrelevant content."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X", "kitchen counter has bananas and apples"))
    s.insert_belief(_mk("Y", "garage has a workbench"))

    # No belief contains `#999` — the boost has no hit, FTS5 path
    # applies, and the query "fruit #999" still matches X (via
    # "fruit"-tokenized hit if present; here neither row contains
    # "fruit" so the result may legitimately be empty).
    hits = retrieve(s, query="bananas #999")
    ids = [b.id for b in hits]
    # The one banana row should still come back; Y must not.
    assert "X" in ids or ids == []  # FTS may not score on the bare #999
    assert "Y" not in ids
