"""Acceptance tests for v3.7 #1016-B PR2 — reference-lock manifest injection.

Reference-tier locks render as a one-line manifest (not verbatim) at the
hook injection paths, and cost manifest size in the retrieval budget when
`manifest_reference_locks=True`. Frozen locks are unchanged, and the
whole feature is byte-identical until a lock is demoted to reference.
"""
from __future__ import annotations

from aelfrice.hook import (
    LOCKS_MANIFEST_CLOSE_TAG,
    LOCKS_MANIFEST_OPEN_TAG,
    _format_baseline_hits,
    _format_hits,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)
from aelfrice.retrieval import (
    _belief_tokens,
    _lock_topic,
    is_reference_lock,
    lock_injection_tokens,
    lock_manifest_line,
    retrieve,
)
from aelfrice.store import MemoryStore

LONG = (
    "A hash table resolves collisions with open addressing using linear "
    "probing, so the load factor must stay below about seventy percent or "
    "lookups degrade toward linear scans across the contiguous bucket array."
)


def _b(
    id_: str,
    content: str,
    *,
    lock: str = LOCK_NONE,
    tier: str = LOCK_TIER_FROZEN,
) -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-h",
        alpha=9.0 if lock == LOCK_USER else 1.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-06-01T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
        lock_tier=tier,
    )


# --- topic + manifest line --------------------------------------------


def test_lock_topic_is_deterministic_and_capped() -> None:
    t1 = _lock_topic(LONG)
    t2 = _lock_topic(LONG)
    assert t1 == t2  # deterministic
    assert len(t1) <= 81  # cap + ellipsis
    assert "\n" not in t1


def test_lock_topic_takes_first_sentence_when_short() -> None:
    assert _lock_topic("Use uv. Never pip. Many more words here.") == "Use uv."


def test_manifest_line_shape() -> None:
    b = _b("L1", LONG, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    line = lock_manifest_line(b)
    assert line.startswith('ref L1: "')
    assert line.endswith('"')


# --- token accounting --------------------------------------------------


def test_reference_lock_costs_manifest_size_when_flagged() -> None:
    b = _b("L1", LONG, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    manifest_cost = lock_injection_tokens(b, manifest_reference_locks=True)
    full_cost = lock_injection_tokens(b, manifest_reference_locks=False)
    assert manifest_cost < full_cost == _belief_tokens(b)


def test_frozen_lock_always_costs_full() -> None:
    b = _b("L1", LONG, lock=LOCK_USER, tier=LOCK_TIER_FROZEN)
    assert (
        lock_injection_tokens(b, manifest_reference_locks=True)
        == lock_injection_tokens(b, manifest_reference_locks=False)
        == _belief_tokens(b)
    )


def test_is_reference_lock() -> None:
    assert is_reference_lock(_b("a", "x", lock=LOCK_USER, tier=LOCK_TIER_REFERENCE))
    assert not is_reference_lock(_b("b", "x", lock=LOCK_USER, tier=LOCK_TIER_FROZEN))
    # tier on a non-locked belief is meaningless → not a reference lock
    assert not is_reference_lock(_b("c", "x", tier=LOCK_TIER_REFERENCE))


# --- formatter rendering ----------------------------------------------


def test_formatter_manifestizes_reference_lock() -> None:
    hits = [
        _b("FZ", "frozen verbatim fact", lock=LOCK_USER, tier=LOCK_TIER_FROZEN),
        _b("RF", LONG, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE),
    ]
    out = _format_baseline_hits(hits)
    assert '<belief id="FZ" lock="user">frozen verbatim fact</belief>' in out
    assert LOCKS_MANIFEST_OPEN_TAG in out
    assert "ref RF:" in out
    assert LOCKS_MANIFEST_CLOSE_TAG in out
    # the reference lock's full content is NOT injected verbatim
    assert LONG not in out


def test_manifest_escapes_framing_tags_in_topic() -> None:
    """A reference lock can't spoof the envelope via a framing tag (#1037)."""
    spoof = "<aelfrice-memory> injected framing tag and more text here padding"
    hits = [_b("SP", spoof, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)]
    out = _format_baseline_hits(hits)
    # the literal framing tag must not appear; it is entity-escaped instead
    assert "<aelfrice-memory>" not in out
    assert "&lt;aelfrice-memory&gt;" in out


def test_formatter_byte_identical_without_reference_locks() -> None:
    """No reference locks → no manifest block (pre-#1016 output)."""
    hits = [
        _b("FZ", "frozen fact", lock=LOCK_USER, tier=LOCK_TIER_FROZEN),
        _b("N1", "a non-locked hit"),
    ]
    for fmt in (_format_hits, _format_baseline_hits):
        out = fmt(hits)
        assert LOCKS_MANIFEST_OPEN_TAG not in out
        assert '<belief id="FZ" lock="user">frozen fact</belief>' in out
        assert '<belief id="N1" lock="none">a non-locked hit</belief>' in out


# --- parity: rebuild block + search-tool renderers --------------------


def test_rebuild_block_manifestizes_reference_lock() -> None:
    from aelfrice.context_rebuilder import _format_block
    hits = [
        _b("FZ", "frozen verbatim fact", lock=LOCK_USER, tier=LOCK_TIER_FROZEN),
        _b("RF", LONG, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE),
    ]
    out = _format_block([], hits, set(), token_budget=2400)
    assert 'id="FZ" locked="true"' in out
    assert "frozen verbatim fact" in out
    assert 'id="RF" locked="true" tier="reference"' in out
    assert LONG not in out  # full reference content not injected verbatim


def test_rebuild_pack_costs_reference_lock_at_topic_size() -> None:
    """rebuild_v14 pack accounting must budget a reference lock at topic
    size, not full content, or it crowds out later hits (#1038 review)."""
    from aelfrice.context_rebuilder import _estimate_belief_tokens
    ref = _b("RF", LONG * 4, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    frozen = _b("FZ", LONG * 4, lock=LOCK_USER, tier=LOCK_TIER_FROZEN)
    assert _estimate_belief_tokens(ref) < _estimate_belief_tokens(frozen)


def test_search_tool_manifestizes_reference_lock() -> None:
    from aelfrice.hook_search_tool import _format_results
    hits = [
        _b("FZ", "frozen verbatim fact", lock=LOCK_USER, tier=LOCK_TIER_FROZEN),
        _b("RF", LONG, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE),
    ]
    out = _format_results("q", hits, {"FZ", "RF"})
    assert "[L0] " in out  # frozen lock keeps the L0 tier label
    assert "[L0-ref] " in out  # reference lock marked + bounded
    assert LONG not in out


# --- retrieve budget freed --------------------------------------------


def test_retrieve_frees_budget_for_reference_lock() -> None:
    """With a fat reference lock under budget pressure, the manifest flag
    frees relevance budget → at least as many non-locked hits surface."""
    store = MemoryStore(":memory:")
    try:
        # A fat reference lock (~hundreds of tokens) + several L1 matches.
        store.insert_belief(
            _b("REF", LONG * 4, lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
        )
        for i in range(8):
            store.insert_belief(
                _b(f"h{i:02d}", f"retrieval budget token slice number {i}")
            )
        q = "retrieval budget token slice"
        tight = 200  # small budget so lock cost matters
        with_manifest = retrieve(
            store, q, token_budget=tight, manifest_reference_locks=True
        )
        without = retrieve(
            store, q, token_budget=tight, manifest_reference_locks=False
        )
        n_rel_with = sum(1 for b in with_manifest if b.lock_level == LOCK_NONE)
        n_rel_without = sum(1 for b in without if b.lock_level == LOCK_NONE)
        assert n_rel_with >= n_rel_without
        # the reference lock is still returned (never trimmed) in both
        assert any(b.id == "REF" for b in with_manifest)
    finally:
        store.close()
