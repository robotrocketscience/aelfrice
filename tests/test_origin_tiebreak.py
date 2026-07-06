"""#1089 axis-2 origin-priority retrieval tie-break.

Covers the canonical priority map (+ its sync guard against
contradiction.precedence_class), the store SQL tie-break, the flag
resolver, the `_origin_priority` helper, and the retrieve_v2 / _l1_hits
integration on both the default rerank path and the posterior_weight==0
short-circuit. Synthetic fixtures only; default-off is byte-identical.
"""
from __future__ import annotations

import pytest

from aelfrice.contradiction import (
    PRECEDENCE_DOCUMENT_RECENT,
    precedence_class,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_AGENT_REMEMBERED,
    ORIGIN_DOCUMENT_RECENT,
    ORIGIN_RETRIEVAL_PRIORITY,
    ORIGIN_RETRIEVAL_PRIORITY_DEFAULT,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.retrieval import (
    _l1_hits,
    _origin_priority,
    is_origin_tiebreak_enabled,
    retrieve_v2,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str, origin: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )


# --- canonical map + sync guard ------------------------------------------


def test_priority_map_in_sync_with_contradiction_precedence() -> None:
    """The retrieval priority MUST equal contradiction.precedence_class
    for every origin it names — the two copies exist so store.py can
    avoid a circular import, and must never drift (#1089)."""
    for origin, priority in ORIGIN_RETRIEVAL_PRIORITY.items():
        b = _mk("b", "x", origin)
        assert precedence_class(b) == priority, origin


def test_priority_default_matches_document_recent_bucket() -> None:
    """Origins absent from the map fall to the default, which must equal
    precedence_class's document_recent bucket (its 'don't know' class)."""
    assert ORIGIN_RETRIEVAL_PRIORITY_DEFAULT == PRECEDENCE_DOCUMENT_RECENT
    for origin in (ORIGIN_DOCUMENT_RECENT, ORIGIN_UNKNOWN, ORIGIN_AGENT_REMEMBERED):
        assert _origin_priority(origin) == ORIGIN_RETRIEVAL_PRIORITY_DEFAULT
        assert precedence_class(_mk("b", "x", origin)) == PRECEDENCE_DOCUMENT_RECENT


def test_origin_priority_curated_beats_conversational() -> None:
    """The whole point: curated user/feedback (user_validated) outranks
    conversational transcript capture (user_transcript)."""
    assert _origin_priority(ORIGIN_USER_VALIDATED) > _origin_priority(
        ORIGIN_USER_TRANSCRIPT
    )
    assert _origin_priority(ORIGIN_USER_TRANSCRIPT) > _origin_priority(
        ORIGIN_AGENT_INFERRED
    )


# --- resolver ------------------------------------------------------------


def test_resolver_default_off() -> None:
    assert is_origin_tiebreak_enabled(None) is False


def test_resolver_kwarg_true() -> None:
    assert is_origin_tiebreak_enabled(True) is True


def test_resolver_env_overrides_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_ORIGIN_TIEBREAK", "0")
    assert is_origin_tiebreak_enabled(True) is False
    monkeypatch.setenv("AELFRICE_ORIGIN_TIEBREAK", "1")
    assert is_origin_tiebreak_enabled(False) is True


# --- store SQL tie-break -------------------------------------------------


def _seed_tie(store: MemoryStore) -> None:
    # Identical content -> identical bm25 -> a tie. The curated belief is
    # given the LATER id so an id-only tie-break would rank it BELOW the
    # transcript belief; only the origin tie-break can lift it.
    store.insert_belief(
        _mk("aaa_transcript", "widget configuration lives here", ORIGIN_USER_TRANSCRIPT)
    )
    store.insert_belief(
        _mk("zzz_curated", "widget configuration lives here", ORIGIN_USER_VALIDATED)
    )


def test_search_beliefs_tiebreak_lifts_curated() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        on = [b.id for b in s.search_beliefs("widget configuration", origin_tiebreak=True)]
    finally:
        s.close()
    assert on == ["zzz_curated", "aaa_transcript"]


def test_search_beliefs_off_is_byte_identical() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        default = [b.id for b in s.search_beliefs("widget configuration")]
        off = [
            b.id
            for b in s.search_beliefs("widget configuration", origin_tiebreak=False)
        ]
    finally:
        s.close()
    assert default == off


def test_search_beliefs_scored_tiebreak_lifts_curated() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        on = [
            b.id
            for b, _ in s.search_beliefs_scored(
                "widget configuration", origin_tiebreak=True
            )
        ]
    finally:
        s.close()
    assert on == ["zzz_curated", "aaa_transcript"]


# --- retrieve_v2 / _l1_hits integration ----------------------------------


def test_retrieve_v2_default_is_byte_identical_off() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        base = [b.id for b in retrieve_v2(s, "widget configuration").beliefs]
        off = [
            b.id
            for b in retrieve_v2(
                s, "widget configuration", use_origin_tiebreak=False
            ).beliefs
        ]
    finally:
        s.close()
    assert base == off


def test_retrieve_v2_on_lifts_curated_on_rerank_path() -> None:
    """The default retrieve_v2 path is the posterior rerank
    (posterior_weight=0.5), so the tie-break must bite in the rerank
    sort, not only the bm25 short-circuit."""
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        off = [
            b.id
            for b in retrieve_v2(
                s, "widget configuration", use_origin_tiebreak=False
            ).beliefs
        ]
        on = [
            b.id
            for b in retrieve_v2(
                s, "widget configuration", use_origin_tiebreak=True
            ).beliefs
        ]
    finally:
        s.close()
    assert set(off) == set(on)  # reorder, never drop
    assert off == ["aaa_transcript", "zzz_curated"]  # id order without the flag
    assert on == ["zzz_curated", "aaa_transcript"]   # curated lifted with it


def test_l1_hits_short_circuit_tiebreak_when_posterior_zero() -> None:
    """posterior_weight=0.0 short-circuits to store.search_beliefs; the
    SQL tie-break must apply there too."""
    s = MemoryStore(":memory:")
    try:
        _seed_tie(s)
        on = _l1_hits(
            s, "widget configuration", l1_limit=10, posterior_weight=0.0,
            use_origin_tiebreak=True,
        )
    finally:
        s.close()
    assert [b.id for b in on] == ["zzz_curated", "aaa_transcript"]
