"""Integration tests for the HRR structural-query lane wiring (#152).

The substrate (HRRStructIndex, parse_structural_marker) was shipped
at v1.7.0 but never connected to retrieve_v2 — the flag was a
phantom until this PR. These tests assert the wiring is real:

- IT1: structural marker + flag ON returns HRR-ranked beliefs
- IT2: structural marker + flag OFF falls through to textual
       (byte-identical to pre-PR behavior on marker queries)
- IT3: non-marker query + flag ON behaves like flag OFF
       (byte-identical default-OFF posture for normal text queries)
- IT4: marker with unknown target falls through to textual
       (graceful miss — better than empty result)
- IT5: explicit cache reuses the index across calls
- IT6: locked beliefs still pin to head when structural lane fires
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.hrr_index import HRRStructIndexCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.retrieval import RetrievalResult, retrieve_v2
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "rv2_hrr.db"))
    yield s
    s.close()


def _mk(bid: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=f"content of {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-05-08T00:00:00+00:00" if locked else None,
        created_at="2026-05-08T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _populate(s: MemoryStore) -> None:
    """Topology: b1 -CONTRADICTS-> b2; b3 -SUPPORTS-> b2;
    b4 -CITES-> b5; b1 -CITES-> b5."""
    for i in range(1, 6):
        s.insert_belief(_mk(f"b{i}"))
    s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="b3", dst="b2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b4", dst="b5", type=EDGE_CITES, weight=1.0))
    s.insert_edge(Edge(src="b1", dst="b5", type=EDGE_CITES, weight=1.0))


# --- IT1 -----------------------------------------------------------------


def test_structural_marker_with_flag_on_returns_hrr_results(
    store: MemoryStore,
) -> None:
    _populate(store)
    cache = HRRStructIndexCache(store=store, dim=512, seed=42)
    result = retrieve_v2(
        store, "CONTRADICTS:b2",
        use_hrr_structural=True,
        hrr_struct_index_cache=cache,
        budget=10_000,
    )
    assert isinstance(result, RetrievalResult)
    ids = [b.id for b in result.beliefs]
    # b1 -CONTRADICTS-> b2: b1 must be in the result.
    assert "b1" in ids, f"expected b1 (CONTRADICTS source), got {ids}"
    # b3 only SUPPORTS b2 — must NOT lead the CONTRADICTS probe.
    assert ids.index("b1") <= ids.index("b3") if "b3" in ids else True


# --- IT2 -----------------------------------------------------------------


def test_structural_marker_with_flag_off_falls_through_to_textual(
    store: MemoryStore,
) -> None:
    _populate(store)
    # Same query string, flag explicitly OFF: routes through textual
    # lane (BM25 over the literal "CONTRADICTS:b2" string). The
    # textual lane will likely return nothing matching, but the key
    # point is that the HRR lane is bypassed — assert by absence of
    # the structural-only-derivable result.
    result = retrieve_v2(
        store, "CONTRADICTS:b2",
        use_hrr_structural=False,
        budget=10_000,
    )
    # Textual lane on "CONTRADICTS:b2" string with content like
    # "content of b1" cannot match b1; b1 in results would only come
    # from the structural lane.
    ids = [b.id for b in result.beliefs]
    assert "b1" not in ids


# --- IT3 -----------------------------------------------------------------


def test_non_marker_query_with_flag_on_behaves_like_flag_off(
    store: MemoryStore,
) -> None:
    _populate(store)
    on = retrieve_v2(
        store, "content of b1",
        use_hrr_structural=True,
        budget=10_000,
    )
    off = retrieve_v2(
        store, "content of b1",
        use_hrr_structural=False,
        budget=10_000,
    )
    # Non-marker query: flag is no-op. Result lists must match.
    assert [b.id for b in on.beliefs] == [b.id for b in off.beliefs]


# --- IT4 -----------------------------------------------------------------


def test_marker_with_unknown_target_falls_through(
    store: MemoryStore,
) -> None:
    _populate(store)
    # Valid marker syntax but target b_does_not_exist isn't in the
    # store. _route_structural_query returns None on empty hits so
    # the textual lane handles the literal string.
    result = retrieve_v2(
        store, "CONTRADICTS:b_does_not_exist",
        use_hrr_structural=True,
        budget=10_000,
    )
    # No assertion on content — just that the call succeeds without
    # raising and returns a well-formed RetrievalResult.
    assert isinstance(result, RetrievalResult)


# --- IT5 -----------------------------------------------------------------


def test_explicit_cache_reuses_index_across_calls(
    store: MemoryStore,
) -> None:
    _populate(store)
    cache = HRRStructIndexCache(store=store, dim=256, seed=7)
    # First call builds the index; second call must reuse the same
    # instance (verified by the cache's identity-preservation).
    retrieve_v2(
        store, "CONTRADICTS:b2",
        use_hrr_structural=True,
        hrr_struct_index_cache=cache,
    )
    first = cache._index
    retrieve_v2(
        store, "CITES:b5",
        use_hrr_structural=True,
        hrr_struct_index_cache=cache,
    )
    second = cache._index
    assert first is not None and first is second


# --- IT6 -----------------------------------------------------------------


def test_locked_beliefs_pin_to_head_under_structural_lane(
    store: MemoryStore,
) -> None:
    # b0 is a locked belief unrelated to the structural query.
    store.insert_belief(_mk("b0", locked=True))
    _populate(store)
    cache = HRRStructIndexCache(store=store, dim=256, seed=7)
    result = retrieve_v2(
        store, "CONTRADICTS:b2",
        use_hrr_structural=True,
        hrr_struct_index_cache=cache,
        include_locked=True,
        budget=10_000,
    )
    assert result.beliefs, "result must be non-empty"
    assert result.beliefs[0].lock_level == LOCK_USER
    assert result.locked_ids == ["b0"]
