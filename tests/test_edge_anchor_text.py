"""Edge.anchor_text round-trips and respects the soft cap."""
from __future__ import annotations

from aelfrice.models import (
    ANCHOR_TEXT_MAX_LEN,
    EDGE_CITES,
    Belief,
    Edge,
    LOCK_NONE,
)
from aelfrice.store import MemoryStore


def _b(id_: str) -> Belief:
    return Belief(
        id=id_,
        content=id_,
        content_hash=id_,
        alpha=1.0,
        beta=1.0,
        type="factual",
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-27T00:00:00+00:00",
        last_retrieved_at=None,
    )


def test_default_anchor_text_is_none() -> None:
    e = Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0)
    assert e.anchor_text is None


def test_anchor_text_persists_through_round_trip() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("a"))
        store.insert_belief(_b("b"))
        e = Edge(
            src="a", dst="b", type=EDGE_CITES, weight=1.0,
            anchor_text="the WAL discussion",
        )
        store.insert_edge(e)
        got = store.get_edge("a", "b", EDGE_CITES)
        assert got is not None
        assert got.anchor_text == "the WAL discussion"
    finally:
        store.close()


def test_none_anchor_text_persists_as_none() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("a"))
        store.insert_belief(_b("b"))
        store.insert_edge(Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0))
        got = store.get_edge("a", "b", EDGE_CITES)
        assert got is not None
        assert got.anchor_text is None
    finally:
        store.close()


def test_anchor_text_cap_truncates_with_no_error() -> None:
    overlong = "x" * (ANCHOR_TEXT_MAX_LEN + 500)
    e = Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0, anchor_text=overlong)
    assert e.anchor_text is not None
    assert len(e.anchor_text) == ANCHOR_TEXT_MAX_LEN


def test_anchor_text_at_cap_unchanged() -> None:
    exact = "y" * ANCHOR_TEXT_MAX_LEN
    e = Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0, anchor_text=exact)
    assert e.anchor_text == exact


def test_update_edge_persists_new_anchor_text() -> None:
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_b("a"))
        store.insert_belief(_b("b"))
        store.insert_edge(Edge(
            src="a", dst="b", type=EDGE_CITES, weight=1.0,
            anchor_text="initial",
        ))
        store.update_edge(Edge(
            src="a", dst="b", type=EDGE_CITES, weight=2.0,
            anchor_text="updated",
        ))
        got = store.get_edge("a", "b", EDGE_CITES)
        assert got is not None
        assert got.weight == 2.0
        assert got.anchor_text == "updated"
    finally:
        store.close()
