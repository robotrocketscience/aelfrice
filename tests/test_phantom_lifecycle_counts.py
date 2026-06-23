"""Phantom (speculative) lifecycle counts surfaced by `aelf status` (#980).

`MemoryStore.count_phantom_lifecycle` partitions every `type='speculative'`
belief into three mutually-exclusive lifecycle states (active / promoted /
retired) plus the latest live-phantom timestamp. These tests pin the
transitions — ingest -> promote -> GC — and the two surfaces that render
them (CLI `aelf status`, MCP `aelf:stats`).
"""
from __future__ import annotations

import argparse
import io

import pytest

from aelfrice.cli import _cmd_stats
from aelfrice.mcp_server import tool_stats
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    PhantomLifecycleCounts,
    RETENTION_FACT,
    Belief,
    Phantom,
)
from aelfrice.promotion import promote
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_ingest


def _constituent(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"constituent content for {bid}",
        content_hash=f"ch_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-01T00:00:00+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _phantom(content: str, *, score: float = 0.75) -> Phantom:
    # wonder_ingest dedups on (sorted constituents, generator), so vary the
    # generator per phantom to persist distinct rows from the same anchors.
    return Phantom(
        constituent_belief_ids=("a", "b"),
        generator=f"gen-{content}",
        content=content,
        score=score,
    )


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_constituent("a"))
    s.insert_belief(_constituent("b"))
    return s


def _speculative_ids(store: MemoryStore) -> list[str]:
    return [
        b.id
        for bid in store.list_belief_ids()
        if (b := store.get_belief(bid)) is not None and b.type == "speculative"
    ]


# ---------------------------------------------------------------------------
# Store method: lifecycle partition
# ---------------------------------------------------------------------------


def test_empty_store_reports_zeroes_and_no_latest(store: MemoryStore) -> None:
    counts = store.count_phantom_lifecycle()
    assert counts == PhantomLifecycleCounts(
        active=0, promoted=0, retired=0, latest=None
    )


def test_ingest_makes_phantoms_active(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1"), _phantom("p2")])
    counts = store.count_phantom_lifecycle()
    assert counts.active == 2
    assert counts.promoted == 0
    assert counts.retired == 0
    assert counts.latest is not None


def test_promotion_moves_active_to_promoted(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1"), _phantom("p2")])
    spec_id = _speculative_ids(store)[0]

    promote(store, spec_id)

    counts = store.count_phantom_lifecycle()
    assert counts.active == 1
    assert counts.promoted == 1
    assert counts.retired == 0


def test_soft_delete_moves_active_to_retired(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1"), _phantom("p2")])
    spec_id = _speculative_ids(store)[0]

    store.soft_delete_belief(spec_id, ts="2026-06-01T00:00:00+00:00")

    counts = store.count_phantom_lifecycle()
    assert counts.active == 1
    assert counts.promoted == 0
    assert counts.retired == 1


def test_states_are_mutually_exclusive_across_full_lifecycle(
    store: MemoryStore,
) -> None:
    wonder_ingest(
        store, [_phantom("p1"), _phantom("p2"), _phantom("p3")]
    )
    ids = _speculative_ids(store)
    promote(store, ids[0])
    store.soft_delete_belief(ids[1], ts="2026-06-01T00:00:00+00:00")

    counts = store.count_phantom_lifecycle()
    # one promoted, one retired, one still active — sum equals ingested.
    assert (counts.active, counts.promoted, counts.retired) == (1, 1, 1)


def test_latest_tracks_most_recent_live_phantom(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1")])
    counts = store.count_phantom_lifecycle()
    # wonder_ingest stamps created_at at ingest time; just assert it is the
    # ISO string of the one live phantom.
    live_id = _speculative_ids(store)[0]
    live = store.get_belief(live_id)
    assert live is not None
    assert counts.latest == live.created_at


def test_retired_phantom_does_not_count_as_latest(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1")])
    spec_id = _speculative_ids(store)[0]
    store.soft_delete_belief(spec_id, ts="2026-06-01T00:00:00+00:00")

    counts = store.count_phantom_lifecycle()
    assert counts.retired == 1
    assert counts.latest is None


# ---------------------------------------------------------------------------
# Surfaces: CLI `aelf status` + MCP `aelf:stats`
# ---------------------------------------------------------------------------


def test_cli_status_prints_phantom_line(
    monkeypatch: pytest.MonkeyPatch, store: MemoryStore
) -> None:
    wonder_ingest(store, [_phantom("p1"), _phantom("p2")])
    promote(store, _speculative_ids(store)[0])

    monkeypatch.setattr("aelfrice.cli._open_store", lambda: store)
    monkeypatch.setattr(store, "close", lambda: None)

    buf = io.StringIO()
    rc = _cmd_stats(argparse.Namespace(), buf)
    assert rc == 0

    out = buf.getvalue()
    assert "phantoms:" in out
    assert "1 active" in out
    assert "1 promoted" in out
    assert "0 retired" in out


def test_cli_status_phantom_line_empty_store(
    monkeypatch: pytest.MonkeyPatch, store: MemoryStore
) -> None:
    monkeypatch.setattr("aelfrice.cli._open_store", lambda: store)
    monkeypatch.setattr(store, "close", lambda: None)

    buf = io.StringIO()
    _cmd_stats(argparse.Namespace(), buf)
    out = buf.getvalue()
    assert "0 active · 0 promoted · 0 retired" in out
    assert "latest: —" in out


def test_mcp_stats_payload_includes_phantoms(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1")])
    payload = tool_stats(store)
    assert payload["phantoms"] == {
        "active": 1,
        "promoted": 0,
        "retired": 0,
        "latest": payload["phantoms"]["latest"],
    }
    assert payload["phantoms"]["latest"] is not None


def test_mcp_stats_markdown_renders_phantom_line(store: MemoryStore) -> None:
    wonder_ingest(store, [_phantom("p1")])
    result = tool_stats(store, response_format="markdown")
    markdown = result["text"]
    assert "Phantoms:" in markdown
    assert "1 active" in markdown
