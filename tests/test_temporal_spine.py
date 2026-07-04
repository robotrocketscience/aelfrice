"""Unit tests for the #1064 temporal-spine writer + ingest wiring.

Covers ``write_temporal_spine`` (per-session TEMPORAL_NEXT chains,
src = successor / dst = predecessor / weight 0.8), the
``session_predecessor_id`` store accessor's ordering contract
(created_at, insertion order as tie-break), the default-off
``write_temporal_spine`` flag resolver, idempotency, and the
byte-identical off-path through ``ingest_turn``.

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.ingest import ingest_turn
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import (
    ENV_TEMPORAL_SPINE_WRITE,
    TEMPORAL_SPINE_EDGE_WEIGHT,
    is_temporal_spine_write_enabled,
    write_temporal_spine,
)


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    session_id: str | None = None,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        session_id=session_id,
    )
    store.insert_belief(b)
    return b


def _spine_edges(store: MemoryStore) -> list[tuple[str, str, float]]:
    """All TEMPORAL_NEXT edges as (src, dst, weight), sorted."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst, weight FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_TEMPORAL_NEXT,),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# Flag resolver precedence
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    # start at an empty dir so no repo .aelfrice.toml is found
    assert is_temporal_spine_write_enabled(start=tmp_path) is False


def test_flag_env_wins_over_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "off")
    assert is_temporal_spine_write_enabled(explicit=True) is False
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "on")
    assert is_temporal_spine_write_enabled(explicit=False) is True


def test_flag_unrecognised_env_not_decisive(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "maybe")
    assert is_temporal_spine_write_enabled(explicit=True, start=tmp_path) is True
    assert is_temporal_spine_write_enabled(start=tmp_path) is False


def test_flag_kwarg_wins_over_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[ingest]\nwrite_temporal_spine = true\n"
    )
    assert is_temporal_spine_write_enabled(explicit=False, start=tmp_path) is False


def test_flag_toml_read(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[ingest]\nwrite_temporal_spine = true\n"
    )
    assert is_temporal_spine_write_enabled(start=tmp_path) is True
    (tmp_path / ".aelfrice.toml").write_text(
        "[ingest]\nwrite_temporal_spine = false\n"
    )
    assert is_temporal_spine_write_enabled(start=tmp_path) is False


def test_flag_malformed_toml_not_decisive(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[ingest]\nwrite_temporal_spine = 'yes'\n"
    )
    assert is_temporal_spine_write_enabled(start=tmp_path) is False


# ---------------------------------------------------------------------------
# session_predecessor_id ordering contract
# ---------------------------------------------------------------------------


def test_predecessor_orders_by_created_at(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="first fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b2", content="second fact",
                 session_id="s1", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="b3", content="third fact",
                 session_id="s1", created_at="2026-01-01T00:00:03Z")
    assert store.session_predecessor_id("b1") is None
    assert store.session_predecessor_id("b2") == "b1"
    assert store.session_predecessor_id("b3") == "b2"


def test_predecessor_tie_breaks_on_insertion_order(store: MemoryStore) -> None:
    # Identical created_at: insertion order (rowid) decides the chain.
    ts = "2026-01-01T00:00:00Z"
    _make_belief(store, belief_id="z-late", content="inserted first",
                 session_id="s1", created_at=ts)
    _make_belief(store, belief_id="a-early", content="inserted second",
                 session_id="s1", created_at=ts)
    assert store.session_predecessor_id("z-late") is None
    assert store.session_predecessor_id("a-early") == "z-late"


def test_predecessor_scoped_to_session(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="session one fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b2", content="session two fact",
                 session_id="s2", created_at="2026-01-01T00:00:02Z")
    assert store.session_predecessor_id("b2") is None


def test_predecessor_null_session_and_missing(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="no session fact",
                 session_id=None)
    assert store.session_predecessor_id("b1") is None
    assert store.session_predecessor_id("nonexistent") is None


# ---------------------------------------------------------------------------
# write_temporal_spine
# ---------------------------------------------------------------------------


def test_writer_chains_session(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="first fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b2", content="second fact",
                 session_id="s1", created_at="2026-01-01T00:00:02Z")

    report = write_temporal_spine(store, new_belief_ids=["b1", "b2"])

    assert report.n_beliefs_seen == 2
    assert report.n_edges_written == 1
    assert report.n_skipped_no_predecessor == 1
    assert _spine_edges(store) == [
        ("b2", "b1", TEMPORAL_SPINE_EDGE_WEIGHT),
    ]


def test_writer_links_batch_to_prior_session_tail(store: MemoryStore) -> None:
    # b1 chained in an earlier turn; a later turn's batch must link its
    # first belief back to the store's existing session tail.
    _make_belief(store, belief_id="b1", content="prior turn fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    write_temporal_spine(store, new_belief_ids=["b1"])

    _make_belief(store, belief_id="b2", content="next turn fact",
                 session_id="s1", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="b3", content="another next turn fact",
                 session_id="s1", created_at="2026-01-01T00:00:03Z")
    report = write_temporal_spine(store, new_belief_ids=["b2", "b3"])

    assert report.n_edges_written == 2
    assert _spine_edges(store) == [
        ("b2", "b1", TEMPORAL_SPINE_EDGE_WEIGHT),
        ("b3", "b2", TEMPORAL_SPINE_EDGE_WEIGHT),
    ]


def test_writer_sessions_isolated(store: MemoryStore) -> None:
    _make_belief(store, belief_id="a1", content="session a first",
                 session_id="sa", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b1", content="session b first",
                 session_id="sb", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="a2", content="session a second",
                 session_id="sa", created_at="2026-01-01T00:00:03Z")

    report = write_temporal_spine(store, new_belief_ids=["a1", "b1", "a2"])

    assert report.n_edges_written == 1
    assert _spine_edges(store) == [
        ("a2", "a1", TEMPORAL_SPINE_EDGE_WEIGHT),
    ]


def test_writer_skips_null_session(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="no session fact",
                 session_id=None)
    report = write_temporal_spine(store, new_belief_ids=["b1", "ghost"])
    assert report.n_beliefs_seen == 2
    assert report.n_skipped_no_session == 2
    assert _spine_edges(store) == []


def test_writer_idempotent(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="first fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b2", content="second fact",
                 session_id="s1", created_at="2026-01-01T00:00:02Z")

    first = write_temporal_spine(store, new_belief_ids=["b1", "b2"])
    second = write_temporal_spine(store, new_belief_ids=["b1", "b2"])

    assert first.n_edges_written == 1
    assert second.n_edges_written == 0
    assert second.n_skipped_existing == 1
    assert len(_spine_edges(store)) == 1


def test_writer_dedupes_input_ids(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content="first fact",
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="b2", content="second fact",
                 session_id="s1", created_at="2026-01-01T00:00:02Z")
    report = write_temporal_spine(store, new_belief_ids=["b2", "b2", "b1"])
    assert report.n_beliefs_seen == 2
    assert report.n_edges_written == 1


# ---------------------------------------------------------------------------
# Ingest wiring
# ---------------------------------------------------------------------------

_TURN_ONE = "The staging database runs on port 5433."
_TURN_TWO = "The staging cache was flushed after the last deploy."


def test_ingest_off_path_writes_no_spine_edges(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    ingest_turn(store, _TURN_ONE, "test-source", session_id="s1")
    ingest_turn(store, _TURN_TWO, "test-source", session_id="s1")
    assert _spine_edges(store) == []


def test_ingest_on_path_chains_consecutive_turns(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "1")
    n1 = ingest_turn(store, _TURN_ONE, "test-source", session_id="s1",
                     created_at="2026-01-01T00:00:01Z")
    n2 = ingest_turn(store, _TURN_TWO, "test-source", session_id="s1",
                     created_at="2026-01-01T00:00:02Z")
    assert n1 == 1 and n2 == 1
    edges = _spine_edges(store)
    assert len(edges) == 1
    src, dst, weight = edges[0]
    assert weight == TEMPORAL_SPINE_EDGE_WEIGHT
    # src is the later turn's belief, dst the earlier turn's belief.
    src_belief = store.get_belief(src)
    dst_belief = store.get_belief(dst)
    assert src_belief is not None and dst_belief is not None
    assert src_belief.created_at > dst_belief.created_at


def test_ingest_on_path_skips_other_sessions(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "1")
    ingest_turn(store, _TURN_ONE, "test-source", session_id="s1",
                created_at="2026-01-01T00:00:01Z")
    ingest_turn(store, _TURN_TWO, "test-source", session_id="s2",
                created_at="2026-01-01T00:00:02Z")
    assert _spine_edges(store) == []
