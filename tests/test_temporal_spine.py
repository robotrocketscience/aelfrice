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
    backfill_temporal_spine,
    clear_temporal_spine,
    is_temporal_spine_write_enabled,
    maybe_backfill_temporal_spine,
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


def test_flag_defaults_on(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Default-ON since the v4.0 #1064 writer flip: no env, no .aelfrice.toml.
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    # start at an empty dir so no repo .aelfrice.toml is found
    assert is_temporal_spine_write_enabled(start=tmp_path) is True


def test_flag_explicit_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The writer is opt-out now that the default is ON; the env var and the
    # TOML key must still be able to force it back off.
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "off")
    assert is_temporal_spine_write_enabled(start=tmp_path) is False
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[ingest]\nwrite_temporal_spine = false\n"
    )
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
    # Non-decisive env falls through to the next rung; prove it against the
    # explicit=False lower rung rather than the (now default-ON) default, so
    # the assertion still isolates "unrecognised env did not decide."
    assert is_temporal_spine_write_enabled(explicit=False, start=tmp_path) is False


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
    # Malformed TOML value is non-decisive; prove it against the explicit=False
    # lower rung rather than the (now default-ON) default.
    assert is_temporal_spine_write_enabled(explicit=False, start=tmp_path) is False


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


def test_ingest_default_on_chains_turns(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default-ON since the #1064 writer flip: env unset → ingest chains.
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    ingest_turn(store, _TURN_ONE, "test-source", session_id="s1",
                created_at="2026-01-01T00:00:01Z")
    ingest_turn(store, _TURN_TWO, "test-source", session_id="s1",
                created_at="2026-01-01T00:00:02Z")
    assert len(_spine_edges(store)) == 1


def test_ingest_explicit_off_writes_no_spine_edges(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Writer is opt-out now that the default is ON: forcing the env off must
    # suppress spine edges at ingest.
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "0")
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


# ---------------------------------------------------------------------------
# backfill_temporal_spine
# ---------------------------------------------------------------------------


def _seed_two_sessions(store: MemoryStore) -> None:
    _make_belief(store, belief_id="a1", content="session a first",
                 session_id="sa", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="a2", content="session a second",
                 session_id="sa", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="a3", content="session a third",
                 session_id="sa", created_at="2026-01-01T00:00:03Z")
    _make_belief(store, belief_id="b1", content="session b first",
                 session_id="sb", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="b2", content="session b second",
                 session_id="sb", created_at="2026-01-01T00:00:04Z")
    _make_belief(store, belief_id="n1", content="no session fact",
                 session_id=None)


def test_backfill_chains_all_sessions(store: MemoryStore) -> None:
    _seed_two_sessions(store)
    report = backfill_temporal_spine(store)
    assert report.n_sessions == 2
    assert report.n_beliefs_in_sessions == 5
    assert report.n_edges_written == 3
    assert report.n_edges_existing == 0
    assert _spine_edges(store) == [
        ("a2", "a1", TEMPORAL_SPINE_EDGE_WEIGHT),
        ("a3", "a2", TEMPORAL_SPINE_EDGE_WEIGHT),
        ("b2", "b1", TEMPORAL_SPINE_EDGE_WEIGHT),
    ]


def test_backfill_dry_run_writes_nothing(store: MemoryStore) -> None:
    _seed_two_sessions(store)
    report = backfill_temporal_spine(store, dry_run=True)
    assert report.n_edges_written == 3
    assert _spine_edges(store) == []


def test_backfill_idempotent(store: MemoryStore) -> None:
    _seed_two_sessions(store)
    first = backfill_temporal_spine(store)
    second = backfill_temporal_spine(store)
    assert first.n_edges_written == 3
    assert second.n_edges_written == 0
    assert second.n_edges_existing == 3
    assert len(_spine_edges(store)) == 3


def test_backfill_matches_writer_output(store: MemoryStore) -> None:
    """A backfilled store and a writer-chained store produce the same
    spine — the migration path and the ingest path are equivalent."""
    _seed_two_sessions(store)
    incremental = MemoryStore(":memory:")
    _seed_two_sessions(incremental)
    for bid in ("a1", "a2", "a3", "b1", "b2", "n1"):
        write_temporal_spine(incremental, new_belief_ids=[bid])

    backfill_temporal_spine(store)
    assert _spine_edges(store) == _spine_edges(incremental)


def test_backfill_empty_store(store: MemoryStore) -> None:
    report = backfill_temporal_spine(store)
    assert report.n_sessions == 0
    assert report.n_beliefs_in_sessions == 0
    assert report.n_edges_written == 0


# ---------------------------------------------------------------------------
# spine_neighbors traversal
# ---------------------------------------------------------------------------

from aelfrice.temporal_spine import spine_neighbors  # noqa: E402


def _chain(store: MemoryStore, ids: list[str], *, session: str = "s1") -> None:
    for i, bid in enumerate(ids):
        _make_belief(store, belief_id=bid, content=f"unique fact number {bid}",
                     session_id=session,
                     created_at=f"2026-01-01T00:00:{i:02d}Z")
    backfill_temporal_spine(store)


def test_neighbors_bidirectional_depth_one(store: MemoryStore) -> None:
    _chain(store, ["b1", "b2", "b3"])
    hits = spine_neighbors(store, ["b2"])
    assert [b.id for b in hits] == ["b3", "b1"]  # successor first, then pred


def test_neighbors_depth_two(store: MemoryStore) -> None:
    _chain(store, ["b1", "b2", "b3", "b4", "b5"])
    hits = spine_neighbors(store, ["b3"], depth=2)
    assert {b.id for b in hits} == {"b1", "b2", "b4", "b5"}


def test_neighbors_budget_caps_output(store: MemoryStore) -> None:
    _chain(store, ["b1", "b2", "b3", "b4", "b5"])
    hits = spine_neighbors(store, ["b3"], depth=2, node_budget=2)
    assert len(hits) == 2
    assert spine_neighbors(store, ["b3"], node_budget=0) == []


def test_neighbors_skip_but_continue_soft_deleted(store: MemoryStore) -> None:
    _chain(store, ["b1", "b2", "b3"])
    store.soft_delete_belief("b2")
    hits = spine_neighbors(store, ["b1"], depth=2)
    # b2 is traversed through (chain integrity) but never emitted.
    assert [b.id for b in hits] == ["b3"]


def test_neighbors_seeds_never_emitted(store: MemoryStore) -> None:
    _chain(store, ["b1", "b2"])
    hits = spine_neighbors(store, ["b1", "b2"], depth=3)
    assert hits == []


# ---------------------------------------------------------------------------
# Retrieval lane (retrieve_v2 wiring)
# ---------------------------------------------------------------------------

from aelfrice.retrieval import (  # noqa: E402
    ENV_TEMPORAL_SPINE,
    is_temporal_spine_enabled,
    last_lane_telemetry,
    resolve_temporal_spine_budget,
    retrieve_v2,
)

# The query shares terms with the anchor belief only; the chronological
# neighbours are lexically disjoint from it (the #1064 mechanism: ~84%
# of missing gold shares zero salient terms with the question).
_ANCHOR = "the kubernetes deployment rollout failed during the canary stage"
_BEFORE = "morning standup covered vacation plans and a birthday cake"
_AFTER = "someone watered the office plants and refilled the coffee pot"
_QUERY = "kubernetes canary rollout failure"


def _seed_lane_store(store: MemoryStore) -> None:
    _make_belief(store, belief_id="before", content=_BEFORE,
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    _make_belief(store, belief_id="anchor", content=_ANCHOR,
                 session_id="s1", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="after", content=_AFTER,
                 session_id="s1", created_at="2026-01-01T00:00:03Z")
    backfill_temporal_spine(store)


def test_lane_default_on_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    # Default-ON since the #1064 lane flip (#1107 Phase 2). Start at an
    # empty dir so no repo .aelfrice.toml is found.
    monkeypatch.delenv(ENV_TEMPORAL_SPINE, raising=False)
    assert is_temporal_spine_enabled(start=tmp_path) is True
    assert is_temporal_spine_enabled(explicit=False, start=tmp_path) is False
    assert resolve_temporal_spine_budget() == 32
    assert resolve_temporal_spine_budget(explicit=7) == 7


def test_lane_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    # The lane is opt-out now that the default is ON; the env var and the
    # TOML key must still force it back off.
    monkeypatch.setenv(ENV_TEMPORAL_SPINE, "off")
    assert is_temporal_spine_enabled(start=tmp_path) is False
    monkeypatch.delenv(ENV_TEMPORAL_SPINE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_temporal_spine = false\n"
    )
    assert is_temporal_spine_enabled(start=tmp_path) is False


def test_lane_off_omits_neighbours(store: MemoryStore) -> None:
    _seed_lane_store(store)
    result = retrieve_v2(store, _QUERY, use_temporal_spine=False)
    ids = {b.id for b in result.beliefs}
    assert "anchor" in ids
    assert "before" not in ids and "after" not in ids
    tel = last_lane_telemetry()
    assert tel.temporal_spine == 0
    assert tel.temporal_spine_candidates == 0


def test_lane_on_appends_chronological_neighbours(store: MemoryStore) -> None:
    _seed_lane_store(store)
    result = retrieve_v2(store, _QUERY, use_temporal_spine=True)
    ids = [b.id for b in result.beliefs]
    assert "anchor" in ids
    assert "before" in ids and "after" in ids
    # Never displaces L1 pre-packing: neighbours come after the anchor.
    assert ids.index("anchor") < ids.index("before")
    assert ids.index("anchor") < ids.index("after")
    tel = last_lane_telemetry()
    assert tel.temporal_spine == 2
    assert tel.temporal_spine_candidates == 2


def test_lane_env_var_enables(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_lane_store(store)
    monkeypatch.setenv(ENV_TEMPORAL_SPINE, "1")
    result = retrieve_v2(store, _QUERY)
    ids = {b.id for b in result.beliefs}
    assert "before" in ids and "after" in ids


def test_lane_noop_guard_without_spine_edges(store: MemoryStore) -> None:
    # Same store shape but NO spine edges: lane on must be byte-identical
    # to lane off (the no-op guard for spineless stores).
    _make_belief(store, belief_id="anchor", content=_ANCHOR,
                 session_id="s1", created_at="2026-01-01T00:00:02Z")
    _make_belief(store, belief_id="other", content=_BEFORE,
                 session_id="s1", created_at="2026-01-01T00:00:01Z")
    off = retrieve_v2(store, _QUERY, use_temporal_spine=False)
    on = retrieve_v2(store, _QUERY, use_temporal_spine=True)
    assert [b.id for b in on.beliefs] == [b.id for b in off.beliefs]
    tel = last_lane_telemetry()
    assert tel.temporal_spine == 0


def test_lane_node_budget_kwarg(store: MemoryStore) -> None:
    _seed_lane_store(store)
    result = retrieve_v2(
        store, _QUERY, use_temporal_spine=True,
        temporal_spine_node_budget=1,
    )
    ids = {b.id for b in result.beliefs}
    # Budget 1 → exactly one neighbour emitted by the traversal.
    assert len(ids & {"before", "after"}) == 1
    tel = last_lane_telemetry()
    assert tel.temporal_spine_candidates == 1


# ---------------------------------------------------------------------------
# G4 — `aelf spine clear` (reversibility) + sentinel-gated auto-backfill
# ---------------------------------------------------------------------------


def _seed_two_session_beliefs(store: MemoryStore) -> None:
    _make_belief(store, belief_id="a1", content="alpha one",
                 session_id="s1", created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="a2", content="alpha two",
                 session_id="s1", created_at="2026-01-01T00:01:00Z")
    _make_belief(store, belief_id="b1", content="beta one",
                 session_id="s2", created_at="2026-01-01T00:00:30Z")
    _make_belief(store, belief_id="b2", content="beta two",
                 session_id="s2", created_at="2026-01-01T00:02:00Z")


def test_clear_temporal_spine_removes_all_edges(store: MemoryStore) -> None:
    _seed_two_session_beliefs(store)
    backfill_temporal_spine(store)
    assert store.has_edge_type(EDGE_TEMPORAL_NEXT)
    removed = clear_temporal_spine(store)
    assert removed == 2                       # one edge per 2-belief session
    assert not store.has_edge_type(EDGE_TEMPORAL_NEXT)


def test_clear_temporal_spine_empty_is_zero(store: MemoryStore) -> None:
    assert clear_temporal_spine(store) == 0


def test_clear_leaves_beliefs_intact(store: MemoryStore) -> None:
    _seed_two_session_beliefs(store)
    backfill_temporal_spine(store)
    clear_temporal_spine(store)
    assert store.get_belief("a1") is not None
    assert store.get_belief("b2") is not None


def test_clear_then_rebuild_is_byte_identical(store: MemoryStore) -> None:
    """Deterministic backfill: clear + re-backfill reproduces the exact
    same spine (the G5 property the reversibility path relies on)."""
    _seed_two_session_beliefs(store)
    backfill_temporal_spine(store)
    before = _spine_edges(store)
    clear_temporal_spine(store)
    backfill_temporal_spine(store)
    assert _spine_edges(store) == before


def test_delete_edges_by_type_counts_and_scopes(store: MemoryStore) -> None:
    _seed_two_session_beliefs(store)
    backfill_temporal_spine(store)
    # A non-spine edge must survive a TEMPORAL_NEXT-scoped delete.
    from aelfrice.models import Edge
    store.insert_edge(Edge(src="a1", dst="b1", type="RELATES_TO", weight=0.5))
    removed = store.delete_edges_by_type(EDGE_TEMPORAL_NEXT)
    assert removed == 2
    assert store.edges_from("a1")  # RELATES_TO edge still present


# --- maybe_backfill_temporal_spine (sentinel-gated auto migration) ---------


def test_auto_backfill_deferred_when_writer_off(store: MemoryStore, tmp_path) -> None:
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    result = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=False,
    )
    assert result.ran is False
    assert result.n_edges == 0
    # Deferred, NOT sentinel-written — so it re-arms after the flip.
    assert not sentinel.exists()
    assert not store.has_edge_type(EDGE_TEMPORAL_NEXT)


def test_auto_backfill_runs_when_enabled_and_writes_sentinel(
    store: MemoryStore, tmp_path,
) -> None:
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    result = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=True,
    )
    assert result.ran is True
    assert result.n_edges == 2
    assert sentinel.exists()
    assert store.has_edge_type(EDGE_TEMPORAL_NEXT)


def test_auto_backfill_sentinel_makes_it_one_shot(
    store: MemoryStore, tmp_path,
) -> None:
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    first = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=True,
    )
    second = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=True,
    )
    assert first.ran is True
    assert second.ran is False
    assert "sentinel" in second.reason


def test_auto_backfill_rearms_after_flip(store: MemoryStore, tmp_path) -> None:
    """Writer-off leaves the check un-armed; a later writer-on call (the
    flip) fires it exactly once."""
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    deferred = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=False,
    )
    fired = maybe_backfill_temporal_spine(
        store, sentinel_path=sentinel, write_enabled=True,
    )
    assert deferred.ran is False
    assert fired.ran is True
    assert fired.n_edges == 2


def test_auto_backfill_default_gate_reads_writer_flag(
    store: MemoryStore, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With write_enabled unset, the gate falls to the real resolver —
    default-ON since the #1064 writer flip means the backfill runs once
    without an explicit kwarg."""
    monkeypatch.delenv(ENV_TEMPORAL_SPINE_WRITE, raising=False)
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    result = maybe_backfill_temporal_spine(store, sentinel_path=sentinel)
    assert result.ran is True
    assert sentinel.exists()


def test_auto_backfill_deferred_when_writer_opted_out(
    store: MemoryStore, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit opt-out (env=0) defers the auto-backfill and leaves no
    sentinel, so re-enabling later still runs it once."""
    monkeypatch.setenv(ENV_TEMPORAL_SPINE_WRITE, "0")
    _seed_two_session_beliefs(store)
    sentinel = tmp_path / "spine-backfilled"
    result = maybe_backfill_temporal_spine(store, sentinel_path=sentinel)
    assert result.ran is False
    assert not sentinel.exists()


# --- CLI dispatch (`aelf spine {backfill,clear}`) --------------------------


def test_cli_spine_clear_dispatches(tmp_path, monkeypatch, capsys) -> None:
    from aelfrice import cli

    db = tmp_path / "store.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    seed = MemoryStore(str(db))
    _seed_two_session_beliefs(seed)
    backfill_temporal_spine(seed)
    seed.close()

    assert cli.main(["spine", "clear"]) == 0
    out = capsys.readouterr().out
    assert "deleted 2 TEMPORAL_NEXT edge(s)" in out

    check = MemoryStore(str(db))
    assert not check.has_edge_type(EDGE_TEMPORAL_NEXT)
    check.close()


def test_cli_spine_clear_empty_store_is_zero(tmp_path, monkeypatch, capsys) -> None:
    from aelfrice import cli

    db = tmp_path / "store.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    MemoryStore(str(db)).close()
    assert cli.main(["spine", "clear"]) == 0
    assert "deleted 0 TEMPORAL_NEXT edge(s)" in capsys.readouterr().out
