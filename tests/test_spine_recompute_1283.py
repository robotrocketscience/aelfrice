"""#1283 AC2: the spine is recomputable from the log, and the gap is named.

The recompute keys on `(created_at, ingest_log ULID, id)` against the
shipped writer's `(created_at, rowid)`, where `rowid` is implicit and
`VACUUM` may renumber it. The ULID component is durable as an *observed
property, not a guarantee* — `ulid.make_generator` is monotone only
within a process — so nothing here may be written as if the key were
guaranteed; see `spine_recompute.recompute_spine_edges` for the measured
exposure.

**A test here must not assert that divergence is zero.** The writer has
not been changed yet, so writer and recompute key on different things
and a zero would mean the recompute had been fitted to the defect. What
these pin instead is that each *bucket* of the gap is what it claims to
be, because only one of the three is a defect anyone can fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.models import EDGE_TEMPORAL_NEXT, Belief, Edge
from aelfrice.spine_recompute import (
    SYNTH_SOURCE_KIND,
    recompute_spine_edges,
    spine_divergence,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def _belief(
    store: MemoryStore, bid: str, *, session: str, created_at: str
) -> None:
    store.insert_belief(Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"hash-{bid}",
        alpha=1.0,
        beta=1.0,
        type="fact",
        lock_level="none",
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        session_id=session,
    ))


def _log(
    store: MemoryStore,
    ulid: str,
    belief_ids: list[str],
    *,
    source_kind: str = "transcript",
) -> None:
    """Plant an ingest_log row directly.

    Written through the connection rather than `record_ingest` because
    the ULID is the thing under test: it has to be chosen, not minted.
    """
    store._conn.execute(
        "INSERT INTO ingest_log (id, source_kind, raw_text, ts, "
        "derived_belief_ids) VALUES (?, ?, '', '2026-01-01T00:00:00Z', ?)",
        (ulid, source_kind, json.dumps(belief_ids)),
    )
    store._conn.commit()


# --- the key ------------------------------------------------------------

def test_ulid_orders_beliefs_that_share_a_created_at(
    store: MemoryStore,
) -> None:
    """The whole point: a `created_at` tie is broken by the log, not rowid.

    All three beliefs carry the same timestamp and are inserted in an
    order that disagrees with their ULIDs, so a recompute that fell back
    to insertion order would chain them b-a-c instead of a-b-c.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("b", "a", "c"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["c"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == set()
    assert edges == {("b", "a"), ("c", "b")}


def test_ulid_beats_belief_id_when_the_two_orders_disagree(
    store: MemoryStore,
) -> None:
    """The ULID is the key — not the belief id that usually agrees with it.

    The sibling test above picks ids whose alphabetical order happens to
    match their ULID order, so `(created_at, id)` and
    `(created_at, ULID, id)` produce the same chain and it cannot tell
    them apart. Replacing the log key with a constant leaves it green.

    Here the two orders are deliberately opposed: `aaa` carries the
    LAST ULID and `ccc` the first, so the log says ccc-bbb-aaa while the
    id says aaa-bbb-ccc. Only a recompute that actually consults the log
    produces the expected set — which matters because 96.5% of this
    store shares a `created_at`, so the id is doing the ordering
    whenever the ULID is ignored.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("aaa", "bbb", "ccc"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["aaa"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["bbb"])
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["ccc"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == set()
    # ULID order ccc < bbb < aaa. An `(created_at, id)` sort would give
    # aaa < bbb < ccc and therefore {("bbb", "aaa"), ("ccc", "bbb")}.
    assert edges == {("bbb", "ccc"), ("aaa", "bbb")}


def test_a_belief_takes_its_earliest_log_row(store: MemoryStore) -> None:
    """Later rows are corroborations; only the first records insertion."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "first", session="s1", created_at=same)
    _belief(store, "second", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["first"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["second"])
    # `first` is corroborated later; that must not reorder it after
    # `second`.
    _log(store, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", ["first"])

    edges, _ = recompute_spine_edges(store)
    assert edges == {("second", "first")}


def test_created_at_dominates_the_ulid(store: MemoryStore) -> None:
    """The key is `(created_at, ULID)`, not the ULID alone.

    Without this, a backfill — whose ULIDs are minted long after the
    content — would reorder a whole session by processing time.
    """
    _belief(store, "early", session="s1", created_at="2026-01-01T00:00:00Z")
    _belief(store, "late", session="s1", created_at="2026-06-01T00:00:00Z")
    # ULIDs deliberately inverted against the timestamps.
    _log(store, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", ["early"])
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["late"])

    edges, _ = recompute_spine_edges(store)
    assert edges == {("late", "early")}


def test_sessions_do_not_chain_into_each_other(store: MemoryStore) -> None:
    same = "2026-03-01T00:00:00Z"
    _belief(store, "a1", session="s1", created_at=same)
    _belief(store, "b1", session="s2", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a1"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b1"])

    edges, _ = recompute_spine_edges(store)
    assert edges == set(), "beliefs from different sessions were chained"


# --- the synth exclusion ------------------------------------------------

def test_synth_rows_supply_no_ordering_key(store: MemoryStore) -> None:
    """A belief covered only by a synth row counts as having no log row.

    The #263 synthesis relabelled `beliefs.rowid` as a ULID, so honouring
    those keys would launder rowid order into the key the contract calls
    durable. Excluded by `source_kind`, a stated column — not by a
    prefix or density heuristic, both of which were measured and failed.
    """
    same = "2026-03-01T00:00:00Z"
    _belief(store, "real", session="s1", created_at=same)
    _belief(store, "synthetic", session="s1", created_at=same)
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["real"])
    _log(
        store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["synthetic"],
        source_kind=SYNTH_SOURCE_KIND,
    )

    edges, no_log = recompute_spine_edges(store)
    assert no_log == {"synthetic"}, (
        "a synth-covered belief was given an ordering key; its ULID is "
        "migration wall clock and its order is rowid relabelled"
    )
    # `synthetic` sorts last by the no-log convention despite its ULID
    # sorting first, which is the observable consequence.
    assert edges == {("synthetic", "real")}


# --- the no-log convention ----------------------------------------------

def test_no_log_beliefs_sort_last_within_their_group(
    store: MemoryStore,
) -> None:
    """A stated forward convention, not a recovery of historical order.

    The module docstring is explicit that the historical ordering of
    this bucket is unreconstructible — 93.9% of it sits inside a
    `created_at` tie where the only other durable column is a
    content-addressed id. What this pins is that the convention is
    applied consistently, so the recompute is deterministic.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("logged", "orphan_b", "orphan_a"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["logged"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == {"orphan_a", "orphan_b"}
    # logged -> orphan_a -> orphan_b: no-log last, then id ASC.
    assert edges == {("orphan_a", "logged"), ("orphan_b", "orphan_a")}


def test_recompute_is_deterministic(store: MemoryStore) -> None:
    """Same store state, same edge set — including the no-log bucket.

    Determinism is the property the no-log convention exists to buy, so
    it is asserted over a store that has one.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("d", "a", "c", "b"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])

    first, first_no_log = recompute_spine_edges(store)
    second, second_no_log = recompute_spine_edges(store)
    assert first == second
    assert first_no_log == second_no_log


# --- divergence attribution ---------------------------------------------

def test_divergence_is_zero_when_the_writer_agrees(
    store: MemoryStore,
) -> None:
    """The control. Without it every bucket assertion below is vacuous —
    a recompute that produced nothing would file every shipped edge under
    some bucket and look like a correct attribution."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "a", session="s1", created_at=same)
    _belief(store, "b", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    store.insert_edge(Edge(src="b", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))

    report = spine_divergence(store)
    assert report.n_shipped == 1
    assert report.n_reproduced == 1
    assert report.reproduced_share == 1.0
    assert report.missing_other == 0


def test_a_no_log_miss_lands_in_its_own_bucket(store: MemoryStore) -> None:
    """Unreconstructible misses must never be counted as drift."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "orphan", session="s1", created_at=same)
    _belief(store, "logged", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["logged"])
    # The writer chained orphan first; the recompute puts it last.
    store.insert_edge(
        Edge(src="logged", dst="orphan", type=EDGE_TEMPORAL_NEXT, weight=1.0)
    )

    report = spine_divergence(store)
    assert report.n_reproduced == 0
    assert report.missing_touching_no_log == 1
    assert report.missing_fan_in == 0
    assert report.missing_other == 0


def test_a_fan_in_miss_lands_in_its_own_bucket(store: MemoryStore) -> None:
    """Two predecessors for one successor is a writer defect, not a key
    disagreement — a chain gives each successor exactly one."""
    for i, bid in enumerate(("a", "b", "c")):
        _belief(store, bid, session="s1", created_at=f"2026-03-0{i+1}T00:00:00Z")
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["c"])
    store.insert_edge(Edge(src="b", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))
    store.insert_edge(Edge(src="c", dst="b", type=EDGE_TEMPORAL_NEXT, weight=1.0))
    # The defect: `c` gains a second predecessor.
    store.insert_edge(Edge(src="c", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))

    report = spine_divergence(store)
    assert report.n_shipped == 3
    assert report.n_reproduced == 2
    assert report.missing_fan_in == 1
    assert report.missing_touching_no_log == 0
    assert report.missing_other == 0


def test_an_empty_store_reports_full_reproduction(
    store: MemoryStore,
) -> None:
    """A zero-edge store is not a 0% reproduction — it is nothing to
    reproduce. Getting this wrong makes a fresh store look broken."""
    report = spine_divergence(store)
    assert report.n_shipped == 0
    assert report.reproduced_share == 1.0
