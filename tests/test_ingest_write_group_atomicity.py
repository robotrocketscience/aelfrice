"""The ingest write group is one transaction and one clock (#1373).

Two properties, both previously unheld, both falsifiable here:

§3 — atomicity. `_process_row` used to issue four independent commits
per ingest row (`insert_or_corroborate`, `insert_edge`,
`link_belief_to_document`, `update_ingest_derived_ids`). A fault between
any two left `ingest_log` and `beliefs` disagreeing. The tests below
inject a fault between the belief insert and the edge insert and assert
that *neither* landed, and that the `was_inserted`-gated audit row is
recoverable on the retry pass.

§7 — one logical event, one timestamp. `record_corroboration` resolved
its own wall clock even though the worker held the log row's `ts`, and
`record_retrieval` let `apply_feedback` and `stamp_retrieved` resolve
two independent clocks for one retrieval. Both defaults truncate to
whole seconds, so a same-second comparison cannot tell a shared
timestamp from two coincident ones. The clock tests therefore install an
advancing fake clock (one hour per read): under the fix exactly one read
happens and the two rows match; under the bug two reads happen and the
rows are an hour apart.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aelfrice import feedback as feedback_mod
from aelfrice import hook_search, store as store_mod
from aelfrice.derivation import DerivationInput, DerivationOutput, derive
from aelfrice.derivation_worker import run_worker
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    EDGE_SUPPORTS,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

_ROW_TS: str = "2020-03-04T05:06:07+00:00"
_CLOCK_BASE: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "atomicity.db"))
    yield s
    s.close()


def _record(
    store: MemoryStore,
    text: str,
    *,
    overrides: dict[str, object] | None = None,
    ts: str = _ROW_TS,
    source_path: str | None = "doc:notes.md",
) -> str:
    raw_meta: dict[str, object] = {
        "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    }
    if overrides is not None:
        raw_meta["route_overrides"] = overrides
    return store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=text,
        raw_meta=raw_meta,
        ts=ts,
    )


def _derive_with_one_edge(inp: DerivationInput) -> DerivationOutput:
    """`derive()` plus a self-edge, so the worker's edge loop runs.

    The shipped rule set emits no edges yet, so the write group's edge
    leg is unreachable without this. The belief is untouched — only the
    edge list is populated — so the row exercises the real insert path.
    """
    out = derive(inp)
    if out.belief is None:
        return out
    return DerivationOutput(
        belief=out.belief,
        edges=[
            Edge(
                src=out.belief.id,
                dst=out.belief.id,
                type=EDGE_SUPPORTS,
                weight=1.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# §3 — atomicity of the ingest write group
# ---------------------------------------------------------------------------


def test_fault_between_belief_and_edge_insert_leaves_neither(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: a fault raised by `insert_edge` rolls the belief back
    with it, and leaves the log row unstamped for a later pass.

    Falsifiable by a surviving belief row: pre-#1373 the belief insert
    had already committed on its own, so the store kept a belief whose
    originating log row claimed nothing was derived."""
    log_id = _record(store, "The write group commits as one unit.")

    monkeypatch.setattr(
        "aelfrice.derivation_worker.derive", _derive_with_one_edge
    )

    def _boom(_edge: Edge) -> None:
        raise RuntimeError("injected fault between belief and edge insert")

    monkeypatch.setattr(store, "insert_edge", _boom)

    with pytest.raises(RuntimeError, match="injected fault"):
        run_worker(store)

    assert store.count_beliefs() == 0
    assert store.count_edges() == 0
    unstamped = store.list_unstamped_ingest_log()
    assert [r["id"] for r in unstamped] == [log_id]


def test_rolled_back_row_re_derives_cleanly_on_the_next_pass(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: the `was_inserted`-gated audit row survives a fault
    later in the same row, because the rollback lets the retry take the
    insert branch again.

    This is the window the issue flagged. It is not merely unchanged by
    the wrapping — it closes. Pre-#1373 the belief committed before the
    fault, so the retry found it by content_hash, took the corroboration
    branch, and the `if was_inserted` guard suppressed the audit row
    permanently. Falsifiable by zero feedback events after the retry."""
    _record(
        store,
        "Route-marked content that faults on its first pass.",
        overrides={
            "belief_type": "factual",
            "origin": ORIGIN_AGENT_INFERRED,
            "alpha": 1.0,
            "beta": 1.0,
            "audit_source": "route_marker_v1",
        },
    )

    monkeypatch.setattr(
        "aelfrice.derivation_worker.derive", _derive_with_one_edge
    )
    calls: list[Edge] = []

    def _boom(edge: Edge) -> None:
        calls.append(edge)
        raise RuntimeError("injected fault after the audit row")

    monkeypatch.setattr(store, "insert_edge", _boom)
    with pytest.raises(RuntimeError):
        run_worker(store)

    assert len(calls) == 1
    assert store.count_feedback_events() == 0

    # Clear the fault; the row is still unstamped, so the next pass
    # re-derives it from the log — the crash-recovery contract.
    monkeypatch.undo()
    result = run_worker(store)

    assert result.beliefs_inserted == 1
    events = store.list_feedback_events()
    assert [e.source for e in events] == ["route_marker_v1"]


def test_successful_row_still_writes_every_leg(store: MemoryStore) -> None:
    """Hypothesis: wrapping the group changes nothing on the happy path
    — belief, corroboration-free insert, doc anchor and log stamp all
    land, and the commit is visible to a fresh handle on the same file.

    Falsifiable by a missing row, or by a group that never committed."""
    log_id = _record(store, "SQLite is the storage engine of record.")

    result = run_worker(store)
    assert result.beliefs_inserted == 1
    assert result.rows_stamped == 1

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    derived = row["derived_belief_ids"]
    assert isinstance(derived, list) and len(derived) == 1

    reopened = MemoryStore(store.db_path)
    try:
        assert reopened.get_belief(derived[0]) is not None
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# §7 — one logical event, one timestamp
# ---------------------------------------------------------------------------


def test_worker_stamps_corroboration_with_the_log_rows_ts(
    store: MemoryStore,
) -> None:
    """Hypothesis: the corroboration row the worker writes carries the
    ingest row's own `ts`, not a second wall-clock read taken when the
    worker happened to run.

    Falsifiable by any timestamp other than `_ROW_TS` — pre-#1373 it was
    `datetime.now()` inside `record_corroboration`, which for a log row
    dated 2020 is off by years."""
    text = "Corroborated content carries the event time."
    _record(store, text)
    run_worker(store)
    bid = _only_belief_id(store)
    assert bid is not None

    # A second log row with the same text corroborates the first belief.
    _record(store, text, ts=_ROW_TS, source_path="doc:other.md")
    run_worker(store)

    rows = store.list_corroborations(bid)
    assert rows, "expected a corroboration row"
    assert [r[0] for r in rows] == [_ROW_TS]


def test_worker_falls_back_to_wall_clock_when_the_row_has_no_ts(
    store: MemoryStore,
) -> None:
    """Hypothesis: an empty `ts` on the log row does not propagate an
    empty string into `belief_corroborations.ingested_at`.

    `inp.ts` is `str(row.get("ts") or "")`, so threading it naively
    would write "" for a row without one. Falsifiable by an empty
    ingested_at."""
    text = "A row with no usable timestamp still records corroboration."
    _record(store, text)
    run_worker(store)
    bid = _only_belief_id(store)
    assert bid is not None

    log_id = _record(store, text, source_path="doc:other.md")
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "UPDATE ingest_log SET ts = '' WHERE id = ?", (log_id,)
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]
    run_worker(store)

    rows = store.list_corroborations(bid)
    assert rows
    assert all(r[0] for r in rows), "ingested_at must never be empty"


def _only_belief_id(store: MemoryStore) -> str | None:
    ids = store.list_belief_ids()
    return ids[0] if len(ids) == 1 else None


class _AdvancingClock(datetime):
    """A `datetime` whose `now()` jumps one hour per read.

    Subclasses the real class so every other constructor
    (`fromisoformat`, arithmetic, comparison) keeps working; only the
    clock read is replaced. One hour is far larger than the shipped
    timestamp format's one-second resolution, so two reads can never
    render as the same string.
    """

    reads: int = 0

    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # noqa: D102
        moment = _CLOCK_BASE + timedelta(hours=_AdvancingClock.reads)
        _AdvancingClock.reads += 1
        return moment


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2019-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _install_advancing_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    _AdvancingClock.reads = 0
    for module in (hook_search, feedback_mod, store_mod):
        monkeypatch.setattr(module, "datetime", _AdvancingClock)


def test_one_retrieval_writes_one_timestamp(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: the feedback_history row and the `last_retrieved_at`
    projection written for one retrieval carry the same timestamp,
    because `record_retrieval` resolves `now` once and threads it to
    both writes.

    Falsifiable by a mismatch. Under the advancing clock the pre-#1373
    code reads twice — once in `apply_feedback`, once in
    `stamp_retrieved` — and the two rows land an hour apart. Asserting
    only that a timestamp exists would pass either way, and so would
    asserting equality under the real clock, whose one-second resolution
    hides the second read."""
    belief = _mk_belief("b-shared-ts", "Retrieval exposure is one event.")
    store.insert_belief(belief)

    _install_advancing_clock(monkeypatch)
    written = hook_search.record_retrieval(store, [belief])
    assert written == 1

    events = store.list_feedback_events()
    assert len(events) == 1
    stored = store.get_belief(belief.id)
    assert stored is not None

    expected = _CLOCK_BASE.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert events[0].created_at == stored.last_retrieved_at
    assert events[0].created_at == expected
    assert _AdvancingClock.reads == 1


def test_record_retrieval_honours_a_caller_supplied_now(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: `now=` overrides the clock entirely, for both writes.

    Falsifiable by either row carrying a clock-derived value, or by the
    clock being read at all."""
    belief = _mk_belief("b-caller-ts", "A caller may stamp the whole batch.")
    store.insert_belief(belief)

    _install_advancing_clock(monkeypatch)
    caller_ts = "1999-12-31T23:59:59Z"
    assert hook_search.record_retrieval(store, [belief], now=caller_ts) == 1

    events = store.list_feedback_events()
    stored = store.get_belief(belief.id)
    assert stored is not None
    assert [e.created_at for e in events] == [caller_ts]
    assert stored.last_retrieved_at == caller_ts
    assert _AdvancingClock.reads == 0
