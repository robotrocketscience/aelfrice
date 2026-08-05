"""Tests for #1354 — populating `ingest_log.derived_edge_ids`.

Each test is a falsifiable hypothesis. The file is organised in the
order the feature is built:

1. the worker's edge-insert guards (a latent crash independent of the
   feature: `store.insert_edge` is a bare INSERT into a table keyed
   `PRIMARY KEY (src, dst, type)` and `run_worker` has no per-row
   `try/except`, so the first `derive()` that emits an edge turns
   re-ingest into an IntegrityError);
2. the `[]`-vs-NULL watermark that makes the column forward-only;
3. `derive()` emitting DERIVED_FROM from the logged `raw_meta` block;
4. the intra-turn ingest writer that populates that block.

The guiding rule for every assertion here is that it must DISTINGUISH
values, not observe presence — a test that passes against a hardcoded
constant is not coverage.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import derivation_worker
from aelfrice.derivation import (
    META_DERIVED_FROM,
    _TRANSCRIPT_SOURCE_LABEL,
    DerivationInput,
    DerivationOutput,
    derive,
)
from aelfrice.derivation_worker import run_worker
from aelfrice.ingest import _ingest_turn_ids
from aelfrice.models import (
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_DERIVED_FROM,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_TRANSCRIPT,
    Edge,
)
from aelfrice.store import MemoryStore

_ABSENT_ID = "ffffffffffffffff"

# Sentinel distinguishing "no block" from "block explicitly set to None".
_UNSET: object = object()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "derived_edge_ids.db"))
    yield s
    s.close()


def _record(
    store: MemoryStore,
    text: str,
    *,
    raw_meta: dict[str, object] | None = None,
) -> str:
    """Append one unstamped transcript log row. Returns the log id."""
    return store.record_ingest(
        source_kind=INGEST_SOURCE_TRANSCRIPT,
        raw_text=text,
        raw_meta=raw_meta or {"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
    )


def _belief_id_of(store: MemoryStore, log_id: str) -> str:
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    ids = entry["derived_belief_ids"]
    assert isinstance(ids, list) and ids
    return str(ids[0])


def _log_ids_in_order(store: MemoryStore) -> list[str]:
    """Every ingest_log id, in insertion order (ULIDs sort lexically)."""
    cur = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT id FROM ingest_log ORDER BY id"
    )
    return [str(r[0]) for r in cur.fetchall()]


def _edge_ids_of(store: MemoryStore, log_id: str) -> object:
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    return entry["derived_edge_ids"]


def _materialise_two_beliefs(store: MemoryStore) -> tuple[str, str]:
    """Land two real beliefs via the worker; return (first, second) ids."""
    a = _record(store, "The cache is invalidated on every write.")
    b = _record(store, "The index rebuild takes about two seconds.")
    run_worker(store)
    return _belief_id_of(store, a), _belief_id_of(store, b)


def _emit(monkeypatch: pytest.MonkeyPatch, edges: list[Edge]) -> None:
    """Force `derive()` to return its real belief plus `edges`.

    Used by the guard tests, which must run before anything in
    `derive()` emits edges of its own.
    """
    real = derive

    def _patched(inp: object) -> DerivationOutput:
        out = real(inp)  # type: ignore[arg-type]
        if out.belief is None:
            return out
        return DerivationOutput(belief=out.belief, edges=list(edges))

    monkeypatch.setattr(derivation_worker, "derive", _patched)


# --- 1. worker edge-insert guards ---------------------------------------


def test_reingesting_a_derived_edge_does_not_raise(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: when `derive()` emits an edge the worker has already
    inserted, the second pass skips the INSERT instead of raising.

    Falsifiable by reverting the loop to a bare `store.insert_edge(edge)`
    — the duplicate then violates `PRIMARY KEY (src, dst, type)` and
    raises IntegrityError, aborting the turn.
    """
    first, second = _materialise_two_beliefs(store)
    edge = Edge(src=second, dst=first, type=EDGE_DERIVED_FROM, weight=1.0)
    _emit(monkeypatch, [edge])

    one = _record(store, "The scheduler retries on a transient failure.")
    run_worker(store)
    assert store.get_edge(second, first, EDGE_DERIVED_FROM) is not None

    two = _record(store, "The retry budget is capped at five attempts.")
    run_worker(store)  # must not raise

    # The edge is logged on BOTH rows — the column records the
    # derivation, not the insert — but exists exactly once in `edges`.
    expected = [[second, first, EDGE_DERIVED_FROM]]
    assert _edge_ids_of(store, one) == expected
    assert _edge_ids_of(store, two) == expected
    assert len(store.edges_from(second)) == 1


def test_edge_with_an_absent_endpoint_is_logged_but_not_inserted(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: an emitted edge naming a belief that does not exist is
    recorded in `derived_edge_ids` but never written to `edges`.

    Pins the log-vs-insert split in both directions. Falsifiable by
    dropping the endpoint guard (the edge would be inserted, dangling),
    or by stamping only what was inserted (the column would read `[]`).
    """
    first, _second = _materialise_two_beliefs(store)
    edge = Edge(src=first, dst=_ABSENT_ID, type=EDGE_DERIVED_FROM, weight=1.0)
    _emit(monkeypatch, [edge])

    log_id = _record(store, "The compaction pass runs once per epoch.")
    run_worker(store)

    assert _edge_ids_of(store, log_id) == [[first, _ABSENT_ID, EDGE_DERIVED_FROM]]
    assert store.get_edge(first, _ABSENT_ID, EDGE_DERIVED_FROM) is None


# --- 2. the []-vs-NULL watermark ----------------------------------------


def test_a_zero_edge_row_stamps_an_empty_list_not_null(
    store: MemoryStore,
) -> None:
    """Hypothesis: a row the worker processed but which derived no edges
    is stamped `[]`, not left NULL.

    This is the watermark. NULL must mean "no edge-aware writer ever saw
    this row" so the replay probe can exempt the historical cohort;
    if a zero-edge row were also NULL the two are indistinguishable.
    Falsifiable by restoring `derived_edge_ids if derived_edge_ids else
    None` — `is not None` then fails.
    """
    log_id = _record(store, "The planner emits one task per shard.")
    run_worker(store)

    assert _edge_ids_of(store, log_id) is not None
    assert _edge_ids_of(store, log_id) == []


def test_a_persist_false_row_also_stamps_an_empty_edge_list(
    store: MemoryStore,
) -> None:
    """Hypothesis: the no-belief early return stamps `derived_edge_ids`
    too, so it does not leave a NULL the probe would read as historical.

    Falsifiable by stamping only `derived_belief_ids` on that path.
    """
    log_id = _record(store, "What happens when the queue drains?")
    run_worker(store)

    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    assert entry["derived_belief_ids"] == []  # persist=False
    assert entry["derived_edge_ids"] == []


def test_a_prestamped_row_is_never_revisited(store: MemoryStore) -> None:
    """Hypothesis: the worker only ever visits rows whose
    `derived_belief_ids` is NULL, so a historical row keeps its NULL
    `derived_edge_ids` forever. This is what makes the change
    forward-only without a date, a ULID boundary or a migration.

    Falsifiable by any backfill sweep, or by widening
    `list_unstamped_ingest_log`'s predicate.
    """
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_TRANSCRIPT,
        raw_text="A row a previous release already stamped.",
        raw_meta={"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
        derived_belief_ids=["0123456789abcdef"],
    )
    run_worker(store)

    assert _edge_ids_of(store, log_id) is None


# --- 3. derive() emits DERIVED_FROM from the logged block ----------------

_PRIOR = "The compaction pass runs once per epoch."
_LATER = "The scheduler retries on a transient failure."


def _transcript_input(
    text: str, *, block: object = _UNSET, role: str = "user"
) -> DerivationInput:
    meta: dict[str, object] = {
        "call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
        "role": role,
    }
    if block is not _UNSET:
        meta[META_DERIVED_FROM] = block
    return DerivationInput(
        raw_text=text,
        source_kind=INGEST_SOURCE_TRANSCRIPT,
        source_path=_TRANSCRIPT_SOURCE_LABEL,
        raw_meta=meta,
        ts="2026-08-05T00:00:00+00:00",
    )


def test_derive_emits_one_derived_from_edge_from_the_block() -> None:
    """Hypothesis: a row carrying a well-formed `derived_from` block
    derives exactly one DERIVED_FROM edge, with every field pinned.

    Falsifiable by reverting the return path to `edges=[]`, by dropping
    the anchor, or by changing the weight.
    """
    out = derive(
        _transcript_input(
            _LATER, block={"prior_text": _PRIOR, "anchor_text": "Notes:"}
        )
    )
    assert out.belief is not None
    assert len(out.edges) == 1
    edge = out.edges[0]
    assert edge.type == EDGE_DERIVED_FROM
    assert edge.weight == 1.0
    assert edge.anchor_text == "Notes:"


def test_derive_emits_no_edge_without_the_block() -> None:
    """Hypothesis: the identical row minus the block derives no edge.

    Pairs with the test above — that one yields 1, this one yields 0 —
    so a helper that emits unconditionally, or synthesises a default
    predecessor, fails here while passing there.
    """
    out = derive(_transcript_input(_LATER))
    assert out.belief is not None
    assert out.edges == []


def test_edge_direction_is_later_to_earlier() -> None:
    """Hypothesis: `src` is the belief being derived and `dst` is the
    belief named by `prior_text`, matching `ingest.py`'s convention.

    Both endpoints are obtained by calling `derive()` rather than by
    hand-computing an id, so a swapped `src`/`dst` cannot survive by
    being wrong in the same direction as the expectation.
    """
    later = derive(_transcript_input(_LATER, block={"prior_text": _PRIOR}))
    earlier = derive(_transcript_input(_PRIOR))
    assert later.belief is not None
    assert earlier.belief is not None

    edge = later.edges[0]
    assert edge.src == later.belief.id
    assert edge.dst == earlier.belief.id
    assert edge.src != edge.dst


def test_a_self_referential_block_emits_no_edge() -> None:
    """Hypothesis: `prior_text` equal to the row's own text derives no
    edge. An edge from a belief to itself is not a relationship, and
    `ingest.py` skips that case explicitly.

    Falsifiable by dropping the `prior == raw_text` arm of the guard.
    """
    out = derive(_transcript_input(_LATER, block={"prior_text": _LATER}))
    assert out.belief is not None
    assert out.edges == []


@pytest.mark.parametrize(
    "block",
    [
        pytest.param({"prior_text": ""}, id="empty-prior"),
        pytest.param({"anchor_text": "Notes:"}, id="no-prior-key"),
        pytest.param({"prior_text": 17}, id="non-string-prior"),
        pytest.param("not-a-dict", id="block-not-a-dict"),
        pytest.param(None, id="block-null"),
    ],
)
def test_a_malformed_block_emits_no_edge(block: object) -> None:
    """Hypothesis: every malformed shape of the block derives no edge and
    raises nothing. `raw_meta` is attacker-adjacent in the sense that it
    round-trips through JSON on the log row, so shape cannot be assumed.
    """
    out = derive(_transcript_input(_LATER, block=block))
    assert out.belief is not None
    assert out.edges == []


@pytest.mark.parametrize(
    "source_kind",
    [
        pytest.param(INGEST_SOURCE_CLI_REMEMBER, id="cli-remember"),
        pytest.param(INGEST_SOURCE_MCP_REMEMBER, id="mcp-remember"),
        pytest.param(INGEST_SOURCE_GIT, id="git-triple"),
    ],
)
def test_the_other_id_schemes_ignore_the_block(source_kind: str) -> None:
    """Hypothesis: the lock/remember and triple-extraction paths never
    attach the edge, even when the block is present and well-formed.

    Those paths mint ids via `_lock_id` / `_triple_belief_id`, so a
    `_belief_id`-scheme `dst` would name a belief that does not exist.
    The `belief is not None` assertion keeps this from passing vacuously
    on a path that simply produced nothing.
    """
    out = derive(
        DerivationInput(
            raw_text=_LATER,
            source_kind=source_kind,
            raw_meta={META_DERIVED_FROM: {"prior_text": _PRIOR}},
            ts="2026-08-05T00:00:00+00:00",
        )
    )
    assert out.belief is not None
    assert out.edges == []


def test_a_persist_false_row_emits_no_edge() -> None:
    """Hypothesis: a row the classifier declines to persist derives no
    edge, even carrying a valid block — there is no `src` belief for it
    to hang from.

    Falsifiable by computing the edges after the `persist` gate in a way
    that leaks them onto the no-belief return.
    """
    out = derive(
        _transcript_input(
            "What happens when the queue drains?",
            block={"prior_text": _PRIOR},
        )
    )
    assert out.belief is None
    assert out.edges == []


# --- 4. the intra-turn ingest writer ------------------------------------

# Fixture shape borrowed from the #809 suite: two full-length sentences
# with a header stub between them, which `_ingest_turn_ids` demotes to
# the intra-turn edge's anchor_text. Paragraph breaks matter — the
# sentence splitter keeps `Header: text` as one sentence otherwise.
_FULL_A = "The configuration file lives at /etc/aelfrice/conf."
_SUB_HEADER = "Acceptance criteria:"
_FULL_B = "Astronomers process supernova imagery nightly using clusters."

_FULL_C = "Radio telescopes calibrate against known pulsar timings."

_SUBFLOOR_TURN = f"{_FULL_A}\n\n{_SUB_HEADER}\n\n{_FULL_B}"
_PLAIN_TURN = f"{_FULL_A}\n\n{_FULL_B}"
# Anchored pair followed by an unanchored third sentence. The third row
# must NOT inherit the second row's block.
_TRAILING_TURN = f"{_FULL_A}\n\n{_SUB_HEADER}\n\n{_FULL_B}\n\n{_FULL_C}"


def test_ingest_stamps_the_edge_on_the_second_row_only(
    store: MemoryStore,
) -> None:
    """Hypothesis: ingesting a turn whose sentences are separated by a
    sub-floor clause stamps `derived_edge_ids` as `[]` on the first row
    and as the one derived edge on the second.

    The two rows must DIFFER — a writer that stamps the same value
    everywhere, or that never consumes `out.edges`, fails here. Ids are
    read back from `derived_belief_ids` rather than hardcoded.
    """
    _ingest_turn_ids(
        store=store,
        text=_SUBFLOOR_TURN,
        source=_TRANSCRIPT_SOURCE_LABEL,
        session_id="s-1354",
        created_at="2026-08-05T00:00:00+00:00",
        role="user",
    )
    rows = [
        store.get_ingest_log_entry(r)
        for r in _log_ids_in_order(store)
    ]
    assert len(rows) == 2
    first, second = rows[0], rows[1]
    assert first is not None and second is not None

    first_bid = first["derived_belief_ids"][0]  # type: ignore[index]
    second_bid = second["derived_belief_ids"][0]  # type: ignore[index]

    assert first["derived_edge_ids"] == []
    assert second["derived_edge_ids"] == [
        [second_bid, first_bid, EDGE_DERIVED_FROM]
    ]


def test_the_live_edge_set_is_unchanged_by_the_stamp(
    store: MemoryStore,
) -> None:
    """Hypothesis: the same turn still produces exactly one DERIVED_FROM
    edge, with both endpoints resolving and the anchor preserved.

    This is the regression guard on the whole change: the column is new,
    the edge table must be byte-identical to pre-#1354 behaviour.
    Falsifiable by either idempotency guard being removed (two edges, or
    an IntegrityError) or by a dangling endpoint.
    """
    _ingest_turn_ids(
        store=store,
        text=_SUBFLOOR_TURN,
        source=_TRANSCRIPT_SOURCE_LABEL,
        session_id="s-1354",
        created_at="2026-08-05T00:00:00+00:00",
        role="user",
    )
    edges = [e for e in store.iter_all_edges() if e.type == EDGE_DERIVED_FROM]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.anchor_text == _SUB_HEADER
    assert edge.weight == 1.0
    assert store.get_belief(edge.src) is not None
    assert store.get_belief(edge.dst) is not None


def test_reingesting_the_same_turn_is_still_idempotent(
    store: MemoryStore,
) -> None:
    """Hypothesis: ingesting the same turn twice raises nothing and
    leaves exactly one edge — the documented re-ingest contract, now
    exercised on a path where `derive()` actually emits an edge.

    Falsifiable by reverting the worker's INSERT guard: the second pass
    violates `PRIMARY KEY (src, dst, type)` inside the turn transaction.
    """
    for _ in range(2):
        _ingest_turn_ids(
            store=store,
            text=_SUBFLOOR_TURN,
            source=_TRANSCRIPT_SOURCE_LABEL,
            session_id="s-1354",
            created_at="2026-08-05T00:00:00+00:00",
            role="user",
        )
    edges = [e for e in store.iter_all_edges() if e.type == EDGE_DERIVED_FROM]
    assert len(edges) == 1


def test_an_unanchored_later_sentence_does_not_inherit_the_block(
    store: MemoryStore,
) -> None:
    """Hypothesis: in a turn (A, header, B, C), only row B carries the
    block — row C, which has no sub-floor clause before it, stamps `[]`.

    This is the per-sentence-copy guard. `raw_meta` is one dict shared
    across the turn; setting the key on it in place would leave it
    visible to every subsequent `record_ingest` in the loop, so C would
    silently derive a spurious edge to A — skipping over its real
    predecessor B. Falsifiable by `row_meta = raw_meta` in place of the
    copy, which no two-sentence fixture can detect.
    """
    _ingest_turn_ids(
        store=store,
        text=_TRAILING_TURN,
        source=_TRANSCRIPT_SOURCE_LABEL,
        session_id="s-1354",
        created_at="2026-08-05T00:00:00+00:00",
        role="user",
    )
    rows = [store.get_ingest_log_entry(r) for r in _log_ids_in_order(store)]
    assert len(rows) == 3
    assert all(r is not None for r in rows)

    a, b, c = rows[0], rows[1], rows[2]
    assert a is not None and b is not None and c is not None
    assert a["derived_edge_ids"] == []
    assert b["derived_edge_ids"] == [
        [
            b["derived_belief_ids"][0],  # type: ignore[index]
            a["derived_belief_ids"][0],  # type: ignore[index]
            EDGE_DERIVED_FROM,
        ]
    ]
    assert c["derived_edge_ids"] == []

    # And the live edge set agrees: exactly one edge, B -> A.
    edges = [e for e in store.iter_all_edges() if e.type == EDGE_DERIVED_FROM]
    assert len(edges) == 1


def test_a_turn_without_subfloor_clauses_stamps_empty_lists(
    store: MemoryStore,
) -> None:
    """Hypothesis: an ordinary two-sentence turn — no sub-floor clause
    between them — carries no block and so stamps `[]` on both rows.

    Pairs with the stamping test above: that turn yields one edge, this
    one yields none, so a writer that attaches the block unconditionally
    fails here.
    """
    _ingest_turn_ids(
        store=store,
        text=_PLAIN_TURN,
        source=_TRANSCRIPT_SOURCE_LABEL,
        session_id="s-1354",
        created_at="2026-08-05T00:00:00+00:00",
        role="user",
    )
    for log_id in _log_ids_in_order(store):
        entry = store.get_ingest_log_entry(log_id)
        assert entry is not None
        assert entry["derived_edge_ids"] == []
    assert [
        e for e in store.iter_all_edges() if e.type == EDGE_DERIVED_FROM
    ] == []
