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
from aelfrice.derivation import DerivationOutput, derive
from aelfrice.derivation_worker import run_worker
from aelfrice.models import (
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_DERIVED_FROM,
    INGEST_SOURCE_TRANSCRIPT,
    Edge,
)
from aelfrice.store import MemoryStore

_ABSENT_ID = "ffffffffffffffff"


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
