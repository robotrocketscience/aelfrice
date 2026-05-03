"""Tests for the v2.x derivation worker (#264).

Each test is a falsifiable hypothesis about the worker's contract:
canonical state matches inline-derive output, idempotency, crash
recovery, and skip-handling for non-persistent classifications.

Worker integration with each entry point lives in subsequent commits
(scanner, classification, triple_extractor, mcp, cli). This file
exercises `run_worker(store)` directly against synthesized log rows.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation_worker import run_worker
from aelfrice.models import (
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_FILESYSTEM,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "worker.db"))
    yield s
    s.close()


def _record_unstamped(
    store: MemoryStore,
    text: str,
    *,
    source_path: str = "transcripts/test.jsonl",
    call_site: str = CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
) -> str:
    return store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=text,
        raw_meta={"call_site": call_site},
        # `derived_belief_ids` deliberately omitted — this is the v2.x
        # log-first contract the worker depends on.
    )


def test_worker_materializes_beliefs_from_unstamped_log_rows(
    store: MemoryStore,
) -> None:
    """Hypothesis: after run_worker(), every unstamped log row that
    derive() classifies as persist=True yields a canonical belief row,
    and the log row's derived_belief_ids is non-empty. Falsifiable by
    a missing belief or an empty stamp."""
    log_id = _record_unstamped(store, "The system uses SQLite for storage.")

    result = run_worker(store)

    assert result.rows_scanned == 1
    assert result.beliefs_inserted == 1
    assert result.rows_stamped == 1

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    derived = row["derived_belief_ids"]
    assert isinstance(derived, list) and len(derived) == 1
    bid = derived[0]
    assert isinstance(bid, str)
    assert store.get_belief(bid) is not None


def test_worker_is_idempotent_on_second_run(store: MemoryStore) -> None:
    """Hypothesis: invoking run_worker twice over the same log produces
    identical canonical state — same belief count, no duplicates, no new
    stamps on the second pass. Falsifiable by either run_worker()
    returning beliefs_inserted > 0 on the second pass, or by a new
    belief row appearing in `beliefs`."""
    _record_unstamped(store, "Locks always inject in full at SessionStart.")
    first = run_worker(store)
    assert first.beliefs_inserted == 1
    assert first.rows_stamped == 1

    belief_count_after_first = store.count_beliefs()

    second = run_worker(store)
    assert second.rows_scanned == 0
    assert second.beliefs_inserted == 0
    assert second.rows_stamped == 0
    assert store.count_beliefs() == belief_count_after_first


def test_worker_recovers_orphan_belief_after_simulated_crash(
    store: MemoryStore,
) -> None:
    """Hypothesis: if the worker dies between `insert_belief` and
    `update_ingest_derived_ids`, the next worker run notices the orphan
    (belief exists, log row unstamped) and stamps it without inserting
    a duplicate belief. Falsifiable by either a duplicate belief row or
    a permanently-unstamped log row."""
    log_id = _record_unstamped(store, "Crash recovery proves the contract.")

    # Materialize the belief inline (skip the log-stamp step) — this
    # reproduces the half-completed worker pass.
    from aelfrice.derivation import DerivationInput, derive
    out = derive(DerivationInput(
        raw_text="Crash recovery proves the contract.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="transcripts/test.jsonl",
    ))
    assert out.belief is not None
    store.insert_or_corroborate(
        out.belief,
        source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    )
    pre_count = store.count_beliefs()
    assert pre_count == 1

    result = run_worker(store)

    assert result.rows_scanned == 1
    assert result.rows_stamped == 1
    # No duplicate belief: insert_or_corroborate hit the existing
    # content_hash and recorded a corroboration instead.
    assert result.beliefs_inserted == 0
    assert result.beliefs_corroborated == 1
    assert store.count_beliefs() == pre_count

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    derived = row["derived_belief_ids"]
    assert isinstance(derived, list) and len(derived) == 1
    assert derived[0] == out.belief.id


def test_worker_stamps_empty_list_when_classifier_skips(
    store: MemoryStore,
) -> None:
    """Hypothesis: when derive() returns no belief (e.g. an empty or
    question-form sentence), the worker still stamps the log row with
    an explicit empty list so subsequent runs don't re-process it.
    Falsifiable by the row remaining at NULL after run_worker()."""
    log_id = _record_unstamped(store, "")  # extraction yields nothing

    result = run_worker(store)

    # The row may or may not have been visited depending on classifier
    # behaviour for empty text; what matters is that the next pass
    # finds no unstamped rows.
    second = run_worker(store)
    assert second.rows_scanned == 0

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    # An explicit JSON [] is a valid stamp; result should be a list.
    assert isinstance(row["derived_belief_ids"], list)
    # Either zero or one belief got persisted depending on classifier;
    # both states are stamped.
    if not row["derived_belief_ids"]:
        assert result.rows_skipped_no_belief >= 1


def test_worker_processes_multiple_rows_in_one_pass(
    store: MemoryStore,
) -> None:
    """Hypothesis: a batch of N unstamped rows all become stamped after
    a single run_worker() call. Falsifiable by any unstamped row
    remaining."""
    n = 5
    log_ids: list[str] = []
    for i in range(n):
        log_ids.append(
            _record_unstamped(store, f"Batch sentence number {i}.")
        )

    result = run_worker(store)
    assert result.rows_scanned == n
    assert result.rows_stamped == n

    for lid in log_ids:
        row = store.get_ingest_log_entry(lid)
        assert row is not None
        assert isinstance(row["derived_belief_ids"], list)
