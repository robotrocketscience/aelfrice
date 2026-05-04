"""Integration tests for the post-#264 ingest_turn → derivation_worker
call shape.

Each test states a falsifiable hypothesis about the new contract:

    ingest_turn writes one log row per sentence (unstamped at the
    moment of write), then run_worker materializes beliefs and stamps
    every log row's `derived_belief_ids`.

These tests anchor the *new* invariants. They will catch any future
regression that re-introduces direct `derive()` / `insert_belief()`
calls in the entry point and bypasses the worker.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_turn
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "ingest-via-worker.db"))
    yield s
    s.close()


def test_every_log_row_is_stamped_after_ingest_turn(
    store: MemoryStore,
) -> None:
    """Hypothesis: after ingest_turn returns, no row in `ingest_log`
    remains unstamped — the worker ran end-of-turn and stamped every
    row written by the entry point. Falsifiable by any row whose
    `derived_belief_ids IS NULL` after the call (the worker did not
    run, or the entry point bypassed the worker for some sentences).
    """
    text = (
        "The configuration file lives at /etc/aelfrice/conf. "
        "The default port is 8080 for the dashboard."
    )
    ingest_turn(store, text, source="user")
    unstamped = store.list_unstamped_ingest_log()
    assert unstamped == [], f"unstamped rows: {unstamped}"


def test_log_row_carries_call_site_metadata(store: MemoryStore) -> None:
    """Hypothesis: the entry point stamps `raw_meta.call_site` =
    `transcript_ingest` so the worker can resolve the corroboration
    source unambiguously. Falsifiable if `raw_meta` is missing or
    carries a different call_site."""
    ingest_turn(store, "The sky is blue.", source="user")
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log"
    ).fetchall()
    assert rows
    import json
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "transcript_ingest"


def test_question_sentence_stamps_log_row_with_empty_derived_ids(
    store: MemoryStore,
) -> None:
    """Hypothesis: a sentence the classifier rejects (persist=False —
    e.g. a question) still produces a log row, and the worker stamps
    that row with an explicit empty list (sentinel for 'visited, no
    belief'). Falsifiable if no log row exists OR if the row remains
    NULL-stamped (which would cause the worker to re-process it on
    every subsequent call)."""
    text = "What is the default port?"
    ingest_turn(store, text, source="user")
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT derived_belief_ids FROM ingest_log "
        "WHERE raw_text = ?",
        ("What is the default port?",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["derived_belief_ids"] == "[]"
    # And no beliefs were inserted.
    assert store.count_beliefs() == 0


def test_replay_full_equality_passes_after_ingest(store: MemoryStore) -> None:
    """Hypothesis (CI gate): after a representative ingest_turn run, the
    full-equality replay probe reports zero drift — every log row
    re-derives to a shape-equal canonical belief. This is the tripwire
    for missed entry points: any caller that bypasses the worker (writes
    `beliefs` directly without a matching log row) shows up as
    `canonical_orphan`; any worker bug that produces a different
    canonical belief shows up as `mismatched` or `derived_orphan`.

    Falsifiable by any non-zero drift counter."""
    text = (
        "The configuration file lives at /etc/aelfrice/conf. "
        "The default port is 8080 for the dashboard. "
        "Aelfrice stores beliefs in a SQLite database. "
        "The scheduler runs every five minutes by default."
    )
    ingest_turn(store, text, source="user", session_id="sess-1")
    # Re-ingest to exercise the corroboration path.
    ingest_turn(store, text, source="user", session_id="sess-2")
    report = replay_full_equality(store)
    assert report.total_log_rows > 0
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0


def test_ingest_turn_idempotent_after_worker_path(store: MemoryStore) -> None:
    """Hypothesis: re-ingesting the same text under the worker path is
    idempotent on the canonical belief set even though it adds new log
    rows. `ingest_turn`'s public return (newly-inserted count) is 0 on
    re-run. Falsifiable if a duplicate belief appears OR if the second
    call's return value is non-zero."""
    text = "Atomic commits beat batched commits."
    n1 = ingest_turn(store, text, source="user")
    assert n1 == 1
    beliefs_after_first = store.count_beliefs()
    n2 = ingest_turn(store, text, source="user")
    assert n2 == 0  # no new beliefs
    assert store.count_beliefs() == beliefs_after_first
    # But two log rows now exist for the same sentence.
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log WHERE raw_text = ?",
        (text,),
    ).fetchone()
    assert rows["n"] == 2
