"""Integration tests for the post-#264 accept_classifications →
derivation_worker call shape.

`accept_classifications` is the host-LLM onboarding terminus: the host
classified each sentence in its own context, returned a `belief_type`
verdict, and now aelfrice materializes the resulting beliefs. Under
slice 2 the entry point appends one unstamped log row per persisted
classification (with `raw_meta.override_belief_type` set), then invokes
`run_worker(store)` once at end-of-batch. Skipped-vs-inserted accounting
is reconstructed post-worker from the stamped log rows.

These tests anchor the *new* invariants. They will catch any future
regression that re-introduces an inline `derive()` / `insert_belief()`
call in the entry point.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.models import BELIEF_FACTUAL
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "classification-via-worker.db"))
    yield s
    s.close()


def _populate_repo(root: Path) -> None:
    """Drop a single .md file with a couple of factual sentences."""
    (root / "facts.md").write_text(
        "The configuration file lives at /etc/aelfrice/conf.\n"
        "The default port is 8080.\n"
    )


def test_every_log_row_is_stamped_after_accept(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: after accept_classifications returns, no log row
    remains unstamped — the worker ran end-of-batch and stamped every
    row. Falsifiable by any row whose `derived_belief_ids IS NULL`."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    accept_classifications(store, result.session_id, cls, now="2026-04-26T01:00:00Z")
    unstamped = store.list_unstamped_ingest_log()
    assert unstamped == [], f"unstamped rows: {unstamped}"


def test_log_row_carries_call_site_and_override_metadata(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: each persisted classification produces a log row
    whose `raw_meta` carries both `call_site=filesystem_ingest` and
    `override_belief_type=<host's verdict>`. The worker reads
    override_belief_type to reconstruct the same DerivationInput on
    replay. Falsifiable if either key is missing."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    accept_classifications(store, result.session_id, cls, now="2026-04-26T01:00:00Z")
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log WHERE source_path IS NOT NULL"
    ).fetchall()
    assert rows
    import json
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "filesystem_ingest"
        assert meta.get("override_belief_type") == BELIEF_FACTUAL


def test_non_persisting_classification_writes_no_log_row(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: a HostClassification with persist=False is dropped
    before record_ingest — no log row is appended. The host's verdict
    is "this isn't worth persisting" and the canonical log shouldn't
    record it. Falsifiable if a log row appears for that index."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    if not result.sentences:
        pytest.skip("fixture produced no sentences")
    cls = [HostClassification(
        index=result.sentences[0].index,
        belief_type=BELIEF_FACTUAL,
        persist=False,
    )]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z",
    )
    assert outcome.skipped_non_persisting == 1
    assert outcome.inserted == 0
    n_rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log",
    ).fetchone()["n"]
    assert n_rows == 0


def test_replay_full_equality_passes_after_accept(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis (CI gate): after a representative accept run, the
    full-equality replay probe reports zero drift — every log row
    re-derives to a shape-equal canonical belief, including the
    host-supplied override_belief_type that rides in raw_meta.

    Falsifiable by any non-zero drift counter."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    accept_classifications(store, result.session_id, cls, now="2026-04-26T01:00:00Z")
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
