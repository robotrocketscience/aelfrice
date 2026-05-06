"""Integration tests for the post-#264 accept_classifications →
derivation_worker call shape.

Falsifiable hypotheses:

    accept_classifications writes one ingest_log row per persisting
    classification with raw_meta carrying both call_site
    (filesystem_ingest) AND override_belief_type, then invokes
    run_worker once. The worker's existing override_belief_type
    plumbing carries the host-decided type into derive().
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.models import BELIEF_FACTUAL, BELIEF_REQUIREMENT
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "classification-via-worker.db"))
    yield s
    s.close()


def _seed_repo(root: Path) -> Path:
    p = root / "DESIGN.md"
    p.write_text(
        "The configuration file lives at /etc/aelfrice/conf.\n"
        "\n"
        "The system must always sign every commit.\n"
        "\n"
        "Aelfrice stores beliefs in a SQLite database.\n",
        encoding="utf-8",
    )
    return root


def test_log_rows_carry_call_site_and_override_belief_type(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: every persisting classification produces one log row
    with raw_meta carrying both `call_site=filesystem_ingest` and
    `override_belief_type` matching the host's decision. Falsifiable
    by a missing key, a wrong call_site, or a row count mismatch."""
    root = _seed_repo(tmp_path)
    started = start_onboard_session(store, root, now="2026-05-06T00:00:00Z")
    sentences = started.sentences
    classifications = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in sentences
    ]
    accept_classifications(
        store, started.session_id, classifications,
        now="2026-05-06T00:00:00Z",
    )
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log"
    ).fetchall()
    assert len(rows) == len(sentences)
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "filesystem_ingest"
        assert meta.get("override_belief_type") == BELIEF_FACTUAL


def test_every_log_row_is_stamped(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: the worker ran end-of-batch and stamped every row."""
    root = _seed_repo(tmp_path)
    started = start_onboard_session(store, root, now="2026-05-06T00:00:00Z")
    classifications = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in started.sentences
    ]
    accept_classifications(
        store, started.session_id, classifications,
        now="2026-05-06T00:00:00Z",
    )
    assert store.list_unstamped_ingest_log() == []


def test_inserted_count_matches_new_canonical_beliefs(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: AcceptOnboardResult.inserted equals the count of
    canonical beliefs newly inserted by the worker. Re-run on the same
    session is rejected; re-run on a fresh session over the same text
    yields inserted=0."""
    root = _seed_repo(tmp_path)
    started1 = start_onboard_session(store, root, now="2026-05-06T00:00:00Z")
    n_sentences = len(started1.sentences)
    classifications1 = [
        HostClassification(
            index=s.index, belief_type=BELIEF_REQUIREMENT, persist=True
        )
        for s in started1.sentences
    ]
    res1 = accept_classifications(
        store, started1.session_id, classifications1,
        now="2026-05-06T00:00:00Z",
    )
    assert res1.inserted == n_sentences
    assert store.count_beliefs() == n_sentences

    started2 = start_onboard_session(store, root, now="2026-05-06T00:00:01Z")
    # The second session may produce 0 sentences if start_onboard_session
    # filters already-known. If it does produce sentences, classifying
    # them must yield inserted=0.
    if started2.sentences:
        classifications2 = [
            HostClassification(
                index=s.index, belief_type=BELIEF_REQUIREMENT, persist=True
            )
            for s in started2.sentences
        ]
        res2 = accept_classifications(
            store, started2.session_id, classifications2,
            now="2026-05-06T00:00:01Z",
        )
        assert res2.inserted == 0
        assert res2.skipped_existing == len(started2.sentences)
    assert store.count_beliefs() == n_sentences


def test_replay_full_equality_passes(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis (CI gate): the full-equality replay probe reports zero
    drift after a host-classified onboarding workload. Tripwire for any
    bypass that writes beliefs without a matching log row."""
    root = _seed_repo(tmp_path)
    started = start_onboard_session(store, root, now="2026-05-06T00:00:00Z")
    classifications = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in started.sentences
    ]
    accept_classifications(
        store, started.session_id, classifications,
        now="2026-05-06T00:00:00Z",
    )
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
