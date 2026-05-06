"""Integration tests for the post-#264 triple_extractor.ingest_triples
→ derivation_worker call shape.

Falsifiable hypotheses:

    ingest_triples writes one ingest_log row per phrase (subject and
    object) with raw_meta.call_site = commit_ingest, invokes run_worker
    once, then reads the stamped belief ids back to insert the relation
    edge.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import Triple, ingest_triples


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "triple-via-worker.db"))
    yield s
    s.close()


def _t(subject: str, relation: str, object_: str) -> Triple:
    return Triple(
        subject=subject, relation=relation, object=object_,
        anchor_text=f"{subject} {relation} {object_}",
    )


def test_log_rows_carry_call_site(store: MemoryStore) -> None:
    """Hypothesis: every triple produces two log rows (subject + object),
    each with raw_meta.call_site = commit_ingest. Falsifiable by missing
    rows or wrong call_site."""
    triples = [
        _t("the worker", "implements", "the spec"),
        _t("aelfrice", "supports", "ingest_log"),
    ]
    ingest_triples(store, triples)
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta, source_kind FROM ingest_log ORDER BY id"
    ).fetchall()
    assert len(rows) == 4  # 2 triples * 2 phrases
    for row in rows:
        assert row["source_kind"] == "git"
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "commit_ingest"


def test_every_log_row_is_stamped(store: MemoryStore) -> None:
    """Hypothesis: the worker ran end-of-call; no orphan log rows."""
    ingest_triples(store, [_t("alpha", "supports", "beta")])
    assert store.list_unstamped_ingest_log() == []


def test_edge_inserted_between_worker_stamped_beliefs(
    store: MemoryStore,
) -> None:
    """Hypothesis: ingest_triples inserts an edge whose endpoints are the
    worker-stamped belief ids — same id-scheme as the pre-#264 path so
    edge id stability holds."""
    res = ingest_triples(store, [_t("alpha", "supports", "beta")])
    assert len(res.new_edges) == 1
    src, dst, relation = res.new_edges[0]
    assert relation == "supports"
    assert store.get_belief(src) is not None
    assert store.get_belief(dst) is not None
    assert store.get_edge(src, dst, relation) is not None


def test_idempotent_on_canonical_state(store: MemoryStore) -> None:
    """Hypothesis: re-ingesting the same triple is idempotent on canonical
    beliefs and edges even though it adds new log rows. Falsifiable by a
    duplicate edge or new_beliefs/new_edges non-empty on the second
    call."""
    triples = [_t("alpha", "supports", "beta")]
    res1 = ingest_triples(store, triples)
    assert len(res1.new_beliefs) == 2
    assert len(res1.new_edges) == 1
    n_beliefs = store.count_beliefs()

    res2 = ingest_triples(store, triples)
    assert res2.new_beliefs == []
    assert res2.new_edges == []
    assert res2.skipped_duplicate_edges == 1
    assert store.count_beliefs() == n_beliefs


def test_replay_full_equality_passes(store: MemoryStore) -> None:
    """Hypothesis (CI gate): full-equality replay reports zero drift after
    a triple-ingest workload."""
    triples = [
        _t("the worker", "implements", "the spec"),
        _t("aelfrice", "supports", "ingest_log"),
        _t("the worker", "supports", "edges"),
    ]
    ingest_triples(store, triples)
    # Re-ingest to exercise the corroboration path inside the worker.
    ingest_triples(store, triples)
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
