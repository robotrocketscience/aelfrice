"""Tests for `aelfrice.auditor` — the v1.1.0 structural auditor.

Three checks: orphan_edges, fts_sync, locked_contradicts. Each test
seeds a state that triggers exactly one check, asserting the report's
`failed` flag and the specific finding's severity / count.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.auditor import (
    CHECK_FTS_SYNC,
    CHECK_LOCKED_CONTRADICTS,
    CHECK_ORPHAN_EDGES,
    SEVERITY_FAIL,
    SEVERITY_INFO,
    audit,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(str(tmp_path / "audit.db"))
    yield s
    s.close()


def _belief(bid: str, *, content: str = None, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content or f"belief content for {bid}",
        content_hash=f"hash-{bid}",
        alpha=9.0 if locked else 1.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-27T00:00:00Z" if locked else None,
        demotion_pressure=0,
        created_at="2026-04-27T00:00:00Z",
        last_retrieved_at=None,
    )


def _find(report, check: str):
    for f in report.findings:
        if f.check == check:
            return f
    raise AssertionError(f"finding for {check!r} not in report")


# --- Empty store -----------------------------------------------------


def test_audit_empty_store_passes_all_checks(store: MemoryStore) -> None:
    report = audit(store)
    assert not report.failed
    for f in report.findings:
        assert f.severity == SEVERITY_INFO


def test_audit_empty_store_metrics_are_zeros(store: MemoryStore) -> None:
    report = audit(store)
    assert report.metrics["beliefs"] == 0
    assert report.metrics["edges"] == 0
    assert report.metrics["locked"] == 0


# --- orphan_edges ----------------------------------------------------


def test_audit_flags_orphan_edge(store: MemoryStore) -> None:
    """An edge whose dst doesn't exist must fail the orphan_edges check."""
    a = _belief("aaaa")
    store.insert_belief(a)
    # Insert an edge to a missing dst by going under the API:
    store._conn.execute(  # type: ignore[attr-defined]
        "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
        (a.id, "ghost", EDGE_SUPPORTS, 1.0),
    )
    store._conn.commit()  # type: ignore[attr-defined]
    report = audit(store)
    assert report.failed
    f = _find(report, CHECK_ORPHAN_EDGES)
    assert f.severity == SEVERITY_FAIL
    assert f.count == 1


def test_audit_clean_edges_pass_orphan_check(store: MemoryStore) -> None:
    a, b = _belief("aaaa"), _belief("bbbb")
    store.insert_belief(a)
    store.insert_belief(b)
    store.insert_edge(Edge(src=a.id, dst=b.id, type=EDGE_SUPPORTS, weight=1.0))
    report = audit(store)
    f = _find(report, CHECK_ORPHAN_EDGES)
    assert f.severity == SEVERITY_INFO
    assert f.count == 0


# --- fts_sync --------------------------------------------------------


def test_audit_flags_fts_drift(store: MemoryStore) -> None:
    """A belief without its FTS mirror row must fail fts_sync."""
    a = _belief("aaaa")
    store.insert_belief(a)
    # Manually delete the FTS row so beliefs and beliefs_fts disagree:
    store._conn.execute(  # type: ignore[attr-defined]
        "DELETE FROM beliefs_fts WHERE id = ?", (a.id,),
    )
    store._conn.commit()  # type: ignore[attr-defined]
    report = audit(store)
    assert report.failed
    f = _find(report, CHECK_FTS_SYNC)
    assert f.severity == SEVERITY_FAIL
    assert f.count == 1


def test_audit_passes_fts_sync_after_normal_inserts(store: MemoryStore) -> None:
    for bid in ("aa", "bb", "cc"):
        store.insert_belief(_belief(bid))
    report = audit(store)
    f = _find(report, CHECK_FTS_SYNC)
    assert f.severity == SEVERITY_INFO


# --- locked_contradicts ---------------------------------------------


def test_audit_flags_locked_contradicts_pair(store: MemoryStore) -> None:
    """Two locked beliefs joined by CONTRADICTS must fail."""
    a = _belief("aaaa", content="rule one is true", locked=True)
    b = _belief("bbbb", content="rule one is false", locked=True)
    store.insert_belief(a)
    store.insert_belief(b)
    store.insert_edge(Edge(src=a.id, dst=b.id, type=EDGE_CONTRADICTS, weight=1.0))
    report = audit(store)
    assert report.failed
    f = _find(report, CHECK_LOCKED_CONTRADICTS)
    assert f.severity == SEVERITY_FAIL
    assert f.count == 1


def test_audit_ignores_unlocked_contradicts(store: MemoryStore) -> None:
    """A CONTRADICTS edge between unlocked beliefs is informational."""
    a = _belief("aaaa", content="rule one is true")
    b = _belief("bbbb", content="rule one is false")
    store.insert_belief(a)
    store.insert_belief(b)
    store.insert_edge(Edge(src=a.id, dst=b.id, type=EDGE_CONTRADICTS, weight=1.0))
    report = audit(store)
    f = _find(report, CHECK_LOCKED_CONTRADICTS)
    assert f.severity == SEVERITY_INFO
    assert not report.failed


def test_audit_locked_contradicts_pair_is_undirected(store: MemoryStore) -> None:
    """CONTRADICTS dst→src is the same finding as src→dst — pair-level."""
    a = _belief("aaaa", content="rule one is true", locked=True)
    b = _belief("bbbb", content="rule one is false", locked=True)
    store.insert_belief(a)
    store.insert_belief(b)
    # Two contradicts edges — one each direction. Should still be ONE pair.
    store.insert_edge(Edge(src=a.id, dst=b.id, type=EDGE_CONTRADICTS, weight=1.0))
    store.insert_edge(Edge(src=b.id, dst=a.id, type=EDGE_CONTRADICTS, weight=1.0))
    report = audit(store)
    f = _find(report, CHECK_LOCKED_CONTRADICTS)
    assert f.count == 1


# --- failed flag aggregation ----------------------------------------


def test_audit_failed_when_any_check_fails(store: MemoryStore) -> None:
    a = _belief("aaaa")
    store.insert_belief(a)
    store._conn.execute(  # type: ignore[attr-defined]
        "DELETE FROM beliefs_fts WHERE id = ?", (a.id,),
    )
    store._conn.commit()  # type: ignore[attr-defined]
    report = audit(store)
    assert report.failed


def test_audit_metrics_include_avg_confidence(store: MemoryStore) -> None:
    a = _belief("aaaa")
    store.insert_belief(a)
    report = audit(store)
    # alpha=1.0, beta=0.5 → posterior mean 0.667
    assert 0.6 <= report.metrics["avg_confidence"] <= 0.7
