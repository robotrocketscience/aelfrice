"""v0.9.0-rc benchmark harness — unit + score-floor tests."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from aelfrice.benchmark import (
    BENCHMARK_NAME,
    BenchmarkReport,
    DEFAULT_TOP_K,
    run_benchmark,
    seed_corpus,
)
from aelfrice.benchmark import _percentile, _rank_of  # type: ignore[reportPrivateUsage]
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import Store


def _fresh_store(tmp_path: Path) -> Store:
    return Store(str(tmp_path / "bench.db"))


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def test_seed_corpus_inserts_16_beliefs(tmp_path: Path) -> None:
    s = _fresh_store(tmp_path)
    try:
        n = seed_corpus(s)
    finally:
        s.close()
    assert n == 16
    s = _fresh_store(tmp_path)
    try:
        # Re-open and confirm round-trip.
        assert s.count_beliefs() == 16
    finally:
        s.close()


def test_run_benchmark_returns_well_formed_report(tmp_path: Path) -> None:
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="0.9.0-rc")
    finally:
        s.close()
    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_name == BENCHMARK_NAME
    assert report.aelfrice_version == "0.9.0-rc"
    assert report.corpus_size == 16
    assert report.query_count == 16
    assert report.top_k == DEFAULT_TOP_K
    for f in fields(BenchmarkReport):
        assert getattr(report, f.name) is not None, f"field {f.name} missing"


def test_run_benchmark_score_floor_hit_at_5(tmp_path: Path) -> None:
    """Locked-in floor: at least 12 of 16 queries find their correct
    belief in the top-5. Below this means BM25 escaping or scoring has
    regressed and the harness will produce an unpublishable result."""
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert report.hit_at_5 >= 0.75, (
        f"hit_at_5={report.hit_at_5:.3f} below floor 0.75 "
        f"({int(report.hit_at_5 * report.query_count)} of "
        f"{report.query_count} hits)"
    )


def test_run_benchmark_mrr_consistency(tmp_path: Path) -> None:
    """MRR must be between hit_at_1 (lower bound — missed queries
    contribute 0) and hit_at_5 (upper bound — at best every hit is
    rank 1)."""
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert report.hit_at_1 <= report.mrr <= report.hit_at_5


def test_run_benchmark_latency_percentiles_ordered(tmp_path: Path) -> None:
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert report.p50_latency_ms <= report.p99_latency_ms


def test_run_benchmark_latency_p99_under_ceiling(tmp_path: Path) -> None:
    """For a 16-belief corpus on any halfway-modern machine, p99 should
    be deep below 100ms. If we cross that the harness has a bug."""
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert report.p99_latency_ms < 100.0, (
        f"p99={report.p99_latency_ms:.2f}ms exceeds 100ms ceiling"
    )


def test_run_benchmark_is_deterministic(tmp_path: Path) -> None:
    """Two runs against fresh stores must produce identical
    accuracy scores. Latencies will differ; that's expected and
    handled at the percentile-only assertion above."""
    s1 = Store(str(tmp_path / "a.db"))
    s2 = Store(str(tmp_path / "b.db"))
    try:
        seed_corpus(s1)
        seed_corpus(s2)
        r1 = run_benchmark(s1, aelfrice_version="test")
        r2 = run_benchmark(s2, aelfrice_version="test")
    finally:
        s1.close()
        s2.close()
    assert r1.hit_at_1 == r2.hit_at_1
    assert r1.hit_at_3 == r2.hit_at_3
    assert r1.hit_at_5 == r2.hit_at_5
    assert r1.mrr == r2.mrr


def test_run_benchmark_rejects_nonpositive_top_k(tmp_path: Path) -> None:
    s = _fresh_store(tmp_path)
    try:
        seed_corpus(s)
        with pytest.raises(ValueError):
            run_benchmark(s, aelfrice_version="test", top_k=0)
        with pytest.raises(ValueError):
            run_benchmark(s, aelfrice_version="test", top_k=-1)
    finally:
        s.close()


def test_report_to_dict_round_trip() -> None:
    """to_dict() output must contain every dataclass field as a
    JSON-serialisable primitive."""
    report = BenchmarkReport(
        benchmark_name="test",
        aelfrice_version="0.0.0",
        corpus_size=1,
        query_count=1,
        top_k=5,
        hit_at_1=1.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        mrr=1.0,
        p50_latency_ms=0.5,
        p99_latency_ms=1.5,
    )
    d = report.to_dict()
    assert set(d.keys()) == {f.name for f in fields(BenchmarkReport)}
    import json
    assert json.dumps(d)  # serialisable


# --- Internal helpers ------------------------------------------------


def test_rank_of_returns_one_indexed_position() -> None:
    results = [_mk("a", "x"), _mk("b", "y"), _mk("c", "z")]
    assert _rank_of("a", results) == 1
    assert _rank_of("b", results) == 2
    assert _rank_of("c", results) == 3
    assert _rank_of("d", results) is None


def test_rank_of_empty_results() -> None:
    assert _rank_of("anything", []) is None


def test_percentile_single_value() -> None:
    assert _percentile([42.0], 0.5) == 42.0
    assert _percentile([42.0], 0.99) == 42.0


def test_percentile_two_values() -> None:
    assert _percentile([1.0, 3.0], 0.0) == 1.0
    assert _percentile([1.0, 3.0], 1.0) == 3.0
    assert _percentile([1.0, 3.0], 0.5) == 2.0


def test_percentile_empty_list_returns_zero() -> None:
    assert _percentile([], 0.5) == 0.0


def test_percentile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError):
        _percentile([1.0, 2.0], -0.1)
    with pytest.raises(ValueError):
        _percentile([1.0, 2.0], 1.1)
