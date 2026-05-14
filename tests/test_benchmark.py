"""v0.9.0-rc benchmark harness — unit + score-floor tests."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from aelfrice.benchmark import (
    BENCHMARK_NAME,
    BenchmarkReport,
    DEFAULT_TOP_K,
    MultiHopBenchmarkReport,
    run_benchmark,
    run_multihop_benchmark,
    seed_corpus,
    seed_multihop_corpus,
)
from aelfrice.benchmark import _percentile, _rank_of  # type: ignore[reportPrivateUsage]
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _fresh_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "bench.db"))


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
    s1 = MemoryStore(str(tmp_path / "a.db"))
    s2 = MemoryStore(str(tmp_path / "b.db"))
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


# --- Multi-hop corpus + three-arm reporter tests ---------------------


def _fresh_multihop_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "bench_mh.db"))


def test_seed_multihop_corpus_inserts_12_beliefs(tmp_path: Path) -> None:
    s = _fresh_multihop_store(tmp_path)
    try:
        n = seed_multihop_corpus(s)
    finally:
        s.close()
    assert n == 12
    s = _fresh_multihop_store(tmp_path)
    try:
        assert s.count_beliefs() == 12
    finally:
        s.close()


def test_seed_multihop_corpus_inserts_9_edges(tmp_path: Path) -> None:
    s = _fresh_multihop_store(tmp_path)
    try:
        seed_multihop_corpus(s)
        counts = s.count_edges_by_type()
    finally:
        s.close()
    total = sum(counts.values())
    assert total == 9, f"expected 9 edges, got {total} ({counts})"


def test_run_multihop_benchmark_returns_well_formed_report(tmp_path: Path) -> None:
    s = _fresh_multihop_store(tmp_path)
    try:
        seed_multihop_corpus(s)
        report = run_multihop_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert isinstance(report, MultiHopBenchmarkReport)
    assert report.aelfrice_version == "test"
    assert report.corpus_size == 12
    assert report.query_count == 8
    assert report.top_k == DEFAULT_TOP_K
    # All arm fields present and in [0, 1]
    for arm_name in ("multihop_l1_only", "multihop_l1_l25", "multihop_full"):
        arm = getattr(report, arm_name)
        assert 0.0 <= arm.hit_at_1 <= 1.0, f"{arm_name}.hit_at_1 out of range"
        assert 0.0 <= arm.hit_at_3 <= 1.0, f"{arm_name}.hit_at_3 out of range"
        assert 0.0 <= arm.mrr <= 1.0, f"{arm_name}.mrr out of range"


def test_multihop_l1_only_hit_at_1_below_sanity_threshold(tmp_path: Path) -> None:
    """L1-only arm must hit@1 < 0.50 — queries are multi-hop, not
    surface-keyword-solvable by BM25 alone. Failure means the corpus
    design is broken (bridge identifier leaked into the target belief)."""
    s = _fresh_multihop_store(tmp_path)
    try:
        seed_multihop_corpus(s)
        report = run_multihop_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    assert report.multihop_l1_only.hit_at_1 < 0.50, (
        f"L1-only hit@1={report.multihop_l1_only.hit_at_1:.3f} >= 0.50 — "
        "queries appear to be surface-keyword-solvable; corpus design needs review"
    )


def test_run_multihop_benchmark_rejects_nonpositive_top_k(tmp_path: Path) -> None:
    s = _fresh_multihop_store(tmp_path)
    try:
        seed_multihop_corpus(s)
        with pytest.raises(ValueError):
            run_multihop_benchmark(s, aelfrice_version="test", top_k=0)
        with pytest.raises(ValueError):
            run_multihop_benchmark(s, aelfrice_version="test", top_k=-1)
    finally:
        s.close()


def test_multihop_report_to_dict_structure(tmp_path: Path) -> None:
    """to_dict() must include all top-level keys and each arm as a dict."""
    s = _fresh_multihop_store(tmp_path)
    try:
        seed_multihop_corpus(s)
        report = run_multihop_benchmark(s, aelfrice_version="test")
    finally:
        s.close()
    d = report.to_dict()
    assert "aelfrice_version" in d
    assert "corpus_size" in d
    assert "query_count" in d
    assert "top_k" in d
    for arm_key in ("multihop_l1_only", "multihop_l1_l25", "multihop_full"):
        assert arm_key in d, f"missing key {arm_key}"
        arm_d = d[arm_key]
        assert isinstance(arm_d, dict), f"{arm_key} should be a dict"
        assert set(arm_d.keys()) == {"hit_at_1", "hit_at_3", "mrr"}
    import json
    assert json.dumps(d)  # fully JSON-serialisable


def test_run_multihop_benchmark_is_deterministic(tmp_path: Path) -> None:
    """Two runs against fresh stores yield identical accuracy scores."""
    s1 = MemoryStore(str(tmp_path / "mh_a.db"))
    s2 = MemoryStore(str(tmp_path / "mh_b.db"))
    try:
        seed_multihop_corpus(s1)
        seed_multihop_corpus(s2)
        r1 = run_multihop_benchmark(s1, aelfrice_version="test")
        r2 = run_multihop_benchmark(s2, aelfrice_version="test")
    finally:
        s1.close()
        s2.close()
    assert r1.multihop_l1_only.hit_at_1 == r2.multihop_l1_only.hit_at_1
    assert r1.multihop_l1_l25.hit_at_1 == r2.multihop_l1_l25.hit_at_1
    assert r1.multihop_full.hit_at_1 == r2.multihop_full.hit_at_1


def test_existing_run_benchmark_unaffected_by_multihop_additions(
    tmp_path: Path,
) -> None:
    """seed_corpus + run_benchmark still produce the same results after
    the multi-hop tables were added. Guards against accidental coupling."""
    s = MemoryStore(str(tmp_path / "compat.db"))
    try:
        seed_corpus(s)
        report = run_benchmark(s, aelfrice_version="compat-test")
    finally:
        s.close()
    assert report.benchmark_name == BENCHMARK_NAME
    assert report.corpus_size == 16
    assert report.query_count == 16
    assert report.hit_at_5 >= 0.75
