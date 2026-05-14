"""Smoke tests for the v3.0 BFS latency bench harness (issue #739).

These tests exercise the corpus generator, query generator,
percentile helper, and end-to-end CLI flow on a tiny (40 beliefs /
4 topics) corpus. Latency numbers are not asserted — only schema,
determinism, and gate-result shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.bfs_latency_v3 import (
    DEFAULT_BELIEF_COUNT,
    GATE_DELTA_P50_MS,
    GATE_DELTA_P95_MS,
    _percentile,
    evaluate_gate,
    main,
    seed_corpus,
    synth_queries,
)
from benchmarks.bfs_latency_v3 import ArmResult
from aelfrice.store import MemoryStore


def _arm(
    label: str,
    *,
    p50: float, p95: float, p99: float,
    max_ms: float, mean: float = 0.0, min_ms: float = 0.0,
) -> ArmResult:
    return ArmResult(
        label=label, samples=100,
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        max_ms=max_ms,
        mean_ms=mean or p50,
        min_ms=min_ms or p50 * 0.5,
    )


def test_percentile_nearest_rank():
    s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert _percentile(s, 50) == 5.0
    assert _percentile(s, 95) == 10.0
    assert _percentile(s, 99) == 10.0
    assert _percentile(s, 0) == 1.0
    assert _percentile(s, 100) == 10.0


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_seed_corpus_size_and_edge_count(tmp_path: Path):
    db = tmp_path / "t.sqlite"
    store = MemoryStore(str(db))
    try:
        spec = seed_corpus(store, belief_count=40, topics=4)
        assert spec.beliefs == 40
        # 2 SUPPORTS per belief + 1 CITES per topic = 40*2 + 4 = 84.
        assert spec.edges == 84
        assert spec.topics == 4
    finally:
        store.close()


def test_seed_corpus_rejects_non_multiple():
    with pytest.raises(ValueError):
        # 100 / 7 is not integer; seeder must reject.
        seed_corpus(MemoryStore(":memory:"), belief_count=100, topics=7)


def test_synth_queries_deterministic_and_precise():
    qs = synth_queries(count=10, topics=20, belief_count=200)
    assert len(qs) == 10
    # Determinism: re-call yields identical tuple.
    assert qs == synth_queries(count=10, topics=20, belief_count=200)
    # Precision: every query starts with a topic_NNN_entity_K token
    # so the #741 prompt-shape gate runs BFS instead of short-
    # circuiting it. (The harness has its own internal guard but
    # we double-check from the test surface.)
    for q in qs:
        first = q.split(None, 1)[0]
        assert first.startswith("topic_"), q


def test_evaluate_gate_passes_under_small_delta():
    off = _arm("bfs_off", p50=100.0, p95=140.0, p99=180.0, max_ms=400.0)
    # delta_p50 = +3 ms ≤ 5; delta_p95 = +25 ms ≤ 50; tail ratio fine.
    on = _arm("bfs_on", p50=103.0, p95=165.0, p99=195.0, max_ms=390.0)
    g = evaluate_gate(off, on)
    assert g.passed is True
    assert g.delta_p50_pass and g.delta_p95_pass and g.tail_ratio_pass
    assert g.delta_p50_ms == pytest.approx(3.0)
    assert g.delta_p95_ms == pytest.approx(25.0)


def test_evaluate_gate_fails_on_delta_p50_regression():
    off = _arm("bfs_off", p50=100.0, p95=140.0, p99=180.0, max_ms=400.0)
    # delta_p50 = GATE_DELTA_P50_MS + 0.1 → fail.
    on = _arm(
        "bfs_on",
        p50=100.0 + GATE_DELTA_P50_MS + 0.1,
        p95=145.0, p99=185.0, max_ms=400.0,
    )
    g = evaluate_gate(off, on)
    assert g.delta_p50_pass is False
    assert g.passed is False


def test_evaluate_gate_fails_on_delta_p95_regression():
    off = _arm("bfs_off", p50=100.0, p95=140.0, p99=180.0, max_ms=400.0)
    # delta_p95 = GATE_DELTA_P95_MS + 0.1 → fail.
    on = _arm(
        "bfs_on",
        p50=101.0,
        p95=140.0 + GATE_DELTA_P95_MS + 0.1,
        p99=185.0, max_ms=400.0,
    )
    g = evaluate_gate(off, on)
    assert g.delta_p95_pass is False
    assert g.passed is False


def test_evaluate_gate_fails_on_tail_ratio():
    off = _arm("bfs_off", p50=10.0, p95=40.0, p99=80.0, max_ms=90.0)
    on = _arm(
        "bfs_on", p50=10.0, p95=41.0, p99=82.0,
        # 20× p50 → tail-ratio fail even with negligible deltas.
        max_ms=200.0,
    )
    g = evaluate_gate(off, on)
    assert g.tail_ratio_pass is False
    assert g.passed is False


def test_main_end_to_end_smoke(tmp_path: Path):
    out = tmp_path / "out.json"
    rc = main([
        "--beliefs", "40",
        "--topics", "4",
        "--queries", "4",
        "--iterations", "10",
        "--warmup", "2",
        "--output", str(out),
    ])
    # Tiny-corpus run on a v3.0 stack comfortably clears the gate;
    # rc==0 also implies the JSON was emitted.
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["harness"] == "bfs_latency_v3"
    assert payload["corpus"]["beliefs"] == 40
    assert payload["corpus"]["edges"] == 4 * 2 * 10 + 4  # 84
    assert set(payload["arms"]) == {"bfs_off", "bfs_on"}
    for arm in ("bfs_off", "bfs_on"):
        a = payload["arms"][arm]
        for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "min_ms"):
            assert isinstance(a[k], (int, float))
            assert a[k] >= 0
    assert isinstance(payload["gate"]["passed"], bool)


def test_default_belief_count_is_gate_size():
    """Regression guard: default belief count is the #739 gate target."""
    assert DEFAULT_BELIEF_COUNT >= 10_000
