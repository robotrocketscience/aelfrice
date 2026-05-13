"""Smoke tests for the v3.0 BFS latency bench harness (issue #739).

These tests exercise the corpus generator, query generator,
percentile helper, and end-to-end CLI flow on a tiny (40 beliefs /
4 topics) corpus. Latency numbers are not asserted — only schema,
determinism, and gate-result shape.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from benchmarks.bfs_latency_v3 import (
    DEFAULT_BELIEF_COUNT,
    GATE_P50_MS,
    _percentile,
    evaluate_gate,
    main,
    seed_corpus,
    synth_queries,
)
from benchmarks.bfs_latency_v3 import ArmResult
from aelfrice.store import MemoryStore


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


def test_evaluate_gate_pass_and_fail():
    good = ArmResult(
        label="bfs_on", samples=100,
        p50_ms=10.0, p95_ms=40.0, p99_ms=80.0,
        max_ms=90.0, mean_ms=12.0, min_ms=8.0,
    )
    g = evaluate_gate(good)
    assert g.passed is True
    assert g.p50_pass and g.p95_pass and g.p99_pass and g.tail_ratio_pass

    bad_p50 = ArmResult(
        label="bfs_on", samples=100,
        p50_ms=GATE_P50_MS + 1.0,
        p95_ms=40.0, p99_ms=80.0,
        max_ms=90.0, mean_ms=30.0, min_ms=8.0,
    )
    assert evaluate_gate(bad_p50).passed is False

    bad_tail = ArmResult(
        label="bfs_on", samples=100,
        p50_ms=10.0, p95_ms=40.0, p99_ms=80.0,
        max_ms=200.0,  # 20× p50 -> tail-ratio fail.
        mean_ms=15.0, min_ms=8.0,
    )
    g_tail = evaluate_gate(bad_tail)
    assert g_tail.tail_ratio_pass is False
    assert g_tail.passed is False


def test_main_end_to_end_smoke(tmp_path: Path):
    out = tmp_path / "out.json"
    rc = main([
        "--beliefs", "40",
        "--topics", "4",
        "--queries", "4",
        "--iterations", "2",
        "--warmup", "1",
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
