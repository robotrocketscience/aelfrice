"""Unit tests for the #1064 G3 latency bench.

Covers the deterministic pieces of ``benchmarks/temporal_spine_latency.py``
— percentile math, the delta gate, corpus/query generation and input
validation — plus one small store-backed check that the spine lane
actually fires (the guard against a vacuous latency-delta pass). No
wall-clock timing is asserted, so these run in the ordinary pytest matrix.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to sys.path so the harness can import benchmarks.*
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aelfrice.store import MemoryStore

from benchmarks.temporal_spine_latency import (
    ArmResult,
    GATE_DELTA_P50_MS,
    GATE_DELTA_P95_MS,
    GATE_MAX_OVER_MEDIAN_RATIO,
    _percentile,
    evaluate_gate,
    probe_lane_fires,
    seed_spine_corpus,
    synth_spine_queries,
)


# --- percentile -------------------------------------------------------------


def test_percentile_empty_is_zero() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_bounds() -> None:
    s = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(s, 0) == 1.0
    assert _percentile(s, 100) == 4.0
    # nearest-rank: rank = ceil(p/100 * n)
    assert _percentile(s, 50) == 2.0   # ceil(0.5*4)=2 -> index 1
    assert _percentile(s, 95) == 4.0   # ceil(0.95*4)=4 -> index 3
    assert _percentile(s, 25) == 1.0   # ceil(0.25*4)=1 -> index 0


def test_percentile_single_sample() -> None:
    assert _percentile([7.0], 50) == 7.0
    assert _percentile([7.0], 95) == 7.0


# --- gate -------------------------------------------------------------------


def _arm(label: str, *, p50: float, p95: float, mx: float) -> ArmResult:
    return ArmResult(
        label=label, samples=100, p50_ms=p50, p95_ms=p95,
        p99_ms=p95, max_ms=mx, mean_ms=p50, min_ms=p50 / 2,
    )


def test_gate_passes_within_band() -> None:
    off = _arm("off", p50=60.0, p95=90.0, mx=100.0)
    on = _arm("on", p50=60.5, p95=110.0, mx=120.0)  # Δp50 .5, Δp95 20
    g = evaluate_gate(off, on)
    assert g.delta_p50_ms == pytest.approx(0.5)
    assert g.delta_p95_ms == pytest.approx(20.0)
    assert g.delta_p50_pass and g.delta_p95_pass and g.tail_ratio_pass
    assert g.passed is True


def test_gate_fails_on_p50_delta() -> None:
    off = _arm("off", p50=60.0, p95=90.0, mx=100.0)
    on = _arm("on", p50=60.0 + GATE_DELTA_P50_MS + 0.1, p95=95.0, mx=110.0)
    g = evaluate_gate(off, on)
    assert g.delta_p50_pass is False
    assert g.passed is False


def test_gate_fails_on_p95_delta() -> None:
    off = _arm("off", p50=60.0, p95=90.0, mx=100.0)
    on = _arm("on", p50=60.5, p95=90.0 + GATE_DELTA_P95_MS + 0.1, mx=160.0)
    g = evaluate_gate(off, on)
    assert g.delta_p95_pass is False
    assert g.passed is False


def test_gate_fails_on_tail_ratio() -> None:
    off = _arm("off", p50=60.0, p95=90.0, mx=100.0)
    # max/p50 = 700/60 ≈ 11.7 > 10 tail guard, deltas otherwise fine.
    on = _arm("on", p50=60.0, p95=95.0, mx=700.0)
    g = evaluate_gate(off, on)
    assert g.tail_ratio > GATE_MAX_OVER_MEDIAN_RATIO
    assert g.tail_ratio_pass is False
    assert g.passed is False


def test_gate_boundary_is_inclusive() -> None:
    off = _arm("off", p50=60.0, p95=90.0, mx=100.0)
    on = _arm(
        "on",
        p50=60.0 + GATE_DELTA_P50_MS,      # exactly +5
        p95=90.0 + GATE_DELTA_P95_MS,      # exactly +50
        mx=(60.0 + GATE_DELTA_P50_MS) * GATE_MAX_OVER_MEDIAN_RATIO,  # 10x
    )
    g = evaluate_gate(off, on)
    assert g.passed is True  # ≤ thresholds, not <


# --- corpus / queries -------------------------------------------------------


def test_synth_queries_shape_and_spread() -> None:
    qs = synth_spine_queries(count=5, sessions=100, per_session=50)
    assert len(qs) == 5
    for q in qs:
        assert q.startswith("sess_")
        assert "relates_to" in q and "condition_" in q


def test_synth_queries_validates_input() -> None:
    with pytest.raises(ValueError):
        synth_spine_queries(count=0, sessions=10, per_session=5)
    with pytest.raises(ValueError):
        synth_spine_queries(count=20, sessions=10, per_session=5)  # count>sessions
    with pytest.raises(ValueError):
        synth_spine_queries(count=2, sessions=10, per_session=0)


def test_seed_corpus_validates_input() -> None:
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError):
            seed_spine_corpus(store, belief_count=0, sessions=5)
        with pytest.raises(ValueError):
            # 10 not a multiple of 3
            seed_spine_corpus(store, belief_count=10, sessions=3)
    finally:
        store.close()


def test_seed_corpus_edge_count_is_beliefs_minus_sessions() -> None:
    """One TEMPORAL_NEXT per belief except the first in each chain."""
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(str(Path(td) / "s.sqlite"))
        try:
            spec = seed_spine_corpus(store, belief_count=100, sessions=5)
            assert spec.beliefs == 100
            assert spec.sessions == 5
            assert spec.per_session == 20
            # 5 chains of 20 → 19 edges each → 95 total.
            assert spec.spine_edges == 100 - 5
        finally:
            store.close()


def test_lane_actually_fires_on_seeded_corpus() -> None:
    """The core not-a-null guard: the spine lane must produce candidates.

    A latency delta is only meaningful if turning the lane on changes
    the work done. This proves the seeded corpus + generated queries
    reach chained beliefs, so the bench's delta is not vacuous.
    """
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(str(Path(td) / "s.sqlite"))
        try:
            seed_spine_corpus(store, belief_count=100, sessions=5)
            queries = synth_spine_queries(
                count=5, sessions=5, per_session=20,
            )
            probe = probe_lane_fires(
                store, queries, budget=1500, l1_limit=50,
            )
            assert probe.queries_total == 5
            assert probe.fired is True
            assert probe.candidates_total > 0
            assert probe.queries_fired > 0
        finally:
            store.close()
