"""Tests for the #507 scoring wiring inside benchmarks.longmemeval_adapter.

Real LongMemEval runs need HuggingFace + retrieval, so these tests
construct synthetic RetrievalResult / CategoryStats / BenchmarkResult
objects and exercise the aggregation derivations directly. The
substring-EM math itself is covered by tests/test_bench_qa_scoring.py.

Spec: #507 acceptance #1 (non-trivial correctness), #3 (determinism).
"""
from __future__ import annotations

import pytest

pytest.importorskip("datasets")

from benchmarks import longmemeval_adapter as lme


def test_category_stats_means_zero_when_empty():
    cat = lme.CategoryStats(question_type="x")
    assert cat.exact_match == 0.0
    assert cat.substring_exact_match == 0.0
    assert cat.f1 == 0.0


def test_category_stats_means_compute_correctly():
    cat = lme.CategoryStats(question_type="x")
    cat.count = 3
    cat.total_exact_match = 1.0  # 1 hit out of 3
    cat.total_substring_exact_match = 2.0
    cat.total_f1 = 1.5
    assert cat.exact_match == pytest.approx(1.0 / 3)
    assert cat.substring_exact_match == pytest.approx(2.0 / 3)
    assert cat.f1 == pytest.approx(0.5)


def test_benchmark_result_means_zero_when_empty():
    r = lme.BenchmarkResult()
    assert r.exact_match == 0.0
    assert r.substring_exact_match == 0.0
    assert r.f1 == 0.0


def test_benchmark_result_means_compute_correctly():
    r = lme.BenchmarkResult()
    r.total_questions = 4
    r.total_exact_match = 2.0
    r.total_substring_exact_match = 3.0
    r.total_f1 = 2.0
    assert r.exact_match == pytest.approx(0.5)
    assert r.substring_exact_match == pytest.approx(0.75)
    assert r.f1 == pytest.approx(0.5)


def test_retrieval_result_default_scores_are_zero():
    qr = lme.RetrievalResult(
        question_id="q1",
        question_type="single-session-user",
        question="?",
        question_date="2025-01-01",
        answer="paris",
        retrieved_context="",
        num_beliefs=0,
        retrieval_latency_ms=0.0,
    )
    assert qr.exact_match == 0.0
    assert qr.substring_exact_match == 0.0
    assert qr.f1 == 0.0


def test_score_keys_match_metric_band_names():
    # Same naming contract as amabench: tolerance bands key off
    # substring matches in the metric leaf name.
    assert "exact_match" in lme._SCORE_KEYS
    assert "substring_exact_match" in lme._SCORE_KEYS
    assert "f1" in lme._SCORE_KEYS
