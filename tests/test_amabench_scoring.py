"""Tests for the #507 scoring wiring inside benchmarks.amabench_adapter.

Real AMA-Bench runs need HuggingFace + retrieval, so these tests build
synthetic EpisodeResult / per_question rows and exercise the
aggregation + per-key breakdown helpers directly. The substring-EM
math itself is covered by tests/test_bench_qa_scoring.py.

Spec: #507 acceptance #1 (non-trivial correctness), #3 (determinism).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Loading without HF: amabench_adapter top-level imports `datasets`,
# which is heavy but installed as a hard dep. Import normally.
import pytest

pytest.importorskip("datasets")

from benchmarks import amabench_adapter as ama


def _row(qa_type: str, domain: str, em: float, sem: float, f1: float) -> dict:
    return {
        "episode_id": "ep0",
        "domain": domain,
        "task_type": "task",
        "question": "q",
        "qa_type": qa_type,
        "qa_type_name": ama.QA_TYPE_NAMES.get(qa_type, "unknown"),
        "question_uuid": f"uuid-{qa_type}-{domain}",
        "context": "ctx",
        "exact_match": em,
        "substring_exact_match": sem,
        "f1": f1,
    }


def _make_episode_result(rows: list[dict], domain: str) -> ama.EpisodeResult:
    r = ama.EpisodeResult(episode_id="ep0", domain=domain)
    r.total_qa = len(rows)
    r.per_question = list(rows)
    return r


def test_aggregate_sums_scores_across_episodes():
    rows = [
        _row("A", "Game", 1.0, 1.0, 1.0),
        _row("A", "Game", 0.0, 1.0, 0.5),
        _row("B", "Game", 0.0, 0.0, 0.0),
    ]
    ep = _make_episode_result(rows, "Game")
    agg = ama.aggregate_results([ep])
    assert agg.total_qa == 3
    assert agg.score_sums["exact_match"] == pytest.approx(1.0)
    assert agg.score_sums["substring_exact_match"] == pytest.approx(2.0)
    assert agg.score_sums["f1"] == pytest.approx(1.5)


def test_aggregate_handles_legacy_rows_without_scores():
    # Pre-#507 per_question rows had no score keys; aggregator must
    # treat them as missing rather than crashing.
    legacy = {
        "episode_id": "old",
        "domain": "Game",
        "task_type": "task",
        "question": "q",
        "qa_type": "A",
        "qa_type_name": ama.QA_TYPE_NAMES["A"],
        "question_uuid": "old",
        "context": "ctx",
    }
    ep = _make_episode_result([legacy], "Game")
    agg = ama.aggregate_results([ep])
    assert agg.total_qa == 1
    for k in ama._SCORE_KEYS:
        assert agg.score_sums[k] == 0.0


def test_accuracy_by_qa_type_means():
    rows = [
        _row("A", "Game", 1.0, 1.0, 1.0),
        _row("A", "Game", 1.0, 1.0, 0.0),
        _row("B", "Web", 0.0, 1.0, 0.5),
    ]
    by_type = ama._accuracy_by_key(rows, "qa_type")
    assert by_type["A"]["count"] == 2
    assert by_type["A"]["exact_match"] == pytest.approx(1.0)
    assert by_type["A"]["f1"] == pytest.approx(0.5)
    assert by_type["B"]["count"] == 1
    assert by_type["B"]["substring_exact_match"] == pytest.approx(1.0)


def test_accuracy_by_domain_groups_independently_of_type():
    rows = [
        _row("A", "Game", 1.0, 1.0, 1.0),
        _row("B", "Game", 0.0, 0.0, 0.0),
        _row("A", "Web", 1.0, 1.0, 0.8),
    ]
    by_domain = ama._accuracy_by_key(rows, "domain")
    assert by_domain["Game"]["count"] == 2
    assert by_domain["Game"]["exact_match"] == pytest.approx(0.5)
    assert by_domain["Web"]["count"] == 1
    assert by_domain["Web"]["f1"] == pytest.approx(0.8)


def test_accuracy_by_key_empty_input():
    assert ama._accuracy_by_key([], "qa_type") == {}


def test_score_keys_match_metric_band_names():
    # Tolerance bands key off substring matches in the metric leaf
    # name. The chosen names ride MAB's canonical band assignments.
    assert "exact_match" in ama._SCORE_KEYS
    assert "substring_exact_match" in ama._SCORE_KEYS
    assert "f1" in ama._SCORE_KEYS
