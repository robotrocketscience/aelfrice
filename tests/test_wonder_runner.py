"""Integration test for the bake-off runner (#228)."""
from __future__ import annotations

import json

from aelfrice.wonder.runner import _build_argparser, main, run_bakeoff


def test_run_bakeoff_smoke() -> None:
    result = run_bakeoff(
        n_topics=3,
        n_atoms_per_topic=5,
        n_walks=10,
        n_sts_samples=10,
        feedback_budget=4,
        seeds=2,
    )
    assert "config" in result
    assert "per_seed" in result
    assert "aggregate" in result
    assert len(result["per_seed"]) == 2
    for seed_result in result["per_seed"]:
        assert "seed" in seed_result
        assert len(seed_result["strategy_metrics"]) == 3
        assert seed_result["verdict"] in {"single", "ensemble", "defer", "drop"}


def test_run_bakeoff_deterministic_for_same_seed() -> None:
    a = run_bakeoff(
        n_topics=3, n_atoms_per_topic=5,
        n_walks=10, n_sts_samples=10,
        feedback_budget=4, seeds=2,
    )
    b = run_bakeoff(
        n_topics=3, n_atoms_per_topic=5,
        n_walks=10, n_sts_samples=10,
        feedback_budget=4, seeds=2,
    )
    assert a["aggregate"] == b["aggregate"]


def test_run_bakeoff_aggregate_shape() -> None:
    result = run_bakeoff(
        n_topics=3, n_atoms_per_topic=5,
        n_walks=5, n_sts_samples=5,
        feedback_budget=4, seeds=2,
    )
    agg = result["aggregate"]
    assert set(agg["strategy_metrics"].keys()) == {"RW", "TC", "STS"}
    assert "RW|TC" in agg["jaccard"]
    assert "RW|STS" in agg["jaccard"]
    assert "STS|TC" in agg["jaccard"]
    assert agg["majority_verdict"] in {"single", "ensemble", "defer", "drop"}


def test_argparser_defaults() -> None:
    p = _build_argparser()
    args = p.parse_args([])
    assert args.seeds == 10
    assert args.feedback_budget == 16


def test_main_writes_json_to_file(tmp_path) -> None:
    out = tmp_path / "result.json"
    rc = main([
        "--n-topics", "3",
        "--n-atoms-per-topic", "5",
        "--n-walks", "5",
        "--n-sts-samples", "5",
        "--feedback-budget", "4",
        "--seeds", "1",
        "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "aggregate" in payload
