"""Tests for benchmarks.context_rebuilder.kappa (#687)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.context_rebuilder.kappa import (
    HOT_START_FIDELITY_THRESHOLD,
    INTER_JUDGE_KAPPA_THRESHOLD,
    JudgeRun,
    cohens_kappa,
    compute_kappa_report,
    load_judge_run,
    report_to_json,
)


# --- cohens_kappa: pure-math edges ----------------------------------------

def test_kappa_perfect_agreement_returns_one() -> None:
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_perfect_disagreement_returns_minus_one() -> None:
    # Two raters, perfectly inverse, balanced classes: κ = -1.
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) == pytest.approx(-1.0)


def test_kappa_empty_vectors_returns_zero() -> None:
    assert cohens_kappa([], []) == 0.0


def test_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([True, False], [True])


def test_kappa_all_one_class_same_returns_one() -> None:
    # Both raters say True for every item — perfect agreement is the
    # documented convention even though p_e=1 would otherwise blow up.
    assert cohens_kappa([True] * 5, [True] * 5) == 1.0


def test_kappa_known_value() -> None:
    # Hand-computed: 10 pairs, confusion matrix
    #   rater_a: T T T T T F F F F F
    #   rater_b: T T T T F F F F T T
    # → a=4, b=1, c=2, d=3
    # p_o = (4+3)/10 = 0.7
    # p_yes_a = 5/10 = 0.5, p_yes_b = 6/10 = 0.6
    # p_e = 0.5*0.6 + 0.5*0.4 = 0.5
    # κ = (0.7 - 0.5) / (1 - 0.5) = 0.4
    rater_a = [True] * 5 + [False] * 5
    rater_b = [True, True, True, True, False, False, False, False, True, True]
    assert cohens_kappa(rater_a, rater_b) == pytest.approx(0.4)


# --- compute_kappa_report: end-to-end -------------------------------------

def _run(run_id: str, verdicts: dict[int, bool]) -> JudgeRun:
    return JudgeRun(run_id=run_id, verdicts=verdicts)


def test_report_calibrated_path() -> None:
    # 3 runs, all near-perfect agreement, fidelity well above 0.80.
    common = {i: True for i in range(10)}
    runs = [
        _run("judge_1", dict(common)),
        _run("judge_2", dict(common)),
        _run("judge_3", dict(common)),
    ]
    report = compute_kappa_report(runs, baseline=None, run_id="calib_test")
    assert report.n_runs == 3
    assert report.n_pairs == 10
    assert report.inter_judge_kappa["min"] == 1.0
    assert report.hot_start_fidelity_mean == 1.0
    assert report.calibrated is True
    assert report.failure_reasons == []


def test_report_rejects_below_min_runs() -> None:
    runs = [
        _run("judge_1", {0: True, 1: True}),
        _run("judge_2", {0: True, 1: True}),
    ]
    report = compute_kappa_report(runs, baseline=None, run_id="too_few_runs")
    assert report.calibrated is False
    assert any("n_runs" in r for r in report.failure_reasons)


def test_report_rejects_below_kappa_threshold() -> None:
    # 3 runs but one rater disagrees on half — drives min pair κ below 0.70.
    runs = [
        _run("judge_1", {i: True for i in range(10)}),
        _run("judge_2", {i: True for i in range(10)}),
        _run("judge_3", {i: (i % 2 == 0) for i in range(10)}),
    ]
    report = compute_kappa_report(runs, baseline=None, run_id="low_kappa")
    assert report.calibrated is False
    assert report.inter_judge_kappa["min"] < INTER_JUDGE_KAPPA_THRESHOLD
    assert any("inter_judge_kappa.min" in r for r in report.failure_reasons)


def test_report_rejects_below_fidelity_threshold() -> None:
    # 3 runs with perfect agreement on a mostly-False set: κ=1 (passes)
    # but fidelity is below the 0.80 floor.
    common = {i: (i < 5) for i in range(10)}  # 50% True
    runs = [_run(f"judge_{i+1}", dict(common)) for i in range(3)]
    report = compute_kappa_report(runs, baseline=None, run_id="low_fidelity")
    assert report.calibrated is False
    assert report.inter_judge_kappa["min"] == 1.0
    assert report.hot_start_fidelity_mean < HOT_START_FIDELITY_THRESHOLD
    assert any("hot_start_fidelity_mean" in r for r in report.failure_reasons)


def test_report_aligns_to_intersection() -> None:
    # Runs disagree on which turn_idxes they cover — only the
    # intersection contributes to κ.
    runs = [
        _run("judge_1", {0: True, 1: True, 2: True, 3: True}),
        _run("judge_2", {1: True, 2: True, 3: True, 4: True}),
        _run("judge_3", {2: True, 3: True, 4: True, 5: True}),
    ]
    report = compute_kappa_report(runs, baseline=None, run_id="alignment")
    # Common turn_idxes are {2, 3} only.
    assert report.n_pairs == 2


def test_report_with_baseline_reports_judge_vs_baseline() -> None:
    runs = [
        _run("judge_1", {0: True, 1: True, 2: False, 3: True}),
        _run("judge_2", {0: True, 1: True, 2: False, 3: True}),
        _run("judge_3", {0: True, 1: True, 2: False, 3: True}),
    ]
    # Baseline only matches 2 of 4 — substring path misses semantic pairs.
    baseline = _run("baseline", {0: True, 1: False, 2: False, 3: False})
    report = compute_kappa_report(
        runs, baseline=baseline, run_id="with_baseline"
    )
    assert "judge_1" in report.judge_vs_baseline_kappa
    assert "mean" in report.judge_vs_baseline_kappa
    # Calibrated check should ignore baseline (judges agree perfectly,
    # fidelity 0.75 — below 0.80 floor → not calibrated, but the
    # baseline κ itself doesn't gate).
    assert report.hot_start_fidelity_mean == 0.75
    # Baseline κ low (judges disagree with substring) — expected.
    assert report.judge_vs_baseline_kappa["mean"] < 0.5


def test_report_to_json_shape_matches_artifact_contract() -> None:
    runs = [_run(f"judge_{i+1}", {0: True}) for i in range(3)]
    report = compute_kappa_report(runs, baseline=None, run_id="shape_test")
    payload = report_to_json(report)
    required = {
        "run_id", "n_runs", "n_pairs", "judge_model", "baseline",
        "inter_judge_kappa", "judge_vs_baseline_kappa",
        "per_run_hot_start_fidelity", "hot_start_fidelity_mean",
        "calibrated", "failure_reasons",
    }
    assert required.issubset(payload.keys())
    # Re-roundtrip through JSON to confirm no non-serialisable types.
    assert json.loads(json.dumps(payload))["run_id"] == "shape_test"


# --- load_judge_run: file-format compatibility ----------------------------

def test_load_judge_run_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "judge_1.jsonl"
    p.write_text(
        '{"turn_idx": 0, "matched": true}\n'
        'not-json\n'
        '{"turn_idx": "wrong-type", "matched": true}\n'
        '{"turn_idx": 1, "matched": false, "rationale": "..."}\n',
        encoding="utf-8",
    )
    run = load_judge_run(p)
    assert run.run_id == "judge_1"
    assert run.verdicts == {0: True, 1: False}


# --- CLI smoke ------------------------------------------------------------

def test_cli_emits_json_and_exits_zero_when_calibrated(tmp_path: Path) -> None:
    # Build 3 perfectly-agreeing judge files + baseline.
    for i in range(1, 4):
        (tmp_path / f"judge_{i}.jsonl").write_text(
            "\n".join(
                json.dumps({"turn_idx": t, "matched": True})
                for t in range(10)
            ) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "baseline.jsonl").write_text(
        "\n".join(
            json.dumps({"turn_idx": t, "matched": False})  # substring misses
            for t in range(10)
        ) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "judge_kappa.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.context_rebuilder.kappa",
            "--runs",
            str(tmp_path / "judge_1.jsonl"),
            str(tmp_path / "judge_2.jsonl"),
            str(tmp_path / "judge_3.jsonl"),
            "--baseline", str(tmp_path / "baseline.jsonl"),
            "--out", str(out),
            "--run-id", "cli_smoke",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["calibrated"] is True
    assert payload["run_id"] == "cli_smoke"
    assert payload["n_runs"] == 3
    assert payload["n_pairs"] == 10


def test_cli_exits_nonzero_when_not_calibrated(tmp_path: Path) -> None:
    # 2 runs only → fails MIN_RUNS gate → exit 1.
    for i in range(1, 3):
        (tmp_path / f"judge_{i}.jsonl").write_text(
            json.dumps({"turn_idx": 0, "matched": True}) + "\n",
            encoding="utf-8",
        )
    out = tmp_path / "judge_kappa.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.context_rebuilder.kappa",
            "--runs",
            str(tmp_path / "judge_1.jsonl"),
            str(tmp_path / "judge_2.jsonl"),
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["calibrated"] is False
    assert any("n_runs" in r for r in payload["failure_reasons"])
