"""Tests for benchmarks.tolerance — relative-with-floor band classifier.

Spec: docs/v2_reproducibility_harness.md (ratified 2026-05-06).
Issue: #437.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import tolerance
from benchmarks.tolerance import Verdict


def test_relative_band_for_f1():
    lower, upper, kind = tolerance.compute_band("f1_avg", 0.5)
    assert kind == "relative"
    assert lower == pytest.approx(0.465)  # 0.5 - 7%
    assert upper == pytest.approx(0.535)


def test_absolute_floor_kicks_in_for_tiny_values():
    """0.001 * 7% = 0.00007, well below 0.02 floor → absolute band."""
    lower, upper, kind = tolerance.compute_band("f1_avg", 0.001)
    assert kind == "absolute"
    assert upper - lower == pytest.approx(0.04)


def test_latency_uses_25pct_band():
    lower, upper, kind = tolerance.compute_band("median_latency_ms", 100.0)
    assert kind == "relative"
    assert (upper - lower) / 2 == pytest.approx(25.0)


def test_unknown_metric_uses_fallback():
    lower, upper, kind = tolerance.compute_band("retrieved_token_count", 100.0)
    assert kind == "relative"
    assert (upper - lower) / 2 == pytest.approx(10.0)  # fallback 10%


def test_override_takes_precedence():
    lower, upper, kind = tolerance.compute_band(
        "f1_avg", 0.5, overrides={"f1_avg": 0.20},
    )
    assert kind == "override"
    assert (upper - lower) / 2 == pytest.approx(0.10)


def test_classify_pass_inside_band():
    v, _ = tolerance.classify(0.5, 0.51, 0.465, 0.535)
    assert v == Verdict.PASS


def test_classify_warn_at_60pct_drift():
    """Drift >50% of band half-width → warn."""
    # band is [0.465, 0.535], half=0.035, 60% drift = 0.021 from canonical
    v, note = tolerance.classify(0.5, 0.523, 0.465, 0.535)
    assert v == Verdict.WARN
    assert "drift" in note


def test_classify_fail_outside_band():
    v, note = tolerance.classify(0.5, 0.6, 0.465, 0.535)
    assert v == Verdict.FAIL
    assert "outside band" in note


def test_classify_zero_width_band():
    v, _ = tolerance.classify(0.5, 0.5, 0.5, 0.5)
    assert v == Verdict.PASS
    v, _ = tolerance.classify(0.5, 0.500001, 0.5, 0.5)
    assert v == Verdict.FAIL


def _canonical(results: dict) -> dict:
    return {
        "schema_version": 2,
        "label": "test canonical",
        "captured_at_utc": "2026-05-06T00:00:00Z",
        "git_commit": "deadbeef",
        "aelfrice_version": "2.0.0",
        "harness_version": "1",
        "headline_cut": {},
        "results": results,
    }


def test_check_report_walks_nested_leaves():
    cano = _canonical({"mab": {"Conflict_Resolution": {"f1_avg": 0.5}}})
    obs = _canonical({"mab": {"Conflict_Resolution": {"f1_avg": 0.51}}})
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].path == ("mab", "Conflict_Resolution", "f1_avg")
    assert checks[0].verdict == Verdict.PASS


def test_check_report_missing_leaf_is_fail():
    cano = _canonical({"mab": {"split_a": {"f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {}}})
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].verdict == Verdict.FAIL
    assert "no leaf" in checks[0].note


def test_check_report_extra_leaves_ignored():
    cano = _canonical({"mab": {"split_a": {"f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"f1": 0.5, "extra": 0.99}}})
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1


def test_check_report_skips_underscore_prefixed_keys():
    """`_status`, `_elapsed_sec`, etc. are metadata, not metrics."""
    cano = _canonical({"mab": {"_status": "ok", "f1": 0.5}})
    obs = _canonical({"mab": {"_status": "ok", "f1": 0.51}})
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].path == ("mab", "f1")


def test_summarize_fail_dominates():
    cano = _canonical({"mab": {"split_a": {"f1": 0.5, "exact_match": 0.3}}})
    obs = _canonical({"mab": {"split_a": {"f1": 0.51, "exact_match": 0.99}}})
    checks = tolerance.check_report(cano, obs)
    overall, counts = tolerance.summarize(checks)
    assert overall == Verdict.FAIL
    assert counts[Verdict.FAIL.value] == 1
    assert counts[Verdict.PASS.value] == 1


def test_summarize_warn_when_no_fail():
    cano = _canonical({"mab": {"split_a": {"f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"f1": 0.523}}})
    checks = tolerance.check_report(cano, obs)
    overall, _ = tolerance.summarize(checks)
    assert overall == Verdict.WARN


def test_load_report_rejects_wrong_schema(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"schema_version": 1, "results": {}}))
    with pytest.raises(ValueError, match="schema_version=2"):
        tolerance.load_report(p)


def test_load_report_accepts_v2(tmp_path):
    p = tmp_path / "new.json"
    p.write_text(json.dumps({"schema_version": 2, "results": {}}))
    data = tolerance.load_report(p)
    assert data["schema_version"] == 2
