"""Tests for benchmarks.tolerance — relative-with-floor band classifier.

Spec: docs/design/v2_reproducibility_harness.md (ratified 2026-05-06).
Issue: #437.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import tolerance
from benchmarks.metric_status import NOT_APPLICABLE
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
    # band is [0.465, 0.535], half=0.035, ~66% drift = 0.023 from canonical
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


def test_walk_leaves_recurses_into_bare_underscore_sentinel():
    """`benchmarks/run.py:241` uses `"_"` as the sub-bucket for
    single-invocation adapters (locomo, longmemeval, amabench).
    The walker must descend into `_` so `output.*` leaves are
    visible to the band-checker. Metadata like `_status` nested
    inside `_` is still skipped by the recursive re-filter.
    """
    obj = {
        "locomo": {
            "_": {
                "_status": "ok",
                "_elapsed_sec": 0.5,
                "output": {"overall_f1": 0.0212, "avg_latency_ms": 5.55},
            },
        },
    }
    leaves = dict(tolerance._walk_leaves(obj))
    assert ("locomo", "_", "output", "overall_f1") in leaves
    assert leaves[("locomo", "_", "output", "overall_f1")] == pytest.approx(0.0212)
    assert ("locomo", "_", "output", "avg_latency_ms") in leaves
    # _status / _elapsed_sec under `_` must still be skipped.
    assert not any("_status" in p for p, _ in leaves.items())
    assert not any("_elapsed_sec" in p for p, _ in leaves.items())


def test_check_report_band_checks_single_invocation_adapter():
    """Without the `_` sentinel fix, `_walk_leaves` skipped the
    bucket and `output.*` leaves were never band-checked. Regression
    case from #490: locomo / longmemeval / amabench could silently
    pass on a 100% latency regression.
    """
    cano = _canonical({
        "longmemeval": {"_": {"_status": "ok", "output": {"avg_latency_ms": 5.0}}},
    })
    # 100% latency regression — far outside the 25% band for latency
    obs = _canonical({
        "longmemeval": {"_": {"_status": "ok", "output": {"avg_latency_ms": 10.0}}},
    })
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].path == ("longmemeval", "_", "output", "avg_latency_ms")
    assert checks[0].verdict == Verdict.FAIL


def test_check_report_passes_inside_band_for_single_invocation_adapter():
    """Mirror of the FAIL case — confirms band classification works
    correctly through the `_` sentinel, not just that the path is
    visible.
    """
    cano = _canonical({
        "amabench": {"_": {"_status": "ok", "output": {"total_qa": 100.0}}},
    })
    obs = _canonical({
        "amabench": {"_": {"_status": "ok", "output": {"total_qa": 100.5}}},
    })
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].path == ("amabench", "_", "output", "total_qa")
    assert checks[0].verdict == Verdict.PASS


def test_summarize_fail_dominates():
    # exact_match regresses 0.3 -> 0.01 while f1 holds. It used to
    # *improve* to 0.99 here, which failed only because bands were
    # two-sided (#1160); the rollup precedence being tested is
    # unchanged, so the leaf now regresses for real.
    cano = _canonical({"mab": {"split_a": {"f1": 0.5, "exact_match": 0.3}}})
    obs = _canonical({"mab": {"split_a": {"f1": 0.51, "exact_match": 0.01}}})
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


def test_check_report_reads_overrides_from_canonical():
    """metric_overrides in canonical JSON applies when caller doesn't override."""
    cano = _canonical({"mab": {"split_a": {"f1_avg": 0.5}}})
    cano["metric_overrides"] = {"f1_avg": 0.20}  # ±20% → band [0.40, 0.60]
    # Without override, ±7% → band [0.465, 0.535] — observed 0.58 is OUT.
    # With ±20% → observed 0.58 is in-band, drift 80% of half-width → WARN.
    obs = _canonical({"mab": {"split_a": {"f1_avg": 0.58}}})
    checks = tolerance.check_report(cano, obs)
    assert checks[0].band_kind == "override"
    assert checks[0].verdict == Verdict.WARN  # in-band but high-drift
    # Confirm without the override it would have FAILed.
    cano2 = _canonical({"mab": {"split_a": {"f1_avg": 0.5}}})
    checks2 = tolerance.check_report(cano2, obs)
    assert checks2[0].verdict == Verdict.FAIL


def test_explicit_overrides_take_precedence_over_canonical():
    cano = _canonical({"mab": {"split_a": {"f1_avg": 0.5}}})
    cano["metric_overrides"] = {"f1_avg": 0.20}
    obs = _canonical({"mab": {"split_a": {"f1_avg": 0.58}}})
    # Caller passes a tighter override (5%) — should override the canonical 20%.
    checks = tolerance.check_report(cano, obs, metric_overrides={"f1_avg": 0.05})
    assert checks[0].verdict == Verdict.FAIL


# --- one-sided bands (#1160) -------------------------------------------


def test_direction_defaults_to_two_sided_for_unknown_metric():
    """Unclassified metrics must stay two-sided.

    A metric given the wrong direction goes blind to regressions in
    its real direction, so the default has to be conservative — a new
    metric fails loudly until someone classifies it.
    """
    assert tolerance.direction_for(("x", "never_seen_before")) is (
        tolerance.Direction.TWO_SIDED
    )
    assert tolerance.direction_for(()) is tolerance.Direction.TWO_SIDED


def test_direction_falls_back_to_parent_for_bucketed_metrics():
    """LoCoMo per-category F1 keys its leaves by bucket id, not name."""
    assert tolerance.direction_for(
        ("locomo", "_", "output", "category_f1", "5")
    ) is tolerance.Direction.HIGHER_IS_BETTER
    assert tolerance.direction_for(
        ("amabench", "_", "output", "type_counts", "A")
    ) is tolerance.Direction.TWO_SIDED


def test_direction_leaf_wins_over_parent():
    """`count.correct` is a score; `temporal-reasoning.count` is a size."""
    assert tolerance.direction_for(
        ("structmemeval", "x", "output", "count", "correct")
    ) is tolerance.Direction.HIGHER_IS_BETTER
    assert tolerance.direction_for(
        ("longmemeval", "_", "output", "temporal-reasoning", "count")
    ) is tolerance.Direction.TWO_SIDED


def test_improvement_beyond_band_warns_not_fails():
    """The defect #1160 names: a real win registered as FAIL.

    Canonical `exact_match = 0.0` with the ±0.02 absolute floor puts
    every improvement outside the band.
    """
    lower, upper, _ = tolerance.compute_band("exact_match", 0.0)
    v, note = tolerance.classify(
        0.0, 0.25, lower, upper,
        direction=tolerance.Direction.HIGHER_IS_BETTER,
    )
    assert v is Verdict.WARN
    assert "improving side" in note
    # Two-sided is the pre-#1160 behaviour and must be unchanged.
    v_two, _ = tolerance.classify(0.0, 0.25, lower, upper)
    assert v_two is Verdict.FAIL


def test_regression_still_fails_on_a_one_sided_metric():
    """One-sided must not mean unguarded."""
    lower, upper, _ = tolerance.compute_band("f1", 0.5)
    v, _ = tolerance.classify(
        0.5, 0.1, lower, upper,
        direction=tolerance.Direction.HIGHER_IS_BETTER,
    )
    assert v is Verdict.FAIL


def test_latency_direction_is_inverted():
    """For cost metrics the regression is the rise, not the drop."""
    lower, upper, _ = tolerance.compute_band("avg_latency_ms", 100.0)
    slower, _ = tolerance.classify(
        100.0, 400.0, lower, upper,
        direction=tolerance.Direction.LOWER_IS_BETTER,
    )
    faster, _ = tolerance.classify(
        100.0, 10.0, lower, upper,
        direction=tolerance.Direction.LOWER_IS_BETTER,
    )
    assert slower is Verdict.FAIL
    assert faster is Verdict.WARN


def test_check_report_applies_direction_end_to_end():
    """The wiring, not just the helper: a LoCoMo cat-5 fix must not FAIL."""
    cano = {"results": {"locomo": {"_": {"output": {
        "category_f1": {"5": 0.0}, "avg_latency_ms": 100.0,
    }}}}}
    obs = {"results": {"locomo": {"_": {"output": {
        "category_f1": {"5": 0.31}, "avg_latency_ms": 100.0,
    }}}}}
    checks = tolerance.check_report(cano, obs)
    by_path = {c.path[-2:]: c for c in checks}
    cat5 = by_path[("category_f1", "5")]
    assert cat5.verdict is Verdict.WARN
    assert cat5.direction is tolerance.Direction.HIGHER_IS_BETTER
    overall, _ = tolerance.summarize(checks)
    assert overall is Verdict.WARN, "a genuine cat-5 fix must not fail the gate"


def test_corpus_size_drift_still_fails_in_both_directions():
    """Invariants stay two-sided: a shrinking corpus is not an 'improvement'."""
    cano = {"results": {"a": {"_": {"output": {"total_questions": 500}}}}}
    for observed in (250, 900):
        checks = tolerance.check_report(
            cano,
            {"results": {"a": {"_": {"output": {
                "total_questions": observed}}}}},
        )
        assert [c.verdict for c in checks] == [Verdict.FAIL], observed


# ---------------------------------------------------------------------------
# Not-applicable leaves (#1160)
# ---------------------------------------------------------------------------


def test_na_leaf_is_not_applicable_not_fail():
    """The distinguishing assert: without the sentinel branch this is FAIL.

    `"n/a"` is a non-numeric leaf, so it would otherwise land on the
    "observed leaf is not numeric" branch and read as a regression — the
    exact misreport the sentinel exists to prevent.
    """
    cano = _canonical({"mab": {"split_a": {"exact_match": 0.0, "f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"exact_match": NOT_APPLICABLE, "f1": 0.5}}})
    checks = tolerance.check_report(cano, obs)
    by_metric = {c.path[-1]: c for c in checks}
    assert by_metric["exact_match"].verdict == Verdict.NOT_APPLICABLE
    assert by_metric["exact_match"].band_kind == "not_applicable"
    assert by_metric["f1"].verdict == Verdict.PASS


def test_na_does_not_raise_the_rollup():
    """#479's rule for SKIP, applied to n/a: a real PASS still wins."""
    cano = _canonical({"mab": {"split_a": {"exact_match": 0.0, "f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"exact_match": NOT_APPLICABLE, "f1": 0.5}}})
    overall, counts = tolerance.summarize(tolerance.check_report(cano, obs))
    assert overall == Verdict.PASS
    assert counts[Verdict.NOT_APPLICABLE.value] == 1
    assert counts[Verdict.PASS.value] == 1


def test_an_all_na_run_is_no_data_not_pass():
    """Declaring every metric uncomputable is not a green nightly."""
    cano = _canonical({"mab": {"split_a": {"exact_match": 0.0, "f1": 0.5}}})
    obs = _canonical({
        "mab": {"split_a": {"exact_match": NOT_APPLICABLE, "f1": NOT_APPLICABLE}},
    })
    overall, counts = tolerance.summarize(tolerance.check_report(cano, obs))
    assert overall == Verdict.NO_DATA
    assert counts[Verdict.NOT_APPLICABLE.value] == 2


def test_na_is_tallied_even_when_something_else_fails():
    cano = _canonical({"mab": {"split_a": {"exact_match": 0.0, "f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"exact_match": NOT_APPLICABLE, "f1": 0.1}}})
    overall, counts = tolerance.summarize(tolerance.check_report(cano, obs))
    assert overall == Verdict.FAIL
    assert counts[Verdict.NOT_APPLICABLE.value] == 1


def test_a_genuinely_non_numeric_leaf_still_fails():
    """The sentinel is a narrow exemption, not a hole in the numeric check."""
    cano = _canonical({"mab": {"split_a": {"f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"f1": "0.5"}}})
    checks = tolerance.check_report(cano, obs)
    assert checks[0].verdict == Verdict.FAIL
    assert "not numeric" in checks[0].note


def test_rank_metrics_are_one_sided():
    """A ranking win must not fail the nightly it was meant to show up in."""
    for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20"):
        assert tolerance.direction_for(("locomo", "retrieval_quality", metric)) == (
            tolerance.Direction.HIGHER_IS_BETTER
        ), metric
