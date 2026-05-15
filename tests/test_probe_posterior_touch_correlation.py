"""Tests for the posterior-touch correlation probe script.

Covers the pure-stdlib Spearman ρ helper and the verdict-band mapping
at anchor + boundary values. The full script run is operator-time
(requires a real DB) and is not bench-gated.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_posterior_touch_correlation.py"
)


@pytest.fixture(scope="module")
def probe_module():
    """Load the probe script as an importable module."""
    spec = importlib.util.spec_from_file_location(
        "probe_posterior_touch_correlation", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_posterior_touch_correlation"] = module
    spec.loader.exec_module(module)
    return module


class TestSpearmanRho:
    def test_perfect_positive_correlation_returns_one(self, probe_module):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert probe_module.spearman_rho(a, b) == pytest.approx(1.0)

    def test_perfect_negative_correlation_returns_minus_one(self, probe_module):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert probe_module.spearman_rho(a, b) == pytest.approx(-1.0)

    def test_constant_b_returns_zero(self, probe_module):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 5.0, 5.0, 5.0]
        assert probe_module.spearman_rho(a, b) == 0.0

    def test_average_rank_tie_handling(self, probe_module):
        # Two ties in `b` at positions paired with a low and a high `a`.
        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 20.0, 30.0]
        rho = probe_module.spearman_rho(a, b)
        # Ranks: a = [1, 2, 3, 4]; b = [1, 2.5, 2.5, 4].
        # With average-rank ties this is a strong positive but not 1.0.
        assert 0.9 < rho < 1.0

    def test_n_less_than_two_returns_zero(self, probe_module):
        assert probe_module.spearman_rho([], []) == 0.0
        assert probe_module.spearman_rho([1.0], [2.0]) == 0.0

    def test_nonlinear_monotonic_returns_one(self, probe_module):
        # Spearman is rank-based: any monotonic-increasing transform
        # of `a` produces ρ = +1, even when the relationship is highly
        # nonlinear. Pins the behaviour against a future implementation
        # that accidentally drops the rank step.
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 4.0, 27.0, 256.0, 3125.0]  # i ** i
        assert probe_module.spearman_rho(a, b) == 1.0

    def test_length_mismatch_raises(self, probe_module):
        # Silent truncation or IndexError on length-mismatched inputs
        # would mis-report ρ for an asymmetric measurement; raise
        # explicitly so callers cannot reach a wrong-but-finite value.
        with pytest.raises(ValueError, match="length mismatch"):
            probe_module.spearman_rho([1.0, 2.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="length mismatch"):
            probe_module.spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0])


class TestVerdictForRho:
    def test_low_rho_returns_build_pipeline(self, probe_module):
        verdict, implication = probe_module.verdict_for_rho(0.0)
        assert verdict == "BUILD_PIPELINE"
        assert "low" in implication.lower()

    def test_robust_threshold_boundary_is_partial(self, probe_module):
        # ρ = 0.30 exactly is *not* < ROBUST_THRESHOLD; falls into PARTIAL.
        verdict, _ = probe_module.verdict_for_rho(0.30)
        assert verdict == "PARTIAL"

    def test_artifact_threshold_boundary_is_ship_h4(self, probe_module):
        # ρ = 0.60 exactly is *not* < ARTIFACT_THRESHOLD; falls into SHIP_H4_ONLY.
        verdict, _ = probe_module.verdict_for_rho(0.60)
        assert verdict == "SHIP_H4_ONLY"

    def test_high_rho_returns_ship_h4_only(self, probe_module):
        verdict, implication = probe_module.verdict_for_rho(0.87)
        assert verdict == "SHIP_H4_ONLY"
        assert "high" in implication.lower()

    def test_negative_rho_returns_build_pipeline(self, probe_module):
        # A negative correlation is below ROBUST_THRESHOLD too.
        verdict, _ = probe_module.verdict_for_rho(-0.50)
        assert verdict == "BUILD_PIPELINE"


class TestThresholdsCarry:
    def test_thresholds_match_documented_framework(self, probe_module):
        assert probe_module.SPEARMAN_CROSSOVER == 0.60
        assert probe_module.ROBUST_THRESHOLD == 0.30
        assert probe_module.ARTIFACT_THRESHOLD == 0.60
