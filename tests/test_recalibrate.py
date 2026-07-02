"""Tests for benchmarks/recalibrate.py — adapter scorer drift check.

Split into two groups:

1. Framework tests — no external dependencies.  These exercise the band
   check primitive and the broken-adapter regression proof.  They run in
   any Python environment (no [benchmarks] extras needed).

2. Adapter oracle tests — verify real scorer functions against the pinned
   oracle fixtures.  The locomo test needs nltk (benchmarks extras);
   the mab test uses benchmarks.qa_scoring (pure stdlib, no extras).
"""
from __future__ import annotations

import pytest

from benchmarks import recalibrate


# ---------------------------------------------------------------------------
# Framework tests — no extras needed
# ---------------------------------------------------------------------------


class TestBandCheck:
    """Unit tests for the _check_oracle_tuple primitive."""

    def test_pass_exact_lower_bound(self) -> None:
        result = recalibrate._check_oracle_tuple(
            adapter="x", label="lbl", observed=0.4,
            expected_lower=0.4, expected_upper=0.6,
        )
        assert result is None

    def test_pass_exact_upper_bound(self) -> None:
        result = recalibrate._check_oracle_tuple(
            adapter="x", label="lbl", observed=0.6,
            expected_lower=0.4, expected_upper=0.6,
        )
        assert result is None

    def test_pass_midpoint(self) -> None:
        result = recalibrate._check_oracle_tuple(
            adapter="x", label="lbl", observed=0.5,
            expected_lower=0.4, expected_upper=0.6,
        )
        assert result is None

    def test_fail_below_lower(self) -> None:
        fail = recalibrate._check_oracle_tuple(
            adapter="mytest", label="too low", observed=0.3,
            expected_lower=0.4, expected_upper=0.6,
        )
        assert fail is not None
        assert fail.adapter == "mytest"
        assert fail.label == "too low"
        assert fail.observed == pytest.approx(0.3)

    def test_fail_above_upper(self) -> None:
        fail = recalibrate._check_oracle_tuple(
            adapter="mytest", label="too high", observed=0.7,
            expected_lower=0.4, expected_upper=0.6,
        )
        assert fail is not None
        assert fail.observed == pytest.approx(0.7)

    def test_point_band_exact_match(self) -> None:
        result = recalibrate._check_oracle_tuple(
            adapter="x", label="lbl", observed=1.0,
            expected_lower=1.0, expected_upper=1.0,
        )
        assert result is None

    def test_point_band_miss(self) -> None:
        fail = recalibrate._check_oracle_tuple(
            adapter="x", label="lbl", observed=0.99,
            expected_lower=1.0, expected_upper=1.0,
        )
        assert fail is not None


class TestBrokenAdapterCaught:
    """Regression test: a deliberately broken scorer is caught by the oracle.

    This proves the mechanism works — if recalibrate.py can be tricked into
    reporting PASS for a broken scorer, the whole guard is useless.
    """

    def test_broken_locomo_scorer_caught(self) -> None:
        """Scorer that always returns 0.0 is caught by locomo oracle tuples
        that expect non-zero scores (e.g., exact matches → 1.0).
        """
        def always_zero(prediction: str, ground_truth: str, category: int) -> float:
            return 0.0

        fixture = recalibrate.load_fixture("locomo")
        fails = recalibrate._run_locomo_oracle(fixture["oracles"], scorer_fn=always_zero)

        # Oracle has tuples with expected_lower=1.0 (exact matches)
        # The broken scorer returns 0.0 for all of them → must be caught.
        assert len(fails) > 0, (
            "Broken scorer (always 0.0) was not caught by locomo oracle. "
            "The recalibration mechanism is not working."
        )
        # Every failure should have observed=0.0
        for f in fails:
            assert f.observed == pytest.approx(0.0)

    def test_broken_locomo_scorer_labels_reported(self) -> None:
        """Failure labels are present and readable in the OracleFail objects."""
        def always_zero(prediction: str, ground_truth: str, category: int) -> float:
            return 0.0

        fixture = recalibrate.load_fixture("locomo")
        fails = recalibrate._run_locomo_oracle(fixture["oracles"], scorer_fn=always_zero)
        labels = {f.label for f in fails}
        # At minimum the exact-match cases (expected 1.0) should appear.
        assert any("perfect match" in lbl or "exact" in lbl.lower() for lbl in labels), (
            f"Expected exact-match label in failures; got: {labels}"
        )

    def test_broken_mab_sem_scorer_caught(self) -> None:
        """Scorer that always returns 0.0 is caught for MAB SEM oracle tuples
        that expect 1.0 (substring match hits).
        """
        def always_zero(prediction: str, ground_truth: str) -> float:
            return 0.0

        fixture = recalibrate.load_fixture("mab")
        # Only simple (non-multi_answer) scorers go through scorer_fn injection.
        # Filter to those tuples to keep assertion clean.
        simple_oracles = [
            o for o in fixture["oracles"] if o["scorer_fn"] != "score_multi_answer"
        ]
        fails = recalibrate._run_mab_oracle(simple_oracles, scorer_fn=always_zero)

        # Oracle has sem/em/f1 hit cases with expected_lower=1.0
        assert len(fails) > 0, (
            "Broken scorer (always 0.0) was not caught by mab oracle. "
            "The recalibration mechanism is not working."
        )

    def test_correct_scorer_passes_all_locomo_oracles(self) -> None:
        """Identity-with-locomo scorer: any scorer that agrees with score_qa
        on the oracle inputs should produce zero failures.

        This sub-test uses a lambda that delegates to the real scorer to
        verify the pass path of the framework without importing nltk directly
        (the import is inside the lambda, which is evaluated lazily; if nltk
        is absent this specific assertion body is also unreachable — but the
        outer test_locomo_oracle_all_pass covers the production path with a
        proper importorskip).
        """
        # If nltk is not available, skip this cross-check.
        nltk = pytest.importorskip("nltk", reason="[benchmarks] extras needed for locomo scorer")
        _ = nltk  # only needed to trigger the skip if absent

        from benchmarks.locomo_adapter import score_qa

        def correct_scorer(prediction: str, ground_truth: str, category: int) -> float:
            return score_qa(prediction, ground_truth, category)

        fixture = recalibrate.load_fixture("locomo")
        fails = recalibrate._run_locomo_oracle(fixture["oracles"], scorer_fn=correct_scorer)
        assert fails == [], f"Correct scorer produced unexpected failures: {fails}"


# ---------------------------------------------------------------------------
# Real adapter oracle tests
# ---------------------------------------------------------------------------


def test_locomo_oracle_all_pass() -> None:
    """Real LoCoMo scorer passes all pinned oracle tuples."""
    pytest.importorskip("nltk", reason="[benchmarks] extras required for LoCoMo scorer")
    fails = recalibrate.run_adapter_oracles("locomo")
    assert fails == [], (
        "LoCoMo scorer oracle failures detected:\n"
        + "\n".join(str(f) for f in fails)
    )


def test_mab_oracle_all_pass() -> None:
    """Real MAB/qa_scoring scorer passes all pinned oracle tuples.

    qa_scoring.py is pure stdlib — no extras needed.
    """
    fails = recalibrate.run_adapter_oracles("mab")
    assert fails == [], (
        "MAB scorer oracle failures detected:\n"
        + "\n".join(str(f) for f in fails)
    )


def test_run_all_oracles_returns_no_fails() -> None:
    """run_all_oracles() passes on the real code-base (requires benchmarks extras)."""
    pytest.importorskip("nltk", reason="[benchmarks] extras required")
    fails = recalibrate.run_all_oracles()
    assert fails == [], (
        "Scorer oracle failures from run_all_oracles():\n"
        + "\n".join(str(f) for f in fails)
    )


# ---------------------------------------------------------------------------
# Registry / fixture-loading sanity
# ---------------------------------------------------------------------------


def test_all_registered_fixtures_exist() -> None:
    """Every entry in ADAPTER_FIXTURES points to an existing JSON file."""
    for name, fname in recalibrate.ADAPTER_FIXTURES.items():
        path = recalibrate._ORACLE_DIR / fname
        assert path.exists(), (
            f"Oracle fixture for adapter {name!r} not found: {path}"
        )


def test_fixture_schema_has_required_fields() -> None:
    """Each oracle fixture has adapter, scorer, description, and oracles fields."""
    for name in recalibrate.ADAPTER_FIXTURES:
        data = recalibrate.load_fixture(name)
        for field in ("adapter", "scorer", "description", "oracles"):
            assert field in data, (
                f"Fixture {name!r} missing required field {field!r}"
            )
        assert isinstance(data["oracles"], list), (
            f"Fixture {name!r} 'oracles' must be a list"
        )
        assert len(data["oracles"]) > 0, (
            f"Fixture {name!r} has empty oracles list"
        )


def test_each_oracle_tuple_has_band() -> None:
    """Each oracle tuple has expected_lower, expected_upper, and label."""
    for name in recalibrate.ADAPTER_FIXTURES:
        data = recalibrate.load_fixture(name)
        for i, oracle in enumerate(data["oracles"]):
            for field in ("label", "expected_lower", "expected_upper"):
                assert field in oracle, (
                    f"Fixture {name!r} oracle[{i}] missing field {field!r}"
                )
            lo = float(oracle["expected_lower"])
            hi = float(oracle["expected_upper"])
            assert lo <= hi, (
                f"Fixture {name!r} oracle[{i}] {oracle['label']!r}: "
                f"expected_lower {lo} > expected_upper {hi}"
            )
