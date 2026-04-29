"""Tests for the posterior-ranking eval harness (issue #151, Slice 1).

Covers MRR uplift, ECE calibration scorers, the runner, default fixture corpus,
and CLI integration.

All tests are deterministic (fixed seeds), use in-memory stores, and
must complete in < 1.5 seconds total.

Test style mirrors tests/test_bayesian_ranking.py and tests/test_benchmarks_dir.py.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_fixture(
    *,
    fid: str = "t1",
    query: str = "asyncio coroutine scheduler",
    known: str = "asyncio coroutine scheduler runs tasks cooperatively",
    noise: list[str] | None = None,
) -> dict[str, object]:
    """A fixture designed so noise items initially outrank the known item.

    Noise items repeat every query term multiple times to get higher BM25
    scores at baseline.  After repeated positive feedback on the known item,
    its posterior rises and the partial_bayesian_score should climb.
    """
    if noise is None:
        noise = [
            # Heavy query-term overlap to score high at BM25 baseline.
            "asyncio coroutine asyncio coroutine scheduler asyncio scheduler",
            "asyncio scheduler coroutine asyncio coroutine scheduler tasks",
            "coroutine scheduler asyncio asyncio coroutine scheduler tasks",
            "asyncio asyncio asyncio coroutine coroutine scheduler tasks io",
        ]
    return {"id": fid, "query": query, "known_belief_content": known, "noise_belief_contents": noise}


# ---------------------------------------------------------------------------
# MRR uplift — smoke test
# ---------------------------------------------------------------------------


def test_mrr_uplift_smoke() -> None:
    """1-fixture, fixed seed, n_seeds=1: mrr_0 < mrr_10, uplift > 0, passed=True."""
    from benchmarks.posterior_ranking.mrr_uplift import run_single_seed

    fx = _minimal_fixture()
    result = run_single_seed([fx], seed=42, threshold=0.05)

    assert result.mrr_0 >= 0.0
    assert result.mrr_10 > 0.0, "known item must appear in top-K after feedback"
    assert result.mrr_uplift > 0.0, "feedback must improve MRR"
    assert result.passed is True
    assert len(result.mrr_per_round) == 10
    assert result.seed == 42


def test_mrr_uplift_result_fields() -> None:
    """MRRUpliftResult has expected shape and mrr_10 property."""
    from benchmarks.posterior_ranking.mrr_uplift import MRRUpliftResult

    r = MRRUpliftResult(
        mrr_0=0.3,
        mrr_per_round=[0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40],
        mrr_uplift=0.10,
        seed=0,
        pass_threshold=0.05,
        passed=True,
    )
    assert r.mrr_10 == pytest.approx(0.40)
    assert r.mrr_uplift == pytest.approx(0.10)
    assert r.passed is True


# ---------------------------------------------------------------------------
# MRR uplift — regression detection
# ---------------------------------------------------------------------------


def test_mrr_uplift_regression_detection() -> None:
    """passed=False when all rounds fall below regression floor."""
    from benchmarks.posterior_ranking.mrr_uplift import (
        REGRESSION_FLOOR_DELTA,
        MRRUpliftResult,
    )

    mrr_0 = 0.5
    floor = mrr_0 - REGRESSION_FLOOR_DELTA
    bad_rounds = [floor - 0.05] * 10
    uplift = bad_rounds[-1] - mrr_0

    r = MRRUpliftResult(
        mrr_0=mrr_0,
        mrr_per_round=bad_rounds,
        mrr_uplift=uplift,
        seed=0,
        pass_threshold=0.05,
        passed=False,
    )
    assert r.passed is False


def test_mrr_uplift_poor_threshold_fails() -> None:
    """uplift < impossibly high threshold -> passed == False."""
    from benchmarks.posterior_ranking.mrr_uplift import run_single_seed

    fx = _minimal_fixture()
    result = run_single_seed([fx], seed=0, threshold=0.99)
    assert result.passed is False
    assert result.mrr_uplift < 0.99


# ---------------------------------------------------------------------------
# MRR multi-seed reproducibility
# ---------------------------------------------------------------------------


def test_mrr_multi_seed_shape() -> None:
    """n_seeds=5: MultiSeedReport has correct shape and deterministic seeds."""
    from benchmarks.posterior_ranking.mrr_uplift import run_multi_seed

    fx = _minimal_fixture()
    report = run_multi_seed([fx], n_seeds=5, threshold=0.05, base_seed=10)

    assert len(report.results) == 5
    seeds = [r.seed for r in report.results]
    assert seeds == [10, 11, 12, 13, 14]
    assert report.uplift_lo <= report.mean_uplift <= report.uplift_hi
    assert report.std_uplift >= 0.0


def test_mrr_multi_seed_deterministic() -> None:
    """Same base_seed produces identical uplift values on repeated calls."""
    from benchmarks.posterior_ranking.mrr_uplift import run_multi_seed

    fx = _minimal_fixture()
    r1 = run_multi_seed([fx], n_seeds=3, threshold=0.05, base_seed=7)
    r2 = run_multi_seed([fx], n_seeds=3, threshold=0.05, base_seed=7)

    for a, b in zip(r1.results, r2.results):
        assert a.mrr_uplift == pytest.approx(b.mrr_uplift)
        assert a.mrr_0 == pytest.approx(b.mrr_0)


def test_mrr_multi_seed_band_formula() -> None:
    """±2σ band: lo = mean - 2*std, hi = mean + 2*std."""
    from benchmarks.posterior_ranking.mrr_uplift import run_multi_seed

    fx = _minimal_fixture()
    report = run_multi_seed([fx], n_seeds=5, threshold=0.05, base_seed=0)

    expected_lo = report.mean_uplift - 2.0 * report.std_uplift
    expected_hi = report.mean_uplift + 2.0 * report.std_uplift
    assert report.uplift_lo == pytest.approx(expected_lo, abs=1e-10)
    assert report.uplift_hi == pytest.approx(expected_hi, abs=1e-10)


# ---------------------------------------------------------------------------
# ECE — smoke test (well-calibrated)
# ---------------------------------------------------------------------------


def test_ece_smoke_well_calibrated() -> None:
    """Well-calibrated stream: predicted ~ actual -> ECE < 0.10, passed=True."""
    from benchmarks.posterior_ranking.ece import compute_ece

    # alpha=5, beta=5 -> posterior_mean = 0.5; half actually positive.
    triples: list[tuple[float, float, float]] = []
    for i in range(100):
        actual = 1.0 if i % 2 == 0 else 0.0
        triples.append((5.0, 5.0, actual))

    result = compute_ece(triples, threshold=0.10)
    assert result.ece < 0.10
    assert result.passed is True
    assert result.n_total == 100


def test_ece_smoke_poorly_calibrated() -> None:
    """Poorly calibrated: predicted ~0.9 but actual rate 0.1 -> ECE > 0.10."""
    from benchmarks.posterior_ranking.ece import compute_ece

    triples: list[tuple[float, float, float]] = []
    for i in range(100):
        actual = 1.0 if i < 10 else 0.0
        triples.append((9.0, 1.0, actual))

    result = compute_ece(triples, threshold=0.10)
    assert result.ece > 0.10
    assert result.passed is False


def test_ece_bucket_sanity() -> None:
    """10 buckets, no NaN, sum of bucket weights == 1.0."""
    from benchmarks.posterior_ranking.ece import N_BUCKETS, compute_ece

    triples: list[tuple[float, float, float]] = []
    for i in range(50):
        alpha = float(i + 1)
        beta = float(50 - i + 1)
        actual = 1.0 if i % 3 == 0 else 0.0
        triples.append((alpha, beta, actual))

    result = compute_ece(triples)

    assert len(result.buckets) == N_BUCKETS

    for b in result.buckets:
        assert not math.isnan(b.mean_predicted), f"NaN in bucket {b.bucket_idx}"
        assert not math.isnan(b.mean_actual), f"NaN in bucket {b.bucket_idx}"
        assert not math.isnan(b.weight), f"NaN weight in bucket {b.bucket_idx}"

    total_weight = sum(b.weight for b in result.buckets)
    assert total_weight == pytest.approx(1.0, abs=1e-10)


def test_ece_empty_observations() -> None:
    """Empty observation list returns ECE=0 and passed=True."""
    from benchmarks.posterior_ranking.ece import compute_ece

    result = compute_ece([], threshold=0.10)
    assert result.ece == 0.0
    assert result.passed is True
    assert result.n_total == 0


# ---------------------------------------------------------------------------
# Helpers for runner / CLI tests
# ---------------------------------------------------------------------------


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _write_fixtures(tmp_path: Path, fixtures: list[dict[str, object]]) -> Path:
    fpath = tmp_path / "fixtures.jsonl"
    with fpath.open("w", encoding="utf-8") as fh:
        for fx in fixtures:
            fh.write(json.dumps(fx) + "\n")
    return fpath


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_runner_integration_clean(tmp_path: Path) -> None:
    """3-fixture file through run(): both MRR and ECE keys present, correct types."""
    from benchmarks.posterior_ranking.run import run

    # All three fixtures use noise items with high BM25 overlap so the
    # known item starts below rank 1, giving room for feedback to lift it.
    fixtures = [
        _minimal_fixture(fid="r1"),
        _minimal_fixture(
            fid="r2",
            query="SQLite WAL concurrent readers",
            known="SQLite WAL concurrent readers read without blocking",
            noise=[
                "SQLite WAL concurrent SQLite WAL concurrent readers WAL readers",
                "WAL readers concurrent SQLite WAL readers SQLite concurrent",
                "concurrent WAL SQLite WAL readers concurrent WAL concurrent",
                "SQLite SQLite WAL WAL concurrent readers readers concurrent readers",
            ],
        ),
        _minimal_fixture(
            fid="r3",
            query="Beta distribution conjugate prior",
            known="Beta distribution conjugate prior updates with observations",
            noise=[
                "Beta distribution conjugate Beta distribution conjugate prior Beta",
                "conjugate prior Beta distribution Beta conjugate distribution prior",
                "Beta Beta distribution distribution conjugate conjugate prior prior",
                "distribution prior Beta conjugate Beta distribution conjugate prior",
            ],
        ),
    ]
    fpath = _write_fixtures(tmp_path, fixtures)

    result = run(fpath, n_seeds=1, base_seed=0)

    assert "mrr" in result
    assert "ece" in result
    assert "overall_pass" in result

    from benchmarks.posterior_ranking.mrr_uplift import MultiSeedReport
    from benchmarks.posterior_ranking.ece import ECEResult

    assert isinstance(result["mrr"], MultiSeedReport)
    assert isinstance(result["ece"], ECEResult)
    assert isinstance(result["overall_pass"], bool)


def test_runner_as_dict(tmp_path: Path) -> None:
    """run_as_dict returns serializable plain dicts."""
    from benchmarks.posterior_ranking.run import run_as_dict

    fixtures = [_minimal_fixture()]
    fpath = _write_fixtures(tmp_path, fixtures)

    result = run_as_dict(fpath, n_seeds=1, base_seed=0)

    json_str = json.dumps(result)
    parsed = json.loads(json_str)

    assert "mrr" in parsed
    assert "ece" in parsed
    assert "overall_pass" in parsed


# ---------------------------------------------------------------------------
# Default fixtures file exists and is valid
# ---------------------------------------------------------------------------


def test_default_fixtures_file_exists_and_readable() -> None:
    """The shipped default.jsonl has >= 5 entries and all parse correctly."""
    default_path = (
        Path(__file__).parent.parent
        / "benchmarks"
        / "posterior_ranking"
        / "fixtures"
        / "default.jsonl"
    )
    assert default_path.is_file(), f"default fixtures file missing: {default_path}"

    entries: list[dict[str, object]] = []
    with default_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries.append(entry)

    assert len(entries) >= 5, f"expected >= 5 fixtures, got {len(entries)}"

    required_keys = {"id", "query", "known_belief_content", "noise_belief_contents"}
    for i, entry in enumerate(entries):
        missing = required_keys - set(entry.keys())
        assert not missing, f"fixture {i} missing keys: {missing}"
        assert isinstance(entry["noise_belief_contents"], list)
        assert len(entry["noise_belief_contents"]) >= 1


def test_load_fixtures(tmp_path: Path) -> None:
    """load_fixtures reads JSONL correctly."""
    from benchmarks.posterior_ranking.mrr_uplift import load_fixtures

    fixtures = [_minimal_fixture(fid=f"f{i}") for i in range(3)]
    fpath = _write_fixtures(tmp_path, fixtures)

    loaded = load_fixtures(fpath)
    assert len(loaded) == 3
    assert loaded[0]["id"] == "f0"
