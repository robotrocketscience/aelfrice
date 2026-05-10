"""Smoke test for benchmarks/temporal_blend_sweep.py (#487).

Verifies that the sweep harness:
- Runs deterministically on the inline two-row fixture.
- Produces a result table with one row per (task, grid_point) combination.
- Emits valid JSON to the output path.
- Re-running with the same fixture produces identical scores (modulo elapsed_sec).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to sys.path so the harness can import benchmarks.*
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.temporal_blend_sweep import (
    SWEEP_GRID,
    TEMPORAL_TASKS,
    GridPointResult,
    _load_fixture_cases,
    run_grid_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sweep_on_fixture(task: str, tmpdir: str) -> list[GridPointResult]:
    """Run the full sweep grid on the inline fixture for one task."""
    cases = _load_fixture_cases(task)
    assert cases, f"_load_fixture_cases({task!r}) must return at least one case"
    results = []
    for label, hl_seconds in SWEEP_GRID:
        result = run_grid_point(
            cases=cases,
            task=task,
            bench=None,
            half_life_label=label,
            half_life_seconds=hl_seconds,
            db_dir=tmpdir,
        )
        result.is_fixture = True
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sweep_grid_has_required_points() -> None:
    """SWEEP_GRID must include the 7 finite values plus the inf sentinel."""
    labels = {label for label, _ in SWEEP_GRID}
    required = {"1h", "6h", "24h", "3d", "7d", "14d", "30d", "inf"}
    assert required.issubset(labels), f"Missing grid labels: {required - labels}"


def test_sweep_grid_inf_sentinel_is_none() -> None:
    """The inf entry must have half_life_seconds=None (no-decay path)."""
    inf_entry = next(
        ((label, hl) for label, hl in SWEEP_GRID if label == "inf"), None
    )
    assert inf_entry is not None, "SWEEP_GRID must contain an 'inf' entry"
    assert inf_entry[1] is None, "inf entry must have half_life_seconds=None"


def test_fixture_loader_returns_cases_for_location() -> None:
    """_load_fixture_cases returns at least one case for the location task."""
    cases = _load_fixture_cases("location")
    assert len(cases) >= 1


def test_fixture_loader_returns_cases_for_accounting() -> None:
    """_load_fixture_cases returns at least one case for the accounting task."""
    cases = _load_fixture_cases("accounting")
    assert len(cases) >= 1


def test_sweep_produces_one_row_per_grid_point(tmp_path: Path) -> None:
    """run_grid_point returns results covering all grid points for location."""
    with tempfile.TemporaryDirectory(prefix="aelf_smoke_") as tmpdir:
        results = _run_sweep_on_fixture("location", tmpdir)
    assert len(results) == len(SWEEP_GRID)


def test_sweep_scores_are_deterministic(tmp_path: Path) -> None:
    """Two back-to-back runs on the fixture produce identical scores."""
    with tempfile.TemporaryDirectory(prefix="aelf_smoke_det1_") as d1:
        results1 = _run_sweep_on_fixture("location", d1)
    with tempfile.TemporaryDirectory(prefix="aelf_smoke_det2_") as d2:
        results2 = _run_sweep_on_fixture("location", d2)
    scores1 = [r.score for r in results1]
    scores2 = [r.score for r in results2]
    assert scores1 == scores2, (
        "Sweep scores must be deterministic across runs on the same fixture"
    )


def test_sweep_scores_in_unit_interval(tmp_path: Path) -> None:
    """All score values must be in [0, 1]."""
    with tempfile.TemporaryDirectory(prefix="aelf_smoke_unit_") as tmpdir:
        results = _run_sweep_on_fixture("location", tmpdir)
    for r in results:
        assert 0.0 <= r.score <= 1.0, (
            f"score={r.score} out of [0,1] for half_life={r.half_life_label}"
        )


def test_main_writes_valid_json(tmp_path: Path) -> None:
    """main() writes a valid JSON file with the expected top-level keys."""
    import argparse
    from unittest.mock import patch

    output_path = tmp_path / "sweep_out.json"

    test_args = argparse.Namespace(
        data="/nonexistent/path",  # triggers fixture fallback
        task="location",
        bench="small",
        budget=2000,
        output=str(output_path),
    )

    with patch("argparse.ArgumentParser.parse_args", return_value=test_args):
        from benchmarks.temporal_blend_sweep import main
        main()

    assert output_path.exists(), "main() must write the output JSON file"
    with output_path.open() as f:
        doc = json.load(f)

    for key in ("_notes", "generated_at_utc", "sweep_grid", "tasks", "verdict", "rows"):
        assert key in doc, f"Output JSON missing required key: {key!r}"

    assert len(doc["rows"]) == len(SWEEP_GRID), (
        "Output JSON must have one row per grid point"
    )
    for row in doc["rows"]:
        for field in ("half_life_label", "half_life_seconds", "task", "score", "n_queries"):
            assert field in row, f"Row missing field: {field!r}"
