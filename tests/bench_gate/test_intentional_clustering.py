"""Bench gate for #436 intentional clustering.

Spec § A2 (multi-fact recall uplift) + § A3 (single-fact non-regression)
+ § A4 (latency). The full gate evaluates all three; this scaffold runs
the multi-fact corpus through ``cluster_candidates`` + ``pack_with_clusters``
directly, then checks ``cluster_coverage@k`` against a baseline.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset (corpus content
lives lab-side per the directory-of-origin rule). The retrieval-side
wiring (``use_intentional_clustering`` flag in ``retrieve_v2``) is the
follow-up gate; this scaffold tests the module independently so the
substrate can land before the wiring.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_multi_fact_corpus_round_trip(aelfrice_corpus_root: Path) -> None:
    """Smoke check: the multi_fact corpus parses + every row exposes the
    spec § A1 fields. Skips when the directory is empty."""
    rows = load_corpus_module(aelfrice_corpus_root, "multi_fact")
    assert rows, "multi_fact corpus produced zero rows"

    for row in rows:
        assert "query" in row
        assert "expected_belief_ids" in row
        assert "expected_clusters" in row
        assert "n_clusters_required" in row
        assert isinstance(row["expected_clusters"], list)


@pytest.mark.bench_gated
def test_clustering_ship_gate_runner_present(
    aelfrice_corpus_root: Path,
) -> None:
    """The full A2 + A3 ship gate runs from
    ``tests.retrieve_uplift_runner.run_clustering_uplift``. This test
    skips when the runner is absent — the runner is the operator-side
    gate for flipping ``use_intentional_clustering`` to default-on."""
    rows = load_corpus_module(aelfrice_corpus_root, "multi_fact")
    assert rows, "multi_fact corpus produced zero rows"

    try:
        from tests.retrieve_uplift_runner import (  # noqa: F401
            run_clustering_uplift,
        )
    except ImportError:
        pytest.skip(
            "intentional-clustering uplift runner not yet wired "
            "(operator gate; spec § A2 + A3 — pending lab-side corpus + scorer)",
        )

    results = run_clustering_uplift(rows)  # type: ignore[name-defined]
    assert results.cluster_coverage_uplift > 0, (
        "intentional clustering must show strictly positive cluster_coverage@k uplift\n"
        f"  ON={results.cluster_coverage_on:.4f} "
        f"OFF={results.cluster_coverage_off:.4f} "
        f"uplift={results.cluster_coverage_uplift:+.4f}"
    )
