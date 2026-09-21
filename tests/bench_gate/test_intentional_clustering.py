"""Bench gate for #436 intentional clustering.

Spec § A2 (multi-fact recall uplift) + § A3 (single-fact non-regression)
+ § A4 (latency). The full gate evaluates all three; this scaffold runs
the multi-fact corpus through ``cluster_candidates`` + ``pack_with_clusters``
directly, then checks ``cluster_coverage@k`` against a baseline.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset (corpus content
lives lab-side per the directory-of-origin rule). The retrieval-side
wiring (``use_intentional_clustering`` in ``retrieve_v2``) already
shipped and defaults on (#436 R6); this scaffold tests the module directly so the
substrate can land before the wiring.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    AblationArms,
    bar_above,
    guard_ablation_gate,
)
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
    record_property: Callable[[str, object], None],
) -> None:
    """The full A2 + A3 ship gate runs from
    ``tests.retrieve_uplift_runner.run_clustering_uplift``. This test
    skips when the runner is absent — the runner is the operator-side
    gate for flipping ``use_intentional_clustering`` to default-on."""
    rows = load_corpus_module(aelfrice_corpus_root, "multi_fact")
    assert rows, "multi_fact corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "intentional-clustering uplift runner not yet wired "
            "(operator gate; spec § A2 + A3 — pending lab-side corpus + scorer)"
        ),
    )

    measured: dict[str, object] = {}

    def arms() -> AblationArms:
        results = runner_mod.run_clustering_uplift(rows)
        measured["results"] = results
        return AblationArms(
            shipped=results.cluster_coverage_on,
            ablated=results.cluster_coverage_off,
            without_row_scores=results.off_row_scores,
        )

    # #1581: the clustering-off arm is this gate's declared null model,
    # and the structural pre-filters run over the row's candidate pool.
    # `k_key` is the gate's own cutoff: the guard's `default_k` of 10
    # exceeds every multi_fact pool, which would read as zero separable
    # rows and reject a corpus that is in fact 14/14 separable at the
    # cutoff the gate scores with.
    guard_ablation_gate(
        module="multi_fact",
        rows=rows,
        arms=arms,
        bar=bar_above(0.0),
        record_property=record_property,
        gold_key="expected_belief_ids",
        pool_key="beliefs",
        k_key="n_clusters_required",
    )

    results = measured["results"]
    assert results.cluster_coverage_uplift > 0, (
        "intentional clustering must show strictly positive cluster_coverage@k uplift\n"
        f"  ON={results.cluster_coverage_on:.4f} "
        f"OFF={results.cluster_coverage_off:.4f} "
        f"uplift={results.cluster_coverage_uplift:+.4f}"
    )
