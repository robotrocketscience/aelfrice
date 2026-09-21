"""Bench gate for #435 doc linker.

Spec acceptance A2 (docs/design/feature-doc-linker.md):

    NDCG@k(with_doc_anchors=ON, anchors_populated=ON)
        > NDCG@k(with_doc_anchors=ON, anchors_populated=OFF)

The anchors-populated case loads each row's seed beliefs, writes the
labelled `expected_doc_uris` against them, then queries with
`with_doc_anchors=True` and lets a downstream rerank read the anchors.
The anchors-EMPTY baseline runs the same query against the same beliefs
without writing the anchors. Strictly positive uplift is the ship
trigger; zero or negative uplift fails the gate.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule (labelled corpus lives only in
``~/projects/aelfrice-lab/tests/corpus/v2_0/doc_linker/``).
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
def test_doc_linker_uplift(
    aelfrice_corpus_root: Path,
    record_property: Callable[[str, object], None],
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "doc_linker")
    assert rows, "doc_linker corpus produced zero rows"

    # The actual uplift driver lives in tests.retrieve_uplift_runner; the
    # doc-linker variant defers until the bench-gate scoring contract
    # for `with_doc_anchors` is written. Until then, the harness skips
    # so a corpus-mounted run still surfaces the wiring as PASS rather
    # than masking a missing scorer.
    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "doc-linker uplift runner not yet wired (operator gate; "
            "spec § A2 — pending lab-side corpus + scorer)"
        ),
    )

    measured: dict[str, object] = {}

    def arms() -> AblationArms:
        results = runner_mod.run_doc_linker_uplift(rows)
        measured["results"] = results
        return AblationArms(
            shipped=results.mean_ndcg_on,
            ablated=results.mean_ndcg_off,
            without_row_scores=results.off_row_scores,
        )

    # #1581: the anchors-off arm is this gate's declared null model. A
    # corpus where nothing is retrievable without a doc anchor reports
    # +1.000 uplift whatever the linker does.
    guard_ablation_gate(
        module="doc_linker",
        rows=rows,
        arms=arms,
        bar=bar_above(0.0),
        record_property=record_property,
        gold_key="expected_belief_ids",
        pool_key="beliefs",
    )

    results = measured["results"]
    detail = (
        f"  NDCG_anchors_off={results.mean_ndcg_off:.4f} "
        f"NDCG_anchors_on={results.mean_ndcg_on:.4f} "
        f"uplift={results.uplift:+.4f}"
    )
    assert results.uplift > 0, (
        f"doc-linker uplift not strictly positive on "
        f"{len(rows)} rows:\n{detail}"
    )
