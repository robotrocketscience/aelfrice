"""Bench gate for #422 — v3 value-comparison contradiction detector.

Acceptance #2: re-run against the labeled adversarial corpus from #201.
Target: recall on ``contradicts`` ≥ 0.5 with precision ≥ 0.7,
calibrated against #201's R2 numbers (recall 0.033, precision 0.667).

Skips cleanly when ``AELFRICE_CORPUS_ROOT`` is unset (public CI),
when the ``contradiction/`` module dir is missing, or when the
corpus has fewer than ``MIN_CONTRADICTS`` ``contradicts``-labeled
rows (the gate requires a row floor before recall is statistically
meaningful).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

RECALL_FLOOR = 0.5  # per #422 acceptance #2
PRECISION_FLOOR = 0.7  # per #422 acceptance #2
MIN_CONTRADICTS = 30  # row floor for stable recall measurement


@pytest.mark.bench_gated
def test_v3_value_comparison_recall_and_precision(
    aelfrice_corpus_root: Path,
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "contradiction")

    contradicts_rows = [r for r in rows if r["label"] == "contradicts"]
    if len(contradicts_rows) < MIN_CONTRADICTS:
        pytest.skip(
            f"contradiction corpus has {len(contradicts_rows)} 'contradicts'-"
            f"labeled rows; gate requires ≥{MIN_CONTRADICTS} for stable "
            f"recall measurement"
        )

    from aelfrice.relationship_detector import classify

    # Confusion matrix on the contradicts vs not-contradicts axis. The
    # corpus also has 'refines' / 'unrelated' labels — for precision
    # we collapse those into "not-contradicts."
    tp = fp = fn = tn = 0
    for r in rows:
        actual_contradicts = r["label"] == "contradicts"
        predicted = classify(
            r["belief_a"], r["belief_b"], use_value_comparison=True
        )
        predicted_contradicts = predicted == "contradicts"
        if actual_contradicts and predicted_contradicts:
            tp += 1
        elif actual_contradicts and not predicted_contradicts:
            fn += 1
        elif not actual_contradicts and predicted_contradicts:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Diagnostic shape: surface both numbers in any failure so the
    # operator can see which dimension is the gating one.
    assert recall >= RECALL_FLOOR and precision >= PRECISION_FLOOR, (
        f"v3 contradiction gate: recall={recall:.3f} (floor {RECALL_FLOOR:.2f}), "
        f"precision={precision:.3f} (floor {PRECISION_FLOOR:.2f}). "
        f"Confusion: tp={tp} fp={fp} fn={fn} tn={tn}, "
        f"n_contradicts={len(contradicts_rows)}, n_total={len(rows)}. "
        f"Per #422 acceptance #2, ship requires recall ≥ {RECALL_FLOOR:.2f} "
        f"AND precision ≥ {PRECISION_FLOOR:.2f}; below either floor blocks "
        f"the v3 default-on flip."
    )
