"""Bench gate for #422 — v3 value-comparison contradiction detector.

Acceptance #2: re-run against the labeled adversarial corpus from #201.
Target: recall on ``contradicts`` ≥ 0.5 with precision ≥ 0.7,
calibrated against #201's R2 numbers (recall 0.033, precision 0.667).

Skips cleanly when ``AELFRICE_CORPUS_ROOT`` is unset (public CI),
when the ``contradiction/`` module dir is missing, or when the
corpus has fewer than ``MIN_CONTRADICTS`` ``contradicts``-labeled
rows (the gate requires a row floor before recall is statistically
meaningful) — reported as *underfilled*, not absent (#1581).

The conjunction of floors is only evidence while a constant predictor
cannot clear it. A corpus whose majority label is ``contradicts``
hands a constant predictor recall 1.0 and precision equal to the
class share, so the null model runs on the same rows here and both
scores are recorded.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    bar_headroom,
    guard_classification_gate,
)
from tests.conftest import load_corpus_module, skip_corpus_module

RECALL_FLOOR = 0.5  # per #422 acceptance #2
PRECISION_FLOOR = 0.7  # per #422 acceptance #2
MIN_CONTRADICTS = 30  # row floor for stable recall measurement

CONTRADICTS = "contradicts"


def _headroom(precision: float, recall: float) -> float:
    """The gate's conjunction as one scalar: non-negative iff both clear."""
    return min(precision - PRECISION_FLOOR, recall - RECALL_FLOOR)


def _constant_headroom(rows: list[dict], label: str) -> float:
    """Score the constant predictor with the gate's own metric."""
    positives = sum(1 for r in rows if r["label"] == CONTRADICTS)
    if label != CONTRADICTS:
        # Predicts nothing positive: tp = fp = 0, so both rates are 0.0.
        return _headroom(0.0, 0.0)
    precision = positives / len(rows)
    recall = 1.0 if positives else 0.0
    return _headroom(precision, recall)


@pytest.mark.bench_gated
def test_v3_value_comparison_recall_and_precision(
    aelfrice_corpus_root: Path,
    record_property: Callable[[str, object], None],
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "contradiction")

    contradicts_rows = [r for r in rows if r["label"] == CONTRADICTS]
    if len(contradicts_rows) < MIN_CONTRADICTS:
        skip_corpus_module(
            "contradiction",
            "underfilled",
            aelfrice_corpus_root,
            detail=(
                f"{len(contradicts_rows)} {CONTRADICTS!r}-labeled rows < "
                f"{MIN_CONTRADICTS} floor for stable recall measurement"
            ),
        )

    from aelfrice.relationship_detector import classify

    confusion: dict[str, float] = {}

    def shipped() -> float:
        # Confusion matrix on the contradicts vs not-contradicts axis. The
        # corpus also has 'refines' / 'unrelated' labels — for precision
        # we collapse those into "not-contradicts."
        tp = fp = fn = tn = 0
        for r in rows:
            actual_contradicts = r["label"] == CONTRADICTS
            predicted = classify(
                r["belief_a"], r["belief_b"], use_value_comparison=True
            )
            predicted_contradicts = predicted == CONTRADICTS
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
        confusion.update(
            tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision
        )
        return _headroom(precision, recall)

    measured = guard_classification_gate(
        module="contradiction",
        rows=rows,
        shipped=shipped,
        bar=bar_headroom(
            f"precision >= {PRECISION_FLOOR:.2f} and recall >= "
            f"{RECALL_FLOOR:.2f} (min headroom >= 0)"
        ),
        score_constant=lambda label: _constant_headroom(rows, label),
        record_property=record_property,
    )

    # Diagnostic shape: surface both numbers in any failure so the
    # operator can see which dimension is the gating one.
    assert measured.shipped >= 0.0, (
        f"v3 contradiction gate: recall={confusion['recall']:.3f} (floor "
        f"{RECALL_FLOOR:.2f}), precision={confusion['precision']:.3f} (floor "
        f"{PRECISION_FLOOR:.2f}). "
        f"Confusion: tp={confusion['tp']:.0f} fp={confusion['fp']:.0f} "
        f"fn={confusion['fn']:.0f} tn={confusion['tn']:.0f}, "
        f"n_contradicts={len(contradicts_rows)}, n_total={len(rows)}. "
        f"Per #422 acceptance #2, ship requires recall ≥ {RECALL_FLOOR:.2f} "
        f"AND precision ≥ {PRECISION_FLOOR:.2f}; below either floor blocks "
        f"the v3 default-on flip."
    )
