"""Bench gate for #374 — H1 directive detection re-entry.

Per `docs/v2_enforcement.md` § H1, H1 unblocks for implementation only when
the candidate detector hits ≥80% precision and ≥60% recall on ≥200 labeled
coding prompts. This test scores `aelfrice.directive_detector.detect_directive`
against the lab-side corpus and asserts the gate.

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset (public CI), when the
`directive_detection/` module dir is missing, or when the corpus has fewer
than 200 rows (the gate requires a 200-row floor before it can fire).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

PRECISION_GATE = 0.80
RECALL_GATE = 0.60
MIN_ROWS = 200


@pytest.mark.bench_gated
def test_directive_detection_gate(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "directive_detection")

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"directive_detection corpus has {len(rows)} rows; gate requires "
            f"≥{MIN_ROWS} per docs/v2_enforcement.md § H1"
        )

    from aelfrice.directive_detector import detect_directive

    tp = fp = fn = tn = 0
    for row in rows:
        actual = row["label"] == "directive"
        predicted = detect_directive(row["prompt"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif (not predicted) and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    assert precision >= PRECISION_GATE, (
        f"directive_detection precision {precision:.3f} below "
        f"{PRECISION_GATE} gate (TP={tp}, FP={fp}, FN={fn}, TN={tn}, "
        f"n={len(rows)}); H1 stays deferred per docs/v2_enforcement.md § H1"
    )
    assert recall >= RECALL_GATE, (
        f"directive_detection recall {recall:.3f} below {RECALL_GATE} gate "
        f"(TP={tp}, FP={fp}, FN={fn}, TN={tn}, n={len(rows)}); "
        f"H1 stays deferred per docs/v2_enforcement.md § H1"
    )
