"""Bench gate for #193 sentiment-from-prose feedback evaluation.

The accuracy floor is only evidence while a constant predictor cannot
clear it, so the majority-label null model runs on the same rows in
this test and both scores are recorded (#1581).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    bar_at_least,
    constant_predictor_accuracy,
    guard_classification_gate,
)
from tests.conftest import load_corpus_module

ACCURACY_FLOOR = 0.5


@pytest.mark.bench_gated
def test_sentiment_detector_against_corpus(
    aelfrice_corpus_root: Path,
    record_property: Callable[[str, object], None],
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "sentiment")
    from aelfrice import sentiment_feedback

    def shipped() -> float:
        correct = sum(
            1
            for row in rows
            if sentiment_feedback.classify(row["user_message"]) == row["label"]
        )
        return correct / len(rows)

    measured = guard_classification_gate(
        module="sentiment",
        rows=rows,
        shipped=shipped,
        bar=bar_at_least(ACCURACY_FLOOR),
        score_constant=lambda label: constant_predictor_accuracy(rows, label),
        record_property=record_property,
    )

    assert measured.shipped >= ACCURACY_FLOOR, (
        f"sentiment accuracy {measured.shipped:.3f} below {ACCURACY_FLOOR} "
        f"floor on {len(rows)} rows (majority-label null model scores "
        f"{measured.null:.3f})"
    )
