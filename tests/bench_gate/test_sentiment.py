"""Bench gate for #193 sentiment-from-prose feedback evaluation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_sentiment_detector_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "sentiment")
    try:
        from aelfrice import sentiment  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("sentiment detector not yet implemented (#193)")

    correct = 0
    for row in rows:
        predicted = sentiment.classify(row["user_message"])
        if predicted == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    assert accuracy >= 0.5, (
        f"sentiment accuracy {accuracy:.3f} below 0.5 floor on {len(rows)} rows"
    )
