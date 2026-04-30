"""Bench gate for #201 semantic contradiction detector."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_contradiction_detector_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "contradiction")
    try:
        from aelfrice import relationship_detector  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("contradiction detector not yet implemented (#201)")

    correct = 0
    for row in rows:
        predicted = relationship_detector.classify(row["belief_a"], row["belief_b"])
        if predicted == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    assert accuracy >= 0.5, (
        f"contradiction accuracy {accuracy:.3f} below 0.5 floor on {len(rows)} rows"
    )
