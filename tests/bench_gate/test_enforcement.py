"""Bench gate for #199 enforcement module."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_enforcement_detector_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "enforcement")
    try:
        from aelfrice import enforcement  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("enforcement detector not yet implemented (#199)")

    correct = 0
    for row in rows:
        predicted = enforcement.classify(row["user_directive"], row["agent_output"])
        if predicted == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    assert accuracy >= 0.5, (
        f"enforcement accuracy {accuracy:.3f} below 0.5 floor on {len(rows)} rows"
    )
