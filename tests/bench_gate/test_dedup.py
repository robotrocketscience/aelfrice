"""Bench gate for #197 deduplication module.

Loads the lab-mounted corpus and runs the dedup detector against the
labels. Skips on public CI (corpus absent) and skips again here until
the detector module from #197 ships.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_dedup_detector_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "dedup")
    try:
        from aelfrice import dedup  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("dedup detector not yet implemented (#197)")

    correct = 0
    for row in rows:
        predicted = dedup.classify(row["belief_a"], row["belief_b"])
        if predicted == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    assert accuracy >= 0.5, (
        f"dedup accuracy {accuracy:.3f} below 0.5 floor on {len(rows)} rows"
    )
