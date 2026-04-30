"""Bench gate for #229 phantom promotion-trigger rule."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_promotion_trigger_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "promotion_trigger")
    try:
        from aelfrice import promotion_trigger  # type: ignore[attr-defined]
    except ModuleNotFoundError as exc:
        if exc.name in {"aelfrice", "aelfrice.promotion_trigger"}:
            pytest.skip("promotion_trigger rule not yet implemented (#229)")
        raise

    correct = 0
    for row in rows:
        predicted = promotion_trigger.decide(row["belief_sequence"])
        if predicted == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    assert accuracy >= 0.5, (
        f"promotion_trigger accuracy {accuracy:.3f} below 0.5 floor on {len(rows)} rows"
    )
