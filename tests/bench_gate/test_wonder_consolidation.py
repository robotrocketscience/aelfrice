"""Bench gate for #228 wonder-consolidation strategy bake-off."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_wonder_consolidation_against_corpus(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "wonder_consolidation")
    try:
        from aelfrice import wonder_consolidation  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip("wonder_consolidation strategy not yet implemented (#228)")

    # Expected human ratings 1-5; strategy outputs a generated phantom
    # which the bake-off rates. v0.1 acceptance is correlation-shaped,
    # not exact-match — leave the metric to #228 to define.
    scored = 0
    for row in rows:
        rating = wonder_consolidation.score(row["seed_belief"], row["retrieved_neighbors"])
        assert isinstance(rating, (int, float))
        scored += 1
    assert scored == len(rows)
