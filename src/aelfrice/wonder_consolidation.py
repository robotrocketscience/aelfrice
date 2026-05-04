"""Bench-gate shim for #228 wonder-consolidation.

The actual generation strategies + harness live in the
``aelfrice.wonder`` package. This module exists only because
``tests/bench_gate/test_wonder_consolidation.py`` was scaffolded
before the bake-off shape was decided and assumes a
``wonder_consolidation.score(seed_belief, retrieved_neighbors)``
signature for per-row sanity checks.

Decision A in the planning memo: keep the stub, wrap it in a
single-phantom relatedness score. The lab-corpus contract for the
bench gate (rows of the form
``{seed_belief, retrieved_neighbors, expected_metric}``) is
follow-up work tracked separately; the score returned here is a
placeholder relatedness scalar in [0, 1] derived from token
overlap. It is *not* read by the bake-off proper — the runner
talks to the strategies directly.
"""
from __future__ import annotations

from typing import Any

_TOKENIZER_DROP = set(",.;:!?\"'()[]{}")


def _tokens(text: str) -> set[str]:
    cleaned = "".join(" " if c in _TOKENIZER_DROP else c for c in text.lower())
    return {t for t in cleaned.split() if t}


def score(seed_belief: Any, retrieved_neighbors: Any) -> float:
    """Return a token-overlap relatedness score in [0, 1].

    Accepts either strings or dict-like structures with a
    ``content`` field for both arguments. ``retrieved_neighbors``
    may also be a list of strings / dicts; in that case the score
    is the mean per-neighbor overlap.

    Returning a float in [0, 1] keeps the bench-gate stub honest
    (its ``isinstance(rating, (int, float))`` assertion holds)
    without pretending this is the bake-off result. The real
    strategy-quality answer lives in
    ``aelfrice.wonder.runner.run_bakeoff``.
    """
    seed_text = _content_of(seed_belief)
    if isinstance(retrieved_neighbors, list):
        if not retrieved_neighbors:
            return 0.0
        scores = [_pairwise(seed_text, _content_of(n)) for n in retrieved_neighbors]
        return sum(scores) / len(scores)
    return _pairwise(seed_text, _content_of(retrieved_neighbors))


def _content_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("content", ""))
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    return str(item)


def _pairwise(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


__all__ = ["score"]
