"""Wonder-consolidation bake-off harness (#228).

Three offline phantom-generation strategies plus a deterministic
synthetic corpus + feedback simulator + evaluator + runner that
together close the v2.0 ship-decision per
``docs/v2_wonder_consolidation.md``.

The harness is a research surface: nothing here writes to a live
``Store`` outside the bake-off. The chosen-strategy production
wiring is a follow-up issue per the spec.
"""
from __future__ import annotations

from .models import Phantom
from .strategies import random_walk, span_topic_sampling, triangle_closure

__all__ = [
    "Phantom",
    "random_walk",
    "span_topic_sampling",
    "triangle_closure",
]
