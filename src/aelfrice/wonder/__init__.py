"""Wonder-consolidation bake-off harness (#228).

Three offline phantom-generation strategies plus a deterministic
synthetic corpus + feedback simulator + evaluator + runner that
together close the v2.0 ship-decision per
``docs/design/v2_wonder_consolidation.md``.

The generation strategies here (``random_walk``, ``triangle_closure``,
``span_topic_sampling``) are research-only and do not write to a live
``Store``. Production wiring ships separately as
``aelfrice.wonder.lifecycle`` (``wonder_ingest`` / ``wonder_gc``,
#548/#549), exposed via ``aelf wonder --persist`` / ``aelf wonder --gc``
and the ``aelf_wonder_persist`` / ``aelf_wonder_gc`` MCP tools.
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
