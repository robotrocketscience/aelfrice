"""Structured return type for the aelf wonder command (#656).

Kept here (not in ``aelfrice.wonder.models``) because ``WonderResult``
is the CLI / Python API surface -- it lives at the boundary between the
wonder internals and callers (MCP, CLI, tests).  ``Phantom`` and the
strategy-level dataclasses remain in the wonder sub-package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WonderResult:
    """Structured output from one ``aelf wonder`` invocation.

    Fields
    ------
    mode:
        ``"graph_walk"`` for the default seed/BFS consolidation path or
        ``"axes"`` when ``--axes QUERY`` is passed.
    coverage:
        Fraction of research axes whose corresponding sub-agent result
        produced at least one ingested phantom.  Defined as
        ``phantoms_created / max(1, len(research_axes))``.  Always
        ``0.0`` in graph-walk mode (the concept does not apply).
    known_beliefs:
        Belief ids (or content snippets) that satisfied parts of the
        gap in axes mode; the seed belief id in graph-walk mode.
    gaps:
        Human-readable gap descriptors from
        ``aelfrice.wonder.dispatch.GapAnalysis``.
        Empty list in graph-walk mode.
    research_axes:
        ``aelfrice.wonder.dispatch.ResearchAxis.to_dict`` rows.
        Empty list in graph-walk mode.
    anchor_speculative_ids:
        Phantom ids whose ingest anchored back to known beliefs
        (``speculative_anchor_ids`` from the dispatch payload).
        Empty list in graph-walk mode.
    phantoms_created:
        Count of phantoms created (axes mode only); ``0`` in
        graph-walk mode.  Sourced from the ``inserted`` count of the
        ingest path when ``--persist-docs FILE`` is used.
    candidates:
        Graph-walk-mode consolidation candidates ranked by combined
        BFS-path-score × token-relatedness.  Each row is
        ``{candidate_id, score, relatedness, suggested_action, path}``
        — preserves the v2.x ``--json`` structured shape so existing
        callers do not lose access to the candidate ranking on the
        v3.0 dataclass cutover.  Empty list in axes mode (the concept
        does not apply).
    """

    mode: Literal["graph_walk", "axes"]
    coverage: float
    known_beliefs: list[str]
    gaps: list[str]
    research_axes: list[dict]
    anchor_speculative_ids: list[str]
    phantoms_created: int
    candidates: list[dict]


__all__ = ["WonderResult"]
