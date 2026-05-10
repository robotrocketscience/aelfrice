"""Research-agent dispatch wonder surface (#551, umbrella #542 track E).

Two pure functions plus dataclasses:

- :func:`analyze_gaps` — given a query, return a :class:`GapAnalysis`
  summarising the candidate set retrieved for that query: known beliefs,
  high-uncertainty beliefs, unresolved ``CONTRADICTS`` pairs, query-term
  coverage, and the named gaps a downstream research pass should target.
- :func:`generate_research_axes` — turn a :class:`GapAnalysis` into 2–6
  orthogonal :class:`ResearchAxis` records sized for ``agent_count``
  parallel research lanes.

The MCP ``wonder()`` tool and the ``aelf wonder <query> --axes`` CLI flag
both wrap these two functions; the skill layer (E4, separate sub-issue)
consumes the axes JSON, fans out research agents, and pipes their results back
through ``wonder_ingest`` (track C).

Determinism: both functions are pure given a fixed store snapshot. No
LLM calls, no randomness, no filesystem writes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from aelfrice.models import (
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_AGENT_REMEMBERED,
    Belief,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

UNCERTAINTY_THRESHOLD: float = 0.7
"""Minimum normalized uncertainty for a belief to count as high-uncertainty.

Normalized to ``[0, 1]`` where ``1.0`` is the max-uncertainty Beta(1, 1)
prior and values approach ``0`` as evidence accumulates either way. The
0.7 threshold matches the umbrella #542 spec for E1.
"""

QUERY_COVERAGE_LOW: float = 0.5
"""Below this query-term coverage we surface a ``low_coverage`` gap."""

AGENT_ONLY_FRACTION: float = 0.5
"""When more than this fraction of candidates are agent-origin we surface
an ``agent_only_dominant`` gap."""


@dataclass(frozen=True)
class GapAnalysis:
    """Snapshot of what the store knows (and doesn't) about a query.

    Fields are populated by :func:`analyze_gaps` and consumed by
    :func:`generate_research_axes`. The MCP tool serialises this via
    :meth:`to_dict`.
    """

    query: str
    known_beliefs: tuple[Belief, ...]
    high_uncertainty_beliefs: tuple[Belief, ...]
    unresolved_contradicts_pairs: tuple[tuple[str, str], ...]
    query_term_coverage: float
    gaps: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "known_beliefs": [_belief_summary(b) for b in self.known_beliefs],
            "high_uncertainty_beliefs": [
                _belief_summary(b) for b in self.high_uncertainty_beliefs
            ],
            "unresolved_contradicts_pairs": [
                list(p) for p in self.unresolved_contradicts_pairs
            ],
            "query_term_coverage": self.query_term_coverage,
            "gaps": list(self.gaps),
        }


@dataclass(frozen=True)
class ResearchAxis:
    """One orthogonal research lane for a research agent to pursue.

    ``search_hints`` is a tuple of short strings the research agent can plug
    into search engines / tool calls verbatim; ``gap_context`` cites
    which :class:`GapAnalysis` field motivated this axis.
    """

    name: str
    description: str
    search_hints: tuple[str, ...]
    gap_context: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "search_hints": list(self.search_hints),
            "gap_context": self.gap_context,
        }


@dataclass(frozen=True)
class DispatchPayload:
    """Top-level shape returned by the MCP tool / CLI ``--axes`` flag."""

    gap_analysis: GapAnalysis
    research_axes: tuple[ResearchAxis, ...]
    agent_count: int
    speculative_anchor_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "gap_analysis": self.gap_analysis.to_dict(),
            "research_axes": [a.to_dict() for a in self.research_axes],
            "agent_count": self.agent_count,
            "speculative_anchor_ids": list(self.speculative_anchor_ids),
        }


# ---------------------------------------------------------------------------
# E1 — analyze_gaps
# ---------------------------------------------------------------------------

_QUERY_TERM_RE = re.compile(r"[A-Za-z0-9_]+")
_AGENT_ORIGINS = frozenset({ORIGIN_AGENT_INFERRED, ORIGIN_AGENT_REMEMBERED})


def _query_terms(query: str) -> list[str]:
    return [t.lower() for t in _QUERY_TERM_RE.findall(query) if len(t) > 1]


def _normalized_uncertainty(b: Belief) -> float:
    """Beta variance scaled so Beta(1, 1) (uniform prior) returns 1.0.

    The umbrella spec (#542 E1) writes ``uncertainty_score > 0.7`` as a
    threshold expression. ``aelfrice.scoring.uncertainty_score`` returns
    Beta differential entropy on a non-positive scale (``≤ 0``), which
    can't be compared against ``0.7`` directly. We use a simple
    ``[0, 1]``-normalized variance proxy here:

    ``12 * αβ / ((α + β)² · (α + β + 1))``

    The factor of 12 normalises ``Beta(1, 1)`` (variance 1/12) to 1.0;
    higher-evidence Beta distributions return values closer to 0.
    """
    a = max(b.alpha, 1e-9)
    bb = max(b.beta, 1e-9)
    s = a + bb
    var = (a * bb) / ((s * s) * (s + 1.0))
    return min(1.0, 12.0 * var)


def _belief_summary(b: Belief) -> dict[str, object]:
    return {
        "id": b.id,
        "content": b.content,
        "alpha": b.alpha,
        "beta": b.beta,
        "origin": b.origin,
        "uncertainty": _normalized_uncertainty(b),
    }


def _unresolved_contradicts(
    store: MemoryStore, candidate_ids: list[str]
) -> list[tuple[str, str]]:
    """``CONTRADICTS`` pairs within the candidate set with no
    ``SUPERSEDES`` edge between the same endpoints (either direction).

    Result is sorted by ``(min(a, b), max(a, b))`` for determinism.
    """
    if not candidate_ids:
        return []
    candidate_set = set(candidate_ids)
    edges = store.edges_for_beliefs(candidate_ids)

    contradicts: set[tuple[str, str]] = set()
    supersedes: set[tuple[str, str]] = set()
    for e in edges:
        if e.src not in candidate_set or e.dst not in candidate_set:
            continue
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        if e.type == EDGE_CONTRADICTS:
            contradicts.add(pair)
        elif e.type == EDGE_SUPERSEDES:
            supersedes.add(pair)

    unresolved = sorted(contradicts - supersedes)
    return unresolved


def analyze_gaps(
    store: MemoryStore,
    query: str,
    budget: int = 24,
    depth: int = 2,
) -> GapAnalysis:
    """Summarise the candidate set retrieved for ``query`` as gaps.

    ``budget`` caps the number of candidates pulled from
    :func:`aelfrice.retrieval.retrieve` (mapped to the L1 limit so the
    candidate set size is bounded). ``depth`` is forwarded as the
    BFS expansion max-depth — accepted for API parity with the spec
    even though :func:`retrieve` defaults BFS off; explicit ``depth``
    enables BFS so the candidate graph is wider than direct hits.

    Empty stores or no-match queries return a :class:`GapAnalysis`
    with empty tuples and ``query_term_coverage = 0.0``. The function
    never raises on degenerate input.
    """
    terms = _query_terms(query)

    candidates: list[Belief]
    if not terms:
        candidates = []
    else:
        candidates = list(
            retrieve(
                store,
                query,
                l1_limit=max(1, budget),
                bfs_enabled=depth > 0,
                bfs_max_depth=max(1, depth),
            )
        )

    high_uncertainty = tuple(
        b for b in candidates
        if _normalized_uncertainty(b) > UNCERTAINTY_THRESHOLD
    )

    candidate_ids = [b.id for b in candidates]
    unresolved = tuple(_unresolved_contradicts(store, candidate_ids))

    coverage = _query_term_coverage(terms, candidates)

    gaps: list[str] = []
    uncovered = _uncovered_terms(terms, candidates)
    if uncovered:
        gaps.append(f"uncovered_terms:{','.join(uncovered)}")
    if terms and coverage < QUERY_COVERAGE_LOW:
        gaps.append(f"low_coverage:{coverage:.2f}")
    if candidates and _agent_only_fraction(candidates) > AGENT_ONLY_FRACTION:
        gaps.append("agent_only_dominant")
    if unresolved:
        gaps.append(f"unresolved_contradictions:{len(unresolved)}")

    return GapAnalysis(
        query=query,
        known_beliefs=tuple(candidates),
        high_uncertainty_beliefs=high_uncertainty,
        unresolved_contradicts_pairs=unresolved,
        query_term_coverage=coverage,
        gaps=tuple(gaps),
    )


def _query_term_coverage(terms: list[str], candidates: list[Belief]) -> float:
    if not terms:
        return 0.0
    haystack = " ".join(b.content for b in candidates).lower()
    hit = sum(1 for t in terms if t in haystack)
    return hit / len(terms)


def _uncovered_terms(terms: list[str], candidates: list[Belief]) -> list[str]:
    if not terms:
        return []
    haystack = " ".join(b.content for b in candidates).lower()
    return [t for t in terms if t not in haystack]


def _agent_only_fraction(candidates: list[Belief]) -> float:
    if not candidates:
        return 0.0
    n = sum(1 for b in candidates if b.origin in _AGENT_ORIGINS)
    return n / len(candidates)


# ---------------------------------------------------------------------------
# E2 — generate_research_axes
# ---------------------------------------------------------------------------

_MIN_AXES = 2
_MAX_AXES = 6


def generate_research_axes(
    gap_analysis: GapAnalysis,
    agent_count: int = 4,
) -> tuple[ResearchAxis, ...]:
    """Build 2–6 orthogonal research axes for parallel research-agent dispatch.

    Always-on axes:

    1. ``domain_research`` — open-ended external search on the query.
    2. ``internal_gap_analysis`` — sweep the local store for adjacent
       beliefs that did not surface in ``known_beliefs``.

    Conditional axes (added only when the trigger holds):

    3. ``contradiction_resolution`` — when ``unresolved_contradicts_pairs``
       is non-empty.
    4. ``uncertainty_deep_dive`` — when ``high_uncertainty_beliefs`` is
       non-empty.
    5. ``coverage_extension`` — when ``query_term_coverage`` is below the
       low-coverage threshold or named uncovered terms exist.

    The result is capped at ``_MAX_AXES`` (6) and floored at ``_MIN_AXES``
    (2). ``agent_count`` is a hint, not a hard budget; the caller can
    spawn fewer agents than axes by zipping or more by reusing axes.
    """
    if agent_count < 1:
        raise ValueError(f"agent_count must be ≥ 1; got {agent_count}")

    axes: list[ResearchAxis] = [
        ResearchAxis(
            name="domain_research",
            description=(
                "Open-ended external research on the query. Surface "
                "primary sources, definitions, and contemporary context."
            ),
            search_hints=(gap_analysis.query, f"{gap_analysis.query} overview"),
            gap_context="always-on",
        ),
        ResearchAxis(
            name="internal_gap_analysis",
            description=(
                "Sweep the local store for beliefs adjacent to the "
                "candidate set that did not surface in retrieve(). "
                "Look for orphaned RELATES_TO neighbours and recently-"
                "ingested beliefs whose entities overlap the query."
            ),
            search_hints=tuple(
                t for t in _query_terms(gap_analysis.query)[:4]
            ) or (gap_analysis.query,),
            gap_context="always-on",
        ),
    ]

    if gap_analysis.unresolved_contradicts_pairs:
        pairs = gap_analysis.unresolved_contradicts_pairs[:3]
        hints = tuple(f"{a} vs {b}" for a, b in pairs)
        axes.append(
            ResearchAxis(
                name="contradiction_resolution",
                description=(
                    "Resolve unresolved CONTRADICTS pairs in the "
                    "candidate set: surface evidence that supersedes "
                    "one side or reframes the contradiction."
                ),
                search_hints=hints,
                gap_context=(
                    f"unresolved_contradicts_pairs="
                    f"{len(gap_analysis.unresolved_contradicts_pairs)}"
                ),
            )
        )

    if gap_analysis.high_uncertainty_beliefs:
        sample = gap_analysis.high_uncertainty_beliefs[:3]
        hints = tuple(b.content[:80] for b in sample)
        axes.append(
            ResearchAxis(
                name="uncertainty_deep_dive",
                description=(
                    "Find evidence for or against the high-uncertainty "
                    "beliefs in the candidate set. The goal is α/β "
                    "movement, not new beliefs."
                ),
                search_hints=hints,
                gap_context=(
                    f"high_uncertainty_count="
                    f"{len(gap_analysis.high_uncertainty_beliefs)}"
                ),
            )
        )

    if (
        gap_analysis.query_term_coverage < QUERY_COVERAGE_LOW
        or any(g.startswith("uncovered_terms:") for g in gap_analysis.gaps)
    ):
        uncovered_blob = next(
            (g[len("uncovered_terms:"):] for g in gap_analysis.gaps
             if g.startswith("uncovered_terms:")),
            "",
        )
        coverage_hints = tuple(
            t for t in uncovered_blob.split(",") if t
        ) or (gap_analysis.query,)
        axes.append(
            ResearchAxis(
                name="coverage_extension",
                description=(
                    "Query-term coverage is below threshold; research "
                    "the uncovered terms specifically and find primary "
                    "definitions for them."
                ),
                search_hints=coverage_hints,
                gap_context=(
                    f"query_term_coverage={gap_analysis.query_term_coverage:.2f}"
                ),
            )
        )

    return tuple(axes[:_MAX_AXES])


# ---------------------------------------------------------------------------
# E3 helper — bundle gap analysis + axes for the MCP tool / CLI flag
# ---------------------------------------------------------------------------


def build_dispatch_payload(
    store: MemoryStore,
    query: str,
    *,
    budget: int = 24,
    depth: int = 2,
    agent_count: int = 4,
) -> DispatchPayload:
    """Convenience wrapper composing :func:`analyze_gaps` and
    :func:`generate_research_axes` into the JSON-serialisable shape
    that the MCP ``wonder()`` tool and ``aelf wonder --axes`` CLI flag
    both return.

    ``speculative_anchor_ids`` is the candidate-belief id list — track
    C's ``wonder_ingest`` will use these as ``RELATES_TO`` targets when
    persisting research agent research as speculative beliefs.
    """
    ga = analyze_gaps(store, query, budget=budget, depth=depth)
    axes = generate_research_axes(ga, agent_count=agent_count)
    anchors = tuple(b.id for b in ga.known_beliefs)
    return DispatchPayload(
        gap_analysis=ga,
        research_axes=axes,
        agent_count=agent_count,
        speculative_anchor_ids=anchors,
    )


__all__ = [
    "AGENT_ONLY_FRACTION",
    "DispatchPayload",
    "GapAnalysis",
    "QUERY_COVERAGE_LOW",
    "ResearchAxis",
    "UNCERTAINTY_THRESHOLD",
    "analyze_gaps",
    "build_dispatch_payload",
    "generate_research_axes",
]
