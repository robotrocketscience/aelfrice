"""Tests for aelfrice.wonder.dispatch (#551).

Coverage:

- analyze_gaps on empty store / no-match query / non-empty store
- high-uncertainty filter (Beta(1, 1) prior is included; Beta(20, 1) is not)
- unresolved CONTRADICTS pairs (with and without SUPERSEDES)
- query-term coverage and uncovered-term gap reporting
- agent-only-dominant gap detection
- generate_research_axes — always-on count, conditional triggers,
  agent_count validation, ≤6 cap
- DispatchPayload.to_dict round-trip is JSON-serialisable

All tests use an in-memory store (no filesystem, no LLM calls,
deterministic).
"""
from __future__ import annotations

import json

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.dispatch import (
    AGENT_ONLY_FRACTION,
    DispatchPayload,
    GapAnalysis,
    QUERY_COVERAGE_LOW,
    ResearchAxis,
    UNCERTAINTY_THRESHOLD,
    _normalized_uncertainty,
    analyze_gaps,
    build_dispatch_payload,
    generate_research_axes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store() -> MemoryStore:
    return MemoryStore(":memory:")


def _belief(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    origin: str = ORIGIN_USER_STATED,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"hash-{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-10T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin=origin,
    )


def _seed_corpus(store: MemoryStore) -> list[str]:
    """Insert a small fixed corpus and return belief ids."""
    beliefs = [
        _belief("b-cats", "Cats are mammals that meow."),
        _belief("b-cats-fly", "Some cats can fly under certain conditions.",
                alpha=2.0, beta=2.0),
        _belief("b-cats-walk", "Cats cannot fly; they walk.",
                alpha=5.0, beta=1.0),
        _belief("b-dogs", "Dogs are mammals that bark."),
        _belief("b-uncertain", "Cats may also purr softly.",
                alpha=1.0, beta=1.0),
    ]
    for b in beliefs:
        store.insert_belief(b)
    return [b.id for b in beliefs]


# ---------------------------------------------------------------------------
# _normalized_uncertainty
# ---------------------------------------------------------------------------


def test_normalized_uncertainty_uniform_prior_is_one() -> None:
    """Beta(1, 1) (no evidence) returns 1.0 — the max-uncertainty case."""
    b = _belief("u-1", "x", alpha=1.0, beta=1.0)
    assert _normalized_uncertainty(b) == pytest.approx(1.0, rel=1e-9)


def test_normalized_uncertainty_strong_evidence_is_low() -> None:
    """Beta(20, 1) (strong yes-evidence) returns near-zero."""
    b = _belief("u-2", "x", alpha=20.0, beta=1.0)
    assert _normalized_uncertainty(b) < 0.1


def test_normalized_uncertainty_threshold_boundary() -> None:
    """Beta(2, 2) is below the 0.7 threshold; Beta(1, 1) is above."""
    b_uniform = _belief("u-3", "x", alpha=1.0, beta=1.0)
    b_balanced = _belief("u-4", "x", alpha=2.0, beta=2.0)
    assert _normalized_uncertainty(b_uniform) > UNCERTAINTY_THRESHOLD
    assert _normalized_uncertainty(b_balanced) < UNCERTAINTY_THRESHOLD


# ---------------------------------------------------------------------------
# analyze_gaps — empty / degenerate input
# ---------------------------------------------------------------------------


def test_analyze_gaps_empty_store() -> None:
    """Empty store returns a GapAnalysis with empty tuples and 0 coverage."""
    store = _fresh_store()
    try:
        ga = analyze_gaps(store, "cats")
        assert ga.query == "cats"
        assert ga.known_beliefs == ()
        assert ga.high_uncertainty_beliefs == ()
        assert ga.unresolved_contradicts_pairs == ()
        assert ga.query_term_coverage == 0.0
    finally:
        store.close()


def test_analyze_gaps_empty_query_no_terms() -> None:
    """A query with no extractable terms returns 0 coverage and no gaps
    that depend on terms."""
    store = _fresh_store()
    try:
        _seed_corpus(store)
        ga = analyze_gaps(store, "   ")
        assert ga.query_term_coverage == 0.0
        assert ga.known_beliefs == ()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# analyze_gaps — non-empty store
# ---------------------------------------------------------------------------


def test_analyze_gaps_finds_known_and_uncertain() -> None:
    store = _fresh_store()
    try:
        _seed_corpus(store)
        ga = analyze_gaps(store, "cats")
        # Must surface at least one cat belief
        ids = {b.id for b in ga.known_beliefs}
        assert "b-cats" in ids or "b-uncertain" in ids
        # The Beta(1, 1) cats belief is high-uncertainty
        hu_ids = {b.id for b in ga.high_uncertainty_beliefs}
        assert "b-uncertain" in hu_ids or "b-cats" in hu_ids
    finally:
        store.close()


def test_analyze_gaps_unresolved_contradiction_surfaces() -> None:
    """A CONTRADICTS edge between two candidates with no SUPERSEDES is
    reported as unresolved."""
    store = _fresh_store()
    try:
        _seed_corpus(store)
        store.insert_edge(
            Edge(src="b-cats-fly", dst="b-cats-walk",
                 type=EDGE_CONTRADICTS, weight=1.0, anchor_text=None)
        )
        ga = analyze_gaps(store, "cats")
        assert ga.unresolved_contradicts_pairs == (
            (min("b-cats-fly", "b-cats-walk"),
             max("b-cats-fly", "b-cats-walk")),
        )
        assert any(
            g.startswith("unresolved_contradictions:") for g in ga.gaps
        )
    finally:
        store.close()


def test_analyze_gaps_supersedes_resolves_contradiction() -> None:
    """A CONTRADICTS pair with a SUPERSEDES edge is NOT reported."""
    store = _fresh_store()
    try:
        _seed_corpus(store)
        store.insert_edge(
            Edge(src="b-cats-fly", dst="b-cats-walk",
                 type=EDGE_CONTRADICTS, weight=1.0, anchor_text=None)
        )
        store.insert_edge(
            Edge(src="b-cats-walk", dst="b-cats-fly",
                 type=EDGE_SUPERSEDES, weight=1.0, anchor_text=None)
        )
        ga = analyze_gaps(store, "cats")
        assert ga.unresolved_contradicts_pairs == ()
        assert not any(
            g.startswith("unresolved_contradictions:") for g in ga.gaps
        )
    finally:
        store.close()


def test_analyze_gaps_uncovered_terms_reported() -> None:
    """A query term that doesn't appear in any candidate is named in gaps."""
    store = _fresh_store()
    try:
        _seed_corpus(store)
        ga = analyze_gaps(store, "cats unicorns")
        # 'unicorns' should not appear in any candidate content
        uncovered = [g for g in ga.gaps if g.startswith("uncovered_terms:")]
        assert uncovered, f"expected uncovered_terms gap; got {ga.gaps}"
        assert "unicorns" in uncovered[0]
    finally:
        store.close()


def test_analyze_gaps_agent_only_dominant_gap() -> None:
    """When >50% of candidates are agent-origin, the gap is reported."""
    store = _fresh_store()
    try:
        for i in range(4):
            store.insert_belief(
                _belief(f"a-{i}", f"agent claim about quarks number {i}",
                        origin=ORIGIN_AGENT_INFERRED)
            )
        store.insert_belief(
            _belief("u-1", "user claim about quarks", origin=ORIGIN_USER_STATED)
        )
        ga = analyze_gaps(store, "quarks")
        assert "agent_only_dominant" in ga.gaps
    finally:
        store.close()


# ---------------------------------------------------------------------------
# generate_research_axes
# ---------------------------------------------------------------------------


def _empty_ga(query: str = "cats", **kwargs: object) -> GapAnalysis:
    base = {
        "query": query,
        "known_beliefs": (),
        "high_uncertainty_beliefs": (),
        "unresolved_contradicts_pairs": (),
        "query_term_coverage": 1.0,
        "gaps": (),
    }
    base.update(kwargs)  # type: ignore[arg-type]
    return GapAnalysis(**base)  # type: ignore[arg-type]


def test_research_axes_always_on_minimum_two() -> None:
    """Empty gap analysis still produces the two always-on axes."""
    ga = _empty_ga()
    axes = generate_research_axes(ga)
    names = [a.name for a in axes]
    assert "domain_research" in names
    assert "internal_gap_analysis" in names
    assert len(axes) >= 2


def test_research_axes_conditional_contradictions() -> None:
    ga = _empty_ga(unresolved_contradicts_pairs=(("a", "b"),))
    axes = generate_research_axes(ga)
    assert "contradiction_resolution" in [a.name for a in axes]


def test_research_axes_conditional_uncertainty() -> None:
    b = _belief("hu-1", "uncertain claim", alpha=1.0, beta=1.0)
    ga = _empty_ga(high_uncertainty_beliefs=(b,))
    axes = generate_research_axes(ga)
    assert "uncertainty_deep_dive" in [a.name for a in axes]


def test_research_axes_conditional_coverage() -> None:
    ga = _empty_ga(
        query_term_coverage=0.2,
        gaps=("uncovered_terms:foo,bar", "low_coverage:0.20"),
    )
    axes = generate_research_axes(ga)
    assert "coverage_extension" in [a.name for a in axes]


def test_research_axes_capped_at_six() -> None:
    """All triggers fire → still ≤ 6 axes."""
    b = _belief("hu-1", "uncertain claim", alpha=1.0, beta=1.0)
    ga = _empty_ga(
        unresolved_contradicts_pairs=(("a", "b"), ("c", "d")),
        high_uncertainty_beliefs=(b, b),
        query_term_coverage=0.0,
        gaps=("uncovered_terms:x,y,z", "low_coverage:0.00"),
    )
    axes = generate_research_axes(ga)
    assert 2 <= len(axes) <= 6


def test_research_axes_invalid_agent_count_raises() -> None:
    with pytest.raises(ValueError):
        generate_research_axes(_empty_ga(), agent_count=0)


# ---------------------------------------------------------------------------
# build_dispatch_payload + JSON round-trip
# ---------------------------------------------------------------------------


def test_dispatch_payload_round_trips_to_json() -> None:
    store = _fresh_store()
    try:
        _seed_corpus(store)
        payload = build_dispatch_payload(store, "cats", agent_count=4)
        # Must JSON-serialise without raising
        s = json.dumps(payload.to_dict())
        round_tripped = json.loads(s)
        assert round_tripped["agent_count"] == 4
        assert round_tripped["gap_analysis"]["query"] == "cats"
        assert isinstance(round_tripped["research_axes"], list)
        assert len(round_tripped["research_axes"]) >= 2
    finally:
        store.close()


def test_dispatch_payload_anchor_ids_match_known_beliefs() -> None:
    store = _fresh_store()
    try:
        _seed_corpus(store)
        payload = build_dispatch_payload(store, "cats")
        known_ids = tuple(b.id for b in payload.gap_analysis.known_beliefs)
        assert payload.speculative_anchor_ids == known_ids
    finally:
        store.close()
