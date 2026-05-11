"""Tests for aelfrice.wonder.result.WonderResult (#656).

Coverage:
- Dataclass field shapes and frozen constraint.
- Round-trip: build a WonderResult, dataclasses.asdict, json.dumps, json.loads
  -> same shape.
- Graph-walk mode always returns mode="graph_walk" and coverage=0.0.
- Axes mode with N axes and K phantoms -> coverage == K / max(1, N).
- Coverage scalar boundary: 0 axes -> denominator clamped to 1.
- All fields serialise to JSON-native types (no extra converters needed).
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from aelfrice.wonder.result import WonderResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_walk_result(**overrides: object) -> WonderResult:
    """Minimal graph-walk WonderResult fixture."""
    base: dict = dict(
        mode="graph_walk",
        coverage=0.0,
        known_beliefs=["b-aaa"],
        gaps=[],
        research_axes=[],
        anchor_speculative_ids=[],
        phantoms_created=0,
    )
    base.update(overrides)
    return WonderResult(**base)  # type: ignore[arg-type]


def _axes_result(n_axes: int, k_phantoms: int) -> WonderResult:
    axes = [
        {"name": f"axis-{i}", "description": f"desc {i}",
         "search_hints": [f"hint-{i}"], "gap_context": "test"}
        for i in range(n_axes)
    ]
    coverage = k_phantoms / max(1, n_axes)
    return WonderResult(
        mode="axes",
        coverage=coverage,
        known_beliefs=["b-seed"],
        gaps=["uncovered_terms:foo"],
        research_axes=axes,
        anchor_speculative_ids=["b-anchor-1"],
        phantoms_created=k_phantoms,
    )


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


def test_wonder_result_is_frozen() -> None:
    """WonderResult is a frozen dataclass; mutation raises FrozenInstanceError."""
    result = _graph_walk_result()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        result.coverage = 0.5  # type: ignore[misc]


def test_wonder_result_field_names() -> None:
    """All seven fields specified in the issue are present."""
    result = _graph_walk_result()
    d = dataclasses.asdict(result)
    expected = {
        "mode", "coverage", "known_beliefs", "gaps",
        "research_axes", "anchor_speculative_ids", "phantoms_created",
    }
    assert expected == set(d.keys())


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_graph_walk_result_json_round_trip() -> None:
    """Graph-walk WonderResult round-trips through dataclasses.asdict + json."""
    result = _graph_walk_result()
    d = dataclasses.asdict(result)
    serialised = json.dumps(d)
    back = json.loads(serialised)
    assert back["mode"] == "graph_walk"
    assert back["coverage"] == 0.0
    assert back["known_beliefs"] == ["b-aaa"]
    assert back["gaps"] == []
    assert back["research_axes"] == []
    assert back["anchor_speculative_ids"] == []
    assert back["phantoms_created"] == 0


def test_axes_result_json_round_trip() -> None:
    """Axes WonderResult with N=4, K=3 round-trips through JSON."""
    result = _axes_result(n_axes=4, k_phantoms=3)
    d = dataclasses.asdict(result)
    serialised = json.dumps(d)
    back = json.loads(serialised)
    assert back["mode"] == "axes"
    assert back["phantoms_created"] == 3
    assert len(back["research_axes"]) == 4
    assert pytest.approx(back["coverage"]) == 3 / 4


def test_json_round_trip_preserves_list_types() -> None:
    """Lists serialise as JSON arrays; no numpy or custom types."""
    result = _axes_result(n_axes=2, k_phantoms=1)
    d = dataclasses.asdict(result)
    raw = json.dumps(d)
    # Must not raise; round-trip preserves list types
    back = json.loads(raw)
    assert isinstance(back["known_beliefs"], list)
    assert isinstance(back["research_axes"], list)
    assert isinstance(back["anchor_speculative_ids"], list)


# ---------------------------------------------------------------------------
# Graph-walk mode guarantees
# ---------------------------------------------------------------------------


def test_graph_walk_mode_field() -> None:
    result = _graph_walk_result()
    assert result.mode == "graph_walk"


def test_graph_walk_coverage_is_zero() -> None:
    result = _graph_walk_result()
    assert result.coverage == 0.0


def test_graph_walk_research_axes_empty() -> None:
    result = _graph_walk_result()
    assert result.research_axes == []


# ---------------------------------------------------------------------------
# Axes mode: coverage scalar
# ---------------------------------------------------------------------------


def test_axes_coverage_n4_k3() -> None:
    """4 axes, 3 phantoms -> coverage == 0.75."""
    result = _axes_result(n_axes=4, k_phantoms=3)
    assert result.coverage == pytest.approx(3 / 4)


def test_axes_coverage_n0_denominator_clamp() -> None:
    """0 axes -> denominator is clamped to 1; coverage == k / 1."""
    result = _axes_result(n_axes=0, k_phantoms=0)
    assert result.coverage == pytest.approx(0.0)


def test_axes_coverage_full() -> None:
    """All axes produce phantoms -> coverage == 1.0."""
    result = _axes_result(n_axes=5, k_phantoms=5)
    assert result.coverage == pytest.approx(1.0)


def test_axes_coverage_zero_phantoms() -> None:
    """Some axes, no phantoms -> coverage == 0.0."""
    result = _axes_result(n_axes=3, k_phantoms=0)
    assert result.coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mode-field values
# ---------------------------------------------------------------------------


def test_axes_mode_field() -> None:
    result = _axes_result(n_axes=2, k_phantoms=1)
    assert result.mode == "axes"


def test_wonder_result_equality() -> None:
    """Two identically-constructed WonderResults are equal (frozen dataclass)."""
    a = _graph_walk_result()
    b = _graph_walk_result()
    assert a == b
