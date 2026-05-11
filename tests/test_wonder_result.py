"""Tests for aelfrice.wonder.result.WonderResult (#656).

Coverage:
- Dataclass field shapes and frozen constraint.
- Round-trip: build a WonderResult, dataclasses.asdict, json.dumps, json.loads
  -> same shape.
- Graph-walk mode always returns mode="graph_walk" and coverage=0.0.
- Axes mode with N axes and K phantoms -> coverage == K / max(1, N).
- Coverage scalar boundary: 0 axes -> denominator clamped to 1.
- All fields serialise to JSON-native types (no extra converters needed).
- CLI: ``aelf wonder --json`` emits valid JSON matching WonderResult shape.
- CLI: ``aelf wonder`` (no --json) human-readable output is unchanged.
"""
from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

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
        candidates=[],
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
        candidates=[],
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
    """All eight fields are present (#656 spec + candidates regression fix)."""
    result = _graph_walk_result()
    d = dataclasses.asdict(result)
    expected = {
        "mode", "coverage", "known_beliefs", "gaps",
        "research_axes", "anchor_speculative_ids", "phantoms_created",
        "candidates",
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


# ---------------------------------------------------------------------------
# CLI integration: aelf wonder --json (#656)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    from aelfrice.cli import main
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed_db(db: Path) -> str:
    """Insert one belief and return its id."""
    from aelfrice.models import (
        BELIEF_FACTUAL,
        EDGE_RELATES_TO,
        LOCK_NONE,
        ORIGIN_AGENT_INFERRED,
        Belief,
        Edge,
    )
    from aelfrice.store import MemoryStore

    s = MemoryStore(str(db))
    bid = "wr-seed-1"
    bid2 = "wr-seed-2"
    try:
        s.insert_belief(Belief(
            id=bid, content="python uses indentation",
            content_hash="h1", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
            demotion_pressure=0, created_at="2026-05-11T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_belief(Belief(
            id=bid2, content="indentation defines code blocks",
            content_hash="h2", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
            demotion_pressure=0, created_at="2026-05-11T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_edge(Edge(src=bid, dst=bid2, type=EDGE_RELATES_TO, weight=1.0))
    finally:
        s.close()
    return bid


def test_wonder_json_flag_emits_valid_json(_isolated_db: Path) -> None:
    """``aelf wonder --json`` exits 0 and produces valid JSON."""
    _seed_db(_isolated_db)
    code, out = _run("wonder", "--json")
    assert code == 0, out
    # Must not raise
    payload = json.loads(out)
    assert isinstance(payload, dict)


def test_wonder_json_flag_matches_wonder_result_shape(_isolated_db: Path) -> None:
    """``aelf wonder --json`` output matches the WonderResult field set."""
    seed_id = _seed_db(_isolated_db)
    code, out = _run("wonder", "--json")
    assert code == 0, out
    payload = json.loads(out)
    expected_keys = {
        "mode", "coverage", "known_beliefs", "gaps",
        "research_axes", "anchor_speculative_ids", "phantoms_created",
        "candidates",
    }
    assert expected_keys == set(payload.keys())
    # Graph-walk guarantees
    assert payload["mode"] == "graph_walk"
    assert payload["coverage"] == 0.0
    assert payload["phantoms_created"] == 0
    assert payload["research_axes"] == []
    assert payload["gaps"] == []
    assert payload["anchor_speculative_ids"] == []
    assert seed_id in payload["known_beliefs"]
    # Candidates are list-of-dicts with the v2.x --json row shape
    assert isinstance(payload["candidates"], list)
    if payload["candidates"]:
        row = payload["candidates"][0]
        assert {"candidate_id", "score", "relatedness",
                "suggested_action", "path"} == set(row.keys())


def test_wonder_json_round_trips_cleanly(_isolated_db: Path) -> None:
    """The JSON output can be loaded and re-dumped without loss."""
    _seed_db(_isolated_db)
    code, out = _run("wonder", "--json")
    assert code == 0, out
    first = json.loads(out)
    second = json.loads(json.dumps(first))
    assert first == second


def test_wonder_no_json_flag_human_output_unchanged(_isolated_db: Path) -> None:
    """Without --json the graph-walk output still starts with 'seed:'."""
    seed_id = _seed_db(_isolated_db)
    code, out = _run("wonder")
    assert code == 0, out
    assert out.startswith(f"seed: {seed_id}:")
    # Output must NOT be JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
