"""cli.main: aelf reason + aelf wonder smoke tests (#389).

Atomic in-process invocation tests against a tiny synthetic belief
graph. Verifies that:
  - `aelf reason` surfaces seeds + expands via outbound edges.
  - `aelf reason --json` produces a parseable payload with the
    expected shape.
  - `aelf reason --seed-id <unknown>` exits nonzero.
  - `aelf wonder` picks a deterministic seed and prints candidates.
  - `aelf wonder --emit-phantoms` produces JSON-shaped Phantom rows.
  - `aelf wonder --seed <unknown>` exits nonzero.

The graph is tiny on purpose — these are smoke tests over the
ratified surface, not bench-gate evidence.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-04T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_chain(db: Path) -> tuple[str, str, str]:
    """Insert a small 3-belief chain a -> b -> c with SUPERSEDES edges."""
    s = MemoryStore(str(db))
    a, b, c = "aaa1", "bbb2", "ccc3"
    try:
        s.insert_belief(_mk_belief(a, "python uses indentation"))
        s.insert_belief(_mk_belief(b, "indentation defines blocks in python"))
        s.insert_belief(_mk_belief(c, "PEP 8 standardizes indentation width"))
        s.insert_edge(Edge(src=a, dst=b, type=EDGE_SUPERSEDES, weight=1.0))
        s.insert_edge(Edge(src=b, dst=c, type=EDGE_RELATES_TO, weight=1.0))
    finally:
        s.close()
    return a, b, c


# --- aelf reason --------------------------------------------------------


def test_reason_with_seed_id_walks_graph(_isolated_db: Path) -> None:
    a, b, c = _seed_chain(_isolated_db)
    code, out = _run("reason", "indentation", "--seed-id", a)
    assert code == 0, out
    assert a in out  # seed listed
    # b should appear as a 1-hop SUPERSEDES expansion
    assert b in out
    assert "SUPERSEDES" in out


def test_reason_json_payload_shape(_isolated_db: Path) -> None:
    a, _, _ = _seed_chain(_isolated_db)
    code, out = _run("reason", "indentation", "--seed-id", a, "--json")
    assert code == 0, out
    payload = json.loads(out)
    assert payload["query"] == "indentation"
    assert payload["seeds"][0]["id"] == a
    assert isinstance(payload["hops"], list)
    assert len(payload["hops"]) >= 1
    assert {"id", "content", "score", "depth", "path"} <= set(payload["hops"][0].keys())


def test_reason_unknown_seed_id_exits_nonzero(_isolated_db: Path) -> None:
    _seed_chain(_isolated_db)
    code, _ = _run("reason", "anything", "--seed-id", "no-such-id")
    assert code == 2


def test_reason_empty_store_returns_zero_with_message(_isolated_db: Path) -> None:
    code, out = _run("reason", "anything")
    assert code == 0
    assert "no seeds" in out


# --- aelf wonder --------------------------------------------------------


def test_wonder_picks_deterministic_seed(_isolated_db: Path) -> None:
    a, b, _ = _seed_chain(_isolated_db)
    code, out = _run("wonder")
    assert code == 0, out
    # a has 1 outbound edge, b has 1, c has 0 — tie between a and b
    # broken by id-asc → a wins.
    assert f"seed: {a}:" in out
    # Candidate listing should include b (1-hop SUPERSEDES).
    assert b in out


def test_wonder_emit_phantoms_yields_phantom_json(_isolated_db: Path) -> None:
    a, b, _ = _seed_chain(_isolated_db)
    code, out = _run("wonder", "--emit-phantoms")
    assert code == 0, out
    rows = json.loads(out)
    assert isinstance(rows, list) and len(rows) >= 1
    row = rows[0]
    assert {"constituent_belief_ids", "generator", "content", "score"} <= set(row.keys())
    assert a in row["constituent_belief_ids"]
    assert b in row["constituent_belief_ids"]
    assert row["generator"] == "bfs+wonder_consolidation"


def test_wonder_unknown_seed_exits_nonzero(_isolated_db: Path) -> None:
    _seed_chain(_isolated_db)
    code, _ = _run("wonder", "--seed", "no-such-id")
    assert code == 2


def test_wonder_empty_store_returns_zero_with_message(_isolated_db: Path) -> None:
    code, out = _run("wonder")
    assert code == 0
    assert "no eligible seeds" in out


def test_wonder_json_payload_shape(_isolated_db: Path) -> None:
    a, _, _ = _seed_chain(_isolated_db)
    code, out = _run("wonder", "--json")
    assert code == 0, out
    payload = json.loads(out)
    assert payload["seed"]["id"] == a
    assert isinstance(payload["candidates"], list)
    assert len(payload["candidates"]) >= 1
    cand = payload["candidates"][0]
    assert {
        "candidate_id", "score", "relatedness", "suggested_action", "path",
    } <= set(cand.keys())
    assert cand["suggested_action"] in {"merge", "supersede", "contradict", "relate"}


# --- aelf wonder --axes (#551) ----------------------------------------


def test_wonder_axes_emits_json_payload(_isolated_db: Path) -> None:
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "--axes", "indentation")
    assert code == 0, out
    payload = json.loads(out)
    assert {"gap_analysis", "research_axes", "agent_count",
            "speculative_anchor_ids"} <= set(payload.keys())
    assert payload["gap_analysis"]["query"] == "indentation"
    assert 2 <= len(payload["research_axes"]) <= 6


def test_wonder_axes_overrides_default_path(_isolated_db: Path) -> None:
    """--axes bypasses the seed/BFS branch and never prints the
    'no eligible seeds' message even on an empty store."""
    code, out = _run("wonder", "--axes", "anything")
    assert code == 0
    payload = json.loads(out)
    assert payload["gap_analysis"]["query"] == "anything"


def test_wonder_axes_respects_agent_count(_isolated_db: Path) -> None:
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "--axes", "python", "--axes-agents", "2")
    assert code == 0
    payload = json.loads(out)
    assert payload["agent_count"] == 2
