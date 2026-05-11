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
    # #645 R1: verdict + impasses surface in --json payload.
    assert "verdict" in payload
    assert payload["verdict"] in (
        "SUFFICIENT",
        "INSUFFICIENT",
        "CONTRADICTORY",
        "UNCERTAIN",
        "PARTIAL",
    )
    assert isinstance(payload["impasses"], list)
    for imp in payload["impasses"]:
        assert {"kind", "belief_ids", "note"} <= set(imp.keys())
        assert imp["kind"] in ("TIE", "GAP", "CONSTRAINT_FAILURE", "NO_CHANGE")
        assert isinstance(imp["belief_ids"], list)


def test_reason_text_output_includes_verdict_footer(_isolated_db: Path) -> None:
    """#645 R1: text mode trails the chain with a `verdict:` line and
    an `impasses:` line (grep-friendly for downstream R3 dispatch)."""
    a, _, _ = _seed_chain(_isolated_db)
    code, out = _run("reason", "indentation", "--seed-id", a)
    assert code == 0, out
    assert "verdict:" in out
    assert "impasses:" in out


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
    """--json emits WonderResult as JSON (#656).

    The output is a dataclasses.asdict(WonderResult) object with the
    seven fields from the dataclass spec.  Graph-walk mode returns
    mode="graph_walk" and coverage=0.0.
    """
    a, _, _ = _seed_chain(_isolated_db)
    code, out = _run("wonder", "--json")
    assert code == 0, out
    payload = json.loads(out)
    # WonderResult shape — seven top-level keys
    assert {
        "mode", "coverage", "known_beliefs", "gaps",
        "research_axes", "anchor_speculative_ids", "phantoms_created",
    } <= set(payload.keys())
    # Graph-walk mode guarantees
    assert payload["mode"] == "graph_walk"
    assert payload["coverage"] == 0.0
    assert payload["phantoms_created"] == 0
    assert payload["research_axes"] == []
    assert payload["gaps"] == []
    assert payload["anchor_speculative_ids"] == []
    # known_beliefs contains the seed id
    assert a in payload["known_beliefs"]


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


# --- aelf wonder QUERY (positional, #645) -------------------------------


def test_wonder_positional_query_routes_to_axes(_isolated_db: Path) -> None:
    """A positional QUERY runs the axes-spawn-ingest flow by default
    (#645 default-flip), matching `aelf wonder --axes QUERY` shape."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "indentation")
    assert code == 0, out
    payload = json.loads(out)
    assert {"gap_analysis", "research_axes", "agent_count",
            "speculative_anchor_ids"} <= set(payload.keys())
    assert payload["gap_analysis"]["query"] == "indentation"


def test_wonder_no_arg_still_graph_walks(_isolated_db: Path) -> None:
    """No QUERY → graph-walk consolidation mode (aelfrice extension
    preserved; #645 acceptance)."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder")
    # graph-walk produces human-readable rows (or the empty-store
    # message), NOT a JSON payload with `gap_analysis`.
    assert code == 0
    try:
        payload = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if payload is not None:
        assert "gap_analysis" not in payload


def test_wonder_axes_flag_overrides_positional(_isolated_db: Path) -> None:
    """When both `QUERY` positional and `--axes Q2` are passed, the
    explicit --axes flag wins (back-compat for callers that pass both)."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "ignored-positional", "--axes", "explicit")
    assert code == 0
    payload = json.loads(out)
    assert payload["gap_analysis"]["query"] == "explicit"


def test_wonder_positional_query_persist_conflict(_isolated_db: Path) -> None:
    """Positional QUERY + --persist must error like --axes + --persist."""
    code, out = _run("wonder", "some query", "--persist")
    assert code == 2
    assert "--persist" in out


# --- aelf wonder agent-count shorthand (#645) ---------------------------


def test_wonder_query_shorthand_quick_2_agent(_isolated_db: Path) -> None:
    """`quick 2-agent` in the query overrides default agent_count=4."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "quick 2-agent wonder about indentation")
    assert code == 0, out
    payload = json.loads(out)
    assert payload["agent_count"] == 2
    # shorthand stripped from the gap-analysis query text
    assert "2-agent" not in payload["gap_analysis"]["query"]
    assert "quick" not in payload["gap_analysis"]["query"]


def test_wonder_query_shorthand_deep_6_agent(_isolated_db: Path) -> None:
    """`deep 6-agent` parses to agent_count=6."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "deep 6-agent wonder on python")
    assert code == 0
    payload = json.loads(out)
    assert payload["agent_count"] == 6


def test_wonder_query_shorthand_bare_n_agent(_isolated_db: Path) -> None:
    """A bare `N-agent` (no quick/deep) also parses."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "3-agent gap analysis on x")
    assert code == 0
    payload = json.loads(out)
    assert payload["agent_count"] == 3


def test_wonder_explicit_axes_agents_overrides_shorthand(
    _isolated_db: Path,
) -> None:
    """An explicit `--axes-agents N` flag wins over query shorthand."""
    _seed_chain(_isolated_db)
    code, out = _run(
        "wonder", "quick 2-agent wonder about x", "--axes-agents", "5",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["agent_count"] == 5


def test_wonder_no_shorthand_keeps_default_agent_count(
    _isolated_db: Path,
) -> None:
    """Queries without shorthand keep the default agent_count=4."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "plain query about python")
    assert code == 0
    payload = json.loads(out)
    assert payload["agent_count"] == 4
    assert payload["gap_analysis"]["query"] == "plain query about python"


# --- _parse_wonder_query_shorthand unit tests (#645) --------------------


@pytest.mark.parametrize(
    "raw,expected_query,expected_count",
    [
        ("quick 2-agent wonder about X", "about X", 2),
        ("deep 6-agent wonder on Y", "on Y", 6),
        ("3-agent gap on Z", "gap on Z", 3),
        ("Quick 4-AGENT analysis", "analysis", 4),  # case-insensitive
        ("plain query about python", "plain query about python", None),
        ("agentic without count", "agentic without count", None),
        # Shorthand in the middle of the query is stripped cleanly.
        ("the deep 5-agent wonder probe", "the probe", 5),
    ],
)
def test_parse_wonder_query_shorthand_unit(
    raw: str, expected_query: str, expected_count: int | None,
) -> None:
    from aelfrice.cli import _parse_wonder_query_shorthand

    cleaned, count = _parse_wonder_query_shorthand(raw)
    assert cleaned == expected_query
    assert count == expected_count


# --- aelf wonder --persist (#549) ----------------------------------------


def test_wonder_persist_writes_phantoms_to_store(_isolated_db: Path) -> None:
    """--persist should write phantoms and print an insert summary."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "--persist")
    assert code == 0, out
    assert "wonder persist:" in out
    assert "inserted=" in out
    assert "skipped=" in out
    assert "edges_created=" in out
    # The store should now have at least one speculative belief.
    s = MemoryStore(str(_isolated_db))
    try:
        all_ids = s.list_belief_ids()
        types = [s.get_belief(bid).type for bid in all_ids if s.get_belief(bid)]  # type: ignore[union-attr]
    finally:
        s.close()
    assert "speculative" in types


def test_wonder_persist_idempotent_on_second_run(_isolated_db: Path) -> None:
    """A second --persist run with the same seed should skip already-ingested phantoms."""
    a, _b, _c = _seed_chain(_isolated_db)
    code1, out1 = _run("wonder", "--persist", "--seed", a)
    assert code1 == 0, out1
    code2, out2 = _run("wonder", "--persist", "--seed", a)
    assert code2 == 0, out2
    # Second run: inserted=0, skipped >= 1
    assert "inserted=0" in out2
    assert "skipped=" in out2


def test_wonder_persist_mutually_exclusive_with_emit_phantoms(_isolated_db: Path) -> None:
    """--persist + --emit-phantoms must exit 2 with an error message."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "--persist", "--emit-phantoms")
    assert code == 2
    assert "--persist" in out and "--emit-phantoms" in out


def test_wonder_persist_mutually_exclusive_with_axes(_isolated_db: Path) -> None:
    """--persist + --axes must exit 2 with an error message."""
    _seed_chain(_isolated_db)
    code, out = _run("wonder", "--persist", "--axes", "python")
    assert code == 2
    assert "--persist" in out and "--axes" in out


# --- aelf wonder gc (#549) ------------------------------------------------


def test_wonder_gc_dry_run_reports_candidates(_isolated_db: Path) -> None:
    """--dry-run reports candidates without deleting."""
    from datetime import datetime, timedelta, timezone
    from aelfrice.models import BELIEF_SPECULATIVE, LOCK_NONE, ORIGIN_SPECULATIVE
    from aelfrice.models import Belief as _Belief

    # Seed a stale speculative belief directly.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    s = MemoryStore(str(_isolated_db))
    try:
        spec = _Belief(
            id="spec1",
            content="speculative phantom",
            content_hash="gc-test-hash-1",
            alpha=0.3,
            beta=1.0,
            type=BELIEF_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=old_ts,
            last_retrieved_at=None,
            origin=ORIGIN_SPECULATIVE,
        )
        s.insert_belief(spec)
    finally:
        s.close()

    code, out = _run("wonder", "--gc", "--gc-dry-run")
    assert code == 0, out
    assert "wonder gc:" in out
    assert "scanned=1" in out
    assert "would delete=0" in out

    # Belief must still be present (dry-run does not delete).
    s2 = MemoryStore(str(_isolated_db))
    try:
        assert s2.get_belief("spec1") is not None
    finally:
        s2.close()


def test_wonder_gc_non_dry_run_deletes_stale(_isolated_db: Path) -> None:
    """Non-dry-run GC must soft-delete stale speculative beliefs."""
    from datetime import datetime, timedelta, timezone
    from aelfrice.models import BELIEF_SPECULATIVE, LOCK_NONE, ORIGIN_SPECULATIVE
    from aelfrice.models import Belief as _Belief

    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    s = MemoryStore(str(_isolated_db))
    try:
        spec = _Belief(
            id="spec2",
            content="stale speculative phantom",
            content_hash="gc-test-hash-2",
            alpha=0.3,
            beta=1.0,
            type=BELIEF_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=old_ts,
            last_retrieved_at=None,
            origin=ORIGIN_SPECULATIVE,
        )
        s.insert_belief(spec)
    finally:
        s.close()

    code, out = _run("wonder", "--gc")
    assert code == 0, out
    assert "wonder gc:" in out
    assert "scanned=1" in out
    assert "deleted=1" in out


def test_wonder_gc_ttl_days_override(_isolated_db: Path) -> None:
    """--ttl-days should control which beliefs are eligible."""
    from datetime import datetime, timedelta, timezone
    from aelfrice.models import BELIEF_SPECULATIVE, LOCK_NONE, ORIGIN_SPECULATIVE
    from aelfrice.models import Belief as _Belief

    # Insert a belief that is 5 days old — older than --ttl-days 3, younger than 14.
    ts_5d = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    s = MemoryStore(str(_isolated_db))
    try:
        spec = _Belief(
            id="spec3",
            content="medium-age phantom",
            content_hash="gc-test-hash-3",
            alpha=0.3,
            beta=1.0,
            type=BELIEF_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=ts_5d,
            last_retrieved_at=None,
            origin=ORIGIN_SPECULATIVE,
        )
        s.insert_belief(spec)
    finally:
        s.close()

    # Default TTL (14 days): not eligible.
    code, out = _run("wonder", "--gc", "--gc-dry-run")
    assert code == 0
    assert "scanned=0" in out

    # TTL 3 days: eligible.
    code2, out2 = _run("wonder", "--gc", "--gc-dry-run", "--gc-ttl-days", "3")
    assert code2 == 0
    assert "scanned=1" in out2
