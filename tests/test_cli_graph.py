"""cli.main: aelf graph smoke tests (#629).

In-process invocation tests against a tiny synthetic belief graph.
Verifies seed resolution (literal id, BM25 fallback, --seed-id, error
paths), format dispatch (dot vs json), edge-type filter validation,
and `--out` file emission.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
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


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-15T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_triangle(db: Path) -> tuple[str, str, str]:
    """Insert a -SUPPORTS-> b -CITES-> c with extra a-RELATES_TO->c."""
    s = MemoryStore(str(db))
    a, b, c = "aaa1", "bbb2", "ccc3"
    try:
        s.insert_belief(_mk(a, "transactions need atomicity for crash safety"))
        s.insert_belief(_mk(b, "WAL provides atomicity in SQLite"))
        s.insert_belief(_mk(c, "fsync controls when WAL frames hit disk"))
        s.insert_edge(Edge(src=a, dst=b, type=EDGE_SUPPORTS, weight=1.0))
        s.insert_edge(Edge(src=b, dst=c, type=EDGE_CITES, weight=1.0))
        s.insert_edge(Edge(src=a, dst=c, type=EDGE_RELATES_TO, weight=1.0))
    finally:
        s.close()
    return a, b, c


def test_graph_dot_default_format_renders_anchor(_isolated_db: Path) -> None:
    a, b, c = _seed_triangle(_isolated_db)
    code, out = _run("graph", a, "--hops", "2")
    assert code == 0, out
    assert out.startswith("digraph aelfrice")
    for bid in (a, b, c):
        assert f'"{bid}"' in out
    # Edges land as `src -> dst` lines.
    assert f'"{a}" -> "{b}"' in out
    # #629 ratification: edges are color-only, no text label.
    edge_lines = [ln for ln in out.splitlines() if " -> " in ln]
    assert all("label=" not in ln for ln in edge_lines)
    assert any("color=" in ln for ln in edge_lines)


def test_graph_json_format_emits_nodes_edges(_isolated_db: Path) -> None:
    a, b, c = _seed_triangle(_isolated_db)
    code, out = _run("graph", a, "--format", "json", "--hops", "2")
    assert code == 0, out
    payload = json.loads(out)
    node_ids = sorted(n["id"] for n in payload["nodes"])
    assert node_ids == sorted([a, b, c])
    edge_types = sorted(e["type"] for e in payload["edges"])
    assert edge_types == sorted([EDGE_SUPPORTS, EDGE_CITES, EDGE_RELATES_TO])


def test_graph_seed_id_overrides_positional(_isolated_db: Path) -> None:
    a, b, _c = _seed_triangle(_isolated_db)
    code, out = _run("graph", "--seed-id", b, "--format", "json", "--hops", "1")
    assert code == 0, out
    payload = json.loads(out)
    # b as anchor reaches c (CITES) but not a (no inbound edge from b).
    assert b in {n["id"] for n in payload["nodes"]}
    assert a not in {n["id"] for n in payload["nodes"]}


def test_graph_unknown_seed_id_returns_nonzero(_isolated_db: Path) -> None:
    _seed_triangle(_isolated_db)
    code, out = _run("graph", "--seed-id", "nope-no-such-id")
    assert code == 2
    assert "seed-id not found" in out


def test_graph_unknown_anchor_falls_back_to_bm25(_isolated_db: Path) -> None:
    # Query "WAL" should resolve to belief b via BM25 search_beliefs.
    _seed_triangle(_isolated_db)
    code, out = _run("graph", "WAL", "--format", "json", "--hops", "1")
    assert code == 0, out
    payload = json.loads(out)
    # The bm25 hit on "WAL" lands on bbb2 ("WAL provides atomicity...").
    assert "bbb2" in {n["id"] for n in payload["nodes"]}


def test_graph_empty_anchor_returns_nonzero(_isolated_db: Path) -> None:
    _seed_triangle(_isolated_db)
    code, out = _run("graph", "no-such-content-anywhere")
    assert code == 2
    assert "no anchor" in out


def test_graph_edge_types_filter_drops_others(_isolated_db: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    code, out = _run(
        "graph", a, "--format", "json", "--hops", "2",
        "--edge-types", "CITES",
    )
    assert code == 0, out
    payload = json.loads(out)
    edge_types = {e["type"] for e in payload["edges"]}
    assert edge_types == {EDGE_CITES}


def test_graph_edge_types_invalid_returns_nonzero(_isolated_db: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    code, out = _run("graph", a, "--edge-types", "NOT_A_TYPE")
    assert code == 2
    assert "unknown edge type" in out


def test_graph_preview_chars_truncates_label(_isolated_db: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    code, out = _run(
        "graph", a, "--format", "json", "--preview-chars", "12",
    )
    assert code == 0, out
    payload = json.loads(out)
    assert all(len(n["label"]) <= 12 for n in payload["nodes"])


def test_graph_out_file_writes_payload(_isolated_db: Path, tmp_path: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    target = tmp_path / "out.dot"
    code, out = _run("graph", a, "--out", str(target))
    assert code == 0, out
    text = target.read_text()
    assert text.startswith("digraph aelfrice")
    assert "wrote " in out
    assert "nodes" in out and "edges" in out


def test_graph_deterministic_repeat(_isolated_db: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    _c1, out1 = _run("graph", a, "--hops", "2")
    _c2, out2 = _run("graph", a, "--hops", "2")
    assert out1 == out2


def test_graph_invalid_format_rejected_by_argparse(_isolated_db: Path) -> None:
    a, _b, _c = _seed_triangle(_isolated_db)
    # argparse SystemExits with status 2 on unknown choices; capture it.
    with pytest.raises(SystemExit) as ei:
        _run("graph", a, "--format", "html")
    assert ei.value.code == 2
