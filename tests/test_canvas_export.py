"""Tests for the JSON Canvas 1.0 exporter (#763).

Covers schema shape against the JSON Canvas 1.0 spec, determinism
across runs, deduplication of edges, and color encoding for the
posterior / valence buckets defined in the module.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.bfs_multihop import ScoredHop, expand_bfs
from aelfrice.canvas_export import (
    COLOR_EDGE_NEGATIVE,
    COLOR_EDGE_POSITIVE,
    COLOR_LOCKED,
    COLOR_POSTERIOR_HIGH,
    COLOR_POSTERIOR_LOW,
    NODE_HEIGHT,
    NODE_WIDTH,
    export_canvas,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _belief(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"hash_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-13T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


def _seed_store(beliefs: list[Belief], edges: list[Edge]) -> MemoryStore:
    """Construct an in-memory store with the given beliefs + edges."""
    store = MemoryStore(":memory:")
    for b in beliefs:
        store.insert_belief(b)
    for e in edges:
        store.insert_edge(e)
    return store


def test_empty_seeds_yields_empty_canvas():
    store = MemoryStore(":memory:")
    try:
        payload = export_canvas([], [], store)
        assert payload == {"nodes": [], "edges": []}
    finally:
        store.close()


def test_single_seed_no_hops_renders_one_node():
    seed = _belief("b1", "first belief", alpha=2.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        payload = export_canvas([seed], [], store)
        assert len(payload["nodes"]) == 1
        assert payload["edges"] == []
        node = payload["nodes"][0]
        assert node["id"] == "b1"
        assert node["type"] == "text"
        assert node["text"] == "first belief"
        assert node["x"] == 0 and node["y"] == 0  # single seed at origin
        assert node["width"] == NODE_WIDTH
        assert node["height"] == NODE_HEIGHT
    finally:
        store.close()


def test_locked_belief_gets_locked_color():
    seed = _belief("locked1", "ground truth", lock_level=LOCK_USER)
    store = _seed_store([seed], [])
    try:
        payload = export_canvas([seed], [], store)
        assert payload["nodes"][0]["color"] == COLOR_LOCKED
    finally:
        store.close()


def test_high_posterior_node_color_high():
    # mu = 5/(5+1) ≈ 0.833 >= 0.75 → high bucket
    seed = _belief("b_high", "strong belief", alpha=5.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        payload = export_canvas([seed], [], store)
        assert payload["nodes"][0]["color"] == COLOR_POSTERIOR_HIGH
    finally:
        store.close()


def test_low_posterior_node_color_low():
    # mu = 0.5/(0.5+5) ≈ 0.091 < 0.25 → low bucket
    seed = _belief("b_low", "disputed", alpha=0.5, beta=5.0)
    store = _seed_store([seed], [])
    try:
        payload = export_canvas([seed], [], store)
        assert payload["nodes"][0]["color"] == COLOR_POSTERIOR_LOW
    finally:
        store.close()


def test_mid_posterior_emits_no_color_key():
    # mu = 1/2 = 0.5 → mid bucket → no color attribute emitted
    seed = _belief("b_mid", "uncertain", alpha=1.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        payload = export_canvas([seed], [], store)
        assert "color" not in payload["nodes"][0]
    finally:
        store.close()


def test_edge_colors_by_valence():
    a = _belief("a", "alpha node")
    b = _belief("b", "beta node")
    c = _belief("c", "gamma node")
    store = _seed_store(
        [a, b, c],
        [
            Edge(src="a", dst="b", type=EDGE_SUPPORTS,
                 weight=1.0, anchor_text=None),
            Edge(src="a", dst="c", type=EDGE_CONTRADICTS,
                 weight=1.0, anchor_text=None),
            Edge(src="b", dst="c", type=EDGE_SUPERSEDES,
                 weight=1.0, anchor_text=None),
        ],
    )
    try:
        # Build hops manually so we don't depend on BFS scoring choices
        # in this unit test.
        hops = [
            ScoredHop(belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
                      belief_id_trail=("a", "b")),
            ScoredHop(belief=c, score=0.5, depth=1, path=[EDGE_CONTRADICTS],
                      belief_id_trail=("a", "c")),
        ]
        payload = export_canvas([a], hops, store)
        edges_by_type = {(e["fromNode"], e["toNode"]): e for e in payload["edges"]}
        assert edges_by_type[("a", "b")]["color"] == COLOR_EDGE_POSITIVE
        assert edges_by_type[("a", "c")]["color"] == COLOR_EDGE_NEGATIVE
        # SUPERSEDES is valence=0 → no color key emitted
        assert "color" not in edges_by_type[("b", "c")]
    finally:
        store.close()


def test_anchor_text_is_truncated_into_edge_label():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    long_anchor = "x" * 200
    store = _seed_store(
        [a, b],
        [Edge(src="a", dst="b", type=EDGE_RELATES_TO,
              weight=1.0, anchor_text=long_anchor)],
    )
    try:
        hops = [ScoredHop(
            belief=b, score=0.3, depth=1, path=[EDGE_RELATES_TO],
            belief_id_trail=("a", "b"),
        )]
        payload = export_canvas([a], hops, store)
        label = payload["edges"][0]["label"]
        # Label starts with edge type then truncated anchor with ellipsis.
        assert label.startswith(f"{EDGE_RELATES_TO}: ")
        assert len(label) <= len(EDGE_RELATES_TO) + 2 + 40  # type + ": " + cap
    finally:
        store.close()


def test_determinism_byte_identical_across_runs():
    a = _belief("a", "alpha node", alpha=4.0, beta=1.0)
    b = _belief("b", "beta node", alpha=1.0, beta=4.0)
    c = _belief("c", "gamma node", lock_level=LOCK_USER)
    store = _seed_store(
        [a, b, c],
        [
            Edge(src="a", dst="b", type=EDGE_SUPPORTS,
                 weight=1.0, anchor_text="because reasons"),
            Edge(src="a", dst="c", type=EDGE_RELATES_TO,
                 weight=1.0, anchor_text=None),
            Edge(src="b", dst="c", type=EDGE_CONTRADICTS,
                 weight=1.0, anchor_text="conflict"),
        ],
    )
    try:
        hops1 = expand_bfs([a], store, max_depth=2,
                            nodes_per_hop=8, total_budget=16)
        hops2 = expand_bfs([a], store, max_depth=2,
                            nodes_per_hop=8, total_budget=16)
        p1 = export_canvas([a], hops1, store)
        p2 = export_canvas([a], hops2, store)
        assert json.dumps(p1, sort_keys=False) == json.dumps(p2, sort_keys=False)
    finally:
        store.close()


def test_edge_dedup_same_src_dst_type_emits_once():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    # Two physical edges with same (src, dst, type) — store would
    # reject the dup by PK, but exercise the de-dup set anyway.
    store = _seed_store(
        [a, b],
        [Edge(src="a", dst="b", type=EDGE_SUPPORTS,
              weight=1.0, anchor_text=None)],
    )
    try:
        hops = [ScoredHop(
            belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
            belief_id_trail=("a", "b"),
        )]
        payload = export_canvas([a], hops, store)
        keys = [(e["fromNode"], e["toNode"], e["label"].split(":")[0])
                for e in payload["edges"]]
        assert len(keys) == len(set(keys))
    finally:
        store.close()


def test_dangling_edges_to_unsurfaced_beliefs_omitted():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    c = _belief("c", "gamma — not in BFS")
    store = _seed_store(
        [a, b, c],
        [
            Edge(src="a", dst="b", type=EDGE_SUPPORTS,
                 weight=1.0, anchor_text=None),
            Edge(src="a", dst="c", type=EDGE_RELATES_TO,
                 weight=1.0, anchor_text=None),
        ],
    )
    try:
        hops = [ScoredHop(
            belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
            belief_id_trail=("a", "b"),
        )]
        payload = export_canvas([a], hops, store)
        # c is in the store but NOT in the hop set → no node, no edge.
        node_ids = {n["id"] for n in payload["nodes"]}
        assert node_ids == {"a", "b"}
        edge_pairs = {(e["fromNode"], e["toNode"]) for e in payload["edges"]}
        assert edge_pairs == {("a", "b")}
    finally:
        store.close()


def test_json_canvas_1_0_required_fields():
    """Spec compliance: every node has id/type/x/y/width/height; every
    edge has id/fromNode/toNode."""
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    store = _seed_store(
        [a, b],
        [Edge(src="a", dst="b", type=EDGE_SUPPORTS,
              weight=1.0, anchor_text=None)],
    )
    try:
        hops = [ScoredHop(
            belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
            belief_id_trail=("a", "b"),
        )]
        payload = export_canvas([a], hops, store)
        for n in payload["nodes"]:
            for k in ("id", "type", "x", "y", "width", "height"):
                assert k in n, f"node {n['id']} missing required field {k}"
            assert n["type"] in ("text", "file", "link", "group")
        for e in payload["edges"]:
            for k in ("id", "fromNode", "toNode"):
                assert k in e, f"edge {e['id']} missing required field {k}"
        # Round-trip through JSON to confirm serialisability.
        text = json.dumps(payload)
        json.loads(text)
    finally:
        store.close()


def test_cli_writes_canvas_to_path(tmp_path: Path):
    """End-to-end: aelf export-canvas --seed-id <id> --out <path>."""
    from io import StringIO
    from aelfrice.cli import main as cli_main

    # Spin up a fresh on-disk store and inject one belief so the CLI
    # can find a seed. Point AELFRICE_DB at a tmp file so we don't
    # touch the operator's real store.
    import os
    db_path = tmp_path / "brain.sqlite"
    prev = os.environ.get("AELFRICE_DB")
    os.environ["AELFRICE_DB"] = str(db_path)
    try:
        from aelfrice.db_paths import _open_store
        store = _open_store()
        try:
            store.insert_belief(_belief("seed_belief", "canvas test seed"))
        finally:
            store.close()

        out = tmp_path / "out.canvas"
        buf = StringIO()
        rc = cli_main(
            argv=[
                "export-canvas",
                "--seed-id", "seed_belief",
                "--depth", "1",
                "--budget", "4",
                "--fanout", "4",
                "--out", str(out),
            ],
            out=buf,
        )
        assert rc == 0
        assert out.exists()
        payload = json.loads(out.read_text())
        assert len(payload["nodes"]) == 1
        assert payload["nodes"][0]["id"] == "seed_belief"
    finally:
        if prev is None:
            os.environ.pop("AELFRICE_DB", None)
        else:
            os.environ["AELFRICE_DB"] = prev
