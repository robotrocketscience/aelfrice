"""Tests for the query-anchored DOT / JSON graph viewer (#629).

Covers determinism, DOT escaping, edge dedup + dangling-edge omission,
edge-type filter, preview truncation, and node/edge color encoding.
Mirrors the test layout of ``test_canvas_export.py`` so reviewers can
diff the two viewer surfaces against the same expectations.
"""
from __future__ import annotations

from aelfrice.bfs_multihop import ScoredHop, expand_bfs
from aelfrice.graph_export import (
    DEFAULT_PREVIEW_CHARS,
    EDGE_DOT_COLOR,
    EDGE_LABEL_ABBR,
    NODE_DOT_COLOR_HIGH,
    NODE_DOT_COLOR_LOCKED,
    NODE_DOT_COLOR_LOW,
    export_dot,
    export_graph_json,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
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
        created_at="2026-05-15T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


def _seed_store(beliefs: list[Belief], edges: list[Edge]) -> MemoryStore:
    store = MemoryStore(":memory:")
    for b in beliefs:
        store.insert_belief(b)
    for e in edges:
        store.insert_edge(e)
    return store


# ---------------------------------------------------------------- empty


def test_empty_seeds_dot_emits_empty_digraph():
    store = MemoryStore(":memory:")
    try:
        out = export_dot([], [], store)
        assert out == "digraph aelfrice {\n}\n"
    finally:
        store.close()


def test_empty_seeds_json_emits_empty_dict():
    store = MemoryStore(":memory:")
    try:
        assert export_graph_json([], [], store) == {"nodes": [], "edges": []}
    finally:
        store.close()


# ---------------------------------------------------------------- shapes


def test_single_seed_no_hops_dot_renders_one_node():
    seed = _belief("b1", "first belief", alpha=2.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        out = export_dot([seed], [], store)
        assert 'digraph aelfrice' in out
        assert '"b1" [label="first belief"' in out
        # No edges => no `->` lines.
        assert "->" not in out
    finally:
        store.close()


def test_single_seed_no_hops_json_renders_one_node():
    seed = _belief("b1", "first belief", alpha=2.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        payload = export_graph_json([seed], [], store)
        assert payload["edges"] == []
        assert len(payload["nodes"]) == 1
        n = payload["nodes"][0]
        assert n["id"] == "b1"
        assert n["label"] == "first belief"
        assert n["locked"] is False
        # alpha=2, beta=1 -> mu=2/3 ≈ 0.667 -> mid bucket -> no color key.
        assert "color" not in n
    finally:
        store.close()


# ---------------------------------------------------------------- colors


def test_locked_belief_gets_locked_color_both_formats():
    seed = _belief("locked1", "anchored truth", lock_level=LOCK_USER)
    store = _seed_store([seed], [])
    try:
        dot = export_dot([seed], [], store)
        assert f'color="{NODE_DOT_COLOR_LOCKED}"' in dot
        # Locked nodes get penwidth=2 for visual emphasis even without color
        # in B/W contexts.
        assert "penwidth=2" in dot
        js = export_graph_json([seed], [], store)
        assert js["nodes"][0]["color"] == NODE_DOT_COLOR_LOCKED
        assert js["nodes"][0]["locked"] is True
    finally:
        store.close()


def test_high_posterior_color_high():
    # alpha=5, beta=1 -> mu ≈ 0.833 >= 0.75 -> high bucket.
    seed = _belief("h", "strong belief", alpha=5.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store)
        assert js["nodes"][0]["color"] == NODE_DOT_COLOR_HIGH
    finally:
        store.close()


def test_low_posterior_color_low():
    # alpha=0.5, beta=5 -> mu ≈ 0.091 < 0.25 -> low bucket.
    seed = _belief("l", "disputed", alpha=0.5, beta=5.0)
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store)
        assert js["nodes"][0]["color"] == NODE_DOT_COLOR_LOW
    finally:
        store.close()


def test_mid_posterior_omits_color_key():
    seed = _belief("m", "uncertain", alpha=1.0, beta=1.0)
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store)
        assert "color" not in js["nodes"][0]
        # And DOT line for that node carries no color= attribute.
        dot = export_dot([seed], [], store)
        node_line = [ln for ln in dot.splitlines() if ln.lstrip().startswith('"m"')][0]
        assert "color=" not in node_line
    finally:
        store.close()


# ---------------------------------------------------------------- edge encoding


def test_edge_label_uses_short_abbreviation():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    store = _seed_store([a, b], [
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [ScoredHop(belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
                          belief_id_trail=("a", "b"))]
        js = export_graph_json([a], hops, store)
        assert js["edges"][0]["label"] == EDGE_LABEL_ABBR[EDGE_SUPPORTS]
        assert js["edges"][0]["label"] == "SUP"
        assert js["edges"][0]["type"] == EDGE_SUPPORTS
    finally:
        store.close()


def test_edge_colors_by_type_not_valence():
    # CITES and RELATES_TO both carry positive valence but get distinct
    # colors here so the viewer can tell them apart.
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    c = _belief("c", "gamma")
    store = _seed_store([a, b, c], [
        Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0, anchor_text=None),
        Edge(src="a", dst="c", type=EDGE_RELATES_TO, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [
            ScoredHop(belief=b, score=0.4, depth=1, path=[EDGE_CITES],
                      belief_id_trail=("a", "b")),
            ScoredHop(belief=c, score=0.3, depth=1, path=[EDGE_RELATES_TO],
                      belief_id_trail=("a", "c")),
        ]
        js = export_graph_json([a], hops, store)
        by_dst = {(e["src"], e["dst"]): e for e in js["edges"]}
        assert by_dst[("a", "b")]["color"] == EDGE_DOT_COLOR[EDGE_CITES]
        assert by_dst[("a", "c")]["color"] == EDGE_DOT_COLOR[EDGE_RELATES_TO]
        assert by_dst[("a", "b")]["color"] != by_dst[("a", "c")]["color"]
    finally:
        store.close()


def test_edge_dedup_same_src_dst_type_emits_once():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    # Store PK prevents physical duplicates; exercise the de-dup set
    # by routing the same edge through both seed-set and hop entry.
    store = _seed_store([a, b], [
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [ScoredHop(belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
                          belief_id_trail=("a", "b"))]
        js = export_graph_json([a], hops, store)
        keys = [(e["src"], e["dst"], e["type"]) for e in js["edges"]]
        assert len(keys) == len(set(keys)) == 1
    finally:
        store.close()


def test_dangling_edges_to_unsurfaced_beliefs_omitted():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    c = _belief("c", "gamma — not surfaced")
    store = _seed_store([a, b, c], [
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
        # `c` is in the store but not in hops -> the a->c edge must be
        # dropped (dangling).
        Edge(src="a", dst="c", type=EDGE_RELATES_TO, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [ScoredHop(belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
                          belief_id_trail=("a", "b"))]
        js = export_graph_json([a], hops, store)
        dsts = [e["dst"] for e in js["edges"]]
        assert dsts == ["b"]
    finally:
        store.close()


# ---------------------------------------------------------------- filter


def test_edge_types_filter_drops_disallowed():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    c = _belief("c", "gamma")
    store = _seed_store([a, b, c], [
        Edge(src="a", dst="b", type=EDGE_CITES, weight=1.0, anchor_text=None),
        Edge(src="a", dst="c", type=EDGE_RELATES_TO, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [
            ScoredHop(belief=b, score=0.4, depth=1, path=[EDGE_CITES],
                      belief_id_trail=("a", "b")),
            ScoredHop(belief=c, score=0.3, depth=1, path=[EDGE_RELATES_TO],
                      belief_id_trail=("a", "c")),
        ]
        js = export_graph_json([a], hops, store,
                               edge_types_filter=frozenset({EDGE_CITES}))
        edge_types = [e["type"] for e in js["edges"]]
        assert edge_types == [EDGE_CITES]
        # Surfaced nodes stay even when their connecting edge was
        # filtered out (display-layer filter, not BFS-level).
        node_ids = sorted(n["id"] for n in js["nodes"])
        assert node_ids == ["a", "b", "c"]
    finally:
        store.close()


# ---------------------------------------------------------------- preview


def test_preview_truncation_default_60_chars():
    long_text = "x" * 200
    seed = _belief("seed", long_text)
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store)
        label = js["nodes"][0]["label"]
        # 60-char cap with ellipsis: 59 chars + "…".
        assert len(label) == DEFAULT_PREVIEW_CHARS
        assert label.endswith("…")
    finally:
        store.close()


def test_preview_truncation_override():
    seed = _belief("seed", "x" * 200)
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store, preview_chars=10)
        assert len(js["nodes"][0]["label"]) == 10
    finally:
        store.close()


def test_preview_chars_zero_emits_empty_label():
    seed = _belief("seed", "anything")
    store = _seed_store([seed], [])
    try:
        js = export_graph_json([seed], [], store, preview_chars=0)
        assert js["nodes"][0]["label"] == ""
    finally:
        store.close()


# ---------------------------------------------------------------- determinism


def test_dot_byte_identical_across_runs():
    a = _belief("a", "alpha", alpha=4.0, beta=1.0)
    b = _belief("b", "beta", alpha=1.0, beta=4.0)
    c = _belief("c", "gamma", lock_level=LOCK_USER)
    store = _seed_store([a, b, c], [
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
        Edge(src="a", dst="c", type=EDGE_RELATES_TO, weight=1.0, anchor_text=None),
        Edge(src="b", dst="c", type=EDGE_CONTRADICTS, weight=1.0, anchor_text=None),
    ])
    try:
        h1 = expand_bfs([a], store, max_depth=2, nodes_per_hop=8, total_budget=16)
        h2 = expand_bfs([a], store, max_depth=2, nodes_per_hop=8, total_budget=16)
        d1 = export_dot([a], h1, store)
        d2 = export_dot([a], h2, store)
        assert d1 == d2
    finally:
        store.close()


def test_json_byte_identical_across_runs():
    import json as _json
    a = _belief("a", "alpha", alpha=4.0, beta=1.0)
    b = _belief("b", "beta", alpha=1.0, beta=4.0)
    store = _seed_store([a, b], [
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
    ])
    try:
        h1 = expand_bfs([a], store, max_depth=2, nodes_per_hop=8, total_budget=16)
        h2 = expand_bfs([a], store, max_depth=2, nodes_per_hop=8, total_budget=16)
        p1 = export_graph_json([a], h1, store)
        p2 = export_graph_json([a], h2, store)
        assert _json.dumps(p1, sort_keys=False) == _json.dumps(p2, sort_keys=False)
    finally:
        store.close()


# ---------------------------------------------------------------- DOT safety


def test_dot_escapes_quotes_and_backslashes_in_content():
    seed = _belief("id-1", 'hostile "content" with \\ backslash')
    store = _seed_store([seed], [])
    try:
        out = export_dot([seed], [], store)
        # The literal escape sequences land in the output; the raw
        # unescaped quote should not appear inside the label.
        assert '\\"content\\"' in out
        assert "\\\\ backslash" in out
    finally:
        store.close()


def test_dot_escapes_newlines_to_literal_n():
    seed = _belief("id-1", "first line\nsecond line")
    store = _seed_store([seed], [])
    try:
        out = export_dot([seed], [], store)
        # Newline becomes \n in the DOT label; the raw newline does
        # not appear inside the quoted label.
        node_line = [ln for ln in out.splitlines() if 'label=' in ln][0]
        assert "\\n" in node_line
    finally:
        store.close()


# ---------------------------------------------------------------- header


def test_dot_carries_layout_directives():
    seed = _belief("a", "x")
    store = _seed_store([seed], [])
    try:
        out = export_dot([seed], [], store)
        # rankdir=LR is the issue's expected layout shape (left-to-right
        # reads naturally for query-anchored expansions); boxes with
        # rounded corners are the conventional belief-card shape used
        # in canvas_export.
        assert "rankdir=LR" in out
        assert "shape=box" in out
        assert "style=rounded" in out
    finally:
        store.close()


# ---------------------------------------------------------------- ordering


def test_nodes_sorted_by_id_in_both_formats():
    # Insert in inverted order; assert output is ascending.
    a = _belief("zzz", "last")
    b = _belief("aaa", "first")
    store = _seed_store([a, b], [])
    try:
        js = export_graph_json([a, b], [], store)
        ids = [n["id"] for n in js["nodes"]]
        assert ids == sorted(ids)
        dot = export_dot([a, b], [], store)
        node_lines = [ln for ln in dot.splitlines() if 'label=' in ln]
        assert node_lines[0].lstrip().startswith('"aaa"')
        assert node_lines[1].lstrip().startswith('"zzz"')
    finally:
        store.close()


def test_edges_sorted_by_src_dst_type():
    a = _belief("a", "alpha")
    b = _belief("b", "beta")
    c = _belief("c", "gamma")
    store = _seed_store([a, b, c], [
        Edge(src="b", dst="c", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
        Edge(src="a", dst="c", type=EDGE_CITES, weight=1.0, anchor_text=None),
        Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0, anchor_text=None),
    ])
    try:
        hops = [
            ScoredHop(belief=b, score=0.6, depth=1, path=[EDGE_SUPPORTS],
                      belief_id_trail=("a", "b")),
            ScoredHop(belief=c, score=0.4, depth=2, path=[EDGE_SUPPORTS, EDGE_SUPPORTS],
                      belief_id_trail=("a", "b", "c")),
        ]
        js = export_graph_json([a], hops, store)
        edge_keys = [(e["src"], e["dst"], e["type"]) for e in js["edges"]]
        assert edge_keys == sorted(edge_keys)
    finally:
        store.close()
