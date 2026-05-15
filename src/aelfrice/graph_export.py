"""Query-anchored graph viewer — DOT and JSON emitters (#629).

`aelf graph <belief-id> --hops N --format dot|json` walks the belief
graph BFS-style from a small anchor set and emits a static, renderable
artifact. Sibling to the JSON Canvas exporter (#763); shares the
determinism contract and the seed-selection pipeline but emits
generic graph formats consuming tools (Graphviz, Cytoscape.js,
sigma.js, custom JS) can render without aelfrice in the loop.

Aligns with the deterministic-narrow-surface decision (PHILOSOPHY
#605): emit data; let renderers render. No JS bundle, no in-tree
HTML emitter (the issue's `--format html` was de-scoped during
ratification — operator verdict 2026-05-15 on #629).

Determinism contract
--------------------

Same store + same seeds + same hops + same filter / preview args
=> byte-identical DOT string and JSON dict. Nodes are sorted by
belief id; edges are sorted by ``(src, dst, type)``. No layout
coordinates are emitted — DOT delegates layout to ``dot``,
``neato``, ``sfdp``; JSON consumers run their own layout.

Encoding policy
---------------

- Node label: belief content truncated to ``preview_chars`` (default
  60), with ``…`` ellipsis. Faraday's Q3 ratification: deterministic
  truncation, not summarisation.
- Node color: locked => cyan; posterior mean buckets (high =>
  green, low => red, mid => no color attribute). Matches
  ``canvas_export.py`` so a user looking at both views reads the
  same visual encoding.
- Edge label: 3-4 char abbreviation of the edge type (``SUP``,
  ``CON``, ``CIT``, ``REL``, ``DRV``, ``IMP``, ``TMP``, ``TST``,
  ``SUPS``, ``RES``, ``STL``). Faraday's Q2 ratification: color
  + truncated label, both required for B/W and screen-reader
  contexts.
- Edge color: per edge type, fixed map (see ``EDGE_DOT_COLOR``).
  Consistent across runs; not derived from valence so the viewer
  distinguishes ``CITES`` from ``RELATES_TO`` even though both
  carry positive valence.
- ``--edge-types`` filter: applied at the display layer (not at
  BFS traversal). Edges of disallowed types are dropped from the
  output; nodes that BFS surfaced through them remain visible.
  Reviewer note: with a narrow filter, the BFS budget may be spent
  on hops that get filtered out at render time. Pragmatic for a
  viewer at default hops=2; if 50k-node stores hit budget pressure
  the filter can move into ``expand_bfs`` in a follow-up.

No new dependencies; stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_RELATES_TO,
    EDGE_RESOLVES,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

DEFAULT_PREVIEW_CHARS: Final[int] = 60

# Posterior buckets (mirrors canvas_export so viewers agree).
POSTERIOR_HIGH_CUTOFF: Final[float] = 0.75
POSTERIOR_LOW_CUTOFF: Final[float] = 0.25

# DOT named colors. Picked from the X11 set Graphviz ships by default
# so the output renders without a custom color scheme. Locked
# beliefs use cyan to match canvas_export; the rest are chosen for
# B/W contrast (greens darker than yellows, reds darker than oranges)
# so SVG export to a colorless context still resolves to readable
# greyscale.
NODE_DOT_COLOR_LOCKED: Final[str] = "cyan3"
NODE_DOT_COLOR_HIGH: Final[str] = "darkgreen"
NODE_DOT_COLOR_LOW: Final[str] = "firebrick"
NODE_DOT_COLOR_MID: Final[str] = ""  # no color attribute

# Edge color is per-type, consistent across runs. Distinct from
# valence so the viewer distinguishes CITES from RELATES_TO even
# though both share a positive sign.
EDGE_DOT_COLOR: Final[dict[str, str]] = {
    EDGE_SUPPORTS: "darkgreen",
    EDGE_CONTRADICTS: "firebrick",
    EDGE_CITES: "royalblue",
    EDGE_RELATES_TO: "gray50",
    EDGE_DERIVED_FROM: "purple",
    EDGE_IMPLEMENTS: "teal",
    EDGE_TEMPORAL_NEXT: "darkorange",
    EDGE_TESTS: "goldenrod",
    EDGE_SUPERSEDES: "black",
    EDGE_RESOLVES: "darkviolet",
    EDGE_POTENTIALLY_STALE: "gray70",
}

# 3-4 char edge-type abbreviations. Distinct prefixes so a user
# reading a dense graph can identify types from the label alone
# (color is the primary channel; the label is the redundant one
# per Q2). SUPS vs SUP keeps SUPERSEDES distinguishable from
# SUPPORTS.
EDGE_LABEL_ABBR: Final[dict[str, str]] = {
    EDGE_SUPPORTS: "SUP",
    EDGE_CONTRADICTS: "CON",
    EDGE_CITES: "CIT",
    EDGE_RELATES_TO: "REL",
    EDGE_DERIVED_FROM: "DRV",
    EDGE_IMPLEMENTS: "IMP",
    EDGE_TEMPORAL_NEXT: "TMP",
    EDGE_TESTS: "TST",
    EDGE_SUPERSEDES: "SUPS",
    EDGE_RESOLVES: "RES",
    EDGE_POTENTIALLY_STALE: "STL",
}


@dataclass(frozen=True)
class GraphNode:
    """Resolved node entry before serialisation.

    Kept distinct from the format-specific row dicts so tests can
    assert on a typed shape without reparsing DOT / JSON.
    """
    id: str
    label: str
    color: str  # "" => no color attribute
    locked: bool
    posterior_mean: float


@dataclass(frozen=True)
class GraphEdge:
    """Resolved edge entry before serialisation."""
    src: str
    dst: str
    edge_type: str
    label: str  # 3-4 char abbreviation
    color: str  # "" => no color attribute


def _posterior_mean(b: Belief) -> float:
    """Beta-Bernoulli posterior mean alpha/(alpha+beta).

    Mirrors ``canvas_export._posterior_mean`` — kept inline so this
    module has zero internal-API surface beyond the data classes.
    """
    denom = b.alpha + b.beta
    if denom <= 0:
        return 0.5
    return b.alpha / denom


def _node_color(b: Belief) -> str:
    if b.lock_level == LOCK_USER:
        return NODE_DOT_COLOR_LOCKED
    mu = _posterior_mean(b)
    if mu >= POSTERIOR_HIGH_CUTOFF:
        return NODE_DOT_COLOR_HIGH
    if mu < POSTERIOR_LOW_CUTOFF:
        return NODE_DOT_COLOR_LOW
    return NODE_DOT_COLOR_MID


def _truncate(s: str, n: int) -> str:
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _build_nodes(
    seeds: list[Belief],
    hops: list[ScoredHop],
    preview_chars: int,
) -> dict[str, GraphNode]:
    """Resolve unique nodes from seeds + hops with sorted-by-id order
    deferred to serialisation. Hops whose belief id collides with a
    seed yield to the seed (seeds always win, matching canvas_export)."""
    out: dict[str, GraphNode] = {}
    for s in seeds:
        out[s.id] = GraphNode(
            id=s.id,
            label=_truncate(s.content or "", preview_chars),
            color=_node_color(s),
            locked=(s.lock_level == LOCK_USER),
            posterior_mean=_posterior_mean(s),
        )
    for h in hops:
        if h.belief.id in out:
            continue
        out[h.belief.id] = GraphNode(
            id=h.belief.id,
            label=_truncate(h.belief.content or "", preview_chars),
            color=_node_color(h.belief),
            locked=(h.belief.lock_level == LOCK_USER),
            posterior_mean=_posterior_mean(h.belief),
        )
    return out


def _collect_edges(
    nodes: dict[str, GraphNode],
    store: MemoryStore,
    edge_types_filter: frozenset[str] | None,
) -> list[GraphEdge]:
    """Pull edges between any pair of surfaced nodes from the store.

    Edges whose dst is outside the surfaced node set are dropped
    (dangling-edge omission matches canvas_export). Edge-type
    filter applies after dedup so the same edge that fails the
    filter doesn't waste a slot.
    """
    node_ids = set(nodes.keys())
    seen: set[tuple[str, str, str]] = set()
    rows: list[GraphEdge] = []
    for src_id in sorted(node_ids):
        for e in store.edges_from(src_id):
            if e.dst not in node_ids:
                continue
            key = (e.src, e.dst, e.type)
            if key in seen:
                continue
            seen.add(key)
            if edge_types_filter is not None and e.type not in edge_types_filter:
                continue
            label = EDGE_LABEL_ABBR.get(e.type, e.type[:4])
            color = EDGE_DOT_COLOR.get(e.type, "")
            rows.append(GraphEdge(
                src=e.src,
                dst=e.dst,
                edge_type=e.type,
                label=label,
                color=color,
            ))
    rows.sort(key=lambda r: (r.src, r.dst, r.edge_type))
    return rows


def _dot_escape(s: str) -> str:
    """Escape a string for safe inclusion in a DOT quoted attribute.

    Per Graphviz lexer: backslash escapes itself and double-quote;
    newlines render as ``\\n`` (Graphviz native line break inside
    a label).
    """
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "")
    )


def export_dot(
    seeds: list[Belief],
    hops: list[ScoredHop],
    store: MemoryStore,
    *,
    edge_types_filter: frozenset[str] | None = None,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """Emit a Graphviz DOT digraph for the seed-anchored subgraph.

    Determinism: nodes emitted in ascending belief-id order; edges
    in ascending ``(src, dst, type)`` order. No layout coordinates;
    the consuming renderer (``dot``, ``neato``, ``sfdp``) computes
    layout.

    The output is a complete DOT document (header + closing brace)
    ready to pipe to ``dot -Tsvg``. Quoting + escaping is conservative:
    every identifier and label is enclosed in double quotes so
    belief ids containing punctuation (`-`, `_`, hex) and labels
    containing arbitrary user content render safely.
    """
    if not seeds:
        return "digraph aelfrice {\n}\n"
    nodes = _build_nodes(seeds, hops, preview_chars)
    edges = _collect_edges(nodes, store, edge_types_filter)
    lines: list[str] = [
        "digraph aelfrice {",
        '  graph [rankdir=LR];',
        '  node [shape=box, style=rounded];',
    ]
    for nid in sorted(nodes.keys()):
        n = nodes[nid]
        attrs: list[str] = [f'label="{_dot_escape(n.label)}"']
        if n.color:
            attrs.append(f'color="{n.color}"')
        if n.locked:
            attrs.append('penwidth=2')
        lines.append(f'  "{_dot_escape(n.id)}" [{", ".join(attrs)}];')
    for e in edges:
        attrs = [f'label="{e.label}"']
        if e.color:
            attrs.append(f'color="{e.color}"')
        lines.append(
            f'  "{_dot_escape(e.src)}" -> "{_dot_escape(e.dst)}" '
            f'[{", ".join(attrs)}];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_graph_json(
    seeds: list[Belief],
    hops: list[ScoredHop],
    store: MemoryStore,
    *,
    edge_types_filter: frozenset[str] | None = None,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict:
    """Emit a renderer-agnostic ``{nodes, edges}`` dict.

    Shape:
    ``{"nodes": [{"id", "label", "color"?, "locked", "posterior_mean"}],
       "edges": [{"src", "dst", "type", "label", "color"?}]}``

    Optional ``color`` is omitted (not set to empty string) when
    the encoding doesn't assign one — so consumers can use plain
    ``"color" in node`` checks.
    """
    if not seeds:
        return {"nodes": [], "edges": []}
    nodes = _build_nodes(seeds, hops, preview_chars)
    edges = _collect_edges(nodes, store, edge_types_filter)
    json_nodes: list[dict] = []
    for nid in sorted(nodes.keys()):
        n = nodes[nid]
        row: dict = {
            "id": n.id,
            "label": n.label,
            "locked": n.locked,
            "posterior_mean": n.posterior_mean,
        }
        if n.color:
            row["color"] = n.color
        json_nodes.append(row)
    json_edges: list[dict] = []
    for e in edges:
        row = {
            "src": e.src,
            "dst": e.dst,
            "type": e.edge_type,
            "label": e.label,
        }
        if e.color:
            row["color"] = e.color
        json_edges.append(row)
    return {"nodes": json_nodes, "edges": json_edges}


def iter_surfaced_beliefs(
    seeds: list[Belief],
    hops: list[ScoredHop],
) -> Iterable[Belief]:
    """Yield seeds then hop beliefs, in surfaced order. Convenience
    for callers that want the same set the viewer rendered without
    re-deriving it from the format output."""
    for s in seeds:
        yield s
    for h in hops:
        yield h.belief
