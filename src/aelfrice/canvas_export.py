"""JSON Canvas 1.0 exporter for the belief graph (#763).

Probe for the Obsidian-mapping question raised by the 2026-05-13
wonder + reason passes. Emits a deterministic ``.canvas`` JSON
document from a seed set + BFS expansion. **One-way: DB -> canvas.**
There is no import path; the canvas is a regenerable visualization
artifact, not a content store.

Format spec: JSON Canvas 1.0
(https://github.com/obsidianmd/jsoncanvas/blob/main/spec/1.0.md).

Determinism contract
--------------------

Same store + same seeds + same expansion params -> byte-identical
JSON output. Positions are computed from belief id hash (stable across
runs), not from any force-directed layout. Honors locked
``c06f8d575fad71fb`` (deterministic narrow surface).

Encoding policy (subject to operator feedback per #763)
-------------------------------------------------------

- Nodes: ``type="text"``, ``text = belief.content[:NODE_TEXT_MAX]``.
  Locked beliefs get preset color ``"5"`` (cyan); otherwise posterior
  mean buckets to ``"4"`` green (mu>=0.75), no color (0.25<=mu<0.75),
  or ``"6"`` red (mu<0.25). Width/height fixed.
- Edges: ``label = edge.type``; optional truncated ``anchor_text``
  appended (cap 40 chars). Color from EDGE_VALENCE sign: positive ->
  ``"4"``, negative -> ``"6"``, zero -> no color.
- Layout: concentric rings by BFS depth from seed centroid. Within
  a ring, angle = ``hash(belief_id) % 360`` so siblings spread but
  reproducibly.

No new dependencies; pure stdlib + existing aelfrice surface.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, Iterable

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    EDGE_VALENCE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

JSON_CANVAS_VERSION: Final[str] = "1.0"

NODE_TEXT_MAX: Final[int] = 200
EDGE_LABEL_ANCHOR_MAX: Final[int] = 40

# Fixed node geometry (JSON Canvas spec requires explicit width/height).
NODE_WIDTH: Final[int] = 320
NODE_HEIGHT: Final[int] = 140

# Concentric-ring layout constants.
RING_RADIUS_STEP: Final[int] = 480  # pixels between successive depths
SEED_CIRCLE_RADIUS: Final[int] = 240  # spread of multiple seeds around origin

# Posterior mean buckets -> JSON Canvas preset color ("" = no color).
POSTERIOR_HIGH_CUTOFF: Final[float] = 0.75
POSTERIOR_LOW_CUTOFF: Final[float] = 0.25
COLOR_LOCKED: Final[str] = "5"   # cyan
COLOR_POSTERIOR_HIGH: Final[str] = "4"  # green
COLOR_POSTERIOR_MID: Final[str] = ""    # default (no color)
COLOR_POSTERIOR_LOW: Final[str] = "6"   # red

COLOR_EDGE_POSITIVE: Final[str] = "4"  # green
COLOR_EDGE_NEGATIVE: Final[str] = "6"  # red
COLOR_EDGE_NEUTRAL: Final[str] = ""    # default


@dataclass(frozen=True)
class CanvasNode:
    """Lightweight node spec before JSON serialisation.

    Kept distinct from the spec's free-form dict so the test surface
    can assert on a typed shape without parsing JSON.
    """
    id: str
    text: str
    x: int
    y: int
    color: str  # "" = no color attribute emitted
    locked: bool
    posterior_mean: float


@dataclass(frozen=True)
class CanvasEdge:
    """Lightweight edge spec before JSON serialisation."""
    id: str
    from_node: str
    to_node: str
    label: str
    color: str  # "" = no color attribute emitted


def _posterior_mean(b: Belief) -> float:
    """Beta-Bernoulli posterior mean alpha/(alpha+beta).

    Mirrors ``aelfrice.scoring.posterior_mean`` but kept inline to
    avoid adding scoring as a canvas dep. With Jeffreys prior (0.5,
    0.5) an unobserved belief reads 0.5.
    """
    denom = b.alpha + b.beta
    if denom <= 0:
        return 0.5
    return b.alpha / denom


def _node_color(b: Belief) -> str:
    if b.lock_level == LOCK_USER:
        return COLOR_LOCKED
    mu = _posterior_mean(b)
    if mu >= POSTERIOR_HIGH_CUTOFF:
        return COLOR_POSTERIOR_HIGH
    if mu < POSTERIOR_LOW_CUTOFF:
        return COLOR_POSTERIOR_LOW
    return COLOR_POSTERIOR_MID


def _edge_color(edge_type: str) -> str:
    valence = EDGE_VALENCE.get(edge_type)
    if valence is None or valence == 0:
        return COLOR_EDGE_NEUTRAL
    return COLOR_EDGE_POSITIVE if valence > 0 else COLOR_EDGE_NEGATIVE


def _stable_angle(belief_id: str) -> float:
    """Deterministic angle in [0, 2*pi) keyed by belief id.

    Uses sha256 of the id so a layout doesn't change when an unrelated
    seed is added/removed — the angle for any given belief id is
    fixed across runs.
    """
    h = hashlib.sha256(belief_id.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:4], "big")
    return (bucket / 0xFFFFFFFF) * 2.0 * math.pi


def _seed_center(seeds: list[Belief]) -> dict[str, tuple[int, int]]:
    """Place seeds around the origin so multi-seed walks fan out.

    Single seed: at (0, 0). Multi-seed: evenly spaced around a circle
    of radius ``SEED_CIRCLE_RADIUS``, sorted by belief id so order is
    deterministic.
    """
    if not seeds:
        return {}
    if len(seeds) == 1:
        return {seeds[0].id: (0, 0)}
    by_id = sorted(seeds, key=lambda b: b.id)
    n = len(by_id)
    out: dict[str, tuple[int, int]] = {}
    for i, b in enumerate(by_id):
        angle = (i / n) * 2.0 * math.pi
        out[b.id] = (
            int(SEED_CIRCLE_RADIUS * math.cos(angle)),
            int(SEED_CIRCLE_RADIUS * math.sin(angle)),
        )
    return out


def _node_position(
    belief_id: str,
    depth: int,
    seed_anchor: tuple[int, int],
) -> tuple[int, int]:
    """Position for a non-seed belief at the given BFS depth.

    Ring radius = ``depth * RING_RADIUS_STEP`` from the seed anchor.
    Angle from a stable hash of belief id so siblings spread out.
    """
    angle = _stable_angle(belief_id)
    r = depth * RING_RADIUS_STEP
    return (
        seed_anchor[0] + int(r * math.cos(angle)),
        seed_anchor[1] + int(r * math.sin(angle)),
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _edge_label(edge_type: str, anchor_text: str | None) -> str:
    if not anchor_text:
        return edge_type
    snippet = _truncate(anchor_text.strip(), EDGE_LABEL_ANCHOR_MAX)
    if not snippet:
        return edge_type
    return f"{edge_type}: {snippet}"


def export_canvas(
    seeds: list[Belief],
    hops: list[ScoredHop],
    store: MemoryStore,
) -> dict:
    """Build a JSON Canvas 1.0 document from seeds + BFS hops.

    Pure: reads the store only to materialise edge labels (one
    ``edges_from`` query per beliefs-in-graph for label resolution).
    Returns the canvas dict ready for ``json.dumps``.

    Edges in the output are deduplicated by (src, dst, type); only
    edges whose both endpoints are in the surfaced node set are
    emitted (so the canvas isn't littered with dangling arrows to
    beliefs that didn't make the BFS cut).
    """
    if not seeds:
        return {"nodes": [], "edges": []}

    seed_positions = _seed_center(seeds)
    # belief_id -> CanvasNode
    nodes: dict[str, CanvasNode] = {}
    # belief_id -> seed_anchor (origin of the ring this hop sits on)
    seed_anchor_for: dict[str, tuple[int, int]] = {}

    for s in seeds:
        pos = seed_positions[s.id]
        seed_anchor_for[s.id] = pos
        nodes[s.id] = CanvasNode(
            id=s.id,
            text=_truncate(s.content or "", NODE_TEXT_MAX),
            x=pos[0],
            y=pos[1],
            color=_node_color(s),
            locked=(s.lock_level == LOCK_USER),
            posterior_mean=_posterior_mean(s),
        )

    # Hops carry belief_id_trail starting at the seed; use trail[0] as
    # the seed anchor for ring placement.
    for h in hops:
        if h.belief.id in nodes:
            continue
        trail_seed = h.belief_id_trail[0] if h.belief_id_trail else seeds[0].id
        anchor = seed_anchor_for.get(trail_seed, seed_positions.get(seeds[0].id, (0, 0)))
        x, y = _node_position(h.belief.id, h.depth, anchor)
        nodes[h.belief.id] = CanvasNode(
            id=h.belief.id,
            text=_truncate(h.belief.content or "", NODE_TEXT_MAX),
            x=x,
            y=y,
            color=_node_color(h.belief),
            locked=(h.belief.lock_level == LOCK_USER),
            posterior_mean=_posterior_mean(h.belief),
        )
        seed_anchor_for[h.belief.id] = anchor

    # Pull edges between any pair of beliefs in the node set. Only
    # local-DB edges; federation peer edges deliberately not surfaced
    # in v0 per #661 read-only-federation lock + #763 scope.
    node_ids = set(nodes.keys())
    seen_edges: set[tuple[str, str, str]] = set()
    canvas_edges: list[CanvasEdge] = []
    for src_id in sorted(node_ids):
        for e in store.edges_from(src_id):
            if e.dst not in node_ids:
                continue
            key = (e.src, e.dst, e.type)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            canvas_edges.append(
                CanvasEdge(
                    id=f"e_{e.src}_{e.dst}_{e.type}",
                    from_node=e.src,
                    to_node=e.dst,
                    label=_edge_label(e.type, e.anchor_text),
                    color=_edge_color(e.type),
                )
            )

    # Serialise to JSON Canvas 1.0 shape. Nodes sorted by id for
    # determinism; edges sorted by (from, to, type).
    json_nodes: list[dict] = []
    for nid in sorted(nodes.keys()):
        n = nodes[nid]
        row: dict = {
            "id": n.id,
            "type": "text",
            "text": n.text,
            "x": n.x,
            "y": n.y,
            "width": NODE_WIDTH,
            "height": NODE_HEIGHT,
        }
        if n.color:
            row["color"] = n.color
        json_nodes.append(row)

    canvas_edges.sort(key=lambda e: (e.from_node, e.to_node, e.label))
    json_edges: list[dict] = []
    for e in canvas_edges:
        row = {
            "id": e.id,
            "fromNode": e.from_node,
            "toNode": e.to_node,
            "toEnd": "arrow",
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
    for callers that want to operate on the same set the canvas
    rendered without re-deriving it from the JSON dict."""
    for s in seeds:
        yield s
    for h in hops:
        yield h.belief
