"""HRR vocabulary-bridge expansion lane (#981).

Restores the deterministic, embeddings-free FFT bind/probe *expansion* lane
that the v3.0.1 calibration notes name as the cause of the LoCoMo delta
(66.1% -> 40.88%). Default-OFF behind ``use_hrr_expand`` on
:func:`aelfrice.retrieval.retrieve_v2`; this module holds the lane mechanics,
the flag resolver lives in :func:`aelfrice.retrieval.is_hrr_expand_enabled`.

The lane is a *candidate-expansion* step, distinct from the #152 structural
lane (a parallel marker-routed query path that replaces the textual lane).
When enabled it seeds from the top FTS5 hits, probes the shared
:class:`~aelfrice.hrr_index.HRRStructIndex` for single-hop semantic
neighbours in both edge directions, and merges the recovered beliefs into the
candidate set *before* scoring/packing — exactly the predecessor's L3
``_hrr_expand`` step (predecessor ``retrieval.py:348-389``), re-expressed on
the current structural-index substrate rather than the removed ``HRRGraph``.

The struct index encodes only *outgoing* edges
(``struct[b] = sum bind(role[e.kind], id[e.dst])``), so:

- **forward** out-neighbours of a seed are recovered by
  ``unbind(role[kind], struct[seed])`` followed by a cleanup matvec against
  the id-vector matrix;
- **reverse** in-neighbours are recovered by the existing
  :meth:`HRRStructIndex.probe`, which ranks beliefs whose outgoing structure
  contains ``bind(role[kind], id[seed])``.

Determinism (#981 AC2, AC5): every operation is a numpy FFT / matvec over the
deterministically-seeded struct matrix. There is **no** ``random`` /
``betavariate`` anywhere in this path. :func:`precompute_expand_neighbors`
materialises a byte-stable ``hrr_expand_neighbors`` SQLite table (row order:
``similarity`` DESC, then ``neighbor_id`` ASC); two builds over the same store
produce an identical table.

This is *implement-and-ablate* (#981): it lands the lane plus its ablation
arm and does **not** flip any default. Flipping the default reverses the
locked #605 determinism philosophy and is routed to a re-opened #897.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Final

import numpy as np

from aelfrice.hrr import unbind
from aelfrice.models import EDGE_TYPES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aelfrice.hrr_index import HRRStructIndex
    from aelfrice.models import Belief
    from aelfrice.store import MemoryStore

# Semantic edge kinds worth probing for vocabulary bridging. The predecessor
# (``retrieval.py:335-345``) used SUPERSEDES / CONTRADICTS / SUPPORTS / CALLS
# / CITES / TESTS / IMPLEMENTS. ``CALLS`` is not in the current
# ``models.EDGE_TYPES`` schema, so the *live* set is the intersection with
# EDGE_TYPES (computed at call time via :func:`hrr_expand_edge_types` so a
# future CALLS edge type joins automatically). Co-occurrence / structural
# edges (RELATES_TO, DERIVED_FROM, TEMPORAL_NEXT, RESOLVES) add noise without
# vocabulary-gap signal and are deliberately excluded.
_HRR_EXPAND_KIND_NAMES: Final[frozenset[str]] = frozenset(
    {
        "SUPERSEDES",
        "CONTRADICTS",
        "SUPPORTS",
        "CALLS",
        "CITES",
        "TESTS",
        "IMPLEMENTS",
    }
)

# Lane defaults, mirroring the predecessor: cap seeds at 5, pull top-3 per
# (seed, kind, direction) probe, return the top-5 merged neighbours. Fixed
# here (not Thompson-sampled — there is no sampling in this lane).
DEFAULT_SEED_CAP: Final[int] = 5
DEFAULT_PER_PROBE_K: Final[int] = 3
DEFAULT_EXPAND_TOP_K: Final[int] = 5
DEFAULT_MAX_NODES: Final[int] = 2000
# Similarity floor on the raw HRR inner product. Single-hop typed edges are
# *bimodal* under this algebra: a present edge recovers its bound term exactly
# (``unbind(role_k, bind(role_k, id_t)) · id_t == 1.0`` forward; the matching
# ``bind`` self-dot ``≈ 1.0`` reverse), independent of the belief's degree —
# only the additive crosstalk from the belief's *other* bindings varies, at
# ~1/sqrt(dim) (~0.044 at the default dim=512) per term. So true matches
# cluster at ~1.0 and absent edges at ~noise; 0.5 sits squarely in the gap,
# keeping every present edge while rejecting cleanup spuria. (Empirically a
# spurious TESTS-typed hit on a no-TESTS-edge belief scores ~0.12-0.17 at
# dim=512 — well below 0.5; a false positive would need ~128 same-role
# crosstalk terms to clear the floor.)
DEFAULT_SIM_FLOOR: Final[float] = 0.5
# ``created_at`` is an audit-only column never read by retrieval. The cache
# is a derived, wholesale-rebuilt table, so its default rebuild timestamp is a
# fixed sentinel (not wall-clock) — two rebuilds over an unchanged store stay
# byte-identical (#981 AC2).
_REBUILD_CREATED_AT: Final[str] = "1970-01-01T00:00:00+00:00"

FORWARD: Final[str] = "forward"
REVERSE: Final[str] = "reverse"


def hrr_expand_edge_types() -> tuple[str, ...]:
    """Semantic edge kinds probed by the expansion lane.

    Returns the intersection of the predecessor's HRR edge set with the live
    :data:`aelfrice.models.EDGE_TYPES`, sorted for deterministic iteration.
    """
    return tuple(sorted(_HRR_EXPAND_KIND_NAMES & EDGE_TYPES))


def _id_matrix(index: "HRRStructIndex") -> np.ndarray:
    """Build the ``(N, dim)`` id-vector matrix aligned to ``belief_ids``.

    ``index.id_vecs`` is keyed by belief id; row ``i`` corresponds to
    ``index.belief_ids[i]`` so a cleanup matvec result indexes straight back
    into ``belief_ids``. Returns an empty ``(0, dim)`` array for an empty
    index.
    """
    bids = index.belief_ids
    if not bids:
        return np.zeros((0, index.dim), dtype=np.float64)
    return np.stack([np.asarray(index.id_vecs[bid]) for bid in bids], axis=0)


def _topk_from_scores(
    scores: np.ndarray,
    belief_ids: list[str],
    *,
    exclude: str,
    top_k: int,
    sim_floor: float,
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(belief_id, score)`` pairs above ``sim_floor``.

    Sorted by score DESC then belief id ASC (the deterministic tie-break used
    everywhere in this lane). ``exclude`` drops the seed itself.
    """
    if scores.size == 0 or top_k <= 0:
        return []
    # Candidate rows above the floor, excluding the seed. Float comparison is
    # deterministic for identical inputs.
    hits: list[tuple[str, float]] = []
    for i in np.nonzero(scores >= sim_floor)[0]:
        bid = belief_ids[int(i)]
        if bid == exclude:
            continue
        hits.append((bid, float(scores[int(i)])))
    hits.sort(key=lambda p: (-p[1], p[0]))
    return hits[:top_k]


def _forward_neighbors(
    index: "HRRStructIndex",
    id_matrix: np.ndarray,
    seed_id: str,
    kind: str,
    *,
    top_k: int,
    sim_floor: float,
) -> list[tuple[str, float]]:
    """Out-neighbours: beliefs the seed points TO via ``kind``.

    ``unbind(role[kind], struct[seed])`` recovers the superposed id vectors of
    ``kind``-typed destinations; the cleanup matvec ``id_matrix @ probe`` ranks
    candidate beliefs. Deterministic.
    """
    role = index.role_vecs.get(kind)
    row = index._index.get(seed_id)
    if role is None or row is None or index.struct.size == 0:
        return []
    probe = unbind(np.asarray(role), np.ascontiguousarray(index.struct[row]))
    scores = id_matrix @ probe
    return _topk_from_scores(
        scores, index.belief_ids, exclude=seed_id, top_k=top_k, sim_floor=sim_floor,
    )


def _reverse_neighbors(
    index: "HRRStructIndex",
    seed_id: str,
    kind: str,
    *,
    top_k: int,
    sim_floor: float,
) -> list[tuple[str, float]]:
    """In-neighbours: beliefs that point TO the seed via ``kind``.

    Delegates to :meth:`HRRStructIndex.probe` (which ranks beliefs whose
    outgoing structure contains ``bind(role[kind], id[seed])``), then applies
    the floor and the deterministic tie-break.
    """
    raw = index.probe(kind, seed_id, top_k=top_k)
    hits = [(bid, score) for bid, score in raw if score >= sim_floor and bid != seed_id]
    hits.sort(key=lambda p: (-p[1], p[0]))
    return hits[:top_k]


def neighbor_rows(
    index: "HRRStructIndex",
    seed_id: str,
    *,
    id_matrix: np.ndarray | None = None,
    kinds: tuple[str, ...] | None = None,
    per_probe_k: int = DEFAULT_PER_PROBE_K,
    sim_floor: float = DEFAULT_SIM_FLOOR,
) -> list[tuple[str, str, str, float]]:
    """Per-direction neighbour rows for one seed.

    Returns ``(neighbor_id, edge_type, direction, similarity)`` tuples across
    every probed kind and both directions, sorted by ``similarity`` DESC then
    ``neighbor_id`` ASC then ``edge_type`` ASC then ``direction`` ASC — a
    total order, so the output (and any table built from it) is byte-stable.
    """
    if kinds is None:
        kinds = hrr_expand_edge_types()
    if id_matrix is None:
        id_matrix = _id_matrix(index)
    rows: list[tuple[str, str, str, float]] = []
    for kind in kinds:
        for nid, sim in _forward_neighbors(
            index, id_matrix, seed_id, kind,
            top_k=per_probe_k, sim_floor=sim_floor,
        ):
            rows.append((nid, kind, FORWARD, sim))
        for nid, sim in _reverse_neighbors(
            index, seed_id, kind, top_k=per_probe_k, sim_floor=sim_floor,
        ):
            rows.append((nid, kind, REVERSE, sim))
    rows.sort(key=lambda r: (-r[3], r[0], r[1], r[2]))
    return rows


def precompute_expand_neighbors(
    store: "MemoryStore",
    index: "HRRStructIndex",
    *,
    per_probe_k: int = DEFAULT_PER_PROBE_K,
    max_nodes: int = DEFAULT_MAX_NODES,
    sim_floor: float = DEFAULT_SIM_FLOOR,
    now_iso: str | None = None,
) -> int:
    """Materialise the ``hrr_expand_neighbors`` table; return rows written.

    For each active belief (``valid_to IS NULL``, ``id`` ASC, capped at
    ``max_nodes``) the lane's forward + reverse single-hop neighbours are
    written across every semantic edge kind. The table is fully replaced each
    run. Insertion order is the total order from :func:`neighbor_rows`, so two
    runs over an unchanged store produce a byte-identical table (#981 AC2).

    ``now_iso`` pins the ``created_at`` column; it defaults to a fixed rebuild
    sentinel (not wall-clock) so two rebuilds over an unchanged store stay
    byte-identical (#981 AC2). ``created_at`` is audit-only and never read by
    retrieval, so this derived cache prefers an idempotent rebuild over a real
    timestamp.
    """
    conn: sqlite3.Connection = store._conn  # noqa: SLF001 — same pattern as replay.py/telemetry.py
    if now_iso is None:
        now_iso = _REBUILD_CREATED_AT
    kinds = hrr_expand_edge_types()
    id_matrix = _id_matrix(index)

    seed_ids: list[str] = [
        str(r[0])
        for r in conn.execute(
            "SELECT id FROM beliefs WHERE valid_to IS NULL "
            "ORDER BY id ASC LIMIT ?",
            (max_nodes,),
        ).fetchall()
    ]

    conn.execute("DELETE FROM hrr_expand_neighbors")
    written = 0
    for seed_id in seed_ids:
        rows = neighbor_rows(
            index, seed_id,
            id_matrix=id_matrix, kinds=kinds,
            per_probe_k=per_probe_k, sim_floor=sim_floor,
        )
        if not rows:
            continue
        conn.executemany(
            "INSERT OR REPLACE INTO hrr_expand_neighbors "
            "(belief_id, neighbor_id, similarity, edge_type, direction, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (seed_id, nid, sim, etype, direction, now_iso)
                for (nid, etype, direction, sim) in rows
            ],
        )
        written += len(rows)
    conn.commit()
    return written


def _neighbors_from_table(
    store: "MemoryStore", seed_ids: list[str], *, sim_floor: float,
) -> list[str] | None:
    """Look up precomputed neighbour ids for the seeds, or ``None`` if the
    table is absent / empty.

    Returns neighbour ids ordered by ``similarity`` DESC then ``neighbor_id``
    ASC (deduplicated, keeping the strongest similarity per neighbour). A
    missing table (pre-migration DB) returns ``None`` so the caller falls back
    to a live probe.
    """
    conn: sqlite3.Connection = store._conn  # noqa: SLF001 — same pattern as replay.py/telemetry.py
    placeholders = ",".join("?" for _ in seed_ids)
    try:
        rows = conn.execute(
            f"SELECT neighbor_id, MAX(similarity) AS sim "
            f"FROM hrr_expand_neighbors "
            f"WHERE belief_id IN ({placeholders}) AND similarity >= ? "
            f"GROUP BY neighbor_id "
            f"ORDER BY sim DESC, neighbor_id ASC",
            (*seed_ids, sim_floor),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None
    return [str(r[0]) for r in rows]


def expand_seeds(
    store: "MemoryStore",
    index: "HRRStructIndex",
    seed_ids: list[str],
    *,
    seed_cap: int = DEFAULT_SEED_CAP,
    per_probe_k: int = DEFAULT_PER_PROBE_K,
    top_k: int = DEFAULT_EXPAND_TOP_K,
    sim_floor: float = DEFAULT_SIM_FLOOR,
) -> list["Belief"]:
    """Single-hop HRR expansion of the FTS5 seeds into extra candidates.

    Caps the seeds at ``seed_cap``, gathers forward + reverse semantic
    neighbours (from the precomputed ``hrr_expand_neighbors`` table when
    present, else a live probe against ``index``), drops seeds themselves and
    soft-deleted / invalid beliefs, and returns up to ``top_k`` beliefs in
    deterministic similarity order. Empty seed list or empty index returns
    ``[]``.
    """
    seeds = [s for s in seed_ids[:seed_cap] if s]
    if not seeds or index.struct.size == 0 or top_k <= 0:
        return []
    seed_set = set(seeds)

    neighbor_ids = _neighbors_from_table(store, seeds, sim_floor=sim_floor)
    if neighbor_ids is None:
        # Live fallback: probe the index directly. Merge across seeds keeping
        # the strongest similarity per neighbour, deterministic tie-break.
        id_matrix = _id_matrix(index)
        best: dict[str, float] = {}
        for seed_id in seeds:
            for nid, _etype, _dir, sim in neighbor_rows(
                index, seed_id,
                id_matrix=id_matrix,
                per_probe_k=per_probe_k, sim_floor=sim_floor,
            ):
                if sim > best.get(nid, float("-inf")):
                    best[nid] = sim
        neighbor_ids = [
            nid for nid, _sim in sorted(
                best.items(), key=lambda p: (-p[1], p[0]),
            )
        ]

    out: list["Belief"] = []
    for nid in neighbor_ids:
        if nid in seed_set:
            continue
        belief = store.get_belief(nid)
        if belief is None or belief.valid_to is not None:
            continue
        out.append(belief)
        if len(out) >= top_k:
            break
    return out
