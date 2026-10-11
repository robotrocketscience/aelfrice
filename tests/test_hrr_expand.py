"""Tests for the HRR vocabulary-bridge expansion lane (#981).

Covers the issue's acceptance criteria:

1. ``use_hrr_expand`` resolves via env > kwarg > TOML > default-OFF; passing
   it never raises.
2. The lane is deterministic — the live probe's neighbour rows are
   byte-equal across two runs over the same store.
3. No regression with the flag off — ``retrieve_v2`` output is byte-identical
   to the pre-lane path.
5. No ``random`` / ``betavariate`` is introduced into the lane.
6. The default stays OFF.
"""
from __future__ import annotations

import pathlib
import struct

import pytest

from aelfrice import hrr_expand as hx
from aelfrice.hrr_index import HRRStructIndex, HRRStructIndexCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
    EDGE_TYPES,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    ENV_HRR_EXPAND,
    is_hrr_expand_enabled,
    retrieve_v2,
)
from aelfrice.store import MemoryStore

_SEED = 7
# Probe repetitions in the determinism test. A per-call coin-flip reorder
# of two rows survives all of them with probability 2**-_REPEATS.
_REPEATS = 10


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _toy_store() -> MemoryStore:
    """b1 -CONTRADICTS-> b2; b3 -SUPPORTS-> b2; b4 -CITES-> b5;
    b1 -RELATES_TO-> b5 (RELATES_TO is *not* a probed semantic kind)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha contradicts gamma"))
    s.insert_belief(_mk("b2", "beta singular target topic"))
    s.insert_belief(_mk("b3", "zeta supports gamma"))
    s.insert_belief(_mk("b4", "delta cites epsilon"))
    s.insert_belief(_mk("b5", "epsilon material"))
    s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="b3", dst="b2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b4", dst="b5", type=EDGE_CITES, weight=1.0))
    s.insert_edge(Edge(src="b1", dst="b5", type=EDGE_RELATES_TO, weight=1.0))
    return s


def _built_index(store: MemoryStore) -> HRRStructIndex:
    idx = HRRStructIndex(dim=512, seed=_SEED)
    idx.build(store, seed=_SEED)
    return idx


# --- AC1 / AC6: resolver --------------------------------------------------


def test_resolver_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)
    assert is_hrr_expand_enabled() is False


def test_resolver_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HRR_EXPAND, "1")
    assert is_hrr_expand_enabled() is True
    monkeypatch.setenv(ENV_HRR_EXPAND, "0")
    assert is_hrr_expand_enabled() is False
    # Env wins over an explicit kwarg.
    monkeypatch.setenv(ENV_HRR_EXPAND, "0")
    assert is_hrr_expand_enabled(True) is False


def test_resolver_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)
    assert is_hrr_expand_enabled(True) is True
    assert is_hrr_expand_enabled(False) is False


def test_resolver_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_hrr_expand = true\n"
    )
    assert is_hrr_expand_enabled(start=tmp_path) is True


def test_resolver_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unrecognised env value falls through to the next rung, not an error.
    monkeypatch.setenv(ENV_HRR_EXPAND, "maybe")
    assert is_hrr_expand_enabled() is False
    assert is_hrr_expand_enabled(True) is True


# --- edge-type set --------------------------------------------------------


def test_edge_types_intersect_live_schema() -> None:
    probed = set(hx.hrr_expand_edge_types())
    # Every probed kind is a real edge type.
    assert probed <= EDGE_TYPES
    # The semantic kinds present in the current schema are probed.
    for kind in ("SUPERSEDES", "CONTRADICTS", "SUPPORTS", "CITES", "TESTS", "IMPLEMENTS"):
        assert kind in probed
    # CALLS is not in the current schema, so it is not probed.
    assert "CALLS" not in probed
    # Co-occurrence / structural kinds are excluded.
    assert "RELATES_TO" not in probed
    # Deterministic (sorted) iteration order.
    assert list(hx.hrr_expand_edge_types()) == sorted(probed)


# --- AC2: determinism -----------------------------------------------------


def _all_rows(
    store: MemoryStore, idx: HRRStructIndex,
) -> list[tuple[str, str, str, str, bytes]]:
    """Every live-probe neighbour row for every active belief, in id order.

    Each row is ``(seed, neighbor, edge_type, direction, similarity)`` with
    the similarity packed as its 8 IEEE-754 bytes, so equality is byte
    equality, not float closeness.
    """
    seeds = [
        str(r[0])
        for r in store._conn.execute(  # noqa: SLF001
            "SELECT id FROM beliefs WHERE valid_to IS NULL ORDER BY id"
        ).fetchall()
    ]
    return [
        (seed, nid, etype, direction, struct.pack("<d", sim))
        for seed in seeds
        for nid, etype, direction, sim in hx.neighbor_rows(idx, seed)
    ]


def test_live_probe_rows_byte_stable_across_runs() -> None:
    store = _toy_store()
    idx = _built_index(store)
    # Repeat the probe so an ordering that varies between calls cannot
    # match by chance: in the toy store only b2 has two neighbour rows, so a
    # single comparison would miss a coin-flip reorder half the time.
    first = _all_rows(store, idx)
    assert first  # the toy store has semantic edges, so rows exist
    for _ in range(_REPEATS):
        assert _all_rows(store, idx) == first
    seeds = ["b1", "b2", "b4"]
    first_ids = [b.id for b in hx.expand_seeds(store, idx, seeds)]
    for _ in range(_REPEATS):
        assert [b.id for b in hx.expand_seeds(store, idx, seeds)] == first_ids


def test_live_probe_rows_stable_across_independent_index_builds() -> None:
    # Two stores with identical content + a fresh index build each produce
    # the same neighbour rows (the index seed is fixed).
    store_a = _toy_store()
    store_b = _toy_store()
    rows_a = _all_rows(store_a, _built_index(store_a))
    rows_b = _all_rows(store_b, _built_index(store_b))
    assert rows_a
    assert rows_a == rows_b


def test_true_edges_dominate_noise_floor() -> None:
    store = _toy_store()
    idx = _built_index(store)
    rows = _all_rows(store, idx)
    # Only the three semantic edges (each surfaced forward + reverse) survive
    # the floor; the RELATES_TO edge and all cleanup noise are rejected.
    pairs = {(r[0], r[1]) for r in rows}
    assert ("b1", "b2") in pairs and ("b2", "b1") in pairs  # CONTRADICTS
    assert ("b2", "b3") in pairs and ("b3", "b2") in pairs  # SUPPORTS
    assert ("b4", "b5") in pairs and ("b5", "b4") in pairs  # CITES
    # RELATES_TO is not a probed kind → never surfaces.
    assert ("b1", "b5") not in pairs and ("b5", "b1") not in pairs
    # Every surviving similarity is near a true bound term (~1.0), well above
    # the per-pair noise floor.
    sims = [struct.unpack("<d", r[4])[0] for r in rows]
    assert all(sim > 0.5 for sim in sims)
    assert min(sims) > idx.noise_floor() * 5


# --- neighbour recovery + expand_seeds ------------------------------------


def test_expand_seeds_surfaces_both_directions() -> None:
    store = _toy_store()
    idx = _built_index(store)
    # b2's in-neighbours are b1 (CONTRADICTS) and b3 (SUPPORTS).
    got = sorted(b.id for b in hx.expand_seeds(store, idx, ["b2"]))
    assert got == ["b1", "b3"]


def test_expand_seeds_returns_strongest_first() -> None:
    """Killed by: sorting the merged neighbours weakest-first. The order
    decides which beliefs survive the ``top_k`` cut."""
    store = _toy_store()
    idx = _built_index(store)
    # b2's in-neighbours: b1 (CONTRADICTS, ~1.077) is stronger than
    # b3 (SUPPORTS, ~1.001).
    assert [b.id for b in hx.expand_seeds(store, idx, ["b2"])] == ["b1", "b3"]
    assert [b.id for b in hx.expand_seeds(store, idx, ["b2"], top_k=1)] == ["b1"]


def test_expand_seeds_sorts_across_seeds() -> None:
    """Killed by: dropping the merge sort, so neighbours come out in the
    order the seeds found them. b4 finds b5 (~0.992) before b2 finds b1
    (~1.077) and b3 (~1.001), so insertion order puts the weakest first."""
    store = _toy_store()
    idx = _built_index(store)
    got = [b.id for b in hx.expand_seeds(store, idx, ["b4", "b2"])]
    assert got == ["b1", "b3", "b5"]
    assert [b.id for b in hx.expand_seeds(store, idx, ["b4", "b2"], top_k=1)] == ["b1"]


def _ordering_store() -> MemoryStore:
    """sa -CONTRADICTS-> hub; sb -SUPPORTS-> hub; sb -CITES-> tgt.

    Two filler beliefs come first because the similarities depend on each
    belief's position in the index. With them, sb's probe appends tgt
    (CITES) before the stronger hub (SUPPORTS), and tgt's similarity falls
    between hub's two, so append order and a keep-weakest merge both give
    a different order from the correct one; with the seeds in both
    orders, so do keep-first and keep-last merges. Only 2 fillers give
    this ordering, which `test_ordering_fixture_discriminates` guards.
    """
    s = MemoryStore(":memory:")
    for k in range(2):
        s.insert_belief(_mk(f"f{k}", f"filler {k}"))
    for bid in ("hub", "sa", "sb", "tgt"):
        s.insert_belief(_mk(bid, f"{bid} content"))
    s.insert_edge(Edge(src="sa", dst="hub", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="sb", dst="hub", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="sb", dst="tgt", type=EDGE_CITES, weight=1.0))
    return s


def _sims(idx: HRRStructIndex, seed: str) -> dict[str, float]:
    return {nid: sim for nid, _e, _d, sim in hx.neighbor_rows(idx, seed)}


def test_ordering_fixture_discriminates() -> None:
    """The precondition the next two tests rely on. If the index math
    changes and this fails, rebuild the fixture; don't loosen the tests."""
    store = _ordering_store()
    idx = _built_index(store)
    via_sa, via_sb = _sims(idx, "sa")["hub"], _sims(idx, "sb")["hub"]
    tgt = _sims(idx, "sb")["tgt"]
    assert via_sa < tgt < via_sb


def test_neighbor_rows_are_strongest_first() -> None:
    """Killed by: dropping or reversing the row sort. sb's probe appends
    tgt before the stronger hub."""
    store = _ordering_store()
    idx = _built_index(store)
    rows = hx.neighbor_rows(idx, "sb")
    assert [r[0] for r in rows] == ["hub", "tgt"]
    sims = [r[3] for r in rows]
    assert sims == sorted(sims, reverse=True)


def test_expand_seeds_keeps_each_neighbours_strongest_similarity() -> None:
    """Killed by: keeping a neighbour's weakest similarity across seeds, or
    the first or last one seen. hub ranks above tgt only on its stronger
    sighting, which comes second in one seed order and first in the other."""
    store = _ordering_store()
    idx = _built_index(store)
    for seeds in (["sa", "sb"], ["sb", "sa"]):
        got = [b.id for b in hx.expand_seeds(store, idx, seeds)]
        assert got == ["hub", "tgt"], seeds


def test_fresh_store_has_no_neighbour_cache_table() -> None:
    # #1658: the schema no longer creates `hrr_expand_neighbors`.
    store = MemoryStore(":memory:")
    row = store._conn.execute(  # noqa: SLF001
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name = 'hrr_expand_neighbors'"
    ).fetchone()
    assert row is None


def test_expand_seeds_ignores_legacy_neighbour_table() -> None:
    # A store created before #1658 still has the `hrr_expand_neighbors`
    # table. expand_seeds must not read it: a stale row naming b4 as b2's
    # neighbour must not reach the result, which comes from the live probe.
    store = _toy_store()
    idx = _built_index(store)
    live_ids = [b.id for b in hx.expand_seeds(store, idx, ["b2"])]
    with store._conn:  # noqa: SLF001
        store._conn.execute(  # noqa: SLF001
            "CREATE TABLE IF NOT EXISTS hrr_expand_neighbors ("
            "belief_id TEXT NOT NULL, neighbor_id TEXT NOT NULL, "
            "similarity REAL NOT NULL, edge_type TEXT NOT NULL, "
            "direction TEXT NOT NULL, created_at TEXT NOT NULL, "
            "PRIMARY KEY (belief_id, neighbor_id, edge_type, direction))"
        )
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO hrr_expand_neighbors VALUES "
            "('b2', 'b4', 1.0, 'CITES', 'forward', '1970-01-01T00:00:00Z')"
        )
    got = [b.id for b in hx.expand_seeds(store, idx, ["b2"])]
    assert sorted(got) == ["b1", "b3"]
    assert got == live_ids


def test_expand_seeds_excludes_seed_and_respects_top_k() -> None:
    store = _toy_store()
    idx = _built_index(store)
    got = hx.expand_seeds(store, idx, ["b2"], top_k=1)
    assert len(got) == 1
    assert all(b.id != "b2" for b in got)


def test_expand_seeds_excludes_soft_deleted() -> None:
    store = _toy_store()
    store.soft_delete_belief("b1")  # CONTRADICTS neighbour of b2
    idx = _built_index(store)
    got = {b.id for b in hx.expand_seeds(store, idx, ["b2"])}
    assert "b1" not in got
    assert "b3" in got


def test_expand_seeds_empty_inputs() -> None:
    store = _toy_store()
    idx = _built_index(store)
    assert hx.expand_seeds(store, idx, []) == []
    empty_idx = HRRStructIndex(dim=512, seed=_SEED)
    empty_idx.build(MemoryStore(":memory:"), seed=_SEED)
    assert hx.expand_seeds(store, empty_idx, ["b2"]) == []


# --- AC3: no regression with the flag off ---------------------------------


def test_retrieve_v2_flag_off_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)
    store = _toy_store()
    baseline = retrieve_v2(store, "gamma", use_hrr_structural=False)
    off = retrieve_v2(
        store, "gamma", use_hrr_structural=False, use_hrr_expand=False,
    )
    assert [b.id for b in off.beliefs] == [b.id for b in baseline.beliefs]


# --- lane wiring: net-new merge + telemetry -------------------------------


def test_retrieve_v2_flag_on_merges_net_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)
    store = _toy_store()
    cache = HRRStructIndexCache(store=store, seed=_SEED)
    # Query matches only b2; BFS disabled so the expansion lane is the sole
    # source of b2's semantic neighbours b1 / b3.
    off = retrieve_v2(
        store, "singular", use_hrr_structural=False,
        use_bfs=False, use_hrr_expand=False,
    )
    on = retrieve_v2(
        store, "singular", use_hrr_structural=False,
        use_bfs=False, use_hrr_expand=True,
        hrr_struct_index_cache=cache,
    )
    off_ids = {b.id for b in off.beliefs}
    on_ids = {b.id for b in on.beliefs}
    assert off_ids == {"b2"}
    assert {"b1", "b3"} <= on_ids
    assert off_ids < on_ids


# --- AC5: determinism — no sampling in the lane ---------------------------


@pytest.mark.source_scan
def test_lane_source_has_no_randomness() -> None:
    # AST scan (not a substring grep — the module docstring legitimately
    # mentions "random"/"betavariate" to document their absence). No
    # `import random` / `from random`, and no `.betavariate` attribute use.
    import ast

    tree = ast.parse(pathlib.Path(hx.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "random"
        if isinstance(node, ast.Attribute):
            assert node.attr != "betavariate"
