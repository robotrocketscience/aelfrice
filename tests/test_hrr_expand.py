"""Tests for the HRR vocabulary-bridge expansion lane (#981).

Covers the issue's acceptance criteria:

1. ``use_hrr_expand`` resolves via env > kwarg > TOML > default-OFF; passing
   it never raises.
2. The lane is deterministic — the precomputed neighbour table is byte-equal
   across two runs over the same store.
3. No regression with the flag off — ``retrieve_v2`` output is byte-identical
   to the pre-lane path.
5. No ``random`` / ``betavariate`` is introduced into the lane.
6. The default stays OFF.
"""
from __future__ import annotations

import pathlib

import numpy as np
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


def _table_rows(store: MemoryStore) -> list[tuple]:
    return [
        tuple(r)
        for r in store._conn.execute(  # noqa: SLF001
            "SELECT belief_id, neighbor_id, similarity, edge_type, direction "
            "FROM hrr_expand_neighbors ORDER BY rowid"
        ).fetchall()
    ]


def test_precompute_table_byte_stable_across_runs() -> None:
    store = _toy_store()
    idx = _built_index(store)
    n1 = hx.precompute_expand_neighbors(store, idx, now_iso="2026-01-01T00:00:00Z")
    rows1 = _table_rows(store)
    n2 = hx.precompute_expand_neighbors(store, idx, now_iso="2026-01-01T00:00:00Z")
    rows2 = _table_rows(store)
    assert n1 == n2
    assert rows1 == rows2


def test_precompute_table_stable_across_independent_index_builds() -> None:
    # Two stores with identical content + a fresh index build each produce
    # the same neighbour table (the index seed is fixed).
    rows_a = _table_rows(_precomputed(_toy_store()))
    rows_b = _table_rows(_precomputed(_toy_store()))
    assert rows_a == rows_b


def _precomputed(store: MemoryStore) -> MemoryStore:
    hx.precompute_expand_neighbors(
        store, _built_index(store), now_iso="2026-01-01T00:00:00Z",
    )
    return store


def test_true_edges_dominate_noise_floor() -> None:
    store = _toy_store()
    idx = _built_index(store)
    hx.precompute_expand_neighbors(store, idx, now_iso="2026-01-01T00:00:00Z")
    rows = _table_rows(store)
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
    assert all(r[2] > 0.5 for r in rows)
    assert min(r[2] for r in rows) > idx.noise_floor() * 5


# --- neighbour recovery + expand_seeds ------------------------------------


def test_expand_seeds_surfaces_both_directions() -> None:
    store = _toy_store()
    idx = _built_index(store)
    # b2's in-neighbours are b1 (CONTRADICTS) and b3 (SUPPORTS).
    got = sorted(b.id for b in hx.expand_seeds(store, idx, ["b2"]))
    assert got == ["b1", "b3"]


def test_expand_seeds_table_and_live_paths_agree() -> None:
    table_store = _precomputed(_toy_store())
    live_store = _toy_store()  # no precompute table populated
    idx_t = _built_index(table_store)
    idx_l = _built_index(live_store)
    table_ids = [b.id for b in hx.expand_seeds(table_store, idx_t, ["b2"])]
    live_ids = [b.id for b in hx.expand_seeds(live_store, idx_l, ["b2"])]
    assert table_ids == live_ids


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
