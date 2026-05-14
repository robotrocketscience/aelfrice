"""Acceptance tests for the v1.3.0 BFS multi-hop graph traversal.

One section per acceptance criterion in `docs/bfs_multihop.md`:

  AC1. Fixture chains (decisional, pruned informational, contradiction)
       surface (or are pruned) per the documented edge-weight table.
  AC2. Edge-type weight ordering: a SUPERSEDES path beats a
       RELATES_TO path of the same length.
  AC3. Cycle detection: a cyclic graph terminates and produces
       deterministic output (visited-set per call).
  AC4. Budget enforcement: total expanded ≤ `total_budget_nodes`
       and per-hop fanout ≤ `nodes_per_hop`.
  AC5. `min_path_score` pruning: paths whose multiplicative score
       drops below the floor are dropped.
  AC6. `max_depth` cap: depth-3 chains do not surface at default
       max_depth=2.
  AC7. Determinism: two runs over the same store and same seeds
       produce byte-identical output.
  AC8. Default-OFF byte-identical retrieve(): with `bfs_enabled=
       False` (the v1.3.0 default), `retrieve()` returns the same
       beliefs in the same order as the v1.3.0 L0+L2.5+L1 baseline.
  AC9. Cache invalidation end-to-end: a new SUPERSEDES edge
       between two cached beliefs is reflected in the next L3
       expansion.
  AC10. Latency regression guard: 1k-belief / 5k-edge synthetic
        store completes one BFS-enabled retrieve under the
        documented band.
  AC11. Temporal-coherence (LIMITATIONS contract): the v1.3 latest-
        serial-per-hop behaviour is asserted explicitly so a future
        v2.0 fix flips the test, not breaks it silently.

All tests deterministic, in-memory SQLite, ≤2 s wall clock each.
The latency-band test uses a wider tolerance because CI hardware
varies; the assertion is a regression band, not a benchmark target.
"""
from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aelfrice.bfs_multihop import (
    BFS_EDGE_WEIGHTS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_PATH_SCORE,
    DEFAULT_NODES_PER_HOP,
    DEFAULT_TOTAL_BUDGET_NODES,
    ScoredHop,
    expand_bfs,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    ENV_BFS,
    ENV_ENTITY_INDEX,
    RetrievalCache,
    is_bfs_enabled,
    retrieve,
    retrieve_v2,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
    created_at: str = "2026-04-26T00:00:00Z",
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at=created_at,
        last_retrieved_at=None,
    )


def _edge(src: str, dst: str, type_: str, weight: float = 1.0) -> Edge:
    return Edge(src=src, dst=dst, type=type_, weight=weight)


@pytest.fixture(autouse=True)
def isolated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Strip env-var interference and chdir into a clean tmp_path so
    no `.aelfrice.toml` from the actual repo bleeds into config
    resolution. Tests that want overrides set them inside the
    `tmp_path` they receive.
    """
    monkeypatch.delenv(ENV_BFS, raising=False)
    monkeypatch.delenv(ENV_ENTITY_INDEX, raising=False)
    # #741 adaptive expansion-gate: these tests cover the BFS lane
    # itself and use single-token / unmarked queries that the gate
    # would otherwise classify as "broad" and short-circuit. Force
    # expansion on so the BFS-internal contracts (AC1-AC11) keep
    # asserting BFS behaviour, not gate behaviour. The gate has its
    # own test surface in tests/test_expansion_gate.py.
    monkeypatch.setenv("AELFRICE_FORCE_EXPANSION", "1")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed_decisional_chain(store: MemoryStore) -> None:
    """S0 -[RELATES_TO]-> S1 -[SUPERSEDES]-> S2.

    Score(S1) = 0.30. Score(S2) = 0.30 * 0.90 = 0.27. Both above
    the 0.10 floor; S1 ranks above S2 because 0.30 > 0.27.
    """
    store.insert_belief(_mk("S0", "decisional seed beta architecture"))
    store.insert_belief(_mk("S1", "decisional intermediate beta v1"))
    store.insert_belief(_mk("S2", "decisional latest beta v2"))
    store.insert_edge(_edge("S0", "S1", EDGE_RELATES_TO))
    store.insert_edge(_edge("S1", "S2", EDGE_SUPERSEDES))


def _seed_pruned_chain(store: MemoryStore) -> None:
    """P0 -[RELATES_TO]-> P1 -[RELATES_TO]-> P2.

    Score(P1) = 0.30 (above 0.10). Score(P2) = 0.30 * 0.30 = 0.09
    (below the 0.10 floor — P2 is pruned).
    """
    store.insert_belief(_mk("P0", "pruned seed gamma topic"))
    store.insert_belief(_mk("P1", "pruned intermediate gamma context"))
    store.insert_belief(_mk("P2", "pruned tail gamma trivia"))
    store.insert_edge(_edge("P0", "P1", EDGE_RELATES_TO))
    store.insert_edge(_edge("P1", "P2", EDGE_RELATES_TO))


def _seed_contradiction(store: MemoryStore) -> None:
    """C0 -[CONTRADICTS]-> C1. Score(C1) = 0.85."""
    store.insert_belief(_mk("C0", "contradiction seed delta claim"))
    store.insert_belief(_mk("C1", "contradiction target delta refutation"))
    store.insert_edge(_edge("C0", "C1", EDGE_CONTRADICTS))


# ---------------------------------------------------------------------------
# AC1: fixture chains
# ---------------------------------------------------------------------------


def test_ac1_decisional_chain_surfaces_both_hops() -> None:
    s = MemoryStore(":memory:")
    _seed_decisional_chain(s)
    seed = s.get_belief("S0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    ids = [h.belief.id for h in hops]
    assert ids == ["S1", "S2"], (
        f"expected S1 (0.30) above S2 (0.27); got {ids}"
    )
    by_id = {h.belief.id: h for h in hops}
    assert abs(by_id["S1"].score - 0.30) < 1e-9
    assert by_id["S1"].depth == 1
    assert by_id["S1"].path == [EDGE_RELATES_TO]
    assert abs(by_id["S2"].score - 0.30 * 0.90) < 1e-9
    assert by_id["S2"].depth == 2
    assert by_id["S2"].path == [EDGE_RELATES_TO, EDGE_SUPERSEDES]


def test_ac1_pruned_informational_chain_drops_tail() -> None:
    s = MemoryStore(":memory:")
    _seed_pruned_chain(s)
    seed = s.get_belief("P0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    ids = {h.belief.id for h in hops}
    # P1 surfaces (0.30 ≥ 0.10). P2 does NOT (0.09 < 0.10).
    assert "P1" in ids
    assert "P2" not in ids


def test_ac1_contradiction_surfaces_at_depth_1() -> None:
    s = MemoryStore(":memory:")
    _seed_contradiction(s)
    seed = s.get_belief("C0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    assert len(hops) == 1
    assert hops[0].belief.id == "C1"
    assert abs(hops[0].score - 0.85) < 1e-9
    assert hops[0].depth == 1
    assert hops[0].path == [EDGE_CONTRADICTS]


# ---------------------------------------------------------------------------
# AC2: edge-type weight ordering
# ---------------------------------------------------------------------------


def test_ac2_supersedes_path_beats_relates_to_path_same_length() -> None:
    """Two depth-1 expansions from the same seed: SUPERSEDES (0.90)
    must rank above RELATES_TO (0.30)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X0", "fork point"))
    s.insert_belief(_mk("XS", "supersession target"))
    s.insert_belief(_mk("XR", "relates target"))
    s.insert_edge(_edge("X0", "XS", EDGE_SUPERSEDES))
    s.insert_edge(_edge("X0", "XR", EDGE_RELATES_TO))
    seed = s.get_belief("X0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    ids = [h.belief.id for h in hops]
    assert ids == ["XS", "XR"], (
        f"SUPERSEDES (0.90) must rank above RELATES_TO (0.30); got {ids}"
    )


def test_ac2_full_edge_weight_table_strict_descending() -> None:
    """Decisional > provenance > evidential > referential > informational."""
    expected_order = [
        EDGE_SUPERSEDES,    # 0.90
        EDGE_CONTRADICTS,   # 0.85
        EDGE_DERIVED_FROM,  # 0.70
        EDGE_SUPPORTS,      # 0.60
        EDGE_CITES,         # 0.40
        EDGE_RELATES_TO,    # 0.30
    ]
    weights = [BFS_EDGE_WEIGHTS[e] for e in expected_order]
    assert weights == sorted(weights, reverse=True)


# ---------------------------------------------------------------------------
# AC3: cycle detection
# ---------------------------------------------------------------------------


def test_ac3_cyclic_graph_terminates_deterministically() -> None:
    """A cycle A -> B -> C -> A must terminate. Visited-set
    initialised from seed ids prevents A from being re-surfaced;
    B and C surface at most once each.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", "node A in cycle"))
    s.insert_belief(_mk("B", "node B in cycle"))
    s.insert_belief(_mk("C", "node C in cycle"))
    s.insert_edge(_edge("A", "B", EDGE_SUPPORTS))
    s.insert_edge(_edge("B", "C", EDGE_SUPPORTS))
    s.insert_edge(_edge("C", "A", EDGE_SUPPORTS))
    seed = s.get_belief("A")
    assert seed is not None
    start = time.monotonic()
    hops = expand_bfs([seed], s)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"cycle should terminate fast, took {elapsed}s"
    ids = [h.belief.id for h in hops]
    # Seed A never re-surfaces (initialised in visited).
    assert "A" not in ids
    # B and C each appear at most once.
    assert ids.count("B") == 1
    assert ids.count("C") <= 1


def test_ac3_self_loop_does_not_loop_forever() -> None:
    """A self-loop A -> A is filtered before scoring (visited-set
    blocks dst = seed).
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", "lone node with self-loop"))
    s.insert_edge(_edge("A", "A", EDGE_SUPPORTS))
    seed = s.get_belief("A")
    assert seed is not None
    hops = expand_bfs([seed], s)
    assert hops == []


# ---------------------------------------------------------------------------
# AC4: budget enforcement
# ---------------------------------------------------------------------------


def test_ac4_total_budget_caps_expansion_count() -> None:
    """A seed with many SUPERSEDES neighbours is capped at
    `total_budget` expansions.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("R", "root with high-fanout"))
    for i in range(50):
        s.insert_belief(_mk(f"N{i:02d}", f"neighbour {i}"))
        s.insert_edge(_edge("R", f"N{i:02d}", EDGE_SUPERSEDES))
    seed = s.get_belief("R")
    assert seed is not None
    hops = expand_bfs([seed], s, total_budget=8)
    assert len(hops) == 8
    # All from depth 1.
    assert all(h.depth == 1 for h in hops)


def test_ac4_nodes_per_hop_caps_fanout() -> None:
    """At a single frontier expansion, no more than
    `nodes_per_hop` candidates are taken even if more are
    available."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("R", "root"))
    for i in range(50):
        s.insert_belief(_mk(f"N{i:02d}", f"neighbour {i}"))
        s.insert_edge(_edge("R", f"N{i:02d}", EDGE_SUPPORTS))
    seed = s.get_belief("R")
    assert seed is not None
    hops = expand_bfs([seed], s, nodes_per_hop=4, total_budget=100)
    assert len(hops) == 4


# ---------------------------------------------------------------------------
# AC5: min_path_score pruning
# ---------------------------------------------------------------------------


def test_ac5_min_path_score_blocks_low_score_chains() -> None:
    """A min_path_score threshold above CITES (0.40) blocks
    CITES expansions outright."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("R", "root"))
    s.insert_belief(_mk("N", "cites target"))
    s.insert_edge(_edge("R", "N", EDGE_CITES))
    seed = s.get_belief("R")
    assert seed is not None
    hops = expand_bfs([seed], s, min_path_score=0.50)
    assert hops == []
    # Sanity: with a lower floor it does surface.
    hops2 = expand_bfs([seed], s, min_path_score=0.10)
    assert {h.belief.id for h in hops2} == {"N"}


# ---------------------------------------------------------------------------
# AC6: max_depth cap
# ---------------------------------------------------------------------------


def test_ac6_max_depth_2_does_not_surface_depth_3() -> None:
    """A depth-3 chain along high-weight edges still scores above
    the floor (0.90^3 = 0.729 > 0.10) but is cut by max_depth=2."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("D0", "depth seed"))
    s.insert_belief(_mk("D1", "depth 1"))
    s.insert_belief(_mk("D2", "depth 2"))
    s.insert_belief(_mk("D3", "depth 3 should not surface"))
    s.insert_edge(_edge("D0", "D1", EDGE_SUPERSEDES))
    s.insert_edge(_edge("D1", "D2", EDGE_SUPERSEDES))
    s.insert_edge(_edge("D2", "D3", EDGE_SUPERSEDES))
    seed = s.get_belief("D0")
    assert seed is not None
    hops = expand_bfs([seed], s, max_depth=2)
    ids = {h.belief.id for h in hops}
    assert ids == {"D1", "D2"}
    # depth=1 surfaces only D1.
    hops1 = expand_bfs([seed], s, max_depth=1)
    assert {h.belief.id for h in hops1} == {"D1"}


# ---------------------------------------------------------------------------
# AC7: determinism
# ---------------------------------------------------------------------------


def test_ac7_two_runs_produce_identical_output() -> None:
    s = MemoryStore(":memory:")
    _seed_decisional_chain(s)
    _seed_pruned_chain(s)
    _seed_contradiction(s)
    seeds_ids = ["S0", "P0", "C0"]
    seeds: list[Belief] = []
    for sid in seeds_ids:
        b = s.get_belief(sid)
        assert b is not None
        seeds.append(b)
    a = expand_bfs(seeds, s)
    b = expand_bfs(seeds, s)
    assert [(h.belief.id, h.score, h.depth, tuple(h.path)) for h in a] == \
           [(h.belief.id, h.score, h.depth, tuple(h.path)) for h in b]


def test_ac7_id_tiebreak_at_equal_score() -> None:
    """Two outbound edges of the same type and same edge.weight from
    one seed: tied on path-score; tie-break on dst id ASC."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("R", "root"))
    s.insert_belief(_mk("Y_late", "y target"))
    s.insert_belief(_mk("X_early", "x target"))
    s.insert_edge(_edge("R", "Y_late", EDGE_SUPPORTS))
    s.insert_edge(_edge("R", "X_early", EDGE_SUPPORTS))
    seed = s.get_belief("R")
    assert seed is not None
    hops = expand_bfs([seed], s)
    # Equal score (0.60); id tie-break: 'X_early' < 'Y_late'.
    assert [h.belief.id for h in hops] == ["X_early", "Y_late"]


# ---------------------------------------------------------------------------
# AC8: default-OFF byte-identical retrieve
# ---------------------------------------------------------------------------


def test_ac8_default_off_byte_identical_retrieve() -> None:
    """With bfs_enabled at its v1.3.0 default (False), retrieve()
    returns the same beliefs in the same order as a call that
    explicitly disables BFS — no graph work, no token budget
    consumption changes.
    """
    s = MemoryStore(":memory:")
    _seed_decisional_chain(s)
    _seed_pruned_chain(s)
    _seed_contradiction(s)
    # Add a content match for the L1 path so retrieve() has
    # something to seed BFS from.
    s.insert_belief(_mk("Q1", "decisional context discussion"))
    s.insert_edge(_edge("Q1", "S0", EDGE_RELATES_TO))

    default_out = retrieve(s, "decisional")
    explicit_off = retrieve(s, "decisional", bfs_enabled=False)
    assert [b.id for b in default_out] == [b.id for b in explicit_off]


def test_ac8_explicit_on_can_add_expansions() -> None:
    """The default-off contract is a contract — turning the flag
    ON does change the output (otherwise we shipped a no-op)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    off = retrieve(s, "bananas", bfs_enabled=False)
    on = retrieve(s, "bananas", bfs_enabled=True)
    assert {b.id for b in on} >= {b.id for b in off}
    assert "S2" in {b.id for b in on}
    assert "S2" not in {b.id for b in off}


def test_ac8_env_var_disables_bfs() -> None:
    """`AELFRICE_BFS=0` forces BFS off even if the kwarg says True."""
    import os

    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    os.environ[ENV_BFS] = "0"
    try:
        out = retrieve(s, "bananas", bfs_enabled=True)
    finally:
        del os.environ[ENV_BFS]
    assert "S2" not in {b.id for b in out}


def test_ac8_env_var_enables_bfs() -> None:
    """`AELFRICE_BFS=1` enables BFS without an explicit kwarg."""
    import os

    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    os.environ[ENV_BFS] = "1"
    try:
        out = retrieve(s, "bananas")
    finally:
        del os.environ[ENV_BFS]
    assert "S2" in {b.id for b in out}


def test_ac8_toml_flag_resolution(tmp_path: Path) -> None:
    """`[retrieval] bfs_enabled = true` in `.aelfrice.toml` enables
    BFS without env or kwarg."""
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nbfs_enabled = true\n", encoding="utf-8")
    assert is_bfs_enabled(start=tmp_path) is True
    cfg.write_text("[retrieval]\nbfs_enabled = false\n", encoding="utf-8")
    assert is_bfs_enabled(start=tmp_path) is False


# ---------------------------------------------------------------------------
# AC9: cache invalidation end-to-end
# ---------------------------------------------------------------------------


def test_ac9_new_supersedes_edge_invalidates_cached_bfs_result() -> None:
    """RetrievalCache wipes on edge mutation; the next BFS-enabled
    retrieve picks up the new edge."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    cache = RetrievalCache(s)
    out1 = cache.retrieve("bananas", bfs_enabled=True)
    assert "S2" not in {b.id for b in out1}
    # Add a SUPERSEDES edge and re-query.
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    out2 = cache.retrieve("bananas", bfs_enabled=True)
    assert "S2" in {b.id for b in out2}


def test_ac9_cache_key_distinguishes_bfs_flag() -> None:
    """Two queries that differ only in `bfs_enabled` are distinct
    cache entries."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    cache = RetrievalCache(s)
    off = cache.retrieve("bananas", bfs_enabled=False)
    on = cache.retrieve("bananas", bfs_enabled=True)
    assert {b.id for b in off} != {b.id for b in on}
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# AC10: latency regression guard (synthetic 1k-belief / 5k-edge store)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
def test_ac10_latency_band_1k_beliefs_5k_edges() -> None:
    """One BFS-enabled retrieve() against a 1k-belief / 5k-edge
    store completes well within the documented band.

    Spec § Latency budget regression band targets p50 ≤ 25 ms,
    p95 ≤ 100 ms on a 10k / 25k store. This test runs at 1/10 the
    scale (so p95 should be much lower) but uses a wider 1.5 s
    upper bound to accommodate CI hardware variability without
    flapping. The point is "doesn't blow up", not a benchmark.
    """
    s = MemoryStore(":memory:")
    n_beliefs = 1000
    n_edges = 5000
    # Insert beliefs with deterministic content that the BM25
    # tokenizer will index. Token "anchor" is the seed term every
    # query keys off; only first 50 carry it so L1 returns a small
    # set we can run BFS on without blowing past `total_budget`.
    for i in range(n_beliefs):
        marker = "anchor" if i < 50 else "filler"
        s.insert_belief(_mk(f"B{i:04d}", f"{marker} content for belief {i}"))
    # 5k random-but-deterministic outbound edges.
    edge_types = [
        EDGE_SUPPORTS, EDGE_CITES, EDGE_RELATES_TO, EDGE_DERIVED_FROM,
    ]
    for k in range(n_edges):
        src = f"B{k % n_beliefs:04d}"
        dst = f"B{(k * 7 + 13) % n_beliefs:04d}"
        if src == dst:
            continue
        et = edge_types[k % len(edge_types)]
        try:
            s.insert_edge(_edge(src, dst, et))
        except Exception:  # pragma: no cover — duplicate-edge defensive
            pass
    start = time.monotonic()
    out = retrieve(s, "anchor", bfs_enabled=True)
    elapsed = time.monotonic() - start
    assert out, "retrieve should return at least one belief"
    assert elapsed < 1.5, (
        f"BFS-enabled retrieve on 1k/5k store should be < 1.5s "
        f"on CI; took {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# AC11: temporal-coherence (LIMITATIONS contract)
# ---------------------------------------------------------------------------


def test_ac11_latest_serial_per_hop_documented_v13_contract() -> None:
    """v1.3 contract (LIMITATIONS § BFS multi-hop temporal coherence):
    each hop resolves to the globally-latest serial of its target
    belief independently. This test asserts that contract.

    Setup: seed S0 from "session 1" (created_at 2026-01-01).
    SUPERSEDES targets S1 (created 2026-02-01) and the chain
    continues to S2 (created 2026-03-01).

    Under the v1.3 contract, the BFS expansion from S0 surfaces
    BOTH S1 and S2 — the fact that S2 postdates S0's session is
    NOT used to filter the expansion. A future v2.0 temporal-
    coherence fix may flip this assertion (filtering S2 when
    `as_of_session_id` ≤ S0's session). When that lands, this test
    is the canary that the v1.3 contract was changed deliberately,
    not silently.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "S0", "decisional seed", created_at="2026-01-01T00:00:00Z",
    ))
    s.insert_belief(_mk(
        "S1", "decisional intermediate", created_at="2026-02-01T00:00:00Z",
    ))
    s.insert_belief(_mk(
        "S2", "decisional tail", created_at="2026-03-01T00:00:00Z",
    ))
    s.insert_edge(_edge("S0", "S1", EDGE_SUPERSEDES))
    s.insert_edge(_edge("S1", "S2", EDGE_SUPERSEDES))
    seed = s.get_belief("S0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    ids = {h.belief.id for h in hops}
    # v1.3 latest-serial-per-hop: BOTH later-dated beliefs surface.
    assert ids == {"S1", "S2"}, (
        f"v1.3 contract: latest-serial-per-hop. Expected both "
        f"S1 and S2 to surface regardless of created_at; got {ids}. "
        f"If v2.0 temporal-coherence work is intentionally landing, "
        f"flip this assertion in the same PR."
    )


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_default_constants_match_spec() -> None:
    """The four default knobs are pinned to the spec values. A
    drift here is a documented benchmark deviation — see
    docs/bfs_multihop.md § Depth cap and budget."""
    assert DEFAULT_MAX_DEPTH == 2
    assert DEFAULT_NODES_PER_HOP == 16
    assert DEFAULT_TOTAL_BUDGET_NODES == 32
    assert DEFAULT_MIN_PATH_SCORE == 0.10


def test_edge_weights_match_spec() -> None:
    assert BFS_EDGE_WEIGHTS == {
        EDGE_SUPERSEDES: 0.90,
        EDGE_CONTRADICTS: 0.85,
        EDGE_DERIVED_FROM: 0.70,
        EDGE_IMPLEMENTS: 0.65,
        EDGE_SUPPORTS: 0.60,
        EDGE_TESTS: 0.55,
        EDGE_CITES: 0.40,
        EDGE_RELATES_TO: 0.30,
        EDGE_TEMPORAL_NEXT: 0.25,
        EDGE_POTENTIALLY_STALE: 0.0,
    }


def test_unknown_edge_type_yields_zero_weight() -> None:
    """An edge type not in the table is ignored (weight 0.0,
    skipped before scoring)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("R", "root"))
    s.insert_belief(_mk("N", "target"))
    # Bypass the model-level frozenset by using a raw insert with a
    # known type that the test then monkeypatches OUT of the table.
    s.insert_edge(_edge("R", "N", EDGE_SUPPORTS))
    seed = s.get_belief("R")
    assert seed is not None
    # With SUPPORTS removed from the weight table, the only edge
    # gets weight 0.0 and is skipped.
    saved = BFS_EDGE_WEIGHTS.pop(EDGE_SUPPORTS)
    try:
        hops = expand_bfs([seed], s)
        assert hops == []
    finally:
        BFS_EDGE_WEIGHTS[EDGE_SUPPORTS] = saved


def test_empty_seeds_returns_empty() -> None:
    s = MemoryStore(":memory:")
    assert expand_bfs([], s) == []


def test_seed_with_no_outbound_edges_returns_empty() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Lonely", "isolated belief"))
    seed = s.get_belief("Lonely")
    assert seed is not None
    assert expand_bfs([seed], s) == []


def test_scoredhop_dataclass_shape() -> None:
    """Sanity check on the dataclass shape — frozen, ordered fields."""
    b = _mk("A", "a")
    h = ScoredHop(belief=b, score=0.5, depth=1, path=[EDGE_SUPPORTS])
    assert h.belief is b
    assert h.score == 0.5
    assert h.depth == 1
    assert h.path == [EDGE_SUPPORTS]
    # belief_id_trail defaults to empty for backwards-compat constructors.
    assert h.belief_id_trail == ()
    with pytest.raises(FrozenInstanceError):
        h.score = 0.9  # type: ignore[misc]


def test_belief_id_trail_threaded_through_two_hop_walk() -> None:
    """#645 R2: ``expand_bfs`` emits a per-hop ``belief_id_trail`` that
    starts at the seed and ends at the hop's belief id, with length
    ``depth + 1``."""
    s = MemoryStore(":memory:")
    _seed_decisional_chain(s)  # S0 -- RELATES_TO --> S1 -- SUPERSEDES --> S2
    seed = s.get_belief("S0")
    assert seed is not None
    hops = expand_bfs([seed], s)
    by_id = {h.belief.id: h for h in hops}
    assert by_id["S1"].belief_id_trail == ("S0", "S1")
    assert by_id["S2"].belief_id_trail == ("S0", "S1", "S2")
    for h in hops:
        assert len(h.belief_id_trail) == h.depth + 1
        assert h.belief_id_trail[-1] == h.belief.id


def test_belief_id_trail_with_multiple_seeds_pins_to_originating_seed() -> None:
    """Each hop's trail begins at the seed it expanded from, not the
    first seed in the input list."""
    s = MemoryStore(":memory:")
    # Two disjoint micro-chains: A -> B and X -> Y.
    s.insert_belief(_mk("A", "a"))
    s.insert_belief(_mk("B", "b"))
    s.insert_belief(_mk("X", "x"))
    s.insert_belief(_mk("Y", "y"))
    s.insert_edge(_edge("A", "B", EDGE_SUPPORTS))
    s.insert_edge(_edge("X", "Y", EDGE_SUPPORTS))
    seeds = [s.get_belief("A"), s.get_belief("X")]
    assert all(seed is not None for seed in seeds)
    hops = expand_bfs([sd for sd in seeds if sd is not None], s)
    by_id = {h.belief.id: h for h in hops}
    assert by_id["B"].belief_id_trail == ("A", "B")
    assert by_id["Y"].belief_id_trail == ("X", "Y")


# ---------------------------------------------------------------------------
# retrieve_v2 surface
# ---------------------------------------------------------------------------


def test_retrieve_v2_use_bfs_populates_bfs_chains() -> None:
    """retrieve_v2 with use_bfs=True returns a RetrievalResult whose
    bfs_chains list mirrors the L3 expansions in the merged output."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    result = retrieve_v2(s, "bananas", use_bfs=True)
    ids = [b.id for b in result.beliefs]
    assert "S2" in ids
    # bfs_chains has one entry per L3 expansion.
    assert result.bfs_chains == [[EDGE_SUPERSEDES]]


def test_retrieve_v2_use_bfs_default_off_yields_empty_chains() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Q1", "the kitchen has bananas"))
    s.insert_belief(_mk("S2", "the new yellow fruit policy"))
    s.insert_edge(_edge("Q1", "S2", EDGE_SUPERSEDES))
    result = retrieve_v2(s, "bananas")
    assert result.bfs_chains == []
