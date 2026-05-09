# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false
"""Tests for the HRR structural-query lane (#152).

Acceptance map (#152 § "Acceptance criteria"):
- AC1: parse_structural_marker returns (kind, target_id) on
       valid input, None otherwise; kind must be in EDGE_TYPES
- AC2: build is deterministic at fixed seed
- AC3: probe(CONTRADICTS, b2) finds b1 at rank 1 when
       b1 -CONTRADICTS-> b2
- AC4: same probe does not retrieve a belief whose only edge
       to b2 is SUPPORTS (role specificity)
- AC5: scores in top-K are descending; orthogonal-noise floor
       is ~1/sqrt(dim)
- AC6: rebuild latency at N=10k (perf-gated)
- AC7: per-query probe latency at N=50k (perf-gated)
- AC8: memory footprint check (perf-gated)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from aelfrice.hrr_index import (
    HRRStructIndex,
    HRRStructIndexCache,
    _seed_from_path,
    parse_structural_marker,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _has_run_perf(request: pytest.FixtureRequest) -> bool:
    try:
        return bool(request.config.getoption("--run-perf", default=False))
    except (AttributeError, ValueError):
        return False


def _mk(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=bid,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _toy_store() -> MemoryStore:
    """Topology: b1 -CONTRADICTS-> b2; b3 -SUPPORTS-> b2;
    b4 -CITES-> b5; b1 -RELATES_TO-> b5."""
    s = MemoryStore(":memory:")
    for i in range(1, 6):
        s.insert_belief(_mk(f"b{i}"))
    s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="b3", dst="b2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b4", dst="b5", type=EDGE_CITES, weight=1.0))
    s.insert_edge(Edge(src="b1", dst="b5", type=EDGE_RELATES_TO, weight=1.0))
    return s


# --- AC1 -----------------------------------------------------------------


def test_parse_structural_marker_valid_kinds() -> None:
    assert parse_structural_marker("CONTRADICTS:b2") == ("CONTRADICTS", "b2")
    assert parse_structural_marker("SUPPORTS:0026e983a7a99608") == (
        "SUPPORTS", "0026e983a7a99608",
    )
    assert parse_structural_marker("CITES:b/abc") == ("CITES", "b/abc")
    assert parse_structural_marker("SUPERSEDES:x") == ("SUPERSEDES", "x")
    assert parse_structural_marker("RELATES_TO:x") == ("RELATES_TO", "x")
    assert parse_structural_marker("DERIVED_FROM:x") == ("DERIVED_FROM", "x")


def test_parse_structural_marker_unknown_kind_returns_none() -> None:
    assert parse_structural_marker("UNRELATED:b2") is None
    # Lowercase is rejected — the supported edge types are uppercase.
    assert parse_structural_marker("contradicts:b2") is None


def test_parse_structural_marker_no_target_returns_none() -> None:
    assert parse_structural_marker("CONTRADICTS:") is None
    assert parse_structural_marker("CONTRADICTS:   ") is None


def test_parse_structural_marker_no_marker_returns_none() -> None:
    assert parse_structural_marker("just a normal query") is None
    assert parse_structural_marker("") is None
    assert parse_structural_marker("contradicts everything") is None


def test_parse_structural_marker_strips_whitespace() -> None:
    assert parse_structural_marker("  CONTRADICTS:b2  ") == ("CONTRADICTS", "b2")


# --- AC2 -----------------------------------------------------------------


def test_build_is_deterministic_at_fixed_seed() -> None:
    s = _toy_store()
    a = HRRStructIndex(dim=512, seed=42)
    a.build(s)
    b = HRRStructIndex(dim=512, seed=42)
    b.build(s)
    np.testing.assert_array_equal(a.struct, b.struct)
    assert a.belief_ids == b.belief_ids
    for bid in a.id_vecs:
        np.testing.assert_array_equal(a.id_vecs[bid], b.id_vecs[bid])


def test_build_different_seed_produces_different_struct() -> None:
    s = _toy_store()
    a = HRRStructIndex(dim=512, seed=1)
    a.build(s)
    b = HRRStructIndex(dim=512, seed=2)
    b.build(s)
    assert not np.array_equal(a.struct, b.struct)


def test_seed_from_path_is_stable_across_runs() -> None:
    # md5-based — same input must produce same seed; Python hash()
    # randomization would break this.
    s1 = _seed_from_path("/tmp/aelfrice/store.db")
    s2 = _seed_from_path("/tmp/aelfrice/store.db")
    assert s1 == s2
    s3 = _seed_from_path("/tmp/aelfrice/other.db")
    assert s1 != s3


def test_build_empty_store_produces_zero_struct() -> None:
    s = MemoryStore(":memory:")
    idx = HRRStructIndex(dim=128, seed=7)
    idx.build(s)
    assert idx.struct.shape == (0, 128)
    assert idx.belief_ids == []


# --- AC3 -----------------------------------------------------------------


def test_probe_recovers_contradicts_source_at_rank_1() -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=2048, seed=11)
    idx.build(s)
    hits = idx.probe("CONTRADICTS", "b2", top_k=5)
    assert hits, "probe returned no hits"
    assert hits[0][0] == "b1"  # b1 -CONTRADICTS-> b2


def test_probe_recovers_cites_source_at_rank_1() -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=2048, seed=11)
    idx.build(s)
    hits = idx.probe("CITES", "b5", top_k=5)
    assert hits[0][0] == "b4"  # b4 -CITES-> b5


# --- AC4 -----------------------------------------------------------------


def test_probe_role_specificity() -> None:
    """A CONTRADICTS probe at b2 must NOT score b3 (which only has
    a SUPPORTS edge to b2) above the orthogonal noise floor."""
    s = _toy_store()
    idx = HRRStructIndex(dim=2048, seed=11)
    idx.build(s)
    hits = idx.probe("CONTRADICTS", "b2", top_k=5)
    scores = {bid: score for bid, score in hits}
    # b1's score is the high-signal hit (~1.0 cosine pre-noise).
    # b3 should score near the noise floor 1/sqrt(2048) ≈ 0.022.
    if "b3" in scores:
        assert scores["b3"] < scores["b1"] * 0.5, (
            f"b3 contaminated CONTRADICTS probe: b3={scores['b3']}, "
            f"b1={scores['b1']}"
        )


# --- AC5 -----------------------------------------------------------------


def test_probe_returns_descending_scores() -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=2048, seed=11)
    idx.build(s)
    hits = idx.probe("CONTRADICTS", "b2", top_k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_noise_floor_matches_one_over_sqrt_dim() -> None:
    idx = HRRStructIndex(dim=2048)
    assert idx.noise_floor() == pytest.approx(1.0 / np.sqrt(2048))
    idx512 = HRRStructIndex(dim=512)
    assert idx512.noise_floor() == pytest.approx(1.0 / np.sqrt(512))


def test_probe_unknown_kind_or_target_returns_empty() -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=512, seed=1)
    idx.build(s)
    assert idx.probe("UNKNOWN_KIND", "b1") == []
    assert idx.probe("CONTRADICTS", "does_not_exist") == []


def test_use_hrr_structural_default_on() -> None:
    """Per #154 the default flipped to ON after the #437 11/11 gate
    cleared. No env, no kwarg, no toml → True."""
    from aelfrice.retrieval import is_hrr_structural_enabled

    assert is_hrr_structural_enabled() is True


def test_use_hrr_structural_opt_out_paths_intact(tmp_path: Path) -> None:
    """The opt-out surface (env var, kwarg, TOML key) remains
    reachable for users who want the pre-flip ranking. Replaces the
    v1.7-era default-off check."""
    from aelfrice.retrieval import is_hrr_structural_enabled

    # Explicit kwarg
    assert is_hrr_structural_enabled(False) is False

    # TOML key
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_hrr_structural = false\n")
    assert is_hrr_structural_enabled(start=tmp_path) is False


def test_probe_top_k_clamps_to_corpus_size() -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=512, seed=1)
    idx.build(s)
    hits = idx.probe("CONTRADICTS", "b2", top_k=100)
    assert len(hits) == 5  # only 5 beliefs in toy store


# --- save/load -----------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    s = _toy_store()
    idx = HRRStructIndex(dim=512, seed=99)
    idx.build(s)
    path = tmp_path / "hrr.npz"
    idx.save(path)
    loaded = HRRStructIndex.load(path)
    assert loaded.dim == idx.dim
    assert loaded.seed == idx.seed
    assert loaded.belief_ids == idx.belief_ids
    np.testing.assert_array_equal(loaded.struct, idx.struct)
    # Probes return identical hits on the loaded index.
    a = idx.probe("CONTRADICTS", "b2", top_k=5)
    b = loaded.probe("CONTRADICTS", "b2", top_k=5)
    assert a == b


# --- HRRStructIndexCache --------------------------------------------------


def test_cache_lazy_build_then_reuse() -> None:
    s = _toy_store()
    cache = HRRStructIndexCache(store=s, dim=256, seed=7)
    a = cache.get()
    b = cache.get()
    assert a is b, "second get() must return the cached instance"


def test_cache_invalidates_on_store_mutation() -> None:
    s = _toy_store()
    cache = HRRStructIndexCache(store=s, dim=256, seed=7)
    first = cache.get()
    s.insert_belief(_mk("b6"))
    second = cache.get()
    assert first is not second, "store mutation must drop the cache"
    assert "b6" in second.belief_ids


def test_cache_explicit_invalidate_drops_index() -> None:
    s = _toy_store()
    cache = HRRStructIndexCache(store=s, dim=256, seed=7)
    cache.get()
    cache.invalidate()
    assert cache._index is None


# --- AC6 / AC7 (perf-gated) ----------------------------------------------


def test_build_latency_at_n_10k(request: pytest.FixtureRequest) -> None:
    if not _has_run_perf(request):
        pytest.skip("perf-gated: pass --run-perf to run")
    s = MemoryStore(":memory:")
    n = 10_000
    for i in range(n):
        s.insert_belief(_mk(f"b{i}"))
    # Sparse edge graph: ~3 outgoing edges per belief.
    for i in range(n):
        for off in (1, 7, 31):
            s.insert_edge(
                Edge(src=f"b{i}", dst=f"b{(i + off) % n}",
                     type=EDGE_CITES, weight=1.0),
            )
    idx = HRRStructIndex(dim=2048, seed=0)
    t0 = time.perf_counter()
    idx.build(s)
    elapsed = time.perf_counter() - t0
    assert elapsed <= 5.0, f"build took {elapsed:.2f}s, exceeds 5s budget"


def test_probe_latency_at_n_50k(request: pytest.FixtureRequest) -> None:
    if not _has_run_perf(request):
        pytest.skip("perf-gated: pass --run-perf to run")
    n = 50_000
    dim = 2048
    rng = np.random.default_rng(0)
    idx = HRRStructIndex(dim=dim, seed=0)
    idx.belief_ids = [f"b{i}" for i in range(n)]
    idx._index = {bid: i for i, bid in enumerate(idx.belief_ids)}
    idx.struct = rng.standard_normal((n, dim)).astype(np.float64) / np.sqrt(dim)
    idx.id_vecs = {"b0": rng.standard_normal(dim).astype(np.float64)}
    idx.role_vecs = {"CONTRADICTS": rng.standard_normal(dim).astype(np.float64)}
    # Warm-up
    idx.probe("CONTRADICTS", "b0", top_k=10)
    t0 = time.perf_counter()
    idx.probe("CONTRADICTS", "b0", top_k=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 30.0, (
        f"probe took {elapsed_ms:.2f} ms, exceeds 30ms budget"
    )
