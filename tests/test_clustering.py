"""Tests for intentional clustering (#436) module + edges_for_beliefs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.clustering import (
    DEFAULT_CLUSTER_DIVERSITY_TARGET,
    DEFAULT_CLUSTER_EDGE_FLOOR,
    RetrievalCluster,
    cluster_candidates,
    pack_with_clusters,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    RETENTION_FACT,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _b(bid: str, content: str) -> Belief:
    ts = datetime.now(timezone.utc).isoformat()
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=ts,
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_AGENT_INFERRED,
        retention_class=RETENTION_FACT,
    )


def _e(src: str, dst: str, t: str, w: float) -> Edge:
    return Edge(src=src, dst=dst, type=t, weight=w)


# --- cluster_candidates ------------------------------------------------


def test_empty_candidates_returns_empty_list() -> None:
    assert cluster_candidates([], {}, edges=[]) == []


def test_singleton_candidates_become_size_1_clusters() -> None:
    """Three isolated candidates → three size-1 clusters, ordered by
    descending score."""
    cands = [_b("a", "x"), _b("b", "y"), _b("c", "z")]
    scores = {"a": 0.9, "b": 0.7, "c": 0.5}

    clusters = cluster_candidates(cands, scores, edges=[])
    assert [c.cluster_id for c in clusters] == [0, 1, 2]
    assert [c.representative_id for c in clusters] == ["a", "b", "c"]
    for c in clusters:
        assert len(c.member_ids) == 1


def test_two_clusters_via_strong_edge() -> None:
    """A SUPPORTS edge (weight≥floor) merges its endpoints into one
    cluster. A separate isolated belief stays its own cluster."""
    cands = [_b("a", "x"), _b("b", "y"), _b("c", "z")]
    scores = {"a": 0.9, "b": 0.4, "c": 0.6}
    edges = [_e("a", "b", EDGE_SUPPORTS, 0.8)]

    clusters = cluster_candidates(cands, scores, edges=edges)
    # Cluster {a, b} has seed_score = 0.9 (a's score, the max member).
    # Cluster {c} has seed_score = 0.6.
    assert len(clusters) == 2
    assert clusters[0].member_ids == ("a", "b")  # ranked by score
    assert clusters[0].representative_id == "a"
    assert clusters[0].seed_score == pytest.approx(0.9)
    assert clusters[1].member_ids == ("c",)
    assert clusters[1].representative_id == "c"


def test_edge_below_floor_does_not_merge() -> None:
    """RELATES_TO at weight 0.3 is below the default 0.4 floor."""
    cands = [_b("a", "x"), _b("b", "y")]
    scores = {"a": 0.9, "b": 0.7}
    edges = [_e("a", "b", EDGE_RELATES_TO, 0.3)]

    clusters = cluster_candidates(cands, scores, edges=edges)
    assert len(clusters) == 2  # not merged


def test_edge_outside_candidate_pool_is_ignored() -> None:
    """Edges with one endpoint outside the candidate set must not
    propagate connections (candidate-induced subgraph per spec)."""
    cands = [_b("a", "x"), _b("b", "y")]
    scores = {"a": 0.9, "b": 0.7}
    # Both a and b cite "outside" — but "outside" is not a candidate,
    # so a and b must not end up in the same cluster.
    edges = [_e("a", "outside", EDGE_CITES, 0.7),
             _e("b", "outside", EDGE_CITES, 0.7)]

    clusters = cluster_candidates(cands, scores, edges=edges)
    assert len(clusters) == 2


def test_cluster_member_order_descending_score() -> None:
    cands = [_b(x, "x") for x in ("a", "b", "c", "d")]
    scores = {"a": 0.5, "b": 0.9, "c": 0.7, "d": 0.3}
    # Chain: all four merged into one cluster via strong SUPPORTS.
    edges = [
        _e("a", "b", EDGE_SUPPORTS, 0.8),
        _e("b", "c", EDGE_SUPPORTS, 0.8),
        _e("c", "d", EDGE_SUPPORTS, 0.8),
    ]

    clusters = cluster_candidates(cands, scores, edges=edges)
    assert len(clusters) == 1
    # Sorted by descending score: b > c > a > d.
    assert clusters[0].member_ids == ("b", "c", "a", "d")
    assert clusters[0].representative_id == "b"
    assert clusters[0].seed_score == pytest.approx(0.9)


def test_tie_breaking_is_deterministic_by_id_asc() -> None:
    cands = [_b("zzz", "x"), _b("aaa", "y")]
    scores = {"zzz": 0.5, "aaa": 0.5}

    clusters = cluster_candidates(cands, scores, edges=[])
    assert [c.representative_id for c in clusters] == ["aaa", "zzz"]


# --- pack_with_clusters -------------------------------------------------


def test_pack_picks_one_representative_per_cluster_until_target() -> None:
    """Stage 1: with diversity_target=2, pack picks the top 2 reps."""
    a = _b("a", "alpha alpha alpha alpha")  # ~5 tokens
    b = _b("b", "beta beta beta beta")
    c = _b("c", "gamma gamma gamma")
    cands = [a, b, c]
    clusters = [
        RetrievalCluster(0, ("a",), "a", 0.9),
        RetrievalCluster(1, ("b",), "b", 0.7),
        RetrievalCluster(2, ("c",), "c", 0.5),
    ]

    out = pack_with_clusters(
        clusters,
        {b.id: b for b in cands},
        token_budget=10_000,
        cluster_diversity_target=2,
    )
    # Stage 1 stops at 2 covered clusters; Stage 2 fills remaining
    # budget with non-rep members (none here, so just c gets added too
    # because it's a singleton-cluster member in the score-ranked tail).
    assert [b.id for b in out] == ["a", "b", "c"]


def test_pack_stage1_yields_to_stage2_on_tight_budget() -> None:
    """When a cluster representative does not fit the remaining budget,
    fallback_to_score=True abandons Stage 1 and Stage 2 fills from
    the score-ranked tail. Default behaviour."""
    big = _b("a", "x" * 200)  # ~50 tokens
    small = _b("b", "y" * 10)  # ~3 tokens
    cands = [big, small]
    clusters = [
        RetrievalCluster(0, ("a",), "a", 0.9),
        RetrievalCluster(1, ("b",), "b", 0.5),
    ]
    # Budget too tight for `big` to fit alongside even one more belief.
    out = pack_with_clusters(
        clusters,
        {b.id: b for b in cands},
        token_budget=5,
    )
    # `a` consumes ~50 tokens, doesn't fit at budget=5.
    # fallback_to_score=True → Stage 1 abandons after the miss; Stage 2
    # picks `b` from the tail.
    assert [b.id for b in out] == ["b"]


def test_pack_strict_diversity_skips_oversize_rep_and_continues() -> None:
    """fallback_to_score=False keeps trying Stage 1 reps even after a
    miss — strict-diversity mode."""
    big = _b("a", "x" * 200)  # too big
    small = _b("b", "y" * 10)
    cands = [big, small]
    clusters = [
        RetrievalCluster(0, ("a",), "a", 0.9),
        RetrievalCluster(1, ("b",), "b", 0.5),
    ]
    out = pack_with_clusters(
        clusters,
        {b.id: b for b in cands},
        token_budget=5,
        fallback_to_score=False,
    )
    assert [b.id for b in out] == ["b"]


def test_pack_fills_stage2_from_remaining_cluster_members() -> None:
    """A 3-member cluster: Stage 1 picks the rep; Stage 2 fills the
    remaining budget with the other two members in score order."""
    a = _b("a", "x" * 20)
    b = _b("b", "y" * 20)
    c = _b("c", "z" * 20)
    cluster = RetrievalCluster(0, ("a", "b", "c"), "a", 0.9)

    out = pack_with_clusters(
        [cluster],
        {x.id: x for x in (a, b, c)},
        token_budget=10_000,
        cluster_diversity_target=1,
    )
    # Stage 1: picks `a` (covers the only cluster, target=1, done).
    # Stage 2: walks cluster members in member_ids order, picks b, c.
    assert [x.id for x in out] == ["a", "b", "c"]


def test_pack_skips_missing_belief_id() -> None:
    """Race: a belief id is in a cluster but missing from belief_by_id
    (deleted between rank and pack). Pack quietly skips it."""
    cluster = RetrievalCluster(0, ("missing",), "missing", 0.9)
    out = pack_with_clusters([cluster], {}, token_budget=10_000)
    assert out == []


def test_pack_no_clusters_returns_empty() -> None:
    out = pack_with_clusters([], {}, token_budget=10_000)
    assert out == []


def test_pack_default_diversity_target_is_three() -> None:
    """Smoke test on the default constant — five clusters, default
    target=3 means stage 1 picks the top 3 reps."""
    cands = [_b(x, "content " * 5) for x in "abcde"]
    clusters = [
        RetrievalCluster(i, (cands[i].id,), cands[i].id, 1.0 - i * 0.1)
        for i in range(5)
    ]
    out = pack_with_clusters(
        clusters,
        {b.id: b for b in cands},
        token_budget=10_000,
        # explicit default for clarity
        cluster_diversity_target=DEFAULT_CLUSTER_DIVERSITY_TARGET,
    )
    # Stage 1 picks first 3 reps in seed_score order; Stage 2 fills
    # remaining budget with the last 2 members.
    assert [b.id for b in out[:3]] == ["a", "b", "c"]
    assert sorted(b.id for b in out) == list("abcde")


# --- edges_for_beliefs --------------------------------------------------


def test_edges_for_beliefs_batched_lookup(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "ef.db"))
    try:
        for bid in ("a", "b", "c", "d"):
            store.insert_belief(_b(bid, "x"))
        store.insert_edge(_e("a", "b", EDGE_SUPPORTS, 0.8))
        store.insert_edge(_e("b", "c", EDGE_CITES, 0.5))
        store.insert_edge(_e("d", "a", EDGE_RELATES_TO, 0.3))

        # Query for {a, b}: should return all three edges (a→b, b→c, d→a)
        # because each touches at least one of {a, b}.
        out = store.edges_for_beliefs(["a", "b"])
        ids = {(e.src, e.dst, e.type) for e in out}
        assert ids == {
            ("a", "b", EDGE_SUPPORTS),
            ("b", "c", EDGE_CITES),
            ("d", "a", EDGE_RELATES_TO),
        }

        # Empty input → empty output, no SQL.
        assert store.edges_for_beliefs([]) == []

        # Query for an unrelated id → empty result.
        store.insert_belief(_b("loner", "x"))
        assert store.edges_for_beliefs(["loner"]) == []
    finally:
        store.close()


def test_default_edge_floor_excludes_relates_to_and_includes_cites() -> None:
    """Defaults are calibrated against `EDGE_VALENCE`: 0.4 floor includes
    CITES (0.5) and excludes RELATES_TO (0.3)."""
    assert DEFAULT_CLUSTER_EDGE_FLOOR == 0.4
