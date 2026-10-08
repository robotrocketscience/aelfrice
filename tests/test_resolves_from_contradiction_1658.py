"""Contradiction resolution writes a RESOLVES edge beside SUPERSEDES (#1658).

Before #1658 the RESOLVES edge type had readers (`aelf introspect` status,
the wonder GC exemption, the HRR structural marker) and no writer, so every
reader took its "absent" branch on every real store. The tie-breaker now
writes RESOLVES winner -> loser, the direction every reader assumes
(`src` resolves `dst`).

The edge must not move free-text retrieval ranking. Its weight is pinned
below the clustering floor and BFS has no weight for its type; the last
tests here pin both reasons and then check `retrieve()` end to end.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aelfrice.bfs_multihop import expand_bfs
from aelfrice.clustering import DEFAULT_CLUSTER_EDGE_FLOOR, cluster_candidates
from aelfrice.contradiction import (
    RESOLVES_WEIGHT,
    auto_resolve_all_contradictions,
    resolve_contradiction,
)
from aelfrice.introspect import STATUS_DECIDED, STATUS_DECIDES, build_report
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    EDGE_CONTRADICTS,
    EDGE_RESOLVES,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_SPECULATIVE,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_gc


def _mk(
    bid: str,
    *,
    content: str | None = None,
    lock: str = LOCK_NONE,
    origin: str = "unknown",
    btype: str = BELIEF_FACTUAL,
    created_at: str = "2026-04-26T00:00:00Z",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content if content is not None else f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=btype,
        lock_level=lock,
        locked_at="2026-04-26T01:00:00Z" if lock == LOCK_USER else None,
        created_at=created_at,
        last_retrieved_at=None,
        origin=origin,
    )


def _contradicting_pair() -> MemoryStore:
    """A (locked) contradicts B; A wins on precedence."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", lock=LOCK_USER))
    s.insert_belief(_mk("B"))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    return s


def _count(s: MemoryStore, edge_type: str) -> int:
    row = s._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM edges WHERE type = ?", (edge_type,),
    ).fetchone()
    return int(row[0])


# --- the writer --------------------------------------------------------------


def test_resolution_writes_supersedes_and_resolves_winner_to_loser() -> None:
    s = _contradicting_pair()
    result = resolve_contradiction(s, "A", "B")
    assert (result.winner_id, result.loser_id) == ("A", "B")
    assert s.get_edge("A", "B", EDGE_SUPERSEDES) is not None
    resolves = s.get_edge("A", "B", EDGE_RESOLVES)
    assert resolves is not None
    assert resolves.weight == RESOLVES_WEIGHT
    # Same direction as SUPERSEDES, never the reverse.
    assert s.get_edge("B", "A", EDGE_RESOLVES) is None
    assert result.resolves_created is True


def test_resolving_twice_writes_one_resolves_edge() -> None:
    s = _contradicting_pair()
    first = resolve_contradiction(s, "A", "B")
    second = resolve_contradiction(s, "A", "B")
    assert first.resolves_created is True
    assert second.resolves_created is False
    assert _count(s, EDGE_RESOLVES) == 1
    assert _count(s, EDGE_SUPERSEDES) == 1


def test_auto_resolve_rerun_writes_no_second_resolves_edge() -> None:
    s = _contradicting_pair()
    assert len(auto_resolve_all_contradictions(s)) == 1
    assert auto_resolve_all_contradictions(s) == []
    assert _count(s, EDGE_RESOLVES) == 1


def test_resolves_written_when_supersedes_came_from_another_writer() -> None:
    """The triple extractor can write SUPERSEDES before the tie-breaker runs.
    The RESOLVES check is independent, so the pair still gets one."""
    s = _contradicting_pair()
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPERSEDES, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.supersedes_created is False
    assert result.resolves_created is True
    assert s.get_edge("A", "B", EDGE_RESOLVES) is not None


def test_self_pair_writes_no_resolves_edge() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("Z"))
    result = resolve_contradiction(s, "Z", "Z")
    assert result.resolves_created is False
    assert _count(s, EDGE_RESOLVES) == 0


# --- the readers see it in the direction they assume ---------------------------


def test_introspect_reports_loser_decided_and_winner_decides() -> None:
    s = _contradicting_pair()
    resolve_contradiction(s, "A", "B")
    status = {sig.id: sig.status for sig in _signals(build_report(s))}
    assert status["B"] == STATUS_DECIDED
    assert status["A"] == STATUS_DECIDES


def _signals(report: object) -> list:  # type: ignore[type-arg]
    out = []
    for group in report.groups:  # type: ignore[attr-defined]
        out.extend(group.beliefs)
    return out


def test_wonder_gc_keeps_a_phantom_that_won_a_contradiction() -> None:
    """The loser already escapes GC through its tie-breaker audit row; the
    winner has no feedback row, so only the RESOLVES edge exempts it."""
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "PH", btype=BELIEF_SPECULATIVE, origin=ORIGIN_SPECULATIVE,
        created_at=old, alpha=0.3, beta=1.0,
    ))
    s.insert_belief(_mk("AI", origin=ORIGIN_AGENT_INFERRED, created_at=old))
    s.insert_edge(Edge(src="PH", dst="AI", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "PH", "AI")
    assert result.winner_id == "PH"
    gc = wonder_gc(s, ttl_days=14, dry_run=True)
    assert gc.scanned == 0


# --- no ranking change --------------------------------------------------------


def test_resolves_edge_never_joins_a_cluster() -> None:
    """Clustering is the one retrieval stage that reads `Edge.weight`.
    A RESOLVES edge on its own must leave the pair as two singletons."""
    assert RESOLVES_WEIGHT < DEFAULT_CLUSTER_EDGE_FLOOR
    s = _contradicting_pair()
    resolve_contradiction(s, "A", "B")
    resolves = s.get_edge("A", "B", EDGE_RESOLVES)
    assert resolves is not None
    a, b = s.get_belief("A"), s.get_belief("B")
    assert a is not None and b is not None
    clusters = cluster_candidates(
        [a, b], {"A": 2.0, "B": 1.0}, edges=[resolves],
    )
    assert len(clusters) == 2


def test_bfs_does_not_walk_a_resolves_edge() -> None:
    s = _contradicting_pair()
    resolve_contradiction(s, "A", "B")
    s.delete_edge("A", "B", EDGE_CONTRADICTS)
    s.delete_edge("A", "B", EDGE_SUPERSEDES)
    seed = s.get_belief("A")
    assert seed is not None
    assert expand_bfs([seed], s) == []


def _retrieval_fixture() -> MemoryStore:
    """Six contradicting pairs on one topic, resolved, SUPERSEDES removed.

    Removing SUPERSEDES isolates RESOLVES: whatever `retrieve()` returns
    differs between the two runs below only if RESOLVES moves it.
    """
    s = MemoryStore(":memory:")
    for i in range(6):
        w, lo = f"W{i}", f"L{i}"
        s.insert_belief(_mk(
            w, lock=LOCK_NONE, origin="user_validated",
            content=f"deploy target region number {i} is alpha",
        ))
        s.insert_belief(_mk(
            lo, origin=ORIGIN_AGENT_INFERRED,
            content=f"deploy target region number {i} is beta",
        ))
        s.insert_edge(Edge(src=w, dst=lo, type=EDGE_CONTRADICTS, weight=0.1))
    results = auto_resolve_all_contradictions(s)
    assert len(results) == 6
    for r in results:
        s.delete_edge(r.winner_id, r.loser_id, EDGE_SUPERSEDES)
    return s


def test_retrieve_ranking_is_identical_with_and_without_resolves() -> None:
    s = _retrieval_fixture()
    assert _count(s, EDGE_RESOLVES) == 6
    query = "deploy target region"
    with_edges = [
        b.id for b in retrieve(s, query, token_budget=120, bfs_enabled=True)
    ]
    for i in range(6):
        s.delete_edge(f"W{i}", f"L{i}", EDGE_RESOLVES)
    assert _count(s, EDGE_RESOLVES) == 0
    without_edges = [
        b.id for b in retrieve(s, query, token_budget=120, bfs_enabled=True)
    ]
    assert with_edges, "fixture retrieved nothing; the comparison is vacuous"
    assert with_edges == without_edges
