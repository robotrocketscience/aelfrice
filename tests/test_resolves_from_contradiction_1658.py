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
import sqlite3
from pathlib import Path

import pytest

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


def test_direct_call_writes_resolves_when_supersedes_already_exists() -> None:
    """`resolve_contradiction` checks RESOLVES on its own, so a direct call
    on a pair that already has SUPERSEDES still writes RESOLVES."""
    s = _contradicting_pair()
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPERSEDES, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.supersedes_created is False
    assert result.resolves_created is True
    assert s.get_edge("A", "B", EDGE_RESOLVES) is not None


def test_aelf_resolve_skips_a_pair_whose_supersedes_came_first() -> None:
    """`aelf resolve` goes through `find_unresolved_contradictions`, which
    skips any pair with SUPERSEDES in either direction. A pair whose
    SUPERSEDES came from another writer (the triple extractor) therefore
    gets no RESOLVES edge from `aelf resolve`."""
    for src, dst in (("A", "B"), ("B", "A")):
        s = _contradicting_pair()
        s.insert_edge(Edge(src=src, dst=dst, type=EDGE_SUPERSEDES, weight=1.0))
        assert auto_resolve_all_contradictions(s) == []
        assert _count(s, EDGE_RESOLVES) == 0


class _InjectedFailure(RuntimeError):
    pass


def _file_pair(tmp_path: Path) -> tuple[MemoryStore, str]:
    path = str(tmp_path / "memory.db")
    s = MemoryStore(path)
    s.insert_belief(_mk("A", lock=LOCK_USER))
    s.insert_belief(_mk("B"))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    return s, path


def _audit_rows(s: MemoryStore) -> int:
    row = s._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM feedback_history WHERE source LIKE ?",
        ("contradiction_tiebreaker:%",),
    ).fetchone()
    return int(row[0])


def test_failed_resolves_insert_leaves_no_supersedes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SUPERSEDES, RESOLVES and the audit row commit together. If the
    RESOLVES insert fails, the SUPERSEDES edge written just before it is
    rolled back too, so `aelf resolve` sees the pair as unresolved and
    retries it instead of skipping it forever."""
    s, path = _file_pair(tmp_path)
    real_insert = s.insert_edge

    def failing_insert(e: Edge) -> bool:
        if e.type == EDGE_RESOLVES:
            raise _InjectedFailure("injected failure on the RESOLVES insert")
        return real_insert(e)

    monkeypatch.setattr(s, "insert_edge", failing_insert)
    with pytest.raises(_InjectedFailure):
        resolve_contradiction(s, "A", "B")
    s.close()

    fresh = MemoryStore(path)
    try:
        assert fresh.get_edge("A", "B", EDGE_SUPERSEDES) is None
        assert fresh.get_edge("A", "B", EDGE_RESOLVES) is None
        assert _audit_rows(fresh) == 0
        # The pair is still listed, so a rerun settles it in full.
        results = auto_resolve_all_contradictions(fresh)
        assert [(r.winner_id, r.loser_id) for r in results] == [("A", "B")]
        assert fresh.get_edge("A", "B", EDGE_SUPERSEDES) is not None
        assert fresh.get_edge("A", "B", EDGE_RESOLVES) is not None
    finally:
        fresh.close()


def test_failed_audit_row_leaves_neither_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s, path = _file_pair(tmp_path)

    def failing_audit(**_kwargs: object) -> int:
        raise _InjectedFailure("injected failure on the audit row")

    monkeypatch.setattr(s, "insert_feedback_event", failing_audit)
    with pytest.raises(_InjectedFailure):
        resolve_contradiction(s, "A", "B")
    s.close()

    fresh = MemoryStore(path)
    try:
        assert _count(fresh, EDGE_SUPERSEDES) == 0
        assert _count(fresh, EDGE_RESOLVES) == 0
    finally:
        fresh.close()


def test_resolution_holds_the_write_lock_before_its_existence_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existence checks run under the write lock, so another writer
    cannot add the same edge between a check and its insert. While the
    first check runs, a second connection's attempt to take the write
    lock fails at once."""
    s, path = _file_pair(tmp_path)
    real_get_edge = s.get_edge
    competitor_blocked: list[bool] = []

    def probing_get_edge(src: str, dst: str, type_: str) -> Edge | None:
        if not competitor_blocked:
            other = sqlite3.connect(path, timeout=0, isolation_level=None)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
                competitor_blocked.append(False)
            except sqlite3.OperationalError:
                competitor_blocked.append(True)
            finally:
                other.close()
        return real_get_edge(src, dst, type_)

    monkeypatch.setattr(s, "get_edge", probing_get_edge)
    try:
        resolve_contradiction(s, "A", "B")
    finally:
        s.close()
    assert competitor_blocked == [True]


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


# No subprocess here, but the #1307 scan reads the bare `run(...)` call as
# one, and importing the benchmark pulls in `aelfrice.cli`.
@pytest.mark.timeout(60)
def test_effect_benchmark_is_deterministic_and_not_vacuous() -> None:
    """`benchmarks/resolves_edge_effect_1658.py` backs the figures quoted
    for the HRR, wonder-seed and random-walk effects. Same config, same
    report, and the synthetic stores really do carry RESOLVES edges."""
    from benchmarks.resolves_edge_effect_1658 import Config, run

    cfg = Config(beliefs=40, edges=160, stores=2, seed=3, rw_walks=10)
    first, vacuous = run(cfg)
    second, _ = run(cfg)
    assert not vacuous
    assert first == second
    assert first["hrr_first_store"] is not None
