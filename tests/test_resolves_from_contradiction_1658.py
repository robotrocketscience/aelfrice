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
import random
import sqlite3
from pathlib import Path

import pytest

from aelfrice.bfs_multihop import expand_bfs
from aelfrice.cli import _wonder_pick_seed  # pyright: ignore[reportPrivateUsage]
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
    EDGE_RELATES_TO,
    EDGE_RESOLVES,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
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
from aelfrice.wonder.strategies import random_walk


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
    SUPERSEDES came from another writer (the commit hook's triple
    extractor) therefore gets no RESOLVES edge from `aelf resolve`."""
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


# --- aelf wonder ignores the edge ---------------------------------------------


def test_wonder_default_seed_does_not_count_resolves_edges() -> None:
    """Two beliefs with one ordinary outgoing edge each tie, and the lower
    id wins the tie. The higher id also won a contradiction, so it has an
    extra RESOLVES edge. That edge must not make it the seed."""
    s = MemoryStore(":memory:")
    for bid in ("a_plain", "b_winner", "t1", "t2", "loser"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="a_plain", dst="t1", type=EDGE_RELATES_TO, weight=1.0))
    s.insert_edge(Edge(src="b_winner", dst="t2", type=EDGE_RELATES_TO, weight=1.0))
    s.insert_edge(Edge(
        src="b_winner", dst="loser", type=EDGE_RESOLVES, weight=RESOLVES_WEIGHT,
    ))
    seed = _wonder_pick_seed(s)
    assert seed is not None
    assert seed.id == "a_plain"  # type: ignore[attr-defined]


def test_wonder_default_seed_still_counts_other_edge_types() -> None:
    """Control for the test above: an extra edge of an ordinary type
    does move the seed, so the fixture can see a counted edge."""
    s = MemoryStore(":memory:")
    for bid in ("a_plain", "b_winner", "t1", "t2", "loser"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="a_plain", dst="t1", type=EDGE_RELATES_TO, weight=1.0))
    s.insert_edge(Edge(src="b_winner", dst="t2", type=EDGE_RELATES_TO, weight=1.0))
    s.insert_edge(Edge(src="b_winner", dst="loser", type=EDGE_SUPPORTS, weight=1.0))
    seed = _wonder_pick_seed(s)
    assert seed is not None
    assert seed.id == "b_winner"  # type: ignore[attr-defined]


def test_random_walk_never_follows_a_resolves_edge() -> None:
    """W has one SUPPORTS edge to T and one RESOLVES edge to L. Over
    many walks, no phantom may contain L."""
    s = MemoryStore(":memory:")
    for bid in ("W", "T", "L"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="W", dst="T", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(
        src="W", dst="L", type=EDGE_RESOLVES, weight=RESOLVES_WEIGHT,
    ))
    phantoms = random_walk(s, rng=random.Random(0), n_walks=200, depth=2)
    assert [p.composition for p in phantoms] == [("T", "W")]


def test_random_walk_dead_ends_on_a_belief_with_only_resolves_edges() -> None:
    s = MemoryStore(":memory:")
    for bid in ("W", "L"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(
        src="W", dst="L", type=EDGE_RESOLVES, weight=RESOLVES_WEIGHT,
    ))
    assert random_walk(s, rng=random.Random(0), n_walks=50, depth=2) == []


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


def _retrieval_fixture(
    edge_type: str = EDGE_RESOLVES, *, loser_matches_query: bool = False,
) -> MemoryStore:
    """Six resolved contradicting pairs, with only one edge left per pair.

    Each winner matches the query. By default each loser shares no query
    term, so a loser can enter the results only through the BFS walk.
    With `loser_matches_query`, the losers match too, so both sides of
    each edge are candidates and a weight change can join them in a
    cluster. The tie-breaker writes CONTRADICTS, SUPERSEDES and RESOLVES.
    The fixture then deletes CONTRADICTS and SUPERSEDES, which BFS walks,
    so the walk's only route to a loser is the remaining edge.
    `edge_type` replaces that edge with another type for the control
    test.
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
            content=(
                f"deploy target region number {i} is beta"
                if loser_matches_query
                else f"rollout zone {i} uses quartz cluster"
            ),
        ))
        s.insert_edge(Edge(src=w, dst=lo, type=EDGE_CONTRADICTS, weight=0.1))
    results = auto_resolve_all_contradictions(s)
    assert len(results) == 6
    for r in results:
        s.delete_edge(r.winner_id, r.loser_id, EDGE_CONTRADICTS)
        s.delete_edge(r.winner_id, r.loser_id, EDGE_SUPERSEDES)
        if edge_type != EDGE_RESOLVES:
            s.delete_edge(r.winner_id, r.loser_id, EDGE_RESOLVES)
            s.insert_edge(Edge(
                src=r.winner_id, dst=r.loser_id, type=edge_type, weight=1.0,
            ))
    return s


_RETRIEVAL_QUERY = "deploy target region"


def _retrieve_ids(s: MemoryStore) -> list[str]:
    return [
        b.id
        for b in retrieve(s, _RETRIEVAL_QUERY, token_budget=400, bfs_enabled=True)
    ]


@pytest.fixture
def force_bfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this, the expansion gate turns BFS off for a query with no
    structural marker, and `retrieve()` never walks an edge."""
    monkeypatch.setenv("AELFRICE_FORCE_EXPANSION", "1")


@pytest.mark.usefixtures("force_bfs")
def test_retrieve_bfs_reaches_losers_through_a_walkable_edge() -> None:
    """Control: with SUPPORTS in place of RESOLVES, BFS brings every loser
    into the results. The fixture can therefore see an edge BFS walks."""
    ids = _retrieve_ids(_retrieval_fixture(EDGE_SUPPORTS))
    assert {f"L{i}" for i in range(6)} <= set(ids)


@pytest.mark.usefixtures("force_bfs")
@pytest.mark.parametrize(
    "loser_matches_query", [False, True], ids=["bfs_route", "both_candidates"],
)
def test_retrieve_ranking_is_identical_with_and_without_resolves(
    loser_matches_query: bool,
) -> None:
    """`bfs_route` catches a BFS walk over RESOLVES; `both_candidates`
    catches a RESOLVES weight that joins two candidates in a cluster."""
    s = _retrieval_fixture(loser_matches_query=loser_matches_query)
    assert _count(s, EDGE_RESOLVES) == 6
    with_edges = _retrieve_ids(s)
    for i in range(6):
        s.delete_edge(f"W{i}", f"L{i}", EDGE_RESOLVES)
    assert _count(s, EDGE_RESOLVES) == 0
    without_edges = _retrieve_ids(s)
    assert with_edges, "fixture retrieved nothing; the comparison is vacuous"
    if not loser_matches_query:
        assert not any(b.startswith("L") for b in with_edges)
    assert with_edges == without_edges


# No subprocess here, but the #1307 scan reads the bare `run(...)` call as
# one, and importing the benchmark pulls in `aelfrice.cli`.
@pytest.mark.timeout(60)
def test_effect_benchmark_is_deterministic_and_not_vacuous() -> None:
    """`benchmarks/resolves_edge_effect_1658.py` backs the figures quoted
    for the HRR effect and the unchanged wonder seed and random walk.
    Same config, same report, and the synthetic stores really do carry
    RESOLVES edges."""
    from benchmarks.resolves_edge_effect_1658 import Config, run

    cfg = Config(beliefs=40, edges=160, stores=2, seed=3, rw_walks=10)
    first, vacuous = run(cfg)
    second, _ = run(cfg)
    assert not vacuous
    assert first == second
    assert first["hrr_first_store"] is not None
    # `aelf wonder` skips RESOLVES, so neither wonder reader may move.
    assert first["wonder_seed"]["seed_changed"] == 0  # type: ignore[index]
    assert first["random_walk"]["phantom_set_changed"] == 0  # type: ignore[index]
