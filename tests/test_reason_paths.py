"""Pure-derivation tests for :func:`aelfrice.reason.derive_paths` (#645 R2).

Covers the :class:`~aelfrice.reason.ConsequencePath` shape and the
``derive_paths`` reducer over synthetic seeds + ScoredHops:

  - compound_confidence = product of per-hop posterior means.
  - weakest_link = argmin posterior mean, deepest-wins tiebreak.
  - fork_from set when terminal edge is EDGE_CONTRADICTS, points at
    parent path's terminal belief id.
  - Seed-only paths emitted alongside hop paths.
"""
from __future__ import annotations

import pytest

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.reason import (
    ConsequencePath,
    derive_paths,
)


def _mk(
    bid: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=bid,
        content_hash=f"h-{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _hop(
    b: Belief,
    path: list[str],
    trail: tuple[str, ...],
    depth: int | None = None,
    score: float = 0.8,
) -> ScoredHop:
    if depth is None:
        depth = len(trail) - 1
    return ScoredHop(
        belief=b,
        score=score,
        depth=depth,
        path=path,
        belief_id_trail=trail,
    )


def test_seed_only_emits_length_one_path() -> None:
    seed = _mk("S", alpha=4.0, beta=1.0)  # mean = 0.8
    paths = derive_paths([seed], [])
    assert len(paths) == 1
    p = paths[0]
    assert p.belief_ids == ("S",)
    assert p.edge_kinds == ()
    assert p.compound_confidence == pytest.approx(0.8)
    assert p.weakest_link_belief_id == "S"
    assert p.fork_from is None


def test_compound_confidence_is_product_of_posterior_means() -> None:
    s = _mk("S", alpha=4.0, beta=1.0)   # 0.8
    a = _mk("A", alpha=3.0, beta=1.0)   # 0.75
    b = _mk("B", alpha=2.0, beta=2.0)   # 0.5
    hops = [
        _hop(a, [EDGE_RELATES_TO], ("S", "A")),
        _hop(b, [EDGE_RELATES_TO, EDGE_SUPERSEDES], ("S", "A", "B")),
    ]
    paths = derive_paths([s], hops)
    # Seed + 2 hops = 3 paths.
    assert len(paths) == 3
    by_terminal = {p.belief_ids[-1]: p for p in paths}
    # path S→A: 0.8 * 0.75 = 0.6
    assert by_terminal["A"].compound_confidence == pytest.approx(0.6)
    # path S→A→B: 0.8 * 0.75 * 0.5 = 0.3
    assert by_terminal["B"].compound_confidence == pytest.approx(0.3)


def test_weakest_link_argmin_with_deepest_tiebreak() -> None:
    # All three beliefs at posterior mean = 0.5 → tie. Tiebreak: deepest.
    # Real BFS emits a hop per intermediate; mirror that in the fixture.
    s = _mk("S", alpha=1.0, beta=1.0)
    a = _mk("A", alpha=1.0, beta=1.0)
    b = _mk("B", alpha=1.0, beta=1.0)
    hops = [
        _hop(a, [EDGE_RELATES_TO], ("S", "A")),
        _hop(b, [EDGE_RELATES_TO, EDGE_RELATES_TO], ("S", "A", "B")),
    ]
    paths = derive_paths([s], hops)
    by_terminal = {p.belief_ids[-1]: p for p in paths}
    # Three-belief tied trail: weakest_link = "B" (deepest).
    assert by_terminal["B"].weakest_link_belief_id == "B"


def test_weakest_link_picks_actual_min_not_just_terminal() -> None:
    s = _mk("S", alpha=4.0, beta=1.0)    # 0.8
    a = _mk("A", alpha=1.0, beta=4.0)    # 0.2   ← weakest
    b = _mk("B", alpha=3.0, beta=1.0)    # 0.75
    hops = [
        _hop(a, [EDGE_RELATES_TO], ("S", "A")),
        _hop(b, [EDGE_RELATES_TO, EDGE_RELATES_TO], ("S", "A", "B")),
    ]
    paths = derive_paths([s], hops)
    by_terminal = {p.belief_ids[-1]: p for p in paths}
    assert by_terminal["B"].weakest_link_belief_id == "A"


def test_fork_from_set_on_contradicts_terminal_edge() -> None:
    s = _mk("S", alpha=5.0, beta=2.0)
    b = _mk("B", alpha=5.0, beta=2.0)
    c = _mk("C", alpha=4.0, beta=3.0)
    hops = [
        _hop(b, [EDGE_RELATES_TO], ("S", "B")),
        _hop(c, [EDGE_RELATES_TO, EDGE_CONTRADICTS], ("S", "B", "C")),
    ]
    paths = derive_paths([s], hops)
    by_terminal = {p.belief_ids[-1]: p for p in paths}
    # Parent path (terminating at B): fork_from is None.
    assert by_terminal["B"].fork_from is None
    # Forked branch (terminating at C via CONTRADICTS): fork_from == "B".
    assert by_terminal["C"].fork_from == "B"
    assert by_terminal["C"].edge_kinds[-1] == EDGE_CONTRADICTS


def test_fork_from_none_when_contradicts_not_terminal() -> None:
    """A CONTRADICTS edge in the middle of the path is not a fork —
    only when it's the terminal edge does the fork rule fire."""
    s = _mk("S", alpha=5.0, beta=2.0)
    a = _mk("A", alpha=5.0, beta=2.0)
    b = _mk("B", alpha=5.0, beta=2.0)
    hops = [
        _hop(a, [EDGE_CONTRADICTS], ("S", "A")),
        _hop(b, [EDGE_CONTRADICTS, EDGE_RELATES_TO], ("S", "A", "B")),
    ]
    paths = derive_paths([s], hops)
    by_terminal = {p.belief_ids[-1]: p for p in paths}
    # First hop (terminal CONTRADICTS) forks from "S".
    assert by_terminal["A"].fork_from == "S"
    # Second hop continues from A via RELATES_TO; terminal edge is not
    # CONTRADICTS so no fork_from on this path.
    assert by_terminal["B"].fork_from is None


def test_empty_hops_yields_only_seed_paths() -> None:
    seeds = [_mk("S1"), _mk("S2")]
    paths = derive_paths(seeds, [])
    assert len(paths) == 2
    assert {p.belief_ids for p in paths} == {("S1",), ("S2",)}
    for p in paths:
        assert p.edge_kinds == ()
        assert p.fork_from is None


def test_skip_hop_with_empty_trail_does_not_crash() -> None:
    """ScoredHops constructed without belief_id_trail (older fixture
    style) are silently skipped rather than asserted, since the
    deriver can't reconstruct a path from depth/path-of-edges alone."""
    s = _mk("S")
    a = _mk("A")
    legacy_hop = ScoredHop(
        belief=a, score=0.8, depth=1, path=[EDGE_RELATES_TO]
    )  # belief_id_trail defaults to ()
    paths = derive_paths([s], [legacy_hop])
    # One seed path only; legacy hop is skipped.
    assert len(paths) == 1
    assert paths[0].belief_ids == ("S",)


def test_consequencepath_is_frozen_and_hashable() -> None:
    p1 = ConsequencePath(
        belief_ids=("A", "B"),
        edge_kinds=(EDGE_RELATES_TO,),
        compound_confidence=0.6,
        weakest_link_belief_id="B",
        fork_from=None,
    )
    p2 = ConsequencePath(
        belief_ids=("A", "B"),
        edge_kinds=(EDGE_RELATES_TO,),
        compound_confidence=0.6,
        weakest_link_belief_id="B",
        fork_from=None,
    )
    assert p1 == p2
    assert hash(p1) == hash(p2)
    with pytest.raises(Exception):
        p1.compound_confidence = 0.9  # type: ignore[misc]


def test_derive_paths_is_deterministic() -> None:
    s = _mk("S", alpha=4.0, beta=1.0)
    a = _mk("A", alpha=3.0, beta=1.0)
    hops = [_hop(a, [EDGE_RELATES_TO], ("S", "A"))]
    p1 = derive_paths([s], hops)
    p2 = derive_paths([s], hops)
    assert p1 == p2
