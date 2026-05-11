"""Pure-derivation tests for :func:`aelfrice.reason.classify` (#645 R1).

Synthetic fixtures cover each verdict + impasse kind on a tiny store.
The classifier is a pure function of ``(seeds, hops, store)`` and
must produce deterministic output, so each test pins the expected
verdict and the expected impasse kinds in order.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.reason import (
    CLOSE_MEAN_DELTA,
    CONFIDENT_TRIALS_MIN,
    Impasse,
    ImpasseKind,
    Verdict,
    classify,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=bid,
        content_hash=f"h-{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-05-11T00:00:00Z" if lock == LOCK_USER else None,
        demotion_pressure=0,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(str(tmp_path / "r.db"))
    yield s
    s.close()


def _hop(b: Belief, path: list[str], depth: int = 1, score: float = 0.8) -> ScoredHop:
    return ScoredHop(belief=b, score=score, depth=depth, path=path)


def test_empty_hops_returns_insufficient_with_no_change(store: MemoryStore) -> None:
    seeds = [_mk("s1"), _mk("s2")]
    verdict, impasses = classify(seeds, [], store)
    assert verdict == Verdict.INSUFFICIENT
    assert len(impasses) == 1
    assert impasses[0].kind == ImpasseKind.NO_CHANGE
    assert impasses[0].belief_ids == ("s1", "s2")


def test_all_low_evidence_yields_no_change_insufficient(store: MemoryStore) -> None:
    # Seeds + hops all at Beta(1, 1) — alpha+beta = 2 < CONFIDENT_TRIALS_MIN.
    seed = _mk("seed")
    h1 = _hop(_mk("h1"), [EDGE_RELATES_TO])
    h2 = _hop(_mk("h2"), [EDGE_RELATES_TO, EDGE_RELATES_TO], depth=2)
    # Mark h1 and h2 as non-leaf so GAP doesn't compete with NO_CHANGE.
    store.insert_belief(h1.belief)
    store.insert_belief(h2.belief)
    store.insert_edge(Edge(src="h1", dst="h2", type=EDGE_RELATES_TO, weight=1.0))
    store.insert_edge(Edge(src="h2", dst="h1", type=EDGE_RELATES_TO, weight=1.0))
    verdict, impasses = classify([seed], [h1, h2], store)
    kinds = [i.kind for i in impasses]
    assert ImpasseKind.NO_CHANGE in kinds
    assert verdict == Verdict.INSUFFICIENT


def test_constraint_failure_on_locked_contradicting_endpoint(
    store: MemoryStore,
) -> None:
    # Seed: low-evidence start. Hop: locked belief reached via CONTRADICTS.
    # The seed is also low-evidence so NO_CHANGE could fire — bump the
    # locked endpoint's trial count so it's NOT low-evidence and the
    # walk has at least one confident node.
    seed = _mk("s", alpha=5.0, beta=2.0)
    locked = _mk("locked", alpha=10.0, beta=1.0, lock=LOCK_USER)
    h = _hop(locked, [EDGE_CONTRADICTS])
    verdict, impasses = classify([seed], [h], store)
    kinds = [i.kind for i in impasses]
    assert ImpasseKind.CONSTRAINT_FAILURE in kinds
    assert verdict == Verdict.PARTIAL


def test_tie_on_two_contradictions_with_similar_means(store: MemoryStore) -> None:
    seed = _mk("seed", alpha=5.0, beta=3.0)
    # Two contradicting endpoints with similar posterior means (~0.5).
    c1 = _mk("c1", alpha=4.0, beta=4.0)  # mean = 0.500
    c2 = _mk("c2", alpha=5.0, beta=4.0)  # mean ≈ 0.556 → delta ≈ 0.056 < 0.15
    assert abs(c1.alpha / (c1.alpha + c1.beta) - c2.alpha / (c2.alpha + c2.beta)) < CLOSE_MEAN_DELTA
    h1 = _hop(c1, [EDGE_CONTRADICTS])
    h2 = _hop(c2, [EDGE_RELATES_TO, EDGE_CONTRADICTS], depth=2)
    verdict, impasses = classify([seed], [h1, h2], store)
    tie = [i for i in impasses if i.kind == ImpasseKind.TIE]
    assert len(tie) == 1
    assert tie[0].belief_ids == ("c1", "c2")
    assert verdict == Verdict.CONTRADICTORY


def test_far_apart_contradictions_do_not_tie(store: MemoryStore) -> None:
    seed = _mk("seed", alpha=5.0, beta=3.0)
    c1 = _mk("c1", alpha=9.0, beta=1.0)  # mean = 0.9
    c2 = _mk("c2", alpha=1.0, beta=9.0)  # mean = 0.1 → delta = 0.8
    h1 = _hop(c1, [EDGE_CONTRADICTS])
    h2 = _hop(c2, [EDGE_CONTRADICTS])
    verdict, impasses = classify([seed], [h1, h2], store)
    assert not any(i.kind == ImpasseKind.TIE for i in impasses)
    # Two confident hops + no impasse → SUFFICIENT (CONTRADICTS edges
    # alone don't make a verdict; only TIE / CONSTRAINT_FAILURE do).
    assert verdict == Verdict.SUFFICIENT


def test_gap_on_low_evidence_leaf(store: MemoryStore) -> None:
    seed = _mk("seed", alpha=5.0, beta=2.0)
    confident = _mk("conf", alpha=5.0, beta=2.0)
    leaf = _mk("leaf")  # alpha+beta = 2 (low evidence)
    # `conf` has an outbound edge; `leaf` is a true leaf.
    store.insert_belief(confident)
    store.insert_belief(leaf)
    store.insert_edge(
        Edge(src="conf", dst="other", type=EDGE_RELATES_TO, weight=1.0)
    )
    h_conf = _hop(confident, [EDGE_SUPERSEDES])
    h_leaf = _hop(leaf, [EDGE_SUPERSEDES, EDGE_RELATES_TO], depth=2)
    verdict, impasses = classify([seed], [h_conf, h_leaf], store)
    gaps = [i for i in impasses if i.kind == ImpasseKind.GAP]
    assert len(gaps) == 1
    assert gaps[0].belief_ids == ("leaf",)
    # confident hop + gap impasse → PARTIAL.
    assert verdict == Verdict.PARTIAL


def test_gap_only_low_evidence_yields_uncertain(store: MemoryStore) -> None:
    seed = _mk("seed", alpha=5.0, beta=2.0)
    leaf = _mk("leaf")  # low evidence, leaf
    h = _hop(leaf, [EDGE_RELATES_TO])
    verdict, impasses = classify([seed], [h], store)
    # Note: this triggers NO_CHANGE first (seed=high, hop=low; not "all
    # low") — actually only NO_CHANGE fires when *both* seed and hop
    # are low. So here NO_CHANGE does not fire; GAP does (leaf + low).
    kinds = [i.kind for i in impasses]
    assert ImpasseKind.GAP in kinds
    assert ImpasseKind.NO_CHANGE not in kinds
    assert verdict == Verdict.UNCERTAIN


def test_sufficient_when_confident_hop_and_no_impasse(store: MemoryStore) -> None:
    seed = _mk("seed", alpha=5.0, beta=2.0)
    confident = _mk("conf", alpha=8.0, beta=2.0)
    # Non-leaf so no GAP fires.
    store.insert_edge(
        Edge(src="conf", dst="x", type=EDGE_RELATES_TO, weight=1.0)
    )
    h = _hop(confident, [EDGE_SUPERSEDES])
    verdict, impasses = classify([seed], [h], store)
    assert verdict == Verdict.SUFFICIENT
    assert impasses == []


def test_classify_is_deterministic(store: MemoryStore) -> None:
    """Two runs with the same inputs must return identical results."""
    seed = _mk("seed", alpha=5.0, beta=2.0)
    c1 = _mk("c1", alpha=4.0, beta=4.0)
    c2 = _mk("c2", alpha=5.0, beta=4.0)
    h1 = _hop(c1, [EDGE_CONTRADICTS])
    h2 = _hop(c2, [EDGE_RELATES_TO, EDGE_CONTRADICTS], depth=2)
    v1, i1 = classify([seed], [h1, h2], store)
    v2, i2 = classify([seed], [h1, h2], store)
    assert v1 == v2
    assert i1 == i2


def test_constants_documented_and_load_bearing() -> None:
    """The module constants must be importable and have the values the
    derivation rules assume."""
    assert CONFIDENT_TRIALS_MIN == 4
    assert CLOSE_MEAN_DELTA == 0.15


def test_impasse_is_frozen_and_hashable() -> None:
    a = Impasse(ImpasseKind.GAP, ("x",), "note")
    b = Impasse(ImpasseKind.GAP, ("x",), "note")
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(Exception):
        a.kind = ImpasseKind.TIE  # type: ignore[misc]


def test_verdict_and_impasse_kind_are_string_valued() -> None:
    # Required for JSON serialisation downstream.
    assert Verdict.SUFFICIENT.value == "SUFFICIENT"
    assert ImpasseKind.NO_CHANGE.value == "NO_CHANGE"
