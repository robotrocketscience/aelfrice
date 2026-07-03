"""Setr broker-confidence attenuation test.

A->B->C chain. B is the broker. Low-confidence B should dampen propagation
into C by ~9x compared to high-confidence B.
"""
from __future__ import annotations

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, alpha: float, beta: float) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-25T00:00:00Z",
        last_retrieved_at=None,
    )


def _build_chain(broker_alpha: float, broker_beta: float) -> MemoryStore:
    s = MemoryStore(":memory:")
    # A is source; broker confidence at A doesn't matter for downstream
    # because propagation multiplier uses dst broker, but give it neutral.
    s.insert_belief(_mk("A", alpha=5.0, beta=5.0))
    s.insert_belief(_mk("B", alpha=broker_alpha, beta=broker_beta))
    # C is the terminus; give it neutral confidence so its broker factor
    # is the same constant in both runs.
    s.insert_belief(_mk("C", alpha=5.0, beta=5.0))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="B", dst="C", type=EDGE_SUPPORTS, weight=1.0))
    return s


def test_low_broker_confidence_attenuates_vs_high() -> None:
    # Low-confidence broker: alpha=1, beta=9 -> 0.1
    low = _build_chain(broker_alpha=1.0, broker_beta=9.0)
    # High-confidence broker: alpha=9, beta=1 -> 0.9
    high = _build_chain(broker_alpha=9.0, broker_beta=1.0)

    deltas_low = low.propagate_valence("A", valence=1.0, max_hops=3,
                                       min_threshold=0.0001)
    deltas_high = high.propagate_valence("A", valence=1.0, max_hops=3,
                                         min_threshold=0.0001)

    assert "C" in deltas_low, f"low: C missing, got {deltas_low}"
    assert "C" in deltas_high, f"high: C missing, got {deltas_high}"

    ratio = deltas_high["C"] / deltas_low["C"]
    # Expected ratio: (0.9 / 0.1) = 9.0. Both runs share the same C broker
    # factor, EDGE_VALENCE[SUPPORTS]=1.0, and source valence, so it cancels.
    assert 8.5 < ratio < 9.5, f"expected ~9x ratio, got {ratio}"


def test_propagate_returns_empty_for_isolated_source() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("solo", alpha=5.0, beta=5.0))
    out = s.propagate_valence("solo", valence=1.0)
    assert out == {}


def test_propagate_respects_max_hops() -> None:
    s = _build_chain(broker_alpha=5.0, broker_beta=5.0)
    # max_hops=1 should reach B but not C.
    out = s.propagate_valence("A", valence=1.0, max_hops=1,
                              min_threshold=0.0001)
    assert "B" in out
    assert "C" not in out


def test_cycle_back_to_source_never_delivers(  # #1058 src-leak fix
) -> None:
    """A->B, B->A: the source must not appear in the returned map.

    Before the fix, delta delivery was unconditional, so the cycle
    handed the source 0.25 of its own signal back — a
    self-reinforcement channel once propagation feeds apply_feedback.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", alpha=5.0, beta=5.0))
    s.insert_belief(_mk("B", alpha=5.0, beta=5.0))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="B", dst="A", type=EDGE_SUPPORTS, weight=1.0))
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.0001)
    assert "A" not in out
    assert out.keys() == {"B"}


def test_contradicts_chain_flips_then_restores_sign() -> None:
    """Characterization: signed propagation multiplies edge signs.

    A -CONTRADICTS-> X -CONTRADICTS-> Y with positive source valence:
    X is penalized (negative delta), Y is *reinforced* (positive delta,
    enemy-of-my-enemy). This is structural-balance semantics, matching
    the signed-graph treatment in the retrieval tier.
    """
    s = MemoryStore(":memory:")
    for bid in ("A", "X", "Y"):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    s.insert_edge(Edge(src="A", dst="X", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.0001)
    assert out["X"] < 0.0
    assert out["Y"] > 0.0


def test_reconvergent_paths_accumulate_delivery_once_onward() -> None:
    """Characterization: diamond A->{B1,B2}->C->D.

    Delivery into C accumulates across both paths, but C's onward
    propagation carries only the first-arriving delta (first-visit
    frontier semantics). D therefore sees half of what naive
    both-paths propagation would give. Documented as-is in #1058;
    changing it would change delta magnitudes downstream.
    """
    s = MemoryStore(":memory:")
    for bid in ("A", "B1", "B2", "C", "D"):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    for src, dst in (("A", "B1"), ("A", "B2"), ("B1", "C"),
                     ("B2", "C"), ("C", "D")):
        s.insert_edge(Edge(src=src, dst=dst, type=EDGE_SUPPORTS, weight=1.0))
    out = s.propagate_valence("A", valence=1.0, max_hops=4,
                              min_threshold=0.0001)
    # Both paths deliver 0.25 into C (1.0 * 1.0 * 0.5 broker, twice over
    # two hops); onward flow into D uses the first 0.25 only.
    assert abs(out["C"] - 0.5) < 1e-9
    assert abs(out["D"] - 0.125) < 1e-9


def test_min_threshold_prunes_weak_deltas() -> None:
    """A delta whose magnitude falls below min_threshold is not
    delivered and does not extend the frontier."""
    s = _build_chain(broker_alpha=5.0, broker_beta=5.0)
    # Hop 1 delta = 0.5; hop 2 delta = 0.25. Threshold between the
    # two keeps B and prunes C.
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.3)
    assert "B" in out
    assert "C" not in out
