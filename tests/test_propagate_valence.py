"""`MemoryStore.propagate_valence` — attenuation, crediting, determinism.

Broker attenuation (#1058): an A->B->C chain where B is the broker. A
low-confidence B should dampen propagation into C by ~9x relative to a
high-confidence B.

Correctness of the walk (#1169): the multiplier is the confidence of the
belief the signal travels *through*, not of the belief being written to;
each belief is credited exactly once per walk regardless of fan-in,
reconvergence, or cycles; total injected mass is capped; and the output is
a function of the graph rather than of edge insertion order.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
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
    # A is the source. Its own confidence scales the first hop (the
    # multiplier is the confidence of the belief the signal travels
    # *through*, #1169), so keep it neutral and identical across runs.
    s.insert_belief(_mk("A", alpha=5.0, beta=5.0))
    # B is the broker for the B->C hop; this is the factor under test.
    s.insert_belief(_mk("B", alpha=broker_alpha, beta=broker_beta))
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


def test_reconvergent_paths_credit_each_belief_once() -> None:
    """Diamond A->{B1,B2}->C->D credits C once, not twice (#1169).

    Before #1169 delivery accumulated per in-edge, outside the visited
    guard, so C received 2x the mass of a single path. Falsifiable by
    out["C"] coming back at 0.5."""
    s = MemoryStore(":memory:")
    for bid in ("A", "B1", "B2", "C", "D"):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    for src, dst in (("A", "B1"), ("A", "B2"), ("B1", "C"),
                     ("B2", "C"), ("C", "D")):
        s.insert_edge(Edge(src=src, dst=dst, type=EDGE_SUPPORTS, weight=1.0))
    out = s.propagate_valence("A", valence=1.0, max_hops=4,
                              min_threshold=0.0001)
    # Every belief here is (5, 5) so every broker factor is 0.5 and every
    # SUPPORTS multiplier is 1.0: each hop halves the carried magnitude.
    assert abs(out["B1"] - 0.5) < 1e-9
    assert abs(out["B2"] - 0.5) < 1e-9
    assert abs(out["C"] - 0.25) < 1e-9
    assert abs(out["D"] - 0.125) < 1e-9


def test_fan_in_does_not_multiply_the_source_signal() -> None:
    """A 5-way fan-in credits the hub once, not 5x (#1169).

    The issue's worked example: one `aelf confirm` on the root of a
    convergent subgraph added alpha += 5.0 to the convergence point.
    Falsifiable by out["T"] exceeding a single path's magnitude."""
    s = MemoryStore(":memory:")
    for bid in ("A", "T", *[f"B{i}" for i in range(5)]):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    for i in range(5):
        s.insert_edge(
            Edge(src="A", dst=f"B{i}", type=EDGE_SUPPORTS, weight=1.0)
        )
        s.insert_edge(
            Edge(src=f"B{i}", dst="T", type=EDGE_SUPPORTS, weight=1.0)
        )
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.0001)
    assert abs(out["T"] - 0.25) < 1e-9, f"hub over-credited: {out}"


def test_cycle_between_non_source_beliefs_delivers_once() -> None:
    """A->B, B->C, C->B credits B once (#1169).

    The pre-#1169 source guard covered only the source, so a back-edge
    into an already-visited non-source belief re-delivered to it — B came
    back at 2.0x. Any bidirectional RELATES_TO or CONTRADICTS pair inside
    the hop radius hit this. Falsifiable by B exceeding one path."""
    s = MemoryStore(":memory:")
    for bid in ("A", "B", "C"):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    for src, dst in (("A", "B"), ("B", "C"), ("C", "B")):
        s.insert_edge(Edge(src=src, dst=dst, type=EDGE_SUPPORTS, weight=1.0))
    out = s.propagate_valence("A", valence=1.0, max_hops=5,
                              min_threshold=0.0001)
    assert abs(out["B"] - 0.5) < 1e-9, f"B re-credited by the cycle: {out}"
    assert abs(out["C"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Attenuation direction (#1169)
# ---------------------------------------------------------------------------


def test_recipient_confidence_does_not_scale_its_own_delta() -> None:
    """The delta a belief receives is independent of its own posterior.

    This is the rich-get-richer defect: attenuating by the *recipient's*
    confidence meant a belief at mu=0.95 absorbed 0.95x the signal while
    one at mu=0.10 absorbed 0.10x, widening the gap with no evidence
    about either. Falsifiable by the two deltas differing."""
    def one_hop(dst_alpha: float, dst_beta: float) -> float:
        s = MemoryStore(":memory:")
        s.insert_belief(_mk("A", alpha=5.0, beta=5.0))
        s.insert_belief(_mk("B", alpha=dst_alpha, beta=dst_beta))
        s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
        out = s.propagate_valence("A", valence=1.0, max_hops=1,
                                  min_threshold=0.0001)
        return out["B"]

    confident = one_hop(9.0, 1.0)    # mu = 0.9
    doubtful = one_hop(1.0, 9.0)     # mu = 0.1
    assert abs(confident - doubtful) < 1e-9, (
        f"recipient posterior still scales its own delta: "
        f"{confident} vs {doubtful}"
    )


def test_low_confidence_belief_still_receives_negative_feedback() -> None:
    """A junk belief is not shielded from a negative signal (#1169).

    With recipient-side attenuation, valence -1.0 into a belief at
    mu=0.08 became -0.08 and fell under the default min_threshold of
    0.05 once any edge multiplier was below 0.625 — low-confidence junk
    was structurally immune to the signal meant to remove it.
    Falsifiable by "J" being absent from the result."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", alpha=9.0, beta=1.0))
    s.insert_belief(_mk("J", alpha=0.5, beta=6.0))   # mu ~= 0.077
    s.insert_edge(Edge(src="A", dst="J", type=EDGE_CITES, weight=1.0))
    out = s.propagate_valence("A", valence=-1.0, max_hops=1)
    assert "J" in out, f"junk shielded from negative feedback: {out}"
    assert out["J"] < 0.0


# ---------------------------------------------------------------------------
# Determinism and the mass cap (#1169)
# ---------------------------------------------------------------------------


def _diamond_in_edge_order(
    edge_order: list[tuple[str, str, str]],
) -> dict[str, float]:
    s = MemoryStore(":memory:")
    for bid in ("A", "B", "C", "D", "E"):
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    for src, dst, etype in edge_order:
        s.insert_edge(Edge(src=src, dst=dst, type=etype, weight=1.0))
    return s.propagate_valence("A", valence=1.0, max_hops=4,
                              min_threshold=0.0001)


def test_output_is_invariant_to_edge_insertion_order() -> None:
    """Same logical graph, different physical row order, same deltas.

    `edges_from` had no ORDER BY, so row order was (src, rowid) —
    insertion order. The issue measured a 3.3x difference in the delta
    delivered to E purely from inserting A-SUPPORTS->B before or after
    A-RELATES_TO->C. Falsifiable by the two dicts differing."""
    from aelfrice.models import EDGE_RELATES_TO

    forward = [
        ("A", "B", EDGE_SUPPORTS),
        ("A", "C", EDGE_RELATES_TO),
        ("B", "D", EDGE_SUPPORTS),
        ("C", "D", EDGE_SUPPORTS),
        ("D", "E", EDGE_SUPPORTS),
    ]
    reversed_order = list(reversed(forward))

    assert _diamond_in_edge_order(forward) == _diamond_in_edge_order(
        reversed_order
    )


def test_total_injected_mass_is_capped(  # AC4
) -> None:
    """One event cannot inject unbounded evidence into a wide graph.

    Falsifiable by the summed absolute delta exceeding the cap."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", alpha=9.0, beta=1.0))
    for i in range(40):
        s.insert_belief(_mk(f"B{i:02d}", alpha=5.0, beta=5.0))
        s.insert_edge(
            Edge(src="A", dst=f"B{i:02d}", type=EDGE_SUPPORTS, weight=1.0)
        )
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.0001)
    total = sum(abs(v) for v in out.values())
    assert total <= 1.0 * 3 + 1e-9, f"mass {total} exceeds the cap"
    # The cap binds here, so not every neighbour is reached — but the ones
    # that are still get a full-strength delta rather than a diluted one.
    assert out, "cap swallowed every delivery"
    assert all(abs(v) > 0.0 for v in out.values())


def test_explicit_max_total_mass_is_honoured_for_negative_valence() -> None:
    """The override binds on negative events as well as positive ones.

    A budget below abs(valence) * max_hops must clip the walk; the deltas
    that survive stay negative. Falsifiable by the summed magnitude
    exceeding the budget or by a positive delta appearing."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", alpha=9.0, beta=1.0))
    for i in range(10):
        s.insert_belief(_mk(f"B{i}", alpha=5.0, beta=5.0))
        s.insert_edge(
            Edge(src="A", dst=f"B{i}", type=EDGE_SUPPORTS, weight=1.0)
        )
    out = s.propagate_valence("A", valence=-1.0, max_hops=3,
                              min_threshold=0.0001, max_total_mass=0.5)
    assert out, "budget swallowed every delivery"
    assert sum(abs(v) for v in out.values()) <= 0.5 + 1e-9
    assert all(v < 0.0 for v in out.values())


def test_explicit_max_total_mass_is_honoured() -> None:
    """An explicit budget overrides the default. Falsifiable by the sum
    exceeding the passed value."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("A", alpha=9.0, beta=1.0))
    for i in range(10):
        s.insert_belief(_mk(f"B{i}", alpha=5.0, beta=5.0))
        s.insert_edge(
            Edge(src="A", dst=f"B{i}", type=EDGE_SUPPORTS, weight=1.0)
        )
    out = s.propagate_valence("A", valence=1.0, max_hops=3,
                              min_threshold=0.0001, max_total_mass=1.0)
    assert sum(abs(v) for v in out.values()) <= 1.0 + 1e-9


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


# ---------------------------------------------------------------------------
# Property test (#1169 AC5): mass is bounded and order-invariant for any
# graph shape hypothesis can build.
# ---------------------------------------------------------------------------

_N_NODES = 6
_NODE_IDS = [f"n{i}" for i in range(_N_NODES)]


def _store_from_edges(
    edges: list[tuple[int, int]], shuffle_seed: int,
) -> MemoryStore:
    s = MemoryStore(":memory:")
    for bid in _NODE_IDS:
        s.insert_belief(_mk(bid, alpha=5.0, beta=5.0))
    # Rotate the insertion order so physical row order differs between
    # the two stores built from the same logical edge set.
    ordered = edges[shuffle_seed:] + edges[:shuffle_seed]
    for src, dst in ordered:
        s.insert_edge(
            Edge(src=_NODE_IDS[src], dst=_NODE_IDS[dst],
                 type=EDGE_SUPPORTS, weight=1.0)
        )
    return s


@given(
    edges=st.lists(
        # Self-loops are NOT filtered: `insert_edge` accepts src == dst
        # (no guard, and the PK permits it), so a real store can hold one
        # and the walk must handle it — a self-loop on the source is
        # dropped by the source guard, one elsewhere by the visited set.
        st.tuples(
            st.integers(min_value=0, max_value=_N_NODES - 1),
            st.integers(min_value=0, max_value=_N_NODES - 1),
        ),
        min_size=0,
        max_size=14,
        unique=True,
    ),
    valence=st.sampled_from([1.0, -1.0, 0.5]),
    max_hops=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=150, deadline=None)
def test_mass_bounded_and_order_invariant_for_any_shape(
    edges: list[tuple[int, int]], valence: float, max_hops: int,
) -> None:
    """For any graph shape: total injected mass stays within the budget,
    the source is never a recipient, and the result does not depend on
    edge insertion order.

    Falsifiable by a mass overrun (the fan-in/diamond/cycle amplification
    classes), by the source appearing, or by the two orderings
    disagreeing."""
    base = _store_from_edges(edges, shuffle_seed=0)
    rotated = _store_from_edges(edges, shuffle_seed=len(edges) // 2)
    try:
        out = base.propagate_valence(
            _NODE_IDS[0], valence=valence, max_hops=max_hops,
            min_threshold=0.0001,
        )
        out_rotated = rotated.propagate_valence(
            _NODE_IDS[0], valence=valence, max_hops=max_hops,
            min_threshold=0.0001,
        )
    finally:
        base.close()
        rotated.close()

    cap = abs(valence) * max_hops
    assert sum(abs(v) for v in out.values()) <= cap + 1e-9
    assert _NODE_IDS[0] not in out
    # Each belief credited at most once, so no delta can exceed the
    # strongest single hop out of the source.
    assert all(abs(v) <= abs(valence) + 1e-9 for v in out.values())
    assert out == pytest.approx(out_rotated)


def test_the_shallower_path_wins_even_when_a_longer_one_is_stronger() -> None:
    """Hypothesis: credit lands at the shallowest hop, not the largest path.

    Pinned because the two are easy to conflate. Magnitude is
    non-increasing *along* a path (every EDGE_VALENCE magnitude and every
    confidence is <= 1), which invites the reading that the first path to
    arrive is also the strongest. It is not: a longer chain of strong
    edges can carry more than a short weak one, and the shallower path is
    still the one taken.

    Falsifiable by X receiving the two-hop magnitude instead of the
    one-hop one."""
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("A", alpha=9.0, beta=1.0))   # confidence 0.9
        s.insert_belief(_mk("B", alpha=9.0, beta=1.0))   # confidence 0.9
        s.insert_belief(_mk("X", alpha=5.0, beta=5.0))
        # Short and weak: RELATES_TO carries 0.3.
        s.insert_edge(Edge(src="A", dst="X", type=EDGE_RELATES_TO, weight=1.0))
        # Long and strong: two SUPPORTS hops, each carrying 1.0.
        s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
        s.insert_edge(Edge(src="B", dst="X", type=EDGE_SUPPORTS, weight=1.0))

        out = s.propagate_valence(
            "A", valence=1.0, max_hops=3, min_threshold=0.0001,
            src_confidence=0.9,
        )
    finally:
        s.close()

    one_hop = 1.0 * 0.3 * 0.9          # EDGE_VALENCE[RELATES_TO] * conf(A)
    two_hop = (1.0 * 1.0 * 0.9) * 1.0 * 0.9   # via B, strictly larger
    assert two_hop > one_hop, "fixture no longer demonstrates the case"

    assert out["X"] == pytest.approx(one_hop), (
        f"X took the {'two-hop' if out['X'] == pytest.approx(two_hop) else 'wrong'} "
        f"path; credit is documented as landing at the shallowest hop"
    )
    # And B, reached only the one way, is unaffected by the tie-break.
    assert out["B"] == pytest.approx(1.0 * 1.0 * 0.9)
