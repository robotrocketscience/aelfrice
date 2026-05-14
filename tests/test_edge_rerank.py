"""Unit tests for `aelfrice.edge_rerank` (#421).

Cover the contract end-to-end without a corpus dependency:

  T1. Empty hops → empty list (no store calls).
  T2. Empty penalty config → identity (re-sort only).
  T3. Default config (None) demotes a belief with at least one
      ``POTENTIALLY_STALE`` incoming edge.
  T4. Single matching edge type fires once even when multiple
      incoming edges of that type are present (set-based).
  T5. Multi-edge-type composition is multiplicative.
  T6. A belief with no incoming edges is unchanged.
  T7. Determinism: re-sort tie-break is ``(-score, belief.id)``.
  T8. Custom penalty config overrides default.
  T9. Penalty value of 0.0 zeroes the score.
"""
from __future__ import annotations

import pytest

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.edge_rerank import (
    DEFAULT_STALE_PENALTY,
    EDGE_TYPE_PENALTIES_DEFAULT,
    apply_edge_type_rerank,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_SUPPORTS,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(belief_id: str) -> Belief:
    return Belief(
        id=belief_id,
        content=belief_id,
        content_hash=f"h_{belief_id}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level="none",
        locked_at=None,
        created_at="2026-05-05T00:00:00Z",
        last_retrieved_at=None,
    )


def _hop(store: MemoryStore, belief_id: str, score: float) -> ScoredHop:
    belief = store.get_belief(belief_id)
    assert belief is not None
    return ScoredHop(belief=belief, score=score, depth=1, path=["SUPPORTS"])


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    for bid in ("STALE", "FRESH", "DOUBLE", "ISOLATED"):
        s.insert_belief(_mk(bid))
    return s


def test_empty_hops_returns_empty_list(store: MemoryStore) -> None:
    assert apply_edge_type_rerank([], store) == []


def test_empty_penalty_config_is_identity_resort(store: MemoryStore) -> None:
    """An explicit empty config skips the per-hop edge query and
    returns the input re-sorted by (-score, belief.id)."""
    hops = [
        _hop(store, "FRESH", 0.6),
        _hop(store, "STALE", 0.8),
    ]
    result = apply_edge_type_rerank(hops, store, penalties={})
    assert [h.belief.id for h in result] == ["STALE", "FRESH"]
    assert [h.score for h in result] == [0.8, 0.6]


def test_default_demotes_potentially_stale(store: MemoryStore) -> None:
    """A belief with one POTENTIALLY_STALE incoming edge is demoted
    by DEFAULT_STALE_PENALTY (0.5)."""
    store.insert_edge(
        Edge(src="SRC", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    hops = [
        _hop(store, "STALE", 0.8),
        _hop(store, "FRESH", 0.6),
    ]
    result = apply_edge_type_rerank(hops, store)
    by_id = {h.belief.id: h.score for h in result}
    assert by_id["STALE"] == pytest.approx(0.8 * DEFAULT_STALE_PENALTY)
    assert by_id["FRESH"] == 0.6
    assert [h.belief.id for h in result] == ["FRESH", "STALE"]


def test_multiple_same_type_edges_fire_once(store: MemoryStore) -> None:
    """Two POTENTIALLY_STALE edges to the same dst apply ONE penalty
    factor, not two — the trigger is presence, not count."""
    store.insert_edge(
        Edge(src="A", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    store.insert_edge(
        Edge(src="B", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    hops = [_hop(store, "STALE", 0.8)]
    result = apply_edge_type_rerank(hops, store)
    assert result[0].score == pytest.approx(0.8 * DEFAULT_STALE_PENALTY)


def test_multiple_distinct_edge_types_compose_multiplicatively(
    store: MemoryStore,
) -> None:
    """Two distinct penalty-keyed edge types compose as the product
    of their penalty factors."""
    store.insert_edge(
        Edge(src="X", dst="DOUBLE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    store.insert_edge(
        Edge(src="Y", dst="DOUBLE", type=EDGE_CONTRADICTS, weight=1.0)
    )
    hops = [_hop(store, "DOUBLE", 1.0)]
    custom = {EDGE_POTENTIALLY_STALE: 0.5, EDGE_CONTRADICTS: 0.4}
    result = apply_edge_type_rerank(hops, store, penalties=custom)
    assert result[0].score == pytest.approx(1.0 * 0.5 * 0.4)


def test_belief_with_no_incoming_edges_unchanged(store: MemoryStore) -> None:
    hops = [_hop(store, "ISOLATED", 0.7)]
    result = apply_edge_type_rerank(hops, store)
    assert result[0].score == 0.7


def test_tiebreak_sort_by_belief_id_ascending(store: MemoryStore) -> None:
    """Equal post-rerank scores tie-break on belief.id ascending."""
    hops = [
        _hop(store, "FRESH", 0.5),
        _hop(store, "ISOLATED", 0.5),
    ]
    result = apply_edge_type_rerank(hops, store, penalties={})
    assert [h.belief.id for h in result] == ["FRESH", "ISOLATED"]


def test_custom_penalty_overrides_default(store: MemoryStore) -> None:
    """Caller-supplied penalty overrides the module default."""
    store.insert_edge(
        Edge(src="X", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    hops = [_hop(store, "STALE", 0.8)]
    result = apply_edge_type_rerank(
        hops, store, penalties={EDGE_POTENTIALLY_STALE: 0.1}
    )
    assert result[0].score == pytest.approx(0.08)


def test_zero_penalty_zeros_score(store: MemoryStore) -> None:
    """A 0.0 penalty zeroes the score; the belief survives in the
    output but ranks last (or ties with other zeroes by id)."""
    store.insert_edge(
        Edge(src="X", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    hops = [
        _hop(store, "STALE", 0.8),
        _hop(store, "FRESH", 0.1),
    ]
    result = apply_edge_type_rerank(
        hops, store, penalties={EDGE_POTENTIALLY_STALE: 0.0}
    )
    by_id = {h.belief.id: h.score for h in result}
    assert by_id["STALE"] == 0.0
    assert by_id["FRESH"] == 0.1
    assert [h.belief.id for h in result] == ["FRESH", "STALE"]


def test_default_config_pins_potentially_stale_only() -> None:
    """The default config keys ONLY POTENTIALLY_STALE; positive-weight
    relational edges (SUPPORTS, etc.) are biased through BFS_EDGE_WEIGHTS
    at expansion time, not here. Drift on this assertion is a
    documented widening of the rerank surface — see #421."""
    assert dict(EDGE_TYPE_PENALTIES_DEFAULT) == {
        EDGE_POTENTIALLY_STALE: DEFAULT_STALE_PENALTY,
    }
    assert EDGE_SUPPORTS not in EDGE_TYPE_PENALTIES_DEFAULT


def test_determinism_byte_identical_repeat(store: MemoryStore) -> None:
    """Two passes over the same store with the same hops produce
    byte-identical output."""
    store.insert_edge(
        Edge(src="X", dst="STALE", type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    hops = [
        _hop(store, "STALE", 0.8),
        _hop(store, "FRESH", 0.6),
        _hop(store, "ISOLATED", 0.6),
    ]
    a = apply_edge_type_rerank(hops, store)
    b = apply_edge_type_rerank(hops, store)
    assert [(h.belief.id, h.score) for h in a] == [
        (h.belief.id, h.score) for h in b
    ]
