"""health.assess_health: regime classifier tests.

Atomic short tests cover compute_features (small/empty stores),
classify_regime (each branch including insufficient_data), the
classification-confidence math, anchor-aware feature scoring, and the
deterministic regime_description framing (no alarm copy).
"""
from __future__ import annotations

import io
import json

from aelfrice.health import (
    MIN_BELIEFS,
    REGIME_IGNORE,
    REGIME_INSUFFICIENT_DATA,
    REGIME_MIXED,
    REGIME_SUPERSEDE,
    HealthFeatures,
    HealthReport,
    assess_health,
    classify_regime,
    compute_features,
    regime_description,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    EDGE_TYPES,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    alpha: float,
    beta: float,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


# --- compute_features: empty + small stores -----------------------------


def test_empty_store_features_zero() -> None:
    s = MemoryStore(":memory:")
    f = compute_features(s)
    assert f.n_beliefs == 0
    assert f.confidence_mean == 0.0
    assert f.mass_mean == 0.0


def test_single_belief_features_match_belief() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", alpha=3.0, beta=1.0))
    f = compute_features(s)
    assert f.n_beliefs == 1
    assert abs(f.confidence_mean - 0.75) < 1e-9
    assert abs(f.mass_mean - 4.0) < 1e-9


def test_three_beliefs_confidence_mean_is_average() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", alpha=4.0, beta=1.0))  # 0.8
    s.insert_belief(_mk("b2", alpha=2.0, beta=2.0))  # 0.5
    s.insert_belief(_mk("b3", alpha=1.0, beta=4.0))  # 0.2
    f = compute_features(s)
    assert abs(f.confidence_mean - 0.5) < 1e-9


def test_odd_count_median_is_middle_element() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", alpha=4.0, beta=1.0))  # 0.8
    s.insert_belief(_mk("b2", alpha=2.0, beta=2.0))  # 0.5
    s.insert_belief(_mk("b3", alpha=1.0, beta=4.0))  # 0.2
    f = compute_features(s)
    assert f.confidence_median == 0.5


def test_even_count_median_averages_middle_two() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", alpha=4.0, beta=1.0))  # 0.8
    s.insert_belief(_mk("b2", alpha=2.0, beta=2.0))  # 0.5
    s.insert_belief(_mk("b3", alpha=1.0, beta=4.0))  # 0.2
    s.insert_belief(_mk("b4", alpha=3.0, beta=1.0))  # 0.75
    f = compute_features(s)
    # sorted: 0.2, 0.5, 0.75, 0.8 -> middle two avg = 0.625
    assert abs(f.confidence_median - 0.625) < 1e-9


def test_lock_per_1000_zero_when_no_locks() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", alpha=1.0, beta=1.0))
    f = compute_features(s)
    assert f.lock_per_1000 == 0.0


def test_lock_per_1000_scaled_correctly() -> None:
    s = MemoryStore(":memory:")
    # 1 of 4 locked -> 250 per 1000.
    s.insert_belief(_mk("b1", 1.0, 1.0))
    s.insert_belief(_mk("b2", 1.0, 1.0))
    s.insert_belief(_mk("b3", 1.0, 1.0))
    s.insert_belief(_mk("b4", 1.0, 1.0,
                        lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z"))
    f = compute_features(s)
    assert abs(f.lock_per_1000 - 250.0) < 1e-9


def test_edge_per_belief_zero_when_no_edges() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", 1.0, 1.0))
    f = compute_features(s)
    assert f.edge_per_belief == 0.0


def test_edge_per_belief_one_when_one_edge_per_belief() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("a", 1.0, 1.0))
    s.insert_belief(_mk("b", 1.0, 1.0))
    s.insert_edge(Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="b", dst="a", type=EDGE_SUPPORTS, weight=1.0))
    f = compute_features(s)
    # 2 beliefs, 2 edges -> 1.0 edge/belief
    assert f.edge_per_belief == 1.0


# --- classify_regime: branches ------------------------------------------


def test_below_min_beliefs_returns_insufficient_data() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS - 1,
        confidence_mean=0.7,
        confidence_median=0.7,
        mass_mean=3.0,
        lock_per_1000=0.2,
        edge_per_belief=0.8,
    )
    r = classify_regime(f)
    assert r.regime == REGIME_INSUFFICIENT_DATA


def test_below_min_beliefs_classification_confidence_zero() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS - 1,
        confidence_mean=0.7,
        confidence_median=0.7,
        mass_mean=3.0,
        lock_per_1000=0.2,
        edge_per_belief=0.8,
    )
    r = classify_regime(f)
    assert r.classification_confidence == 0.0


def test_supersede_anchors_classify_supersede() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS,
        confidence_mean=0.65,
        confidence_median=0.65,
        mass_mean=3.0,
        lock_per_1000=0.2,
        edge_per_belief=0.8,
    )
    r = classify_regime(f)
    assert r.regime == REGIME_SUPERSEDE


def test_ignore_anchors_classify_ignore() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS,
        confidence_mean=0.377,
        confidence_median=0.375,
        mass_mean=1.616,
        lock_per_1000=0.0,
        edge_per_belief=2.41,
    )
    r = classify_regime(f)
    assert r.regime == REGIME_IGNORE


def test_boundary_features_classify_mixed() -> None:
    """Halfway between the supersede and ignore anchors lands the
    aggregate score near 0.5 -> mixed."""
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS,
        confidence_mean=0.45,  # between 0.377 and 0.501
        confidence_median=0.45,
        mass_mean=1.85,  # between 1.616 and 2.007
        lock_per_1000=0.05,
        edge_per_belief=1.7,  # between 1.139 and 2.410
    )
    r = classify_regime(f)
    assert r.regime == REGIME_MIXED


# --- Classification confidence math -------------------------------------


def test_clear_supersede_confidence_high() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS,
        confidence_mean=0.75,
        confidence_median=0.75,
        mass_mean=4.0,
        lock_per_1000=0.3,
        edge_per_belief=0.5,
    )
    r = classify_regime(f)
    assert r.classification_confidence > 0.8


def test_clear_ignore_confidence_high() -> None:
    f = HealthFeatures(
        n_beliefs=MIN_BELIEFS,
        confidence_mean=0.30,
        confidence_median=0.30,
        mass_mean=1.5,
        lock_per_1000=0.0,
        edge_per_belief=2.5,
    )
    r = classify_regime(f)
    assert r.classification_confidence > 0.8


def test_classification_confidence_in_unit_range() -> None:
    """For any valid feature set the classification confidence stays
    in [0, 1]."""
    for cm in (0.1, 0.5, 0.9):
        f = HealthFeatures(
            n_beliefs=MIN_BELIEFS,
            confidence_mean=cm,
            confidence_median=cm,
            mass_mean=2.0,
            lock_per_1000=0.1,
            edge_per_belief=1.0,
        )
        r = classify_regime(f)
        assert 0.0 <= r.classification_confidence <= 1.0


# --- assess_health end-to-end --------------------------------------------


def test_assess_health_empty_store_insufficient_data() -> None:
    s = MemoryStore(":memory:")
    r = assess_health(s)
    assert r.regime == REGIME_INSUFFICIENT_DATA


def test_assess_health_returns_health_report_typed() -> None:
    s = MemoryStore(":memory:")
    r = assess_health(s)
    assert isinstance(r, HealthReport)


def test_assess_health_n_beliefs_matches_store() -> None:
    s = MemoryStore(":memory:")
    for i in range(7):
        s.insert_belief(_mk(f"b{i}", 1.0, 1.0))
    r = assess_health(s)
    assert r.features.n_beliefs == 7


def test_assess_health_under_threshold_yields_insufficient_data() -> None:
    s = MemoryStore(":memory:")
    for i in range(MIN_BELIEFS - 1):
        s.insert_belief(_mk(f"b{i}", 1.0, 1.0))
    r = assess_health(s)
    assert r.regime == REGIME_INSUFFICIENT_DATA


def test_assess_health_at_threshold_yields_a_real_regime() -> None:
    s = MemoryStore(":memory:")
    for i in range(MIN_BELIEFS):
        s.insert_belief(_mk(f"b{i}", 4.0, 1.0))  # supersede-shaped
    r = assess_health(s)
    assert r.regime != REGIME_INSUFFICIENT_DATA


# --- Determinism --------------------------------------------------------


def test_repeated_assessment_returns_same_regime() -> None:
    s = MemoryStore(":memory:")
    for i in range(MIN_BELIEFS):
        s.insert_belief(_mk(f"b{i}", 4.0, 1.0))
    one = assess_health(s)
    two = assess_health(s)
    assert one.regime == two.regime
    assert one.mean_score == two.mean_score
    assert one.classification_confidence == two.classification_confidence


# --- Regime description framing -----------------------------------------


def test_supersede_description_no_alarm_copy() -> None:
    desc = regime_description(REGIME_SUPERSEDE)
    forbidden = ("wrong", "defective", "broken", "error", "failure", "bad")
    desc_lower = desc.lower()
    for word in forbidden:
        assert word not in desc_lower


def test_ignore_description_no_alarm_copy() -> None:
    desc = regime_description(REGIME_IGNORE)
    forbidden = ("wrong", "defective", "broken", "error", "failure", "bad")
    desc_lower = desc.lower()
    for word in forbidden:
        assert word not in desc_lower


def test_ignore_description_frames_as_legitimate() -> None:
    desc = regime_description(REGIME_IGNORE).lower()
    assert "legitimate" in desc


def test_supersede_description_describes_active_mode() -> None:
    desc = regime_description(REGIME_SUPERSEDE).lower()
    assert "active" in desc


def test_mixed_description_returned_for_mixed() -> None:
    assert regime_description(REGIME_MIXED) != regime_description(REGIME_SUPERSEDE)
    assert regime_description(REGIME_MIXED) != regime_description(REGIME_IGNORE)


def test_insufficient_data_description_directs_user_to_onboard() -> None:
    desc = regime_description(REGIME_INSUFFICIENT_DATA).lower()
    assert "onboard" in desc


# --- edges_by_type: new field on HealthFeatures (#452) -------------------

# Full registry that edges_by_type must enumerate.
_FULL_EDGE_REGISTRY: frozenset[str] = EDGE_TYPES | {EDGE_POTENTIALLY_STALE}


def test_empty_store_edges_by_type_zero_padded() -> None:
    """Empty store: edges_by_type present with 0 for every registered type."""
    s = MemoryStore(":memory:")
    f = compute_features(s)
    assert isinstance(f.edges_by_type, dict)
    assert set(f.edges_by_type.keys()) == _FULL_EDGE_REGISTRY
    assert all(v == 0 for v in f.edges_by_type.values())


def test_seeded_store_edges_by_type_counts_correctly() -> None:
    """Seeded store with one edge of each type: counts must be 1 each."""
    s = MemoryStore(":memory:")
    # Provide enough beliefs for all edges.
    belief_ids = [f"b{i}" for i in range(20)]
    for bid in belief_ids:
        s.insert_belief(_mk(bid, 1.0, 1.0))

    edge_iter = iter(zip(belief_ids, belief_ids[1:]))
    one_of_each = [
        (EDGE_SUPPORTS, *next(edge_iter)),
        (EDGE_CITES, *next(edge_iter)),
        (EDGE_CONTRADICTS, *next(edge_iter)),
        (EDGE_SUPERSEDES, *next(edge_iter)),
        (EDGE_RELATES_TO, *next(edge_iter)),
        (EDGE_DERIVED_FROM, *next(edge_iter)),
        (EDGE_IMPLEMENTS, *next(edge_iter)),
        (EDGE_TEMPORAL_NEXT, *next(edge_iter)),
        (EDGE_TESTS, *next(edge_iter)),
        (EDGE_POTENTIALLY_STALE, *next(edge_iter)),
    ]
    for etype, src, dst in one_of_each:
        s.insert_edge(Edge(src=src, dst=dst, type=etype, weight=1.0))

    f = compute_features(s)
    assert set(f.edges_by_type.keys()) == _FULL_EDGE_REGISTRY
    for etype, _src, _dst in one_of_each:
        assert f.edges_by_type[etype] == 1, f"expected 1 for {etype}"


def test_edges_by_type_multiple_of_same_type() -> None:
    """Multiple edges of one type sum correctly; others remain 0."""
    s = MemoryStore(":memory:")
    for i in range(5):
        s.insert_belief(_mk(f"x{i}", 1.0, 1.0))
    # Insert 3 SUPPORTS edges.
    s.insert_edge(Edge(src="x0", dst="x1", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="x1", dst="x2", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="x2", dst="x3", type=EDGE_SUPPORTS, weight=1.0))
    f = compute_features(s)
    assert f.edges_by_type[EDGE_SUPPORTS] == 3
    assert f.edges_by_type[EDGE_CITES] == 0


def test_edges_by_type_not_in_per_feature_scores() -> None:
    """edges_by_type must not appear in per_feature_scores (not a classifier input)."""
    s = MemoryStore(":memory:")
    for i in range(MIN_BELIEFS):
        s.insert_belief(_mk(f"b{i}", 4.0, 1.0))
    r = classify_regime(compute_features(s))
    assert "edges_by_type" not in r.per_feature_scores


def test_classify_regime_per_feature_scores_unchanged() -> None:
    """Regime classifier still produces exactly the original five feature keys."""
    expected_keys = {
        "confidence_mean",
        "confidence_median",
        "mass_mean",
        "lock_per_1000",
        "edge_per_belief",
    }
    s = MemoryStore(":memory:")
    for i in range(MIN_BELIEFS):
        s.insert_belief(_mk(f"b{i}", 4.0, 1.0))
    r = classify_regime(compute_features(s))
    assert set(r.per_feature_scores.keys()) == expected_keys
