"""Regime classifier for `aelf:health`.

Computes five features over the store and classifies the project's
working mode as `supersede`, `ignore`, `mixed`, or `insufficient_data`.
Per the round-9-through-12 R&D, regime is a stable property of how a
project uses memory — not a defect to be flagged. Output framing
treats both supersede and ignore modes as legitimate working patterns.

Features (per E37 anchors):
  confidence_mean      supersede 0.501-0.750  ignore 0.377
  confidence_median    supersede 0.500-0.750  ignore 0.375
  mass_mean            supersede 2.007-4.056  ignore 1.616
  lock_per_1000        supersede 0.113-0.355  ignore 0.000
  edge_per_belief      supersede 0.464-1.139  ignore 2.410  (inverted)

Classifier (per E38):
  per-feature score against (supersede_range, ignore_value); aggregate
  mean across the five features; threshold >= 0.7 -> supersede,
  <= 0.3 -> ignore, else -> mixed. Below MIN_BELIEFS the regime is
  reported as `insufficient_data` regardless of features (per E43).

Classification confidence reflects how far the aggregate score is from
the supersede/ignore boundary at 0.5, normalized to [0, 1]. Boundary-
near projects (mean_score around 0.5) get lower confidence so the
output can phrase the regime as "leaning X" rather than committing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from aelfrice.models import EDGE_POTENTIALLY_STALE, EDGE_TYPES
from aelfrice.store import MemoryStore

# E43: minimum belief count below which the classifier reports
# `insufficient_data` instead of forcing a regime label. Lowered from
# the placeholder 1000 to 100 after subsampling held 5/5 stable at n=100
# on every full-confidence production DB tested.
MIN_BELIEFS: Final[int] = 100

# Anchor ranges per E37. (lo, hi) means the supersede observed range;
# `ignore` is the single-DB anchor from the agentmemory production DB.
@dataclass(frozen=True)
class _Anchor:
    supersede_lo: float
    supersede_hi: float
    ignore_value: float
    higher_is_supersede: bool


_FEATURE_ANCHORS: Final[dict[str, _Anchor]] = {
    "confidence_mean": _Anchor(0.501, 0.750, 0.377, higher_is_supersede=True),
    "confidence_median": _Anchor(0.500, 0.750, 0.375, higher_is_supersede=True),
    "mass_mean": _Anchor(2.007, 4.056, 1.616, higher_is_supersede=True),
    "lock_per_1000": _Anchor(0.113, 0.355, 0.000, higher_is_supersede=True),
    "edge_per_belief": _Anchor(0.464, 1.139, 2.410, higher_is_supersede=False),
}

REGIME_SUPERSEDE: Final[str] = "supersede"
REGIME_IGNORE: Final[str] = "ignore"
REGIME_MIXED: Final[str] = "mixed"
REGIME_INSUFFICIENT_DATA: Final[str] = "insufficient_data"

_SUPERSEDE_THRESHOLD: Final[float] = 0.7
_IGNORE_THRESHOLD: Final[float] = 0.3


@dataclass
class HealthFeatures:
    """Numeric snapshot of the store at one point in time."""

    n_beliefs: int
    confidence_mean: float
    confidence_median: float
    mass_mean: float
    lock_per_1000: float
    edge_per_belief: float
    # Per-edge-type count breakdown; keys are every member of EDGE_TYPES ∪
    # {EDGE_POTENTIALLY_STALE} so the shape is stable across stores (absent
    # types appear with value 0).  Not fed into the regime classifier —
    # additive observability field only.
    edges_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Classifier output, ready for direct user-facing rendering."""

    regime: str
    classification_confidence: float
    mean_score: float
    per_feature_scores: dict[str, float]
    features: HealthFeatures


def _confidence(alpha: float, beta: float) -> float:
    total = alpha + beta
    if total <= 0.0:
        return 0.5
    return alpha / total


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0


def _zero_padded_edge_counts(raw: dict[str, int]) -> dict[str, int]:
    """Merge *raw* (SQL result, may omit zero-count types) into a dict
    initialised over the full registry: EDGE_TYPES ∪ {EDGE_POTENTIALLY_STALE}.

    POTENTIALLY_STALE is stored in the same `edges` table as the structural
    edge types (confirmed in #387) so the single `count_edges_by_type` query
    captures it automatically.
    """
    full: dict[str, int] = {t: 0 for t in EDGE_TYPES}
    full[EDGE_POTENTIALLY_STALE] = 0
    full.update(raw)
    return full


def compute_features(store: MemoryStore) -> HealthFeatures:
    """Roll up the store into the five-feature vector the classifier
    consumes. Pure function of the current store state.

    Empty store returns all zeros — the classifier downstream will then
    surface `insufficient_data`.
    """
    n_beliefs = store.count_beliefs()
    edges_by_type = _zero_padded_edge_counts(store.count_edges_by_type())
    if n_beliefs == 0:
        return HealthFeatures(
            n_beliefs=0,
            confidence_mean=0.0,
            confidence_median=0.0,
            mass_mean=0.0,
            lock_per_1000=0.0,
            edge_per_belief=0.0,
            edges_by_type=edges_by_type,
        )

    pairs = store.alpha_beta_pairs()
    confidences = sorted(_confidence(a, b) for a, b in pairs)
    masses = [a + b for a, b in pairs]

    n_edges = store.count_edges()
    n_locked = store.count_locked()

    return HealthFeatures(
        n_beliefs=n_beliefs,
        confidence_mean=sum(confidences) / n_beliefs,
        confidence_median=_median(confidences),
        mass_mean=sum(masses) / n_beliefs,
        lock_per_1000=(n_locked / n_beliefs) * 1000.0,
        edge_per_belief=n_edges / n_beliefs,
        edges_by_type=edges_by_type,
    )


def _score_feature(value: float, anchor: _Anchor) -> float:
    """Return per-feature score in [0, 1].

    1.0 = at-or-inside the supersede range.
    0.0 = at-or-past the ignore anchor.
    Linear interpolation between the supersede edge and the ignore
    anchor in between.
    """
    if anchor.higher_is_supersede:
        if value >= anchor.supersede_lo:
            return 1.0
        if value <= anchor.ignore_value:
            return 0.0
        span = anchor.supersede_lo - anchor.ignore_value
        if span <= 0.0:
            return 0.0
        return (value - anchor.ignore_value) / span
    # higher value = more ignore-like
    if value <= anchor.supersede_hi:
        return 1.0
    if value >= anchor.ignore_value:
        return 0.0
    span = anchor.ignore_value - anchor.supersede_hi
    if span <= 0.0:
        return 0.0
    return (anchor.ignore_value - value) / span


def _classify_from_score(mean_score: float) -> str:
    if mean_score >= _SUPERSEDE_THRESHOLD:
        return REGIME_SUPERSEDE
    if mean_score <= _IGNORE_THRESHOLD:
        return REGIME_IGNORE
    return REGIME_MIXED


def _confidence_from_score(mean_score: float) -> float:
    """Distance from the 0.5 boundary, normalized to [0, 1].

    A score of 1.0 (full supersede) -> confidence 1.0.
    A score of 0.5 (boundary) -> confidence 0.0.
    A score of 0.0 (full ignore) -> confidence 1.0.
    """
    return min(1.0, max(0.0, abs(mean_score - 0.5) * 2.0))


def classify_regime(features: HealthFeatures) -> HealthReport:
    """Score the features and produce a HealthReport.

    Below MIN_BELIEFS the regime is `insufficient_data` regardless of
    features (per E43). Empty stores fall through the same path.
    """
    if features.n_beliefs < MIN_BELIEFS:
        return HealthReport(
            regime=REGIME_INSUFFICIENT_DATA,
            classification_confidence=0.0,
            mean_score=0.0,
            per_feature_scores={},
            features=features,
        )
    per_feature_scores: dict[str, float] = {}
    for name, anchor in _FEATURE_ANCHORS.items():
        value = getattr(features, name)
        per_feature_scores[name] = _score_feature(value, anchor)
    mean_score = sum(per_feature_scores.values()) / len(per_feature_scores)
    return HealthReport(
        regime=_classify_from_score(mean_score),
        classification_confidence=_confidence_from_score(mean_score),
        mean_score=mean_score,
        per_feature_scores=per_feature_scores,
        features=features,
    )


def assess_health(store: MemoryStore) -> HealthReport:
    """Single-call convenience: compute features then classify.

    Pure function over the current store state. Deterministic for any
    snapshot.
    """
    return classify_regime(compute_features(store))


# --- User-facing rendering -----------------------------------------------

_SUPERSEDE_DESCRIPTION: Final[str] = (
    "Active mode — beliefs accumulate evidence; contradictions trigger "
    "wholesale supersession rather than gradual Bayesian adjustment. "
    "Active mode is the typical operating pattern for projects in "
    "regular development."
)

_IGNORE_DESCRIPTION: Final[str] = (
    "Skeptical mode — most feedback is rejected as low-quality. "
    "Beliefs stay near prior. The graph carries structure but few "
    "beliefs reach high confidence. Skeptical mode is a legitimate "
    "working pattern for projects that maintain skepticism toward "
    "incoming signals; whether this matches your project's intent is "
    "for you to judge."
)

_MIXED_DESCRIPTION: Final[str] = (
    "Mixed mode — features sit near the supersede/ignore boundary. "
    "The brain shows characteristics of both modes; the dominant "
    "pattern may emerge as more usage accumulates."
)

_INSUFFICIENT_DATA_DESCRIPTION: Final[str] = (
    "Insufficient data — fewer than the minimum belief count needed "
    "for stable classification. Onboard more sources or accumulate "
    "more feedback events; the regime descriptor will resolve once "
    "the brain crosses the threshold."
)


def regime_description(regime: str) -> str:
    """Return a one-paragraph description of the regime for display.

    Output framing follows the round-9 regime-as-feature decision: no
    alarm copy, no defective-mode language, both supersede and ignore
    modes presented as legitimate working patterns.
    """
    if regime == REGIME_SUPERSEDE:
        return _SUPERSEDE_DESCRIPTION
    if regime == REGIME_IGNORE:
        return _IGNORE_DESCRIPTION
    if regime == REGIME_MIXED:
        return _MIXED_DESCRIPTION
    return _INSUFFICIENT_DATA_DESCRIPTION
