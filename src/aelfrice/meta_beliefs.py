"""Adaptive meta-belief substrate (#755, umbrella #480).

A meta-belief is a Bayesian-tracked retrieval-side parameter (e.g.
``meta:retrieval.temporal_half_life_seconds``) that lives in a
**separate** SQL table from regular beliefs. The separation prevents
meta-beliefs from leaking into BFS, edge composition, or any other
brain-graph path (umbrella #480 §1 ratification, locked
``c06f8d575fad71fb`` PHILOSOPHY narrow-surface).

This module ships only the substrate:

1. The per-signal-class Beta-Bernoulli update math.
2. The independent exponential decay engine that pulls each
   sub-posterior back toward a per-meta-belief ``static_default``
   at its own ``half_life_seconds`` cadence (umbrella #480 §2).
3. The weighted-combination read path that surfaces a single
   ``value`` per meta-belief from its per-signal-class sub-posteriors
   (umbrella #480 §3 "type-aware update" — precision and latency are
   not commensurable, so each signal class carries its own
   sub-posterior and the surfaced value is a weighted blend).

No retrieval-side consumption ships here. That lives in #480 sub-tasks
B–F. This module's contract is observable state + decay + update; the
retrieval code that *reads* meta-beliefs is downstream of this PR.

Determinism contract
--------------------

Same ``(initial state, evidence sequence, observation timestamps)``
deterministically yields the same final state. Decay is a function of
``Δt = now - last_updated_ts`` — wall-clock independent (locked
``c06f8d575fad71fb``); ``now`` is supplied by the caller so tests can
pin it.

Signal-class enum (umbrella #480 §3)
------------------------------------

Four signal classes are admitted today. Each meta-belief subscribes to
a subset at install time (its ``signal_classes`` row); ``update_meta_belief``
calls for a class not in the subscription list are recorded as no-op
audit events rather than silently dropped.

- ``SIGNAL_RELEVANCE`` — close-the-loop relevance signal (#365). Did
  the consumer reference / contradict / confirm the surfaced beliefs?
- ``SIGNAL_LATENCY`` — retrieval wall-time per query.
- ``SIGNAL_BFS_DEPTH`` — observed BFS depth distribution.
- ``SIGNAL_BM25_L0_RATIO`` — BM25F-vs-L0 hit-ratio per query.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Signal-class enum
# ---------------------------------------------------------------------------

SIGNAL_RELEVANCE: Final[str] = "relevance"
SIGNAL_LATENCY: Final[str] = "latency"
SIGNAL_BFS_DEPTH: Final[str] = "bfs_depth"
SIGNAL_BM25_L0_RATIO: Final[str] = "bm25_l0_ratio"

SIGNAL_CLASSES: Final[frozenset[str]] = frozenset(
    {SIGNAL_RELEVANCE, SIGNAL_LATENCY, SIGNAL_BFS_DEPTH, SIGNAL_BM25_L0_RATIO}
)


def is_valid_signal_class(name: str) -> bool:
    """Return True iff ``name`` is a known signal-class identifier."""
    return name in SIGNAL_CLASSES


# ---------------------------------------------------------------------------
# Prior derived from static_default
# ---------------------------------------------------------------------------

# Total mass of the "pull-toward" prior in alpha+beta units. Chosen so
# the decay target carries about as much weight as a single observation
# — strong enough to dominate stale tails, weak enough that one fresh
# evidence point moves the posterior visibly.
PRIOR_MASS: Final[float] = 1.0


def prior_alpha_beta(static_default: float) -> tuple[float, float]:
    """Decay target prior (alpha0, beta0) for a meta-belief whose pull is
    toward ``static_default``.

    Holds ``alpha0/(alpha0+beta0) == static_default`` and
    ``alpha0 + beta0 == PRIOR_MASS``. Clamps ``static_default`` to
    ``(eps, 1-eps)`` so the prior is always strictly inside the
    posterior simplex.
    """
    eps = 1e-9
    mu = max(eps, min(1.0 - eps, static_default))
    return (PRIOR_MASS * mu, PRIOR_MASS * (1.0 - mu))


# ---------------------------------------------------------------------------
# Update math
# ---------------------------------------------------------------------------


def apply_evidence(
    alpha: float, beta: float, evidence: float
) -> tuple[float, float]:
    """Beta-Bernoulli posterior update for continuous ``evidence`` in [0, 1].

    The standard Bernoulli update treats one observation of "success" as
    ``alpha += 1`` and one observation of "failure" as ``beta += 1``.
    For a continuous score ``e in [0, 1]`` we add ``e`` to alpha and
    ``1 - e`` to beta — the same expected-value update with fractional
    weights. Clamps ``evidence`` to ``[0, 1]`` first.
    """
    e = max(0.0, min(1.0, evidence))
    return (alpha + e, beta + 1.0 - e)


def decay_toward_default(
    alpha: float,
    beta: float,
    age_seconds: float,
    half_life_seconds: float,
    static_default: float,
) -> tuple[float, float]:
    """Exponential pull of ``(alpha, beta)`` toward the static-default prior.

    For factor ``f = 2^(-age/half_life)``::

        alpha_new = alpha0 + (alpha - alpha0) * f
        beta_new  = beta0  + (beta  - beta0)  * f

    where ``(alpha0, beta0) = prior_alpha_beta(static_default)``.
    Zero-evidence series converges to ``(alpha0, beta0)`` as
    ``age >> half_life``. Symmetric to ``aelfrice.scoring.decay`` but
    targets ``static_default`` rather than the Jeffreys (0.5, 0.5)
    prior.

    Pass-through when ``age <= 0`` or ``half_life <= 0`` so callers
    can apply a same-tick read without distortion.
    """
    if age_seconds <= 0.0 or half_life_seconds <= 0.0:
        return (alpha, beta)
    factor = math.pow(2.0, -age_seconds / half_life_seconds)
    a0, b0 = prior_alpha_beta(static_default)
    return (a0 + (alpha - a0) * factor, b0 + (beta - b0) * factor)


def posterior_mean(alpha: float, beta: float) -> float:
    """Beta-Bernoulli posterior mean alpha/(alpha+beta).

    Degenerate ``alpha + beta <= 0`` falls back to 0.5; should not
    occur in practice (sub-posteriors start at the static-default
    prior whose mass is strictly positive).
    """
    total = alpha + beta
    if total <= 0.0:
        return 0.5
    return alpha / total


# ---------------------------------------------------------------------------
# In-memory state objects (round-tripped via the store layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalPosterior:
    """Per-signal-class Beta-Bernoulli sub-posterior."""
    signal_class: str
    alpha: float
    beta: float
    last_updated_ts: int


@dataclass(frozen=True)
class MetaBeliefState:
    """Materialised meta-belief — config plus current sub-posteriors.

    Surfaced ``value`` is the weight-normalised blend of each
    sub-posterior's mean; weights come from the ``signal_classes``
    configuration (defaults to equal weight per subscribed class).
    """
    key: str
    static_default: float
    half_life_seconds: int
    last_updated_ts: int
    signal_weights: dict[str, float]
    posteriors: dict[str, SignalPosterior] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.surfaced_value(self.posteriors, self.signal_weights,
                                   static_default=self.static_default)

    @staticmethod
    def surfaced_value(
        posteriors: dict[str, SignalPosterior],
        weights: dict[str, float],
        *,
        static_default: float,
    ) -> float:
        """Weight-normalised blend over subscribed signal-class means.

        Falls back to ``static_default`` when no subscribed posterior
        has accumulated above-prior mass — handles the cold-start case
        without surfacing a meaningless 0.5.
        """
        num = 0.0
        denom = 0.0
        for cls, weight in weights.items():
            p = posteriors.get(cls)
            if p is None or weight <= 0.0:
                continue
            num += weight * posterior_mean(p.alpha, p.beta)
            denom += weight
        if denom <= 0.0:
            return static_default
        return num / denom


def parse_signal_weights(signal_classes_json: str) -> dict[str, float]:
    """Decode the ``signal_classes`` SQL column.

    Accepts either a bare JSON array of class names (equal weights) or
    a JSON array of ``{"class": <name>, "weight": <float>}`` objects.
    Unknown signal classes are dropped with no error — the row may
    have been written by a future version that knows more classes.
    """
    raw = json.loads(signal_classes_json) if signal_classes_json else []
    weights: dict[str, float] = {}
    for entry in raw:
        if isinstance(entry, str):
            if entry in SIGNAL_CLASSES:
                weights[entry] = 1.0
        elif isinstance(entry, dict):
            cls = entry.get("class")
            if isinstance(cls, str) and cls in SIGNAL_CLASSES:
                w = entry.get("weight", 1.0)
                if isinstance(w, (int, float)) and w > 0:
                    weights[cls] = float(w)
    return weights


def encode_signal_weights(weights: dict[str, float]) -> str:
    """Inverse of ``parse_signal_weights`` — stable JSON for round-trip.

    Output keys sorted by class name so two stores with the same
    subscriptions produce byte-identical column values (determinism
    property — same as the canvas exporter's sort discipline).
    """
    rows = [{"class": cls, "weight": weights[cls]}
            for cls in sorted(weights.keys())]
    return json.dumps(rows, separators=(",", ":"))


def decay_state(state: MetaBeliefState, *, now_ts: int) -> MetaBeliefState:
    """Return a new ``MetaBeliefState`` with every sub-posterior pulled
    toward its static-default prior by ``decay_toward_default``.

    Read-side helper: callers that want a fresh decayed snapshot pass
    the materialised state from ``MemoryStore.read_meta_belief_state``
    plus the current timestamp; the store's persisted rows are not
    rewritten unless the caller explicitly persists the result.
    """
    decayed: dict[str, SignalPosterior] = {}
    for cls, p in state.posteriors.items():
        age = max(0, now_ts - p.last_updated_ts)
        a, b = decay_toward_default(
            p.alpha, p.beta,
            age_seconds=float(age),
            half_life_seconds=float(state.half_life_seconds),
            static_default=state.static_default,
        )
        decayed[cls] = SignalPosterior(
            signal_class=cls,
            alpha=a,
            beta=b,
            last_updated_ts=now_ts,
        )
    return MetaBeliefState(
        key=state.key,
        static_default=state.static_default,
        half_life_seconds=state.half_life_seconds,
        last_updated_ts=now_ts,
        signal_weights=state.signal_weights,
        posteriors=decayed,
    )
