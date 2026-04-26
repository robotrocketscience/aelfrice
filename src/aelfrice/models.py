"""Domain dataclasses for aelfrice v0.1.0.

Per [redacted]: load-bearing fields only. 4 belief types, 5 edge types,
2 lock levels. No multi-source tagging, no rigor tier, no bitemporal
event_time -- those are deferred to v1.x.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- Belief types (4, per [redacted]) ---
BELIEF_FACTUAL: Final[str] = "factual"
BELIEF_CORRECTION: Final[str] = "correction"
BELIEF_PREFERENCE: Final[str] = "preference"
BELIEF_REQUIREMENT: Final[str] = "requirement"

BELIEF_TYPES: Final[frozenset[str]] = frozenset({
    BELIEF_FACTUAL,
    BELIEF_CORRECTION,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
})

# --- Edge types (5, per [redacted]) ---
EDGE_SUPPORTS: Final[str] = "SUPPORTS"
EDGE_CITES: Final[str] = "CITES"
EDGE_CONTRADICTS: Final[str] = "CONTRADICTS"
EDGE_SUPERSEDES: Final[str] = "SUPERSEDES"
EDGE_RELATES_TO: Final[str] = "RELATES_TO"

# Edge-type valence multipliers for propagation.
# Positive = propagate same sign; negative = invert; 0.0 = no propagation.
EDGE_VALENCE: Final[dict[str, float]] = {
    EDGE_SUPPORTS: 1.0,
    EDGE_CITES: 0.5,
    EDGE_CONTRADICTS: -0.5,
    EDGE_SUPERSEDES: 0.0,
    EDGE_RELATES_TO: 0.3,
}

EDGE_TYPES: Final[frozenset[str]] = frozenset(EDGE_VALENCE.keys())

# --- Lock levels (2, per [redacted]; promoted/freeze deferred to v1.3) ---
LOCK_NONE: Final[str] = "none"
LOCK_USER: Final[str] = "user"

LOCK_LEVELS: Final[frozenset[str]] = frozenset({LOCK_NONE, LOCK_USER})


@dataclass
class Belief:
    """A unit of memory with Bayesian confidence and lock state.

    Fields per [redacted]: (id, content, content_hash, alpha, beta, type,
    lock_level, locked_at, demotion_pressure, created_at, last_retrieved_at).
    """

    id: str
    content: str
    content_hash: str
    alpha: float
    beta: float
    type: str
    lock_level: str
    locked_at: str | None
    demotion_pressure: int
    created_at: str
    last_retrieved_at: str | None


@dataclass
class Edge:
    """A typed, weighted directed link between two beliefs."""

    src: str
    dst: str
    type: str
    weight: float
