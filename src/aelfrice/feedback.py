"""Feedback endpoint: the single Bayesian-update path at runtime.

`apply_feedback(store, belief_id, valence, source)` is the only function
that mutates a belief's alpha/beta after onboarding. It also writes an
audit row to `feedback_history` for every successful update so the
project's feedback regime is recoverable after the fact.

Valence propagation (#1058): after a direct update, the signal walks
outbound edges via `MemoryStore.propagate_valence` (broker-confidence
attenuation) and each attenuated delta is applied back through this
same function — so propagated updates keep the one-row-per-α-write
audit invariant. Propagated applications never re-propagate (the BFS
already did the multi-hop walk). Kill switch:
`AELFRICE_VALENCE_PROPAGATION=0`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from aelfrice.models import Belief
from aelfrice.store import MemoryStore

POSITIVE: Final[str] = "positive"
NEGATIVE: Final[str] = "negative"

# Emergency off-switch for valence propagation, matching the
# AELFRICE_BM25F=0 / AELFRICE_ENTITY_INDEX=0 convention: on by
# default, "0" disables. Read per call so long-lived processes
# honor a flip without restart.
ENV_VALENCE_PROPAGATION: Final[str] = "AELFRICE_VALENCE_PROPAGATION"

# Prefix for feedback_history.source on propagated events; the direct
# event's source is appended so provenance chains stay readable
# (e.g. "propagation:user").
PROPAGATION_SOURCE_PREFIX: Final[str] = "propagation:"


def _propagation_enabled() -> bool:
    return os.environ.get(ENV_VALENCE_PROPAGATION, "1") != "0"


@dataclass
class FeedbackResult:
    """What apply_feedback did, returned to the caller for introspection."""

    belief_id: str
    event_id: int
    prior_alpha: float
    prior_beta: float
    new_alpha: float
    new_beta: float
    valence: float
    source: str
    # Downstream updates applied by valence propagation (#1058); empty
    # when propagation is disabled, suppressed, or found no recipients.
    propagated: list["FeedbackResult"] = field(
        default_factory=list["FeedbackResult"],
    )


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp suffixed Z. Stable across hosts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bayesian_update(b: Belief, valence: float) -> tuple[float, float]:
    """Beta-Bernoulli update: positive valence -> alpha; negative -> beta.

    Magnitude is the increment amount; ±1.0 is conventional but fractional
    valences are honored for weighted feedback sources (e.g., propagated
    signals attenuated by broker confidence).
    """
    if valence > 0.0:
        return (b.alpha + valence, b.beta)
    return (b.alpha, b.beta + (-valence))


def apply_feedback(
    store: MemoryStore,
    belief_id: str,
    valence: float,
    source: str,
    now: str | None = None,
    propagate: bool = True,
) -> FeedbackResult:
    """Apply one feedback event to one belief.

    1. Resolve the belief; raise ValueError if missing.
    2. Reject zero valence: a no-update event is not a successful update,
       and pre-commit #5 says feedback_history records every successful
       update — so a zero call has no row to write.
    3. Bayesian-update alpha or beta by valence sign.
    4. Persist the new posterior on the belief row.
    5. Append one row to feedback_history (created_at = `now` or UTC now).
    6. Propagate the signal through outbound edges (#1058): each
       attenuated delta from `store.propagate_valence` is applied via a
       recursive call with `propagate=False`, so downstream beliefs get
       their own posterior update AND feedback_history row, and the
       walk happens exactly once. Skipped when `propagate` is False,
       when AELFRICE_VALENCE_PROPAGATION=0, or when the store raises
       for a recipient (foreign federated / concurrently-deleted
       beliefs are expected mid-walk and must not fail the direct
       event).
    7. Return a FeedbackResult with prior + new posteriors, the row id,
       and any propagated results.
    """
    if valence == 0.0:
        raise ValueError("valence must be nonzero")
    if not source:
        raise ValueError("source must be a non-empty string")

    # #655 read-only federation: reject mutations on foreign belief ids
    # at the API surface. Raised as ForeignBeliefError (a ValueError
    # subclass) so existing `except ValueError` blocks in CLI / MCP
    # surfaces continue to flag the call without special handling.
    store.assert_local_ownership(belief_id)

    b: Belief | None = store.get_belief(belief_id)
    if b is None:
        raise ValueError(f"belief not found: {belief_id}")

    prior_alpha: float = b.alpha
    prior_beta: float = b.beta
    new_alpha, new_beta = _bayesian_update(b, valence)

    b.alpha = new_alpha
    b.beta = new_beta
    store.update_belief(b)

    timestamp: str = now if now is not None else _utc_now_iso()
    event_id: int = store.insert_feedback_event(
        belief_id=belief_id,
        valence=valence,
        source=source,
        created_at=timestamp,
    )

    result = FeedbackResult(
        belief_id=belief_id,
        event_id=event_id,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        new_alpha=new_alpha,
        new_beta=new_beta,
        valence=valence,
        source=source,
    )

    if propagate and _propagation_enabled():
        deltas = store.propagate_valence(belief_id, valence)
        # Sorted for a deterministic feedback_history row order
        # regardless of edge-iteration order inside the BFS.
        for dst_id, delta in sorted(deltas.items()):
            try:
                downstream = apply_feedback(
                    store,
                    dst_id,
                    delta,
                    source=f"{PROPAGATION_SOURCE_PREFIX}{source}",
                    now=timestamp,
                    propagate=False,
                )
            except ValueError:
                # ForeignBeliefError (read-only federation) or a belief
                # deleted between the walk and the apply. Fail-soft:
                # the direct event already committed; a skipped
                # recipient just doesn't receive the echo.
                continue
            result.propagated.append(downstream)

    return result
