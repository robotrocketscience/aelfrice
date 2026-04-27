"""Feedback endpoint: the single Bayesian-update path at runtime.

`apply_feedback(store, belief_id, valence, source)` is the only function
that mutates a belief's alpha/beta after onboarding. It also writes an
audit row to `feedback_history` for every successful update so the
project's feedback regime is recoverable after the fact.

Demotion-pressure propagation and auto-demote at threshold land in
follow-up commits; this module ships the core update + audit path
first so the contract is stable before the propagation layer hooks in.

`DEMOTION_THRESHOLD` is exposed here as a module-level configurable
constant (default 5, per round-6 E22). Callers may rebind it for tests
or domain-specific tuning. The auto-demote path that consumes it lands
next.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from aelfrice.models import Belief
from aelfrice.store import Store

DEMOTION_THRESHOLD: int = 5

POSITIVE: Final[str] = "positive"
NEGATIVE: Final[str] = "negative"


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
    store: Store,
    belief_id: str,
    valence: float,
    source: str,
    now: str | None = None,
) -> FeedbackResult:
    """Apply one feedback event to one belief.

    1. Resolve the belief; raise ValueError if missing.
    2. Reject zero valence: a no-update event is not a successful update,
       and pre-commit #5 says feedback_history records every successful
       update — so a zero call has no row to write.
    3. Bayesian-update alpha or beta by valence sign.
    4. Persist the new posterior on the belief row.
    5. Append one row to feedback_history (created_at = `now` or UTC now).
    6. Return a FeedbackResult with prior + new posteriors and the row id.
    """
    if valence == 0.0:
        raise ValueError("valence must be nonzero")
    if not source:
        raise ValueError("source must be a non-empty string")

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

    return FeedbackResult(
        belief_id=belief_id,
        event_id=event_id,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        new_alpha=new_alpha,
        new_beta=new_beta,
        valence=valence,
        source=source,
    )
