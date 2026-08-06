"""Feedback endpoint: the primary Bayesian-update path at runtime.

`apply_feedback(store, belief_id, valence, source)` is the runtime
endpoint for moving a belief's alpha/beta. It writes an audit row to
`feedback_history` for every event — including events whose posterior
move was deliberately suppressed — so the project's feedback regime is
recoverable after the fact. The posterior write and that audit row share
one `BEGIN IMMEDIATE` transaction, and the write itself is an atomic
`SET alpha = alpha + ?` (`store.bump_posterior`), so concurrent hook
processes cannot lose each other's evidence (#1168).

A user lock (`lock_level == LOCK_USER`) is a floor: passive feedback
records the event and leaves the posterior alone. Correcting a lock is an
explicit act — `aelf unlock` first.

Three other paths write alpha directly and do NOT come through here
(#1168 AC4). This module is the *primary* writer, not the only one:

  * `deferred_feedback.sweep_deferred_feedback` — the #191/#256 implicit
    lane. Deliberately separate: it applies a much smaller epsilon after
    a grace window that any explicit correction cancels, and it owns its
    own per-row `BEGIN IMMEDIATE` plus queue-status bookkeeping. It now
    honours the same lock floor and federation-ownership check as this
    module, and writes its own `feedback_history` row.
  * `clamp_ghosts.clamp_ghost_alpha` — a one-shot migration clamp over
    pre-migration rows with no feedback and no corroboration history.
    Asserts `lock_level='none'` in its own WHERE clause and writes a
    reversing `feedback_history` row; see that module's docstring.
  * the consolidation dedup pass in `store` — sums alpha/beta across a
    duplicate group when collapsing it, which is a merge of existing
    evidence rather than new evidence, so there is nothing to audit.

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

from aelfrice.models import LOCK_USER, Belief
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
    # False when the event was audited but the posterior deliberately not
    # moved — either `update_posterior=False` (#1086 exposure lane) or the
    # #1168 lock floor. `new_alpha`/`new_beta` equal the priors in that case.
    posterior_applied: bool = True
    # True only for the lock-floor case, so callers can warn the user that
    # their feedback was recorded but did not move a lock they own.
    skipped_locked: bool = False


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp suffixed Z. Stable across hosts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bayesian_delta(valence: float) -> tuple[float, float]:
    """Beta-Bernoulli increment: positive valence -> alpha; negative -> beta.

    Returns the `(d_alpha, d_beta)` to add, not the resulting values —
    the addition itself happens inside SQLite (`store.bump_posterior`)
    so concurrent writers cannot lose each other's evidence (#1168).

    Magnitude is the increment amount; ±1.0 is conventional but fractional
    valences are honored for weighted feedback sources (e.g., propagated
    signals attenuated by broker confidence).
    """
    if valence > 0.0:
        return (valence, 0.0)
    return (0.0, -valence)


def _bayesian_update(b: Belief, valence: float) -> tuple[float, float]:
    """Absolute-value form of `_bayesian_delta`, applied to a snapshot.

    Retained for callers that need the projected posterior without
    writing it (previews, tests). Not used on the write path — see
    `_bayesian_delta`.
    """
    d_alpha, d_beta = _bayesian_delta(valence)
    return (b.alpha + d_alpha, b.beta + d_beta)


def apply_feedback(
    store: MemoryStore,
    belief_id: str,
    valence: float,
    source: str,
    now: str | None = None,
    propagate: bool = True,
    update_posterior: bool = True,
    respect_lock: bool = True,
) -> FeedbackResult:
    """Apply one feedback event to one belief.

    1. Resolve the belief; raise ValueError if missing.
    2. Reject zero valence: a no-update event is not a successful update,
       and pre-commit #5 says feedback_history records every successful
       update — so a zero call has no row to write.
    3. Bayesian-update alpha or beta by valence sign — UNLESS
       `update_posterior` is False (audit-only; #1086: retrieval exposure
       records an event for the recurrence axis without being treated as
       truth-evidence) or the belief carries a user lock and
       `respect_lock` is True (#1168: a lock is a floor, and the docs
       promise passive feedback cannot move it — pass
       `respect_lock=False` from an explicit-affirmation surface such as
       `aelf confirm`). Either way `posterior_applied` is False on the
       result and the priors are echoed back as the new values.
    4. Persist the increment with an atomic `SET alpha = alpha + ?`, and
    5. append one row to feedback_history (created_at = `now` or UTC now)
       — both inside one `BEGIN IMMEDIATE` transaction, so the log and
       the projection cannot disagree and concurrent writers serialise
       instead of overwriting each other (#1168).
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
    # subclass) so existing `except ValueError` blocks in the CLI
    # surfaces continue to flag the call without special handling.
    store.assert_local_ownership(belief_id)

    b: Belief | None = store.get_belief(belief_id)
    if b is None:
        raise ValueError(f"belief not found: {belief_id}")

    # Lock floor (#1168). A user lock is ground truth the user asserted
    # explicitly; docs/user/LIMITATIONS.md and docs/user/PRIVACY.md both
    # state that passive feedback does not move one. The floor lived only
    # in `scoring.decay()`, so `aelf feedback <locked-id> harmful` — and,
    # far worse, every sentiment-derived turn valence, since L0 locks are
    # injected on every prompt and so sit in every turn's pending set —
    # moved locked posteriors anyway. Correcting a lock stays an explicit
    # act (`aelf unlock`, then feedback); the event is still audited.
    #
    # `respect_lock=False` is the opt-out for the one surface the docs
    # define as explicit user affirmation rather than passive feedback:
    # `aelf confirm` / `aelf_confirm`, which docs/user/COMMANDS.md calls
    # out as "distinct from ... implicit retrieval feedback". Every other
    # caller — CLI `aelf feedback`, sentiment,
    # retrieval exposure, valence propagation — takes the floor.
    locked: bool = respect_lock and b.lock_level == LOCK_USER
    posterior_applied: bool = update_posterior and not locked

    prior_alpha: float = b.alpha
    prior_beta: float = b.beta
    timestamp: str = now if now is not None else _utc_now_iso()

    # One transaction for the posterior write and its audit row (#1168):
    # they commit together or not at all, so the append-only log can never
    # claim evidence the projection never took (or vice versa). BEGIN
    # IMMEDIATE takes the write lock up front rather than on first write.
    with store.transaction(immediate=True):
        if posterior_applied:
            d_alpha, d_beta = _bayesian_delta(valence)
            # Atomic `SET alpha = alpha + ?` rather than a whole-row write
            # of the snapshot read above: the read-modify-write in Python
            # lost 180 of 240 concurrent events and could revert a lock
            # committed by another process between the read and the write.
            bumped = store.bump_posterior(belief_id, d_alpha, d_beta)
            if bumped is None:
                raise ValueError(f"belief not found: {belief_id}")
            new_alpha, new_beta = bumped
            # The snapshot's alpha may already be stale under concurrency;
            # the authoritative prior is what the atomic write landed on.
            prior_alpha = new_alpha - d_alpha
            prior_beta = new_beta - d_beta
        else:
            # Audit-only: record the event so exposure frequency stays
            # recoverable (the recurrence axis, #1086) and so a suppressed
            # move on a locked belief is visible, but do NOT move the
            # posterior. A retrieval is exposure, not endorsement.
            new_alpha, new_beta = prior_alpha, prior_beta

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
        posterior_applied=posterior_applied,
        skipped_locked=locked and update_posterior,
    )

    if posterior_applied and propagate and _propagation_enabled():
        # #1168: gated on `posterior_applied`, not `update_posterior` — a
        # locked belief holds its posterior, and propagating from a move
        # that did not happen would leak the held signal into its
        # neighbours by another route.
        #
        # #1169: the first hop is attenuated by the source's confidence as
        # it was *before* this event. Letting propagate_valence read the
        # row back would fold this event's own increment into the strength
        # of its own propagation. In this branch `prior_alpha`/`prior_beta`
        # are the authoritative pre-event pair, recomputed from what the
        # atomic write actually landed on.
        prior_denom = prior_alpha + prior_beta
        deltas = store.propagate_valence(
            belief_id,
            valence,
            src_confidence=(
                (prior_alpha / prior_denom) if prior_denom > 0 else 0.0
            ),
        )
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
