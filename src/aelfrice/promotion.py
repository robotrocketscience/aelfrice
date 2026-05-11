"""Origin-tier promotion for v1.2.0.

Implements the `agent_inferred -> user_validated` promotion path
designed in `docs/promotion_path.md`. The CLI surface is
`aelf validate <belief_id>`; the MCP surface is `aelf:validate`.

Provenance change only — no math change. The user clicking validate
does not move alpha/beta. Justification: validation is a UI act,
not a math act. See § 3 of the design memo.

Reversibility lives in `cli._cmd_demote`: when given a
`lock_level=none, origin=user_validated` belief, demote flips origin
back to `agent_inferred` and writes a `promotion:revert_to_agent_inferred`
audit row.

`unlock()` clears a user-lock without touching origin. It is the
pure inverse of `aelf lock`. Writes a `lock:unlock` audit row.
Idempotent — re-unlocking an already-unlocked belief is a no-op.

Audit row shape (every successful promote, devalidate, and unlock):
  - belief_id = subject's id
  - valence   = 0.0  (replay-safe; ignored by feedback math)
  - source    = 'promotion:user_validated' or
                'promotion:revert_to_agent_inferred' or
                'lock:unlock'
  - created_at = ISO-8601 UTC, Z-suffixed
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_VALIDATED,
)
from aelfrice.store import MemoryStore

SOURCE_PROMOTE_USER_VALIDATED: Final[str] = "promotion:user_validated"
SOURCE_REVERT_TO_AGENT_INFERRED: Final[str] = (
    "promotion:revert_to_agent_inferred"
)
SOURCE_LOCK_UNLOCK: Final[str] = "lock:unlock"


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of one `promote` call.

    `audit_event_id` is None on the idempotent already-validated path
    (no row is written). `prior_origin` lets the caller see the
    pre-promotion tier without re-reading the store.
    """
    belief_id: str
    prior_origin: str
    new_origin: str
    audit_event_id: int | None
    already_validated: bool


@dataclass(frozen=True)
class UnlockResult:
    """Outcome of one `unlock` call.

    `audit_event_id` is None when `already_unlocked=True`
    (idempotent path — no row is written).
    """
    belief_id: str
    already_unlocked: bool
    audit_event_id: int | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def promote(
    store: MemoryStore,
    belief_id: str,
    *,
    source_label: str = SOURCE_PROMOTE_USER_VALIDATED,
    now: str | None = None,
) -> PromotionResult:
    """Promote `belief_id` to origin=user_validated.

    Provenance flip only — alpha/beta/type/lock_level are unchanged.
    Idempotent: a second call on an already-validated belief is a
    no-op that returns `already_validated=True` and writes no audit
    row.

    Refusal cases (raise ValueError):
      - belief not found
      - belief is locked (lock is a strictly stronger tier; demote
        first via `aelf demote` to drop the lock, then validate)
      - belief is origin=user_stated without lock (data
        inconsistency; same handling as locked)

    The `source_label` argument exists so a UI client can tag its
    promotion distinctly while keeping the `promotion:` prefix —
    pass `"promotion:mcp_validate"` for example.
    """
    # #655: reject foreign belief ids at the mutation surface.
    store.assert_local_ownership(belief_id)
    belief = store.get_belief(belief_id)
    if belief is None:
        raise ValueError(f"belief not found: {belief_id}")
    if belief.lock_level == LOCK_USER:
        raise ValueError(
            f"cannot validate locked belief: {belief_id} "
            "(locks already exceed user_validated; use 'aelf demote' "
            "to drop the lock first)"
        )
    if belief.origin == ORIGIN_USER_STATED:
        raise ValueError(
            f"cannot validate user_stated belief: {belief_id} "
            "(origin already at higher tier; use 'aelf demote' to "
            "drop the lock first)"
        )
    if belief.origin == ORIGIN_USER_VALIDATED:
        return PromotionResult(
            belief_id=belief_id,
            prior_origin=ORIGIN_USER_VALIDATED,
            new_origin=ORIGIN_USER_VALIDATED,
            audit_event_id=None,
            already_validated=True,
        )
    prior = belief.origin
    belief.origin = ORIGIN_USER_VALIDATED
    store.update_belief(belief)
    timestamp = now if now is not None else _utc_now_iso()
    audit_id = store.insert_feedback_event(
        belief_id=belief_id,
        valence=0.0,
        source=source_label,
        created_at=timestamp,
    )
    return PromotionResult(
        belief_id=belief_id,
        prior_origin=prior,
        new_origin=ORIGIN_USER_VALIDATED,
        audit_event_id=audit_id,
        already_validated=False,
    )


def devalidate(
    store: MemoryStore,
    belief_id: str,
    *,
    source_label: str = SOURCE_REVERT_TO_AGENT_INFERRED,
    now: str | None = None,
) -> PromotionResult:
    """Reverse `promote`. Flips origin=user_validated -> agent_inferred.

    Called by `aelf demote` when given a `lock_level=none,
    origin=user_validated` belief. Writes a
    `promotion:revert_to_agent_inferred` audit row so the round-trip
    is visible in the log.

    Returns `already_validated=False` on success. Raises ValueError
    if the belief is missing, locked, or not currently
    `user_validated`.
    """
    # #655: reject foreign belief ids at the mutation surface.
    store.assert_local_ownership(belief_id)
    belief = store.get_belief(belief_id)
    if belief is None:
        raise ValueError(f"belief not found: {belief_id}")
    if belief.lock_level == LOCK_USER:
        raise ValueError(
            f"cannot devalidate locked belief: {belief_id} "
            "(drop the lock first with 'aelf demote')"
        )
    if belief.origin != ORIGIN_USER_VALIDATED:
        raise ValueError(
            f"belief is not user_validated: {belief_id} "
            f"(origin={belief.origin})"
        )
    prior = belief.origin
    belief.origin = ORIGIN_AGENT_INFERRED
    store.update_belief(belief)
    timestamp = now if now is not None else _utc_now_iso()
    audit_id = store.insert_feedback_event(
        belief_id=belief_id,
        valence=0.0,
        source=source_label,
        created_at=timestamp,
    )
    return PromotionResult(
        belief_id=belief_id,
        prior_origin=prior,
        new_origin=ORIGIN_AGENT_INFERRED,
        audit_event_id=audit_id,
        already_validated=False,
    )


def unlock(
    store: MemoryStore,
    belief_id: str,
    *,
    source_label: str = SOURCE_LOCK_UNLOCK,
    now: str | None = None,
) -> UnlockResult:
    """Clear the user-lock on a belief. Pure inverse of `aelf lock`.

    Does not touch origin — the belief's origin tier is unchanged.
    Clears `lock_level`, `locked_at`, and `demotion_pressure`.

    Idempotent: unlocking an already-unlocked belief is a no-op
    (returns `already_unlocked=True`, writes no audit row).

    Refusal cases (raise ValueError):
      - belief not found

    Writes a `lock:unlock` audit row on the active (non-idempotent)
    path so the lock-management history is complete.
    """
    # #655: reject foreign belief ids at the mutation surface.
    store.assert_local_ownership(belief_id)
    belief = store.get_belief(belief_id)
    if belief is None:
        raise ValueError(f"belief not found: {belief_id}")
    if belief.lock_level != LOCK_USER:
        return UnlockResult(
            belief_id=belief_id,
            already_unlocked=True,
            audit_event_id=None,
        )
    belief.lock_level = LOCK_NONE
    belief.locked_at = None
    belief.demotion_pressure = 0
    store.update_belief(belief)
    timestamp = now if now is not None else _utc_now_iso()
    audit_id = store.insert_feedback_event(
        belief_id=belief_id,
        valence=0.0,
        source=source_label,
        created_at=timestamp,
    )
    return UnlockResult(
        belief_id=belief_id,
        already_unlocked=False,
        audit_event_id=audit_id,
    )
