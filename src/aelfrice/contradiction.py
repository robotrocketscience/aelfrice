"""Contradiction tie-breaker.

When the graph holds a `CONTRADICTS` edge between two beliefs, the
tie-breaker picks one as the winner via a deterministic precedence
rule and supersedes the loser. The result is a normal `SUPERSEDES`
edge from winner → loser, plus an audit row in `feedback_history`
recording which rule fired.

## Precedence (v1.2+, five classes)

1. **`user_stated`** — `lock_level == LOCK_USER` (short-circuit) or
   `origin == 'user_stated'`. The user explicitly asserted this
   belief as ground truth via `aelf lock` or `aelf:lock`. Highest.
2. **`user_corrected`** — `type == BELIEF_CORRECTION` (legacy path)
   or `origin == 'user_corrected'`. Explicit correction signal.
3. **`user_validated`** — `origin == 'user_validated'`. The user
   acknowledged an onboard belief without locking it.
4. **`document_recent`** — `origin in {'document_recent', 'unknown',
   'agent_remembered'}`. Within-class breaks by recency.
5. **`agent_inferred`** — `origin == 'agent_inferred'`. Onboard
   scanner output that has not been validated. Lowest.

When two beliefs have the same precedence class, **more recent
`created_at` wins**. When created_at also matches (rare; collision
implies identical-second insertion), **higher `id` wins** as a
deterministic tie-breaker.

## v1.0/v1.1 absorption

v1.0 stores opening on v1.2 are migrated by store.py: locked rows
become `user_stated`, correction rows become `user_corrected`,
everything else stays `unknown`. The `unknown` origin is treated as
the same precedence class as `document_recent`, preserving v1.0.1
tie-breaker behaviour for un-tagged content.

## When this fires

The function `resolve_contradiction(store, a, b)` runs on demand;
the v1.0.1 integration is a CLI command (`aelf resolve`) and direct
library use. v1.0 has no automatic write path that creates
CONTRADICTS edges, so there is no existing trigger to integrate with.
Hooks into `scan_repo`, retrieval, and the relationship-detector
land in v1.x once the corresponding write paths exist.

## Audit

Every successful resolution writes one row to `feedback_history`:

- `belief_id` = loser's id (the belief being demoted)
- `valence` = `0.0` (bookkeeping; not a real feedback signal — this
  is intentional and replay logic should ignore zero-valence rows)
- `source` = `f"contradiction_tiebreaker:{rule}"` where `rule`
  encodes the winner's class, e.g.
  `"contradiction_tiebreaker:user_stated_beats_document_recent"`.
- `created_at` = now (or supplied)

Reading the audit log answers "which rule fired and against what?"
without needing to recompute precedence at query time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from aelfrice.models import (
    BELIEF_CORRECTION,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_VALIDATED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

# Precedence classes. Higher value wins. Names are stable wire-format
# strings — they appear in feedback_history.source — so do not rename
# without a migration.
PRECEDENCE_USER_STATED: Final[int] = 5
PRECEDENCE_USER_CORRECTED: Final[int] = 4
PRECEDENCE_USER_VALIDATED: Final[int] = 3
PRECEDENCE_DOCUMENT_RECENT: Final[int] = 2
PRECEDENCE_AGENT_INFERRED: Final[int] = 1

CLASS_NAMES: Final[dict[int, str]] = {
    PRECEDENCE_USER_STATED: "user_stated",
    PRECEDENCE_USER_CORRECTED: "user_corrected",
    PRECEDENCE_USER_VALIDATED: "user_validated",
    PRECEDENCE_DOCUMENT_RECENT: "document_recent",
    PRECEDENCE_AGENT_INFERRED: "agent_inferred",
}

# Audit-row source prefix. Anything written to feedback_history by the
# tie-breaker has this prefix; the suffix is the rule that fired.
SOURCE_PREFIX: Final[str] = "contradiction_tiebreaker"

# Edge weight for SUPERSEDES edges created by the tie-breaker. Per
# `models.EDGE_VALENCE`, SUPERSEDES has valence 0.0 (structural, no
# propagation), so the weight is informational only — kept at 1.0 by
# convention to match other structural-edge insert paths.
SUPERSEDES_WEIGHT: Final[float] = 1.0


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of one tie-breaker call.

    `rule_fired` is the wire-format string written to the audit row;
    callers can read it without re-computing precedence. Format:
    `<winner_class>_beats_<loser_class>` — e.g.
    `user_stated_beats_document_recent`,
    `user_stated_beats_user_stated_by_recency`,
    `document_recent_beats_document_recent_by_id` for the
    final-tiebreak case.

    `supersedes_created` is True when the function inserted a new
    SUPERSEDES edge; False when an equivalent edge already existed
    (idempotent re-resolve). The audit row is written regardless.

    `audit_event_id` is the rowid of the feedback_history row.
    """
    winner_id: str
    loser_id: str
    rule_fired: str
    supersedes_created: bool
    audit_event_id: int


def precedence_class(belief: Belief) -> int:
    """Return the precedence class for `belief`.

    v1.2+ five-class precedence (highest first): user_stated,
    user_corrected, user_validated, document_recent, agent_inferred.

    Resolution order:
      1. lock_level=user short-circuits to user_stated regardless of
         the origin field. Preserves the v1.0.1 invariant that locks
         always win the contradiction.
      2. Origin string maps directly to its class for the explicit
         origins set by v1.2+ producers.
      3. Legacy fallback: type=correction maps to user_corrected
         (covers correction beliefs whose origin is still 'unknown'
         pre-backfill on a v1.0/v1.1 store).
      4. Anything else (origin in {document_recent, unknown,
         agent_remembered}) maps to document_recent. Honest 'don't
         know' bucket; preserves v1.0.1 absorption.
    """
    if belief.lock_level == LOCK_USER:
        return PRECEDENCE_USER_STATED
    if belief.origin == ORIGIN_USER_STATED:
        return PRECEDENCE_USER_STATED
    if belief.origin == ORIGIN_USER_CORRECTED:
        return PRECEDENCE_USER_CORRECTED
    if belief.origin == ORIGIN_USER_VALIDATED:
        return PRECEDENCE_USER_VALIDATED
    if belief.origin == ORIGIN_AGENT_INFERRED:
        return PRECEDENCE_AGENT_INFERRED
    if belief.type == BELIEF_CORRECTION:
        return PRECEDENCE_USER_CORRECTED
    return PRECEDENCE_DOCUMENT_RECENT


def precedence_class_name(belief: Belief) -> str:
    """Return the human-readable class name for `belief`."""
    return CLASS_NAMES[precedence_class(belief)]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_winner(a: Belief, b: Belief) -> tuple[Belief, Belief, str]:
    """Return (winner, loser, rule_fired_suffix).

    Class wins first; if tied, more recent `created_at` wins; if also
    tied, higher `id` wins (deterministic). The rule_fired string
    encodes which axis decided.
    """
    a_cls = precedence_class(a)
    b_cls = precedence_class(b)
    if a_cls != b_cls:
        if a_cls > b_cls:
            return (
                a, b,
                f"{CLASS_NAMES[a_cls]}_beats_{CLASS_NAMES[b_cls]}",
            )
        return (
            b, a,
            f"{CLASS_NAMES[b_cls]}_beats_{CLASS_NAMES[a_cls]}",
        )

    # Same class — break by recency.
    if a.created_at != b.created_at:
        if a.created_at > b.created_at:
            return (
                a, b,
                f"{CLASS_NAMES[a_cls]}_beats_"
                f"{CLASS_NAMES[a_cls]}_by_recency",
            )
        return (
            b, a,
            f"{CLASS_NAMES[a_cls]}_beats_"
            f"{CLASS_NAMES[a_cls]}_by_recency",
        )

    # Same class, same timestamp — break by id (alphabetical).
    if a.id > b.id:
        return (
            a, b,
            f"{CLASS_NAMES[a_cls]}_beats_"
            f"{CLASS_NAMES[a_cls]}_by_id",
        )
    return (
        b, a,
        f"{CLASS_NAMES[a_cls]}_beats_"
        f"{CLASS_NAMES[a_cls]}_by_id",
    )


def resolve_contradiction(
    store: MemoryStore,
    a_id: str,
    b_id: str,
    *,
    now: str | None = None,
) -> ResolutionResult:
    """Resolve a contradiction between two beliefs.

    Picks a winner via the precedence rules (see module docstring),
    inserts a SUPERSEDES edge from winner → loser if not already
    present, and writes one audit row to feedback_history tagged
    `source=f"{SOURCE_PREFIX}:{rule_fired}"`.

    Raises ValueError if either belief is missing. Idempotent: a
    second call with the same args inserts no new SUPERSEDES edge
    but still writes a fresh audit row (so re-resolves leave a
    visible trace).
    """
    a = store.get_belief(a_id)
    if a is None:
        raise ValueError(f"belief not found: {a_id}")
    b = store.get_belief(b_id)
    if b is None:
        raise ValueError(f"belief not found: {b_id}")

    winner, loser, rule = _pick_winner(a, b)

    existing = store.get_edge(
        winner.id, loser.id, EDGE_SUPERSEDES,
    )
    if existing is None:
        store.insert_edge(Edge(
            src=winner.id,
            dst=loser.id,
            type=EDGE_SUPERSEDES,
            weight=SUPERSEDES_WEIGHT,
        ))
        supersedes_created = True
    else:
        supersedes_created = False

    timestamp = now if now is not None else _utc_now_iso()
    audit_id = store.insert_feedback_event(
        belief_id=loser.id,
        valence=0.0,  # bookkeeping; replay logic ignores zero-valence
        source=f"{SOURCE_PREFIX}:{rule}",
        created_at=timestamp,
    )

    return ResolutionResult(
        winner_id=winner.id,
        loser_id=loser.id,
        rule_fired=rule,
        supersedes_created=supersedes_created,
        audit_event_id=audit_id,
    )


def find_unresolved_contradictions(
    store: MemoryStore,
) -> list[tuple[str, str]]:
    """Find CONTRADICTS edges that lack a matching SUPERSEDES edge.

    Returns a deduplicated list of (a_id, b_id) pairs ordered for
    determinism (lexicographic on the smaller id, then larger).
    A CONTRADICTS edge in either direction (A→B or B→A) is treated
    as the same logical pair. A pair is "resolved" when a SUPERSEDES
    edge exists between the two beliefs in either direction.
    """
    pairs: set[tuple[str, str]] = set()
    for b in store.list_beliefs() if hasattr(store, "list_beliefs") else []:
        # Defensive: list_beliefs is not in the v1.0 store. Iterate via SQL.
        _ = b  # never reached; placeholder for future API
    # Fall through to direct SQL since list_beliefs is not exposed
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst FROM edges WHERE type = ?",
        (EDGE_CONTRADICTS,),
    ).fetchall()
    for row in rows:
        a, b = row["src"], row["dst"]
        # Canonicalise pair (lower id first) so A→B and B→A dedupe.
        if a < b:
            pairs.add((a, b))
        else:
            pairs.add((b, a))
    out: list[tuple[str, str]] = []
    for a, b in sorted(pairs):
        if (
            store.get_edge(a, b, EDGE_SUPERSEDES) is not None
            or store.get_edge(b, a, EDGE_SUPERSEDES) is not None
        ):
            continue
        out.append((a, b))
    return out


def auto_resolve_all_contradictions(
    store: MemoryStore,
    *,
    now: str | None = None,
) -> list[ResolutionResult]:
    """Resolve every unresolved CONTRADICTS edge in the store.

    Returns the list of ResolutionResult objects in the order they
    were resolved. Already-resolved pairs (those with an existing
    SUPERSEDES edge between them) are skipped — no audit row written.
    Pairs whose endpoint beliefs no longer exist are silently
    skipped.
    """
    out: list[ResolutionResult] = []
    for a_id, b_id in find_unresolved_contradictions(store):
        try:
            out.append(resolve_contradiction(store, a_id, b_id, now=now))
        except ValueError:
            # Either endpoint missing — skip, do not abort batch.
            continue
    return out
