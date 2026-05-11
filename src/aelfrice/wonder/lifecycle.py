"""Wonder lifecycle: ingest and GC for speculative phantom beliefs (#548).

Two entry points:

* ``wonder_ingest`` — persists in-memory ``Phantom`` candidates to the
  store as ``type='speculative'`` beliefs with ``origin=ORIGIN_SPECULATIVE``,
  Bayesian prior α=0.3 / β=1.0, ``RELATES_TO`` edges to each constituent
  belief, and a ``wonder_ingest`` corroboration row for audit.

* ``wonder_gc`` — soft-deletes stale speculative beliefs by setting
  ``valid_to`` on candidates that have received no feedback and whose
  priors are still at the ingest defaults.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from aelfrice.models import (
    BELIEF_SPECULATIVE,
    CORROBORATION_SOURCE_WONDER_INGEST,
    EDGE_RELATES_TO,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    RETENTION_SNAPSHOT,
    Belief,
    Edge,
    Phantom,
)
from aelfrice.ulid import ulid

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Default Bayesian priors for freshly ingested speculative beliefs.
# Calibrated conservatively: α=0.3 gives a weak positive prior;
# β=1.0 reflects genuine uncertainty. GC uses these as the "unchanged"
# threshold — any α-update above the epsilon band means a feedback event
# has touched the belief, so it survives.
_INGEST_ALPHA: float = 0.3
_INGEST_BETA: float = 1.0


_CONSTITUENT_KEY_VERSION: str = "v2"


def _constituent_key(
    constituent_belief_ids: tuple[str, ...],
    generator: str,
) -> str:
    """SHA-256 of the sorted constituent IDs + generator — the idempotency key.

    Keyed on the sorted constituent tuple *and* the generator so two
    phantoms produced from the same constituent set under different
    generators (e.g. an ``--axes`` dispatch run that returns N axis
    documents over the same anchor set) persist as N distinct rows
    rather than collapsing to one. Phantoms from *different* constituent
    sets with *identical* text remain distinct — content hash is not
    the dedup axis here.

    Key format prefix is ``wonder_ingest:v2:`` (v3.0 #644). The v1
    layout was generator-agnostic; existing speculative rows are
    rehashed on first open by
    ``MemoryStore._maybe_rehash_speculative_v2`` using the generator
    stored in the wonder_ingest corroboration row.
    """
    raw = (
        f"wonder_ingest:{_CONSTITUENT_KEY_VERSION}:"
        + ":".join(sorted(constituent_belief_ids))
        + "|" + generator
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WonderIngestResult:
    """Summary returned by ``wonder_ingest``."""

    inserted: int
    skipped: int
    edges_created: int


@dataclass(frozen=True)
class WonderGCResult:
    """Summary returned by ``wonder_gc``."""

    scanned: int
    deleted: int
    surviving: int


def wonder_ingest(
    store: "MemoryStore",
    phantoms: list[Phantom],
    session_id: str | None = None,
) -> WonderIngestResult:
    """Persist speculative phantom candidates to the store.

    For each ``Phantom``:

    1. Derive a deterministic ``content_hash`` from the sorted
       ``constituent_belief_ids`` **and** the ``generator`` (v3.0 #644);
       if a belief with that hash already exists, skip insertion.
       Idempotent on the (constituent-set, generator) pair: re-running
       the same dispatch is a no-op; running a *different* generator
       over the same constituents produces a distinct phantom.
    2. Insert a ``Belief`` with ``type='speculative'``,
       ``origin=ORIGIN_SPECULATIVE``, α=0.3, β=1.0.
    3. Insert ``RELATES_TO`` edges from the new belief to every
       constituent.
    4. Record a ``wonder_ingest`` corroboration row; ``source_path_hash``
       encodes ``"<generator>@<score:.4f>"`` for audit.

    Returns ``WonderIngestResult(inserted, skipped, edges_created)``.
    """
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    skipped = 0
    edges_created = 0

    for phantom in phantoms:
        key = _constituent_key(
            phantom.constituent_belief_ids, phantom.generator
        )
        existing = store.get_belief_by_content_hash(key)
        if existing is not None:
            skipped += 1
            continue

        belief_id = ulid()
        belief = Belief(
            id=belief_id,
            content=phantom.content,
            content_hash=key,
            alpha=_INGEST_ALPHA,
            beta=_INGEST_BETA,
            type=BELIEF_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=now,
            last_retrieved_at=None,
            session_id=session_id,
            origin=ORIGIN_SPECULATIVE,
            retention_class=RETENTION_SNAPSHOT,
        )
        store.insert_belief(belief)

        for constituent_id in phantom.constituent_belief_ids:
            store.insert_edge(Edge(
                src=belief_id,
                dst=constituent_id,
                type=EDGE_RELATES_TO,
                weight=1.0,
            ))
            edges_created += 1

        audit_meta = f"{phantom.generator}@{phantom.score:.4f}"
        store.record_corroboration(
            belief_id,
            source_type=CORROBORATION_SOURCE_WONDER_INGEST,
            session_id=session_id,
            source_path_hash=audit_meta,
        )

        inserted += 1

    return WonderIngestResult(
        inserted=inserted,
        skipped=skipped,
        edges_created=edges_created,
    )


def wonder_gc(
    store: "MemoryStore",
    ttl_days: int = 14,
    dry_run: bool = False,
) -> WonderGCResult:
    """Soft-delete stale speculative beliefs that have received no feedback.

    Candidates must satisfy ALL of:
    - ``type = 'speculative'`` and ``origin = ORIGIN_SPECULATIVE``
    - ``valid_to IS NULL`` (still active)
    - ``created_at`` older than ``ttl_days`` days ago
    - α ≤ 0.3 + ε and β ≤ 1.0 + ε (priors unchanged from ingest defaults)
    - no ``feedback_history`` rows (``apply_feedback`` never called)
    - no ``RESOLVES`` edges (incoming or outgoing)

    If ``dry_run`` is True, reports candidates without mutating the store.
    The second run in non-dry-run mode finds zero new candidates
    (idempotent because ``soft_delete_belief`` guards on ``valid_to IS NULL``).

    Returns ``WonderGCResult(scanned, deleted, surviving)``.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    cutoff_ts = cutoff.isoformat()

    candidate_ids = store.query_wonder_gc_candidates(cutoff_ts=cutoff_ts)
    scanned = len(candidate_ids)

    if dry_run:
        return WonderGCResult(scanned=scanned, deleted=0, surviving=scanned)

    now = datetime.now(timezone.utc).isoformat()
    for belief_id in candidate_ids:
        store.soft_delete_belief(belief_id, ts=now)

    return WonderGCResult(
        scanned=scanned,
        deleted=scanned,
        surviving=0,
    )


__all__ = [
    "WonderGCResult",
    "WonderIngestResult",
    "wonder_gc",
    "wonder_ingest",
]
