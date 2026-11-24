"""Feedback rows that let a fixture belief qualify on the core posterior arm.

The posterior arm of `aelf core` counts posterior-moving feedback events
(`CORE_MIN_POSTERIOR_FEEDBACK_EVENTS`), not alpha+beta alone, since ingest
type priors such as Beta(3, 1) clear the mean and mass thresholds with no
evidence. Fixtures that seed a belief at, say, Beta(9, 1) to put it in core
call `record_core_evidence` so the store also holds the feedback that such
a posterior stands for.

The rows go in through `MemoryStore.insert_feedback_event`, so the seeded
alpha and beta stay exactly as the fixture wrote them.
"""
from __future__ import annotations

from typing import Final

from aelfrice.models import CORE_MIN_POSTERIOR_FEEDBACK_EVENTS, LOCK_NONE
from aelfrice.sentiment_feedback import SENTIMENT_INFERRED_SOURCE
from aelfrice.store import MemoryStore

_EVIDENCE_TS: Final[str] = "2026-08-01T09:00:00+00:00"


def record_core_evidence(
    store: MemoryStore,
    belief_id: str,
    n: int = CORE_MIN_POSTERIOR_FEEDBACK_EVENTS,
) -> None:
    """Append `n` positive `sentiment_inferred` feedback rows for one belief."""
    assert n >= 0, "n is a count of rows"
    for _ in range(n):
        store.insert_feedback_event(
            belief_id, 1.0, SENTIMENT_INFERRED_SOURCE, _EVIDENCE_TS,
        )


def record_core_evidence_for_store(store: MemoryStore) -> int:
    """Give every unlocked belief seeded above alpha 1 its core evidence.

    For fixtures that insert beliefs at several sites; a belief whose mean
    or mass misses the posterior arm still misses it. Returns the number of
    beliefs given rows.
    """
    given = 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None or b.lock_level != LOCK_NONE or b.alpha <= 1.0:
            continue
        record_core_evidence(store, bid)
        given += 1
    return given
