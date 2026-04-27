"""Ingest pipeline: split a conversation turn into sentences,
classify each, and insert classified sentences as beliefs.

This module is a deliberate **shim**, not a literal port of lab
v2.0.0's ingest pipeline. The public v1.0.0 schema does not model
observations, sessions, evidence, or audit logs as separate
tables; lab v2.0.0 does. Rather than invert v1.0.0's deliberate
slim-down, ingest_turn here lowers conversation turns directly to
the existing `beliefs` table via the public classification API
(`classify_sentence`, one sentence at a time) instead of lab's
`classify_sentences_offline` batch path.

The signature is kept compatible with the lab adapters in
`benchmarks/` so they import successfully without modification:

    ingest_turn(store, text, source, session_id, created_at, source_id)

`session_id` and `source_id` are accepted but not persisted at
v1.0.x. They reappear as belief metadata when (or if) the schema
gains source-tracking columns.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from aelfrice.classification import classify_sentence
from aelfrice.extraction import extract_sentences
from aelfrice.models import LOCK_NONE, Belief
from aelfrice.store import MemoryStore

_BELIEF_ID_HEX_LEN: int = 16


def _belief_id(text: str, source: str) -> str:
    """Stable id derived from (source, text). Matches the scheme used
    by classification._derive_belief_id and scanner._derive_belief_id
    so re-ingesting an identical (text, source) pair is idempotent."""
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest_turn(
    store: MemoryStore,
    text: str,
    source: str,
    session_id: str | None = None,  # noqa: ARG001
    created_at: str | None = None,
    source_id: str = "",  # noqa: ARG001
) -> int:
    """Ingest a single conversation turn.

    Steps:
      1. Sentence-split via :func:`aelfrice.extraction.extract_sentences`.
      2. Classify each sentence via :func:`aelfrice.classification.classify_sentence`.
      3. Insert classifications with `persist=True` as beliefs.

    Idempotent on (source, sentence) pairs: re-ingesting the same turn
    triggers `INSERT OR IGNORE` semantics in the store (belief id
    derives from the sha256 of source + sentence). Sentences whose
    classification returns `persist=False` (questions, empty text) are
    skipped.

    `session_id` and `source_id` are accepted for adapter parity with
    lab v2.0.0 but are not stored at v1.0.x.

    Returns the number of beliefs inserted (or that would have been
    inserted if not already present).
    """
    sentences = extract_sentences(text)
    if not sentences:
        return 0

    ts = created_at or _now_utc_iso()
    inserted = 0
    for sentence in sentences:
        result = classify_sentence(sentence, source)
        if not result.persist:
            continue
        belief_id = _belief_id(sentence, source)
        if store.get_belief(belief_id) is not None:
            continue  # idempotent: same (source, sentence) already ingested
        belief = Belief(
            id=belief_id,
            content=sentence,
            content_hash=_content_hash(sentence),
            alpha=result.alpha,
            beta=result.beta,
            type=result.belief_type,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=ts,
            last_retrieved_at=None,
        )
        store.insert_belief(belief)
        inserted += 1
    return inserted
