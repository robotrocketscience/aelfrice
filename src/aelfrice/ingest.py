"""Ingest pipeline: split a conversation turn into sentences,
classify each, and insert classified sentences as beliefs.

The signature is compatible with the lab adapters in `benchmarks/`:

    ingest_turn(store, text, source, session_id, created_at, source_id)

`session_id` is persisted on every belief inserted under the call
(v1.2+). `source_id` is still accepted for adapter parity but
remains unpersisted pending its own schema slot.
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
    session_id: str | None = None,
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

    When `session_id` is provided it is written to `beliefs.session_id`
    on every newly inserted row (v1.2+). Calls without a session leave
    the column NULL — downstream session-coherent retrieval skips
    NULL rows, no false positives on legacy data.

    `source_id` is still accepted for adapter parity but is not yet
    persisted.

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
            session_id=session_id,
        )
        store.insert_belief(belief)
        inserted += 1
    return inserted
