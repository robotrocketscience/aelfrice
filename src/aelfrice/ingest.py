"""Ingest pipeline: split a conversation turn into sentences,
classify each, and insert classified sentences as beliefs.

The signature is compatible with the lab adapters in `benchmarks/`:

    ingest_turn(store, text, source, session_id, created_at, source_id)

`session_id` is persisted on every belief inserted under the call
(v1.2+). `source_id` is still accepted for adapter parity but
remains unpersisted pending its own schema slot.

`ingest_jsonl` (v1.2+) reads a turns.jsonl file produced by the
transcript-logger hooks and ingests each line; consecutive turns
within a session get DERIVED_FROM edges so the conversation
structure is recoverable downstream by the v1.4.0 context rebuilder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aelfrice.classification import classify_sentence
from aelfrice.extraction import extract_sentences
from aelfrice.models import (
    ANCHOR_TEXT_MAX_LEN,
    EDGE_DERIVED_FROM,
    LOCK_NONE,
    Belief,
    Edge,
)
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
    return len(_ingest_turn_ids(
        store=store, text=text, source=source,
        session_id=session_id, created_at=created_at,
    ))


def _ingest_turn_ids(
    store: MemoryStore,
    text: str,
    source: str,
    session_id: str | None = None,
    created_at: str | None = None,
) -> list[str]:
    """Internal variant of ingest_turn returning the inserted belief ids.

    `ingest_jsonl` uses the id list to wire DERIVED_FROM edges
    between turns within a session. Public callers should use
    `ingest_turn` (which discards the list and returns the count).
    """
    sentences = extract_sentences(text)
    if not sentences:
        return []

    ts = created_at or _now_utc_iso()
    inserted: list[str] = []
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
        inserted.append(belief_id)
    return inserted


@dataclass(frozen=True)
class IngestJsonlResult:
    """Aggregate counts from one ingest_jsonl run."""

    lines_read: int
    turns_ingested: int
    beliefs_inserted: int
    edges_inserted: int
    skipped_lines: int


def ingest_jsonl(
    store: MemoryStore,
    jsonl_path: Path | str,
    *,
    source_label: str = "transcript",
) -> IngestJsonlResult:
    """Ingest a turns.jsonl produced by the transcript-logger hooks.

    For each `{"role": "user"|"assistant", "text": ..., "session_id":
    ..., "turn_id": ...}` line the function calls `ingest_turn` with
    `session_id` propagated through. Within a session, consecutive
    turns are linked with DERIVED_FROM edges from the most recently
    inserted belief of turn N+1 back to the most recently inserted
    belief of turn N, with `anchor_text` set to the prior turn's
    text (truncated to ANCHOR_TEXT_MAX_LEN).

    Idempotency: ingest_turn dedupes per (source_label, sentence), so
    re-running on the same file produces zero new beliefs. The edge
    insert path is wrapped in a duplicate-PK guard for the same
    reason.

    Lines without role/text (compaction markers, malformed) are
    counted under `skipped_lines` and ignored without raising.
    """
    path = Path(jsonl_path)
    lines_read = 0
    turns_ingested = 0
    beliefs_inserted = 0
    edges_inserted = 0
    skipped = 0
    last_per_session: dict[str, tuple[str, str]] = {}
    # session_id -> (last_belief_id_inserted, last_turn_text)

    if not path.is_file():
        return IngestJsonlResult(0, 0, 0, 0, 0)

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            lines_read += 1
            line = raw.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            role = obj.get("role")
            text = obj.get("text")
            if not isinstance(role, str) or not isinstance(text, str) or not text:
                # compaction markers (event=...) and malformed lines
                skipped += 1
                continue
            sess = obj.get("session_id")
            sess_str = sess if isinstance(sess, str) and sess else None
            ts = obj.get("ts")
            created_at = ts if isinstance(ts, str) else None
            ids = _ingest_turn_ids(
                store=store, text=text, source=source_label,
                session_id=sess_str, created_at=created_at,
            )
            turns_ingested += 1
            beliefs_inserted += len(ids)
            if not ids or sess_str is None:
                continue
            head_id = ids[-1]
            prior = last_per_session.get(sess_str)
            if prior is not None:
                prior_id, prior_text = prior
                anchor = prior_text[:ANCHOR_TEXT_MAX_LEN]
                edge = Edge(
                    src=head_id, dst=prior_id, type=EDGE_DERIVED_FROM,
                    weight=1.0, anchor_text=anchor,
                )
                if store.get_edge(edge.src, edge.dst, edge.type) is None:
                    store.insert_edge(edge)
                    edges_inserted += 1
            last_per_session[sess_str] = (head_id, text)

    return IngestJsonlResult(
        lines_read=lines_read,
        turns_ingested=turns_ingested,
        beliefs_inserted=beliefs_inserted,
        edges_inserted=edges_inserted,
        skipped_lines=skipped,
    )
