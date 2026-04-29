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
from typing import cast

from aelfrice.classification import classify_sentence
from aelfrice.extraction import extract_sentences
from aelfrice.models import (
    ANCHOR_TEXT_MAX_LEN,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_DERIVED_FROM,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
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
    *,
    bulk: bool = False,
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

    `bulk` (v1.6+, keyword-only) signals the call is part of a
    multi-turn batch (e.g. a research-line `wonder_ingest` pipeline).
    When True, per-turn side effects that are redundant inside a batch
    are skipped: currently `store.record_corroboration` for duplicate
    sentences. The canonical `beliefs` row and the `ingest_log` write
    (the v2.0 source-of-truth ingest record) are always produced —
    `bulk=True` produces the same final belief state as `bulk=False`
    for the same input sequence; only the corroboration audit table
    differs. Reserved for future per-turn-cost reductions as new side
    effects land. Default-off preserves current behavior.

    Returns the number of beliefs inserted (or that would have been
    inserted if not already present).
    """
    return len(_ingest_turn_ids(
        store=store, text=text, source=source,
        session_id=session_id, created_at=created_at,
        bulk=bulk,
    ))


def _ingest_turn_ids(
    store: MemoryStore,
    text: str,
    source: str,
    session_id: str | None = None,
    created_at: str | None = None,
    *,
    bulk: bool = False,
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
        ch = _content_hash(sentence)
        if store.get_belief(belief_id) is not None:
            # Exact (source, sentence) duplicate: record a corroboration
            # so re-assertions are observable. Canonical row unchanged.
            # bulk=True skips this — corroboration noise is uninteresting
            # in batch ingest and is the only currently-redundant per-turn
            # side effect.
            if not bulk:
                store.record_corroboration(
                    belief_id,
                    source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                    session_id=session_id,
                )
            continue
        # v2.0 #205 parallel-write: log the classifier input before
        # materializing the belief. belief_id is deterministic on
        # (source, sentence) so derived_belief_ids is known up-front.
        store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=source,
            raw_text=sentence,
            derived_belief_ids=[belief_id],
            session_id=session_id,
            ts=ts,
        )
        belief = Belief(
            id=belief_id,
            content=sentence,
            content_hash=ch,
            alpha=result.alpha,
            beta=result.beta,
            type=result.belief_type,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=ts,
            last_retrieved_at=None,
            session_id=session_id,
            origin=ORIGIN_AGENT_INFERRED,
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


def _normalize_jsonl_turn(obj: dict[str, object]) -> dict[str, str | None] | None:
    """Normalize one JSONL line to `{text, session_id, ts}` or None.

    Two shapes are recognised; everything else (compaction markers,
    file-history snapshots, tool results) returns None and is counted
    as skipped by the caller. None of the v2-shape Claude Code
    sub-fields are required to be present -- only `text` matters.

    Always preserves the literal `text` field; ts/session_id flow
    into the belief row when present.
    """
    # Shape 1: aelfrice transcript-logger turns.jsonl
    role = obj.get("role")
    text = obj.get("text")
    if isinstance(role, str) and isinstance(text, str) and text:
        sess = obj.get("session_id")
        ts = obj.get("ts")
        return {
            "text": text,
            "session_id": sess if isinstance(sess, str) and sess else None,
            "ts": ts if isinstance(ts, str) else None,
        }
    # Shape 2: Claude Code internal session JSONL
    type_field = obj.get("type")
    if type_field not in ("user", "assistant"):
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    msg_typed = cast(dict[str, object], msg)
    content = msg_typed.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # v2 content array: [{"type":"text","text":"..."}, ...].
        # Concatenate every "text"-type chunk in order.
        parts: list[str] = []
        for chunk in cast(list[object], content):
            if not isinstance(chunk, dict):
                continue
            chunk_typed = cast(dict[str, object], chunk)
            if chunk_typed.get("type") != "text":
                continue
            t = chunk_typed.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
        text = "\n".join(parts)
    else:
        return None
    if not isinstance(text, str) or not text:
        return None
    sess = obj.get("sessionId")
    ts = obj.get("timestamp")
    return {
        "text": text,
        "session_id": sess if isinstance(sess, str) and sess else None,
        "ts": ts if isinstance(ts, str) else None,
    }


def ingest_jsonl(
    store: MemoryStore,
    jsonl_path: Path | str,
    *,
    source_label: str = "transcript",
) -> IngestJsonlResult:
    """Ingest a turns.jsonl produced by the transcript-logger hooks
    OR a Claude Code internal session JSONL from `~/.claude/projects/`.

    Auto-detects format on a per-line basis. Two shapes are handled:

      * transcript-logger turns.jsonl (aelfrice's own format from
        v1.2.0): `{"role": "user"|"assistant", "text": ...,
        "session_id": ..., "ts": ...}`.
      * Claude Code session JSONL (v1.x onwards): `{"type":
        "user"|"assistant", "message": {"role": ..., "content":
        ...}, "sessionId": ..., "timestamp": ..., "cwd": ...}`. The
        adapter pulls the text out of either a string `content` or
        the v2-shape `[{"type": "text", "text": ...}]` content
        array, falling back to the empty string when neither is
        usable (issue #115 — retroactive ingestion of historical
        Claude Code session logs).

    Within a session, consecutive turns are linked with DERIVED_FROM
    edges from turn N+1's last belief back to turn N's last belief,
    `anchor_text` set to the prior turn's text (truncated to
    ANCHOR_TEXT_MAX_LEN).

    Idempotency: ingest_turn dedupes per (source_label, sentence), so
    re-running on the same file produces zero new beliefs. Edge
    inserts are wrapped in a duplicate-PK guard for the same reason.

    Lines without role/text (compaction markers, file-history
    snapshots, tool-result entries, malformed) are counted under
    `skipped_lines` and ignored without raising.
    """
    from aelfrice.inedible import is_inedible

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
    if is_inedible(path):
        # Privacy opt-out — file is excluded from the brain graph by
        # name. No content read, zero side effects on the store.
        return IngestJsonlResult(0, 0, 0, 0, 0)

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            lines_read += 1
            line = raw.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)  # pyright: ignore[reportAny]
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            obj_typed = cast(dict[str, object], obj)
            normalized = _normalize_jsonl_turn(obj_typed)
            if normalized is None:
                # compaction markers, file-history snapshots, tool
                # results, malformed -- not user/assistant text.
                skipped += 1
                continue
            text = normalized["text"]
            sess_str = normalized["session_id"]
            created_at = normalized["ts"]
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


@dataclass
class IngestJsonlBatchResult:
    """Aggregate counts from one ingest_jsonl_dir run.

    `files_ingested` is the count of files that contributed at least
    one ingestable line (including no-ingest files when the dir was
    walked but all lines were skipped). `files_skipped_age` is the
    count of files filtered out by `--since`. Per-file counts add up
    to the same totals `ingest_jsonl` would have produced if called
    per file.
    """

    files_walked: int
    files_ingested: int
    files_skipped_age: int
    lines_read: int
    turns_ingested: int
    beliefs_inserted: int
    edges_inserted: int
    skipped_lines: int
    files_skipped_inedible: int = 0


def ingest_jsonl_dir(
    store: MemoryStore,
    directory: Path | str,
    *,
    since: datetime | None = None,
    source_label: str = "transcript",
    pattern: str = "**/*.jsonl",
) -> IngestJsonlBatchResult:
    """Walk `directory` and ingest every JSONL file via `ingest_jsonl`.

    `since` filters out files whose mtime is before the cutoff (file
    mtime is the cheapest robust proxy for "this session was last
    appended after X" — Claude Code session JSONLs are
    append-only). `pattern` is the glob pattern relative to
    `directory`; default is recursive `**/*.jsonl`.

    Aggregates per-file counts; idempotent on re-run thanks to
    `ingest_jsonl`'s per-line dedup. Skips files that do not exist
    or cannot be stat'd without raising. Issue #115.
    """
    from aelfrice.inedible import is_inedible

    root = Path(directory)
    if not root.is_dir():
        return IngestJsonlBatchResult(0, 0, 0, 0, 0, 0, 0, 0)
    cutoff_ts = since.timestamp() if since is not None else None
    walked = 0
    ingested = 0
    skipped_age = 0
    skipped_inedible = 0
    lines_total = 0
    turns_total = 0
    beliefs_total = 0
    edges_total = 0
    skipped_lines_total = 0
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        walked += 1
        if is_inedible(path):
            skipped_inedible += 1
            continue
        if cutoff_ts is not None:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff_ts:
                skipped_age += 1
                continue
        result = ingest_jsonl(store, path, source_label=source_label)
        if result.lines_read > 0:
            ingested += 1
        lines_total += result.lines_read
        turns_total += result.turns_ingested
        beliefs_total += result.beliefs_inserted
        edges_total += result.edges_inserted
        skipped_lines_total += result.skipped_lines
    return IngestJsonlBatchResult(
        files_walked=walked,
        files_ingested=ingested,
        files_skipped_age=skipped_age,
        files_skipped_inedible=skipped_inedible,
        lines_read=lines_total,
        turns_ingested=turns_total,
        beliefs_inserted=beliefs_total,
        edges_inserted=edges_total,
        skipped_lines=skipped_lines_total,
    )
