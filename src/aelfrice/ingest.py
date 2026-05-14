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

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from aelfrice.derivation_worker import run_worker
from aelfrice.extraction import extract_sentences
from aelfrice.noise_filter import is_transcript_noise
from aelfrice.session_resolution import resolve_session_id
from aelfrice.models import (
    ANCHOR_TEXT_MAX_LEN,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_DERIVED_FROM,
    INGEST_SOURCE_FILESYSTEM,
    Edge,
)
from aelfrice.store import MemoryStore

# #809 / #785 § 3: pattern-based subfloor-noise detector.
#
# `retrieval-corpus-bloat` R0/R2 (lab campaign, 2026-05-11) attributed
# 19% of short-reinforced beliefs in the alpha+beta >= 10 stratum to
# three specific prose-fragment classes that the sentence splitter
# emits as freestanding sentences but that carry no semantic claim:
# code-fence boundaries (` ```bash`, `` ``` ``), header stubs (lines
# ending with `:` — "Acceptance criteria:", "Pipeline composition, in
# order of evidence:"), and bullet stubs (`- run tests`, `* foo`).
#
# These patterns are filtered at the ingest boundary: a matched
# sentence does not become a freestanding belief row. When it sits
# between two full-length-belief sentences within the same turn, the
# clause attaches as `anchor_text` on an intra-turn DERIVED_FROM edge,
# preserving relational meaning without inflating the belief-row
# count. Unanchored matches (no surrounding full-length belief in the
# same turn) are silently dropped.
#
# Pattern-based rather than length-based per the operator-ratified
# scope for #809: a strict length floor (spec literal: 80 chars) drops
# legitimate short factual claims ("The config file lives at
# /etc/aelf.") alongside the noise, breaking the conservative ingest
# contract that the existing test suite encodes. Pattern-matching
# closes only the named noise classes; legit short claims survive.
#
# Acknowledged false-positive risk: "ends with `:`" can fire on real
# prose ("He said:", "The reasons are these:"). The lab campaign
# named this pattern explicitly; trade-off accepted at empirical
# scope. Re-measure if production data surfaces a non-trivial miss
# rate.
_SUBFLOOR_BULLET_PREFIX = re.compile(r"^[-*+]\s")


def _looks_like_subfloor_noise(sentence: str) -> bool:
    """True when `sentence` matches one of the named noise patterns
    from `retrieval-corpus-bloat` R2: code-fence boundary, header stub
    ending in `:`, or markdown bullet stub. See module docstring for
    rationale and false-positive scope."""
    stripped = sentence.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return True
    if stripped.endswith(":"):
        return True
    if _SUBFLOOR_BULLET_PREFIX.match(stripped):
        return True
    return False


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

    `bulk` (v1.6+, keyword-only) is now a no-op (kept for API parity).
    Under #264 the entry point no longer owns dedup-or-corroborate
    decisions — every raw sentence appends a row to `ingest_log` and
    the derivation worker handles canonical-belief insert vs
    corroborate via `insert_or_corroborate`. The pre-#264 bulk
    fast-path that suppressed `record_corroboration` on duplicates is
    gone; both `bulk=True` and `bulk=False` now produce identical
    audit-table state. The keyword-only signature is preserved so
    existing batch callers do not need to change.

    Returns the number of beliefs inserted (or that would have been
    inserted if not already present).

    When the caller omits `session_id`, the helper
    :func:`aelfrice.session_resolution.resolve_session_id` reads
    ``$AELF_SESSION_ID`` as the inference fallback (#192 Q1.a). If
    neither is set, the call proceeds with a NULL session_id and
    emits a one-shot stderr warn keyed on the entry-point name.
    """
    return len(_ingest_turn_ids(
        store=store, text=text, source=source,
        session_id=resolve_session_id(session_id, surface_name="ingest_turn"),
        created_at=created_at,
        bulk=bulk,
    ))


def _ingest_turn_ids(
    store: MemoryStore,
    text: str,
    source: str,
    session_id: str | None = None,
    created_at: str | None = None,
    *,
    bulk: bool = False,  # noqa: ARG001
) -> list[str]:
    """Internal variant of ingest_turn returning the derived belief ids.

    Under #264 the entry point's job collapses to: split the turn into
    sentences, append one log row per sentence (unstamped), then invoke
    the derivation worker once at end-of-batch. The worker handles
    `derive()` + `insert_or_corroborate` + log-stamping. The returned
    list is the per-sentence derived belief id (in input order, with
    duplicates dropped) — `ingest_jsonl` uses the last entry to wire
    DERIVED_FROM edges between consecutive turns within a session.

    #809 adds a pattern-based subfloor filter: sentences matching
    `_looks_like_subfloor_noise` (code-fence prefix, header ending in
    `:`, bullet stub) do not become belief rows. When a sub-floor
    sentence sits between two full-length-belief sentences in the
    same turn, it attaches as `anchor_text` on an intra-turn
    DERIVED_FROM edge between the surrounding beliefs; unanchored
    sub-floor sentences are silently dropped.
    """
    sentences = extract_sentences(text)
    sentences = [s for s in sentences if not is_transcript_noise(s)]
    if not sentences:
        return []

    # #809: partition sentences into full-length belief candidates and
    # sub-floor clauses pending demotion to edge anchor_text. The
    # `subfloor_between[i]` list holds sub-floor clauses (in original
    # order) that preceded `full_sentences[i]`; entries before the
    # first full sentence and after the last full sentence are
    # unanchored and silently dropped.
    full_sentences: list[str] = []
    subfloor_between: list[list[str]] = []
    pending_subfloor: list[str] = []
    for sentence in sentences:
        if _looks_like_subfloor_noise(sentence):
            pending_subfloor.append(sentence)
        else:
            full_sentences.append(sentence)
            subfloor_between.append(pending_subfloor)
            pending_subfloor = []
    # Anything left in pending_subfloor has no following full sentence —
    # unanchored, silently dropped.
    if not full_sentences:
        return []

    ts = created_at or _now_utc_iso()
    # Snapshot the canonical belief set so we can identify which derived
    # ids in this turn correspond to brand-new inserts (vs corroborations
    # of already-known beliefs). Preserves the pre-#264 public contract
    # that `ingest_turn` returns the count of newly-inserted beliefs.
    ids_before: set[str] = set(store.list_belief_ids())

    log_ids: list[str] = []
    for sentence in full_sentences:
        log_id = store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=source,
            raw_text=sentence,
            session_id=session_id,
            ts=ts,
            raw_meta={"call_site": CORROBORATION_SOURCE_TRANSCRIPT_INGEST},
        )
        log_ids.append(log_id)

    # Worker is idempotent and scans all unstamped rows; calling it once
    # at end-of-turn is the per-batch invocation pattern from the spec.
    run_worker(store)

    # Resolve each log_id to its canonical belief id once, in input
    # order. Used twice: (a) for the public return value (newly
    # inserted beliefs, deduped), (b) for the #809 intra-turn edge
    # wiring below (per-sentence belief id, position-preserving).
    log_belief_ids: list[str | None] = []
    inserted: list[str] = []
    seen: set[str] = set()
    for log_id in log_ids:
        entry = store.get_ingest_log_entry(log_id)
        bid: str | None = None
        if entry is not None:
            ids = entry.get("derived_belief_ids") or []
            if isinstance(ids, list) and ids:
                head = ids[0]
                if isinstance(head, str):
                    bid = head
        log_belief_ids.append(bid)
        if bid is not None and bid not in ids_before and bid not in seen:
            seen.add(bid)
            inserted.append(bid)

    # #809: wire intra-turn DERIVED_FROM edges between consecutive
    # full-length beliefs whose original-prose ordering was separated
    # by one or more sub-floor clauses. Edge direction matches the
    # inter-turn DERIVED_FROM convention in `ingest_jsonl` (src is the
    # later belief, dst is the earlier one — "this is derived from
    # that earlier one"). Anchor_text is the joined sub-floor clauses,
    # truncated to ANCHOR_TEXT_MAX_LEN.
    for i in range(1, len(log_belief_ids)):
        between = subfloor_between[i]
        if not between:
            continue
        prior_bid = log_belief_ids[i - 1]
        curr_bid = log_belief_ids[i]
        if prior_bid is None or curr_bid is None or prior_bid == curr_bid:
            continue
        anchor = " | ".join(between)[:ANCHOR_TEXT_MAX_LEN]
        if store.get_edge(curr_bid, prior_bid, EDGE_DERIVED_FROM) is not None:
            continue
        store.insert_edge(Edge(
            src=curr_bid, dst=prior_bid,
            type=EDGE_DERIVED_FROM, weight=1.0,
            anchor_text=anchor,
        ))

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
    """Normalize one JSONL line to `{role, text, session_id, ts}` or None.

    Two shapes are recognised; everything else (compaction markers,
    file-history snapshots, tool results) returns None and is counted
    as skipped by the caller. None of the v2-shape Claude Code
    sub-fields are required to be present -- only `text` matters.

    Always preserves the literal `text` field; role flows through so
    `ingest_jsonl` can apply the #785 speaker-attribution gate;
    ts/session_id flow into the belief row when present.
    """
    # Shape 1: aelfrice transcript-logger turns.jsonl
    role = obj.get("role")
    text = obj.get("text")
    if isinstance(role, str) and isinstance(text, str) and text:
        sess = obj.get("session_id")
        ts = obj.get("ts")
        return {
            "role": role,
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
        "role": cast(str, type_field),
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
            role = normalized["role"]
            text = normalized["text"]
            sess_str = normalized["session_id"]
            created_at = normalized["ts"]
            # #785 §1 speaker-attribution gate: assistant-role rows are
            # read for normalization (so the rebuilder's transcript view
            # stays whole) but excluded from belief creation. The
            # last_per_session pointer keeps tracking the prior USER
            # belief, so DERIVED_FROM edges between consecutive user
            # turns are built across any intervening assistant rows.
            # Pre-existing assistant rows in the store are not
            # back-purged here — see spec "Non-decisions" for the
            # separate stratum-aware cleanup campaign.
            if role == "assistant":
                skipped += 1
                continue
            ids = _ingest_turn_ids(
                store=store, text=cast(str, text), source=source_label,
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
            last_per_session[sess_str] = (head_id, cast(str, text))

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
