"""Sentence classification: assign one of the four belief types and a
source-adjusted Beta prior.

v1.0 ships the synchronous regex/keyword fallback only. This is the
single classifier used in onboarding, CI, and any environment without a
host LLM. The polymorphic onboard handshake — where a host LLM
classifies sentences in its own context and returns results — lands in
v0.6.0 alongside the MCP server, when there's actually a host present.

Aelfrice never imports the `anthropic` SDK at any point in v1.0
(pre-commit #7). Classification calls flow through the host's existing
LLM context via the polymorphic protocol, never via aelfrice's own
network calls.

Type priors (alpha, beta) are calibrated for user-sourced content. Non-
user sources (scanner extracts, document text, agent-inferred) get
deflated alpha so the feedback loop earns confidence rather than
inheriting it.

The four belief types correspond exactly to the v1.0 surface in
`models.py` — `factual`, `correction`, `preference`, `requirement`.
The richer agentmemory-v4 type catalog (DECISION, ASSUMPTION, ANALYSIS,
TODO etc.) is collapsed into `factual` here; v1.x can re-expand the
catalog if usage data justifies it.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aelfrice.classification_core import (
    ClassificationResult,
    TYPE_PRIORS,
    USER_SOURCE,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.models import (
    BELIEF_TYPES,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    INGEST_SOURCE_FILESYSTEM,
    ONBOARD_STATE_PENDING,
    OnboardSession,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Re-exports for backward compatibility — `aelfrice.classification` was
# the historical entry point for these symbols. New code should import
# from `aelfrice.classification_core` directly (see #499).
__all__ = [
    "ClassificationResult",
    "TYPE_PRIORS",
    "USER_SOURCE",
    "classify_sentence",
    "get_source_adjusted_prior",
    "HostClassification",
    "OnboardSentence",
    "StartOnboardResult",
    "AcceptOnboardResult",
    "OnboardCheckResult",
    "start_onboard_session",
    "accept_classifications",
    "check_onboard_candidates",
]


# --- Polymorphic onboard handshake (v0.6.0) -----------------------------
#
# The host LLM is asked to classify scanner candidates in its own context
# (where it already has the surrounding repo context loaded), then hands
# the typed results back. Classifications produced this way carry
# `pending_classification=False` — they were typed by an actual LLM, not
# by the regex fallback.
#
# Three persisted states transit through onboard_sessions:
#
#   start_onboard_session(repo_path)
#       -> session_id + list of OnboardSentence(index, text, source)
#       -> row written with state=PENDING, candidates_json populated
#
#   accept_classifications(session_id, [HostClassification, ...])
#       -> beliefs inserted with refined types
#       -> row updated to state=COMPLETED, completed_at populated
#
# A full circular import with scanner is avoided by importing extractors
# lazily inside `start_onboard_session`; classification.py is imported by
# scanner.py at module-load, and the reverse import only happens when an
# onboard session is actually started.

_ONBOARD_SESSION_ID_BYTES: Final[int] = 12
_ONBOARD_BELIEF_ID_HEX_LEN: Final[int] = 16


@dataclass
class OnboardSentence:
    """One scanner candidate awaiting host classification.

    `index` is the position used by HostClassification to refer back; it
    is stable across the JSON round-trip in `onboard_sessions.candidates_json`.
    """

    index: int
    text: str
    source: str


@dataclass
class StartOnboardResult:
    """Output of `start_onboard_session`.

    `sentences` is what the host needs to classify. `n_already_present`
    counts candidates that were dropped before the host saw them because
    a belief with the deterministic id already exists — re-running
    onboard on a tree the brain has already seen does not re-ask the
    host to classify the same content. `n_already_rejected` (#801)
    counts candidates the host previously rejected with persist=False;
    they sit in the rejection ledger and are bypassed unless the caller
    passes `force=True`.
    """

    session_id: str
    sentences: list[OnboardSentence]
    n_already_present: int
    n_already_rejected: int = 0


@dataclass
class HostClassification:
    """One classification result from the host LLM, addressed by index.

    `persist` is the host's verdict: True to insert as a belief, False to
    drop (questions, meta-commentary, anything ephemeral). Mirrors the
    `persist` field on ClassificationResult so the regex-fallback and
    host-handshake paths share semantics.
    """

    index: int
    belief_type: str
    persist: bool


@dataclass
class OnboardCheckResult:
    """Output of `check_onboard_candidates`.

    Read-only pre-scan: runs the three extractors and counts how many
    candidates would be filtered as already-present vs handed to the
    classifier, without writing an onboard_sessions row or inserting
    any beliefs. Lets callers decide whether re-onboard is worth the
    LLM/CPU cost before dispatching classification (#761).

    `n_already_rejected` (#801) counts candidates currently in the
    rejection ledger (host previously verdict persist=False); they are
    excluded from `n_new` unless the caller passes `force=True`.
    """

    n_already_present: int
    n_new: int
    repo_path: str
    n_already_rejected: int = 0


@dataclass
class AcceptOnboardResult:
    """Output of `accept_classifications`.

    Mirrors `scanner.ScanResult` shape so callers can render either path
    uniformly.
    """

    session_id: str
    inserted: int
    skipped_non_persisting: int
    skipped_existing: int
    skipped_unclassified: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_session_id() -> str:
    return secrets.token_hex(_ONBOARD_SESSION_ID_BYTES)


def _derive_belief_id(text: str, source: str) -> str:
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_ONBOARD_BELIEF_ID_HEX_LEN]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def start_onboard_session(
    store: "MemoryStore",
    repo_path: Path,
    *,
    now: str | None = None,
    force: bool = False,
) -> StartOnboardResult:
    """Run the three scanner extractors against `repo_path`, filter out
    candidates whose deterministic belief id is already in the store
    (or in the rejection ledger when `force=False`), persist the rest
    as a pending onboard_sessions row, and return the payload the host
    should classify.

    Idempotent: re-calling against the same tree returns a fresh
    session_id whose `sentences` list excludes anything already present
    *and* anything the host previously rejected with persist=False.
    The host can answer with an empty list of classifications and the
    session will close cleanly.

    `force=True` (#801) bypasses the rejection ledger so previously
    rejected candidates are re-emitted for the host to re-classify.
    Already-present beliefs are still filtered — `force` only opts back
    in to noise the host already saw, not to duplicate-stored content.
    """
    # Lazy import: scanner imports classification at module-load (it
    # calls `classify_sentence`); importing scanner at top-level here
    # would form a circular import. Doing it inside the function defers
    # resolution until the cycle is already broken.
    from aelfrice.scanner import (
        extract_ast,
        extract_filesystem,
        extract_git_log,
    )

    timestamp = now if now is not None else _utc_now_iso()
    candidates = (
        extract_filesystem(repo_path)
        + extract_git_log(repo_path)
        + extract_ast(repo_path)
    )

    rejected_ids: set[str] = (
        set() if force else store.list_onboard_rejection_ids()
    )

    pending_sentences: list[OnboardSentence] = []
    n_already_present = 0
    n_already_rejected = 0
    for c in candidates:
        bid = _derive_belief_id(c.text, c.source)
        if store.get_belief(bid) is not None:
            n_already_present += 1
            continue
        if bid in rejected_ids:
            n_already_rejected += 1
            continue
        pending_sentences.append(
            OnboardSentence(
                index=len(pending_sentences),
                text=c.text,
                source=c.source,
            )
        )

    session_id = _new_session_id()
    candidates_json = json.dumps(
        [
            {"index": s.index, "text": s.text, "source": s.source}
            for s in pending_sentences
        ]
    )
    store.insert_onboard_session(
        OnboardSession(
            session_id=session_id,
            repo_path=str(repo_path),
            state=ONBOARD_STATE_PENDING,
            candidates_json=candidates_json,
            created_at=timestamp,
            completed_at=None,
        )
    )
    return StartOnboardResult(
        session_id=session_id,
        sentences=pending_sentences,
        n_already_present=n_already_present,
        n_already_rejected=n_already_rejected,
    )


def check_onboard_candidates(
    store: "MemoryStore",
    repo_path: Path,
    *,
    force: bool = False,
) -> OnboardCheckResult:
    """Pre-scan a repo without persisting a session or inserting beliefs.

    Runs the same extractor + dedup-by-id pipeline as
    `start_onboard_session` but discards the candidate list and returns
    only counts. No `onboard_sessions` row is written and no beliefs are
    touched, so the call is side-effect free and safe to repeat.

    Surfaces the same `n_already_present` signal the polymorphic
    handshake exposes via `--emit-candidates`, but at the human-facing
    `aelf onboard <path> --check` entry — letting callers see what a
    re-onboard would do before paying the classification cost (#761).

    `n_already_rejected` (#801) reports candidates currently in the
    rejection ledger; they are excluded from `n_new` unless `force=True`.
    """
    from aelfrice.scanner import (
        extract_ast,
        extract_filesystem,
        extract_git_log,
    )

    candidates = (
        extract_filesystem(repo_path)
        + extract_git_log(repo_path)
        + extract_ast(repo_path)
    )

    rejected_ids: set[str] = (
        set() if force else store.list_onboard_rejection_ids()
    )

    n_already_present = 0
    n_already_rejected = 0
    n_new = 0
    for c in candidates:
        bid = _derive_belief_id(c.text, c.source)
        if store.get_belief(bid) is not None:
            n_already_present += 1
        elif bid in rejected_ids:
            n_already_rejected += 1
        else:
            n_new += 1

    return OnboardCheckResult(
        n_already_present=n_already_present,
        n_new=n_new,
        repo_path=str(repo_path),
        n_already_rejected=n_already_rejected,
    )


def accept_classifications(
    store: "MemoryStore",
    session_id: str,
    classifications: list[HostClassification],
    *,
    now: str | None = None,
) -> AcceptOnboardResult:
    """Apply host-provided classifications to a pending onboard session.

    Insert one belief per (sentence, classification) pair where the host
    set `persist=True`. Beliefs land with `pending_classification=False`
    semantics — they were typed by an actual LLM, not the regex
    fallback. Sentences with no matching classification index are
    counted as `skipped_unclassified` (host chose to elide them).

    Raises ValueError on:
    - unknown session_id
    - session already in COMPLETED state (re-accepting is rejected so
      callers can't double-insert by replaying a stale message)
    - any classification whose `belief_type` is not in BELIEF_TYPES
    """
    session = store.get_onboard_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    if session.state != ONBOARD_STATE_PENDING:
        raise ValueError(
            f"session not pending (state={session.state}): {session_id}"
        )
    for c in classifications:
        if c.belief_type not in BELIEF_TYPES:
            raise ValueError(f"unknown belief_type: {c.belief_type}")

    timestamp = now if now is not None else _utc_now_iso()
    sentences_data = json.loads(session.candidates_json)
    by_index: dict[int, HostClassification] = {c.index: c for c in classifications}

    inserted = 0
    skipped_non_persisting = 0
    skipped_existing = 0
    skipped_unclassified = 0

    # Lazy import: derivation_worker -> derivation -> classification (this
    # module). Importing at module load forms a cycle.
    from aelfrice.derivation_worker import run_worker  # noqa: PLC0415

    # #264 slice 2: route through the derivation worker. Each persisting
    # sentence appends one ingest_log row carrying the host-decided
    # `override_belief_type` and the `call_site` for the corroboration
    # audit; one worker invocation at end-of-batch derives + writes
    # canonical beliefs and stamps the rows. Snapshot the canonical id
    # set so the post-worker count of newly-inserted beliefs matches the
    # pre-#264 contract.
    ids_before: set[str] = set(store.list_belief_ids())
    persisting_log_ids: list[str] = []

    for sd in sentences_data:
        idx = int(sd["index"])
        text = str(sd["text"])
        source = str(sd["source"])
        c = by_index.get(idx)
        if c is None:
            skipped_unclassified += 1
            continue
        if not c.persist:
            # #801: record the rejection so the next `aelf onboard` pass
            # filters this candidate out instead of re-classifying it.
            store.insert_onboard_rejection(
                _derive_belief_id(text, source), text, source, timestamp,
            )
            skipped_non_persisting += 1
            continue
        # #801: a persisted sentence may have been previously rejected
        # and surfaced again via --force. Drop the stale ledger entry so
        # the ledger only carries currently-rejected candidates.
        store.delete_onboard_rejection(_derive_belief_id(text, source))
        log_id = store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=source,
            raw_text=text,
            session_id=session_id,
            ts=timestamp,
            raw_meta={
                "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
                "override_belief_type": c.belief_type,
            },
        )
        persisting_log_ids.append(log_id)

    if persisting_log_ids:
        run_worker(store)
        for log_id in persisting_log_ids:
            entry = store.get_ingest_log_entry(log_id)
            if entry is None:
                continue
            ids = entry.get("derived_belief_ids") or []
            if not isinstance(ids, list):
                continue
            for bid in ids:
                if not isinstance(bid, str):
                    continue
                if bid in ids_before:
                    skipped_existing += 1
                else:
                    inserted += 1
                    # Subsequent log rows that resolve to the same bid
                    # in this batch should count as skipped_existing,
                    # matching the pre-#264 per-iteration contract.
                    ids_before.add(bid)

    store.complete_onboard_session(session_id, timestamp)
    return AcceptOnboardResult(
        session_id=session_id,
        inserted=inserted,
        skipped_non_persisting=skipped_non_persisting,
        skipped_existing=skipped_existing,
        skipped_unclassified=skipped_unclassified,
    )
