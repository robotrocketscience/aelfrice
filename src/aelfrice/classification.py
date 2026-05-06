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

from aelfrice.correction import detect_correction
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
    BELIEF_TYPES,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    INGEST_SOURCE_FILESYSTEM,
    ONBOARD_STATE_PENDING,
    OnboardSession,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# --- Priors (per Exp 61, restricted to v1.0's 4-type catalog) -----------

TYPE_PRIORS: Final[dict[str, tuple[float, float]]] = {
    BELIEF_REQUIREMENT: (9.0, 0.5),  # 94.7% prior — hard constraints
    BELIEF_CORRECTION: (9.0, 0.5),   # 94.7% — user corrections
    BELIEF_PREFERENCE: (7.0, 1.0),   # 87.5% — user preferences
    BELIEF_FACTUAL: (3.0, 1.0),      # 75.0% — stated facts and analyses
}

# Deflation factor for non-user sources. TYPE_PRIORS are calibrated for
# user-stated content; agent-inferred or document-extracted content
# starts at lower alpha so the feedback loop earns the rest of the
# confidence rather than inheriting it. Without this, scanner-extracted
# beliefs would cluster at 90-95% confidence on day one and Thompson
# sampling would lose discriminative power.
_AGENT_INFERRED_DEFLATION: Final[float] = 0.2
_DEFLATED_ALPHA_FLOOR: Final[float] = 0.5

USER_SOURCE: Final[str] = "user"

# --- Heuristic keyword sets ---------------------------------------------

_REQUIREMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "must",
    "require",
    "mandatory",
    "hard cap",
    "constraint",
    "hard rule",
)

_PREFERENCE_KEYWORDS: Final[tuple[str, ...]] = (
    "prefer",
    "favorite",
    "always use",
    "never use",
    "i like",
    "i hate",
    "i want",
)

_QUESTION_PREFIXES: Final[tuple[str, ...]] = (
    "what ",
    "how ",
    "why ",
    "when ",
    "where ",
    "can ",
    "does ",
    "is there",
    "should ",
    "would ",
    "could ",
)


# --- Output ---------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Output of classify_sentence.

    Fields:
    - belief_type: one of `factual / correction / preference / requirement`.
      For non-persistable sentences (questions, coordination, meta), still
      returns `factual` but `persist=False`.
    - alpha, beta: Beta-Bernoulli prior, source-adjusted.
    - persist: False for ephemeral content (questions, etc.); True
      otherwise. Caller (scanner / onboarding) is responsible for
      skipping non-persisting sentences before insertion.
    - pending_classification: True when this was the regex-fallback path
      and a future host-LLM pass could refine the type. Always True in
      v1.0; the polymorphic-host-handshake path that flips this to False
      lands in v0.6.0.
    """

    belief_type: str
    alpha: float
    beta: float
    persist: bool
    pending_classification: bool


# --- Helpers --------------------------------------------------------------


def get_source_adjusted_prior(
    belief_type: str,
    source: str,
) -> tuple[float, float]:
    """Resolve the Beta prior for a (type, source) pair.

    User-sourced content gets the full TYPE_PRIORS value. Non-user
    sources get alpha deflated by `_AGENT_INFERRED_DEFLATION`, with a
    `_DEFLATED_ALPHA_FLOOR` so the deflated alpha never drops below 0.5
    (which would make posterior_mean numerically degenerate at low beta).
    """
    prior = TYPE_PRIORS.get(belief_type)
    if prior is None:
        # Unknown type collapses to factual prior — keeps the function
        # total without surfacing a partial-failure mode to callers.
        prior = TYPE_PRIORS[BELIEF_FACTUAL]
    alpha, beta = prior
    if source != USER_SOURCE:
        alpha = max(_DEFLATED_ALPHA_FLOOR, alpha * _AGENT_INFERRED_DEFLATION)
    return (alpha, beta)


def _is_question(text_lower: str) -> bool:
    return text_lower.startswith(_QUESTION_PREFIXES) and text_lower.endswith("?")


def _has_any(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text_lower for kw in keywords)


# --- Public API -----------------------------------------------------------


def classify_sentence(text: str, source: str) -> ClassificationResult:
    """Synchronous, deterministic classification.

    Pipeline (in evaluation order):
    1. Empty / whitespace-only -> factual, persist=False.
    2. Question form -> factual, persist=False.
    3. User source + requirement keywords -> requirement.
    4. User source + correction-detector positive -> correction.
    5. Preference keywords -> preference.
    6. Default -> factual.

    Always sets pending_classification=True in v1.0; the host-handshake
    path that flips it to False ships at v0.6.0.

    Pure function. No I/O, no third-party deps, deterministic for any
    (text, source) pair.
    """
    text_lower = text.lower().strip()

    # 1. Empty.
    if not text_lower:
        alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
        return ClassificationResult(
            belief_type=BELIEF_FACTUAL,
            alpha=alpha,
            beta=beta,
            persist=False,
            pending_classification=True,
        )

    # 2. Questions don't persist.
    if _is_question(text_lower):
        alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
        return ClassificationResult(
            belief_type=BELIEF_FACTUAL,
            alpha=alpha,
            beta=beta,
            persist=False,
            pending_classification=True,
        )

    # 3. Requirement keywords (must, mandatory, hard rule, ...)
    #    Originally gated on `source == USER_SOURCE` to suppress
    #    document-text false positives. Per #226: that gate caused
    #    every onboard-extracted requirement (source = `doc:…`,
    #    `ast:…`, `git:…`) to mis-classify, producing zero
    #    requirement counts on the labeled corpus. The
    #    source-prior-deflation in `get_source_adjusted_prior`
    #    already lowers alpha for non-user sources, so the
    #    false-positive risk is handled at the scoring layer
    #    rather than by gating classification.
    if _has_any(text_lower, _REQUIREMENT_KEYWORDS):
        alpha, beta = get_source_adjusted_prior(BELIEF_REQUIREMENT, source)
        return ClassificationResult(
            belief_type=BELIEF_REQUIREMENT,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 4. Correction (per the no-LLM detector). Same #226 reasoning
    #    — the user-only gate suppressed onboard-extracted
    #    corrections; deflated alpha at non-user sources handles
    #    the false-positive concern.
    cresult = detect_correction(text)
    if cresult.is_correction:
        alpha, beta = get_source_adjusted_prior(BELIEF_CORRECTION, source)
        return ClassificationResult(
            belief_type=BELIEF_CORRECTION,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 5. Preference keywords (any source).
    if _has_any(text_lower, _PREFERENCE_KEYWORDS):
        alpha, beta = get_source_adjusted_prior(BELIEF_PREFERENCE, source)
        return ClassificationResult(
            belief_type=BELIEF_PREFERENCE,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 6. Default: factual.
    alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
    return ClassificationResult(
        belief_type=BELIEF_FACTUAL,
        alpha=alpha,
        beta=beta,
        persist=True,
        pending_classification=True,
    )


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
    host to classify the same content.
    """

    session_id: str
    sentences: list[OnboardSentence]
    n_already_present: int


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
) -> StartOnboardResult:
    """Run the three scanner extractors against `repo_path`, filter out
    candidates whose deterministic belief id is already in the store,
    persist the rest as a pending onboard_sessions row, and return the
    payload the host should classify.

    Idempotent: re-calling against the same tree returns a fresh
    session_id whose `sentences` list excludes anything already present.
    The host can answer with an empty list of classifications and the
    session will close cleanly.
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

    pending_sentences: list[OnboardSentence] = []
    n_already_present = 0
    for c in candidates:
        bid = _derive_belief_id(c.text, c.source)
        if store.get_belief(bid) is not None:
            n_already_present += 1
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

    # #264 slice 2: route through the derivation worker. The host's
    # belief-type verdict rides in `raw_meta.override_belief_type` so
    # the worker reconstructs the same DerivationInput on replay.
    # Pre-existing beliefs at the same id surface as `skipped_existing`
    # via post-worker stamp inspection rather than an inline
    # `get_belief` short-circuit — every host verdict appends a log row,
    # so replay sees the full ingest history (#205 parallel-write
    # becomes canonical).
    from aelfrice.derivation_worker import run_worker  # noqa: PLC0415

    ids_before: set[str] = set(store.list_belief_ids())
    log_ids: list[str] = []

    for sd in sentences_data:
        idx = int(sd["index"])
        text = str(sd["text"])
        source = str(sd["source"])
        c = by_index.get(idx)
        if c is None:
            skipped_unclassified += 1
            continue
        if not c.persist:
            skipped_non_persisting += 1
            continue
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
        log_ids.append(log_id)

    if log_ids:
        run_worker(store)
        for log_id in log_ids:
            entry = store.get_ingest_log_entry(log_id)
            if entry is None:
                continue
            ids = entry.get("derived_belief_ids") or []
            if not isinstance(ids, list) or not ids:
                # `override_belief_type` forces persist=True in derive();
                # an empty list here means the row failed to materialize.
                # Surface as skipped_non_persisting so the count is
                # observable rather than silently lost.
                skipped_non_persisting += 1
                continue
            bid = str(ids[0])
            if bid in ids_before:
                skipped_existing += 1
            else:
                inserted += 1
                ids_before.add(bid)  # convergent dupes within this batch

    store.complete_onboard_session(session_id, timestamp)
    return AcceptOnboardResult(
        session_id=session_id,
        inserted=inserted,
        skipped_non_persisting=skipped_non_persisting,
        skipped_existing=skipped_existing,
        skipped_unclassified=skipped_unclassified,
    )
