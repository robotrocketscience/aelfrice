"""Mechanical (subject, relation, object) triple extraction from prose.

Pure regex over a fixed pattern bank. No POS tagger, no embedding,
no LLM. Reusable by every v1.x ingest caller that has prose at
hand: the v1.2.0 commit-ingest hook, transcript-ingest, manual
`aelf remember` calls, and the v1.3.0 entity-index path.

Two-piece API per docs/triple_extractor.md:

- `extract_triples(text)` -> list[Triple]: pure, no store side-effects.
- `ingest_triples(store, triples, session_id=None)` -> IngestResult:
  applies the triples, creating beliefs by content-hash lookup and
  inserting `EDGE_TYPES` edges with `anchor_text` populated from the
  citing prose.

The split lets callers inspect or filter triples between extraction
and ingest. Tests exercise extract_triples without any store.

Pattern set is intentionally narrow (six relation families). The
roadmap notes a minimal set + a future health metric that surfaces
"X messages parsed; Y triples produced" so users can see when their
corpus needs broader patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    CORROBORATION_SOURCE_COMMIT_INGEST,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    INGEST_SOURCE_GIT,
    Edge,
)
from aelfrice.store import MemoryStore

ANCHOR_CONTEXT_TARGET: Final[int] = 80
"""Soft target for anchor_text length — match span plus a little
surrounding prose if the match itself is shorter. Spec calls for
~80 chars; the absolute cap is enforced by `Edge.anchor_text` itself
(ANCHOR_TEXT_MAX_LEN at the dataclass boundary)."""

TRIPLE_BELIEF_SOURCE: Final[str] = "triple"
"""Source label written into the belief-id sha256 for triple-extracted
beliefs. Sharing a source label across all triple ingest sites means
the same noun phrase always resolves to the same belief id — even
across different commits, transcripts, or call sites."""


@dataclass(frozen=True)
class Triple:
    """One extracted (subject, relation, object) tuple.

    `subject` and `object` are short noun phrases drawn from the
    source prose. `relation` is one of `EDGE_TYPES`. `anchor_text`
    is the citing prose's literal phrasing for the relationship —
    typically the matched substring of `text` plus a little context.
    """
    subject: str
    relation: str
    object: str
    anchor_text: str


@dataclass
class IngestResult:
    """Per-call outcome of `ingest_triples`."""
    new_beliefs: list[str] = field(default_factory=list[str])
    new_edges: list[tuple[str, str, str]] = field(
        default_factory=list[tuple[str, str, str]]
    )
    skipped_duplicate_edges: int = 0
    skipped_no_subject_or_object: int = 0


# --- Pattern bank --------------------------------------------------------

# A noun phrase is up to 5 word-tokens, optionally led by an article /
# possessive. Word tokens permit dashes and underscores so identifiers
# (`session_id`, `aelf-hook`) can be subjects/objects without being
# split. The bound at 5 tokens keeps matches local — long sentences
# rarely have a 6+ token noun phrase that the pattern would misread.
_DET = (
    r"(?:the|a|an|our|their|its|this|that|these|those|my|your|his|her)"
)
_TOKEN = r"[A-Za-z][\w-]*"
_NP = (
    rf"(?:(?:{_DET})\s+)?{_TOKEN}(?:\s+{_TOKEN}){{0,4}}"
)

# Public alias of the internal noun-phrase pattern. Re-exported so
# downstream extractors (the v1.3.0 entity-index path) can compose
# the same NP shape without reaching into the underscore name. The
# string itself is what's exposed; consumers compile it themselves
# with whatever flags they need.
NOUN_PHRASE_PATTERN: Final[str] = _NP


def _np(group_name: str) -> str:
    """Build a named non-greedy noun-phrase capture group."""
    return rf"(?P<{group_name}>{_NP})"


# Each entry: (compiled regex, edge_type, swap_subject_object).
# swap=True flips subject/object so passive forms ("X is supported by Y")
# emit the canonical (active-voice) ordering with X as subject and Y as
# object — matching the active-voice form's semantics.
@dataclass(frozen=True)
class _RelationPattern:
    regex: re.Pattern[str]
    edge_type: str
    swap: bool


def _build_pattern(template: str, edge_type: str, *, swap: bool = False) -> _RelationPattern:
    """Compile a template like 'supports' or 'is supported by' into a
    full regex with NP captures around it. The verb phrase tokens are
    space-separated; `\\s+` is inserted between each so multi-space
    inputs still match.
    """
    verb_pattern = r"\s+".join(re.escape(t) for t in template.split())
    full = rf"(?<![\w-]){_np('subject')}\s+{verb_pattern}\s+{_np('object')}"
    return _RelationPattern(
        regex=re.compile(full, re.IGNORECASE),
        edge_type=edge_type,
        swap=swap,
    )


_PATTERNS: Final[tuple[_RelationPattern, ...]] = (
    _build_pattern("supports", EDGE_SUPPORTS),
    _build_pattern("is supported by", EDGE_SUPPORTS, swap=True),
    _build_pattern("cites", EDGE_CITES),
    _build_pattern("mentions", EDGE_CITES),
    _build_pattern("contradicts", EDGE_CONTRADICTS),
    _build_pattern("disagrees with", EDGE_CONTRADICTS),
    _build_pattern("supersedes", EDGE_SUPERSEDES),
    _build_pattern("replaces", EDGE_SUPERSEDES),
    _build_pattern("relates to", EDGE_RELATES_TO),
    _build_pattern("is related to", EDGE_RELATES_TO),
    _build_pattern("is derived from", EDGE_DERIVED_FROM),
    _build_pattern("is based on", EDGE_DERIVED_FROM),
    _build_pattern("extends", EDGE_DERIVED_FROM),
    # IMPLEMENTS: source = implementation, target = spec/claim being implemented.
    # "implements" / "is an implementation of" / "realizes" / "fulfills" all
    # express that the subject satisfies or concretizes the object spec.
    _build_pattern("implements", EDGE_IMPLEMENTS),
    _build_pattern("is an implementation of", EDGE_IMPLEMENTS),
    _build_pattern("realizes", EDGE_IMPLEMENTS),
    _build_pattern("fulfills", EDGE_IMPLEMENTS),
)


# --- Extraction -----------------------------------------------------------


def _normalize_phrase(phrase: str) -> str:
    """Trim + collapse whitespace. Used for content-hash lookup so two
    triples that disagree only in whitespace share a belief id."""
    return " ".join(phrase.split())


def _expand_anchor(text: str, start: int, end: int) -> str:
    """Return `text[start:end]` widened to ~ANCHOR_CONTEXT_TARGET chars
    if the raw match is shorter, breaking on word boundaries.

    The widening keeps the citing prose's framing intact when the
    match itself is just "X supports Y" — useful for downstream
    anchor-aware retrieval that wants the surrounding clause.
    """
    span = text[start:end]
    if len(span) >= ANCHOR_CONTEXT_TARGET:
        return span
    extra = ANCHOR_CONTEXT_TARGET - len(span)
    left_budget = extra // 2
    right_budget = extra - left_budget
    new_start = max(0, start - left_budget)
    new_end = min(len(text), end + right_budget)
    while new_start > 0 and text[new_start].isalnum():
        new_start -= 1
    while new_end < len(text) and text[new_end].isalnum():
        new_end += 1
    return text[new_start:new_end].strip()


def extract_triples(text: str) -> list[Triple]:
    """Extract (subject, relation, object) triples from `text`.

    Returns the triples in left-to-right order of their match start
    in the input. Overlapping matches from different patterns are
    each reported (downstream `ingest_triples` dedups by edge tuple).
    Empty / non-relational input returns an empty list.
    """
    if not text:
        return []
    triples: list[Triple] = []
    for pat in _PATTERNS:
        for m in pat.regex.finditer(text):
            subj_raw = m.group("subject")
            obj_raw = m.group("object")
            subj = _normalize_phrase(subj_raw)
            obj = _normalize_phrase(obj_raw)
            if not subj or not obj:
                continue
            anchor = _expand_anchor(text, m.start(), m.end())
            if pat.swap:
                subj, obj = obj, subj
            triples.append(Triple(
                subject=subj, relation=pat.edge_type,
                object=obj, anchor_text=anchor,
            ))
    triples.sort(key=lambda t: text.find(t.anchor_text))
    return triples


# --- Ingest ---------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_or_create_belief(
    store: MemoryStore, phrase: str, *,
    session_id: str | None,
    created_ids: list[str],
) -> str:
    """Return the id of the belief representing `phrase`, creating
    one with the project default prior (alpha=1.0, beta=1.0) if none
    exists. The first creator within a call stamps `session_id`.

    When the belief already exists (same id = same normalised phrase),
    a corroboration row is recorded so the re-assertion is observable.
    """
    # derive() id-scheme matches _belief_id_for_phrase; compute once.
    ts = _now_iso()
    out = derive(DerivationInput(
        raw_text=phrase,
        source_kind=INGEST_SOURCE_GIT,
        session_id=session_id,
        ts=ts,
    ))
    # INGEST_SOURCE_GIT always produces a belief (no classifier skip).
    assert out.belief is not None
    bid = out.belief.id
    existing = store.get_belief(bid)
    if existing is not None:
        store.record_corroboration(
            bid,
            source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
            session_id=session_id,
        )
        return bid
    # v2.0 #205 parallel-write: log the raw phrase before materialization.
    # source_kind=git because the commit-ingest path emits triples from
    # commit messages; source_path is unknown at this layer (callers
    # have it). Future commits could thread the commit SHA through.
    store.record_ingest(
        source_kind=INGEST_SOURCE_GIT,
        raw_text=phrase,
        derived_belief_ids=[bid],
        session_id=session_id,
        ts=ts,
    )
    actual_id, was_inserted = store.insert_or_corroborate(
        out.belief,
        source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
        session_id=session_id,
    )
    if was_inserted:
        created_ids.append(actual_id)
    return actual_id


def ingest_triples(
    store: MemoryStore,
    triples: list[Triple],
    session_id: str | None = None,
) -> IngestResult:
    """Persist `triples` to `store`. Idempotent.

    For each triple:
      1. Resolve / create the subject belief by content hash.
      2. Resolve / create the object belief.
      3. Insert an Edge(subject_id -> object_id, type=relation,
         anchor_text=triple.anchor_text). Skip if already present.

    Self-edges (subject and object resolve to the same id) are
    counted under `skipped_no_subject_or_object` and dropped.
    """
    result = IngestResult()
    for triple in triples:
        if not triple.subject or not triple.object:
            result.skipped_no_subject_or_object += 1
            continue
        subj_id = _resolve_or_create_belief(
            store, triple.subject,
            session_id=session_id, created_ids=result.new_beliefs,
        )
        obj_id = _resolve_or_create_belief(
            store, triple.object,
            session_id=session_id, created_ids=result.new_beliefs,
        )
        if subj_id == obj_id:
            result.skipped_no_subject_or_object += 1
            continue
        if store.get_edge(subj_id, obj_id, triple.relation) is not None:
            result.skipped_duplicate_edges += 1
            continue
        store.insert_edge(Edge(
            src=subj_id, dst=obj_id,
            type=triple.relation, weight=1.0,
            anchor_text=triple.anchor_text,
        ))
        result.new_edges.append((subj_id, obj_id, triple.relation))
    return result
