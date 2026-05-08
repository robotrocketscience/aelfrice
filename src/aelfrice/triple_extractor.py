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

# `np_pattern` is a leaf module (no aelfrice imports) so
# `entity_extractor` can share the NP regex without closing a
# store ↔ extractors cycle through this module (#499).
from aelfrice.derivation_worker import run_worker
from aelfrice.np_pattern import NOUN_PHRASE_PATTERN, _NP
from aelfrice.models import (
    CORROBORATION_SOURCE_COMMIT_INGEST,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
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
    # TEMPORAL_NEXT: source = temporal successor, target = temporal predecessor.
    # "follows" / "comes after" / "is after" / "succeeds" all express
    # that the subject belief occurred or applies chronologically after
    # the object belief. Patterns require multi-token verb phrases or
    # unambiguous verbs to avoid over-firing on common prose ("after"
    # alone, "next" alone, bare "follows" with a direct object).
    _build_pattern("follows", EDGE_TEMPORAL_NEXT),
    _build_pattern("comes after", EDGE_TEMPORAL_NEXT),
    _build_pattern("is after", EDGE_TEMPORAL_NEXT),
    _build_pattern("succeeds", EDGE_TEMPORAL_NEXT),
    # TESTS: source = test belief, target = spec/claim under test.
    # "tests" / "is a test for" / "is test of" / "covers" all express
    # the same evidential relationship — the subject exercises the object.
    _build_pattern("tests", EDGE_TESTS),
    _build_pattern("is a test for", EDGE_TESTS),
    _build_pattern("is test of", EDGE_TESTS),
    _build_pattern("covers", EDGE_TESTS),
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

    #264 slice 2: subject + object beliefs are materialized via the
    derivation worker. Triple-edges remain entry-point-owned because
    they connect *pairs* of log rows (worker stamps per-row derived
    edges only). Two-pass shape:
      pass A — append one ingest_log row per phrase, remembering each
        phrase's log_id and the relation/anchor that pairs them.
      pass B — invoke run_worker(store) once. For each remembered
        triple, read the stamped derived_belief_ids on the subject /
        object log rows and insert the edge.
    """
    result = IngestResult()
    if not triples:
        return result

    # Snapshot of canonical ids so we can flag fresh inserts via the
    # post-worker walk. Within a single call, the worker may corroborate
    # one log row's bid against an earlier same-phrase log row in this
    # batch — both would resolve to the same actual_id, but only the
    # first should count toward `new_beliefs`.
    ids_before: set[str] = set(store.list_belief_ids())
    ts = _now_iso()

    # Pending edges: (subject_log_id, object_log_id, relation, anchor).
    pending_edges: list[tuple[str, str, str, str]] = []
    # Indices in `triples` we kept (skipped ones are already counted).
    for triple in triples:
        if not triple.subject or not triple.object:
            result.skipped_no_subject_or_object += 1
            continue
        subj_log = store.record_ingest(
            source_kind=INGEST_SOURCE_GIT,
            raw_text=triple.subject,
            session_id=session_id,
            ts=ts,
            raw_meta={"call_site": CORROBORATION_SOURCE_COMMIT_INGEST},
        )
        obj_log = store.record_ingest(
            source_kind=INGEST_SOURCE_GIT,
            raw_text=triple.object,
            session_id=session_id,
            ts=ts,
            raw_meta={"call_site": CORROBORATION_SOURCE_COMMIT_INGEST},
        )
        pending_edges.append(
            (subj_log, obj_log, triple.relation, triple.anchor_text)
        )

    if pending_edges:
        run_worker(store)

    seen_new: set[str] = set()
    for subj_log, obj_log, relation, anchor in pending_edges:
        subj_id = _belief_id_from_log(store, subj_log)
        obj_id = _belief_id_from_log(store, obj_log)
        if subj_id is None or obj_id is None:
            # Worker should have stamped both; absence is a bug worth
            # surfacing. Skip the edge so we don't insert a dangling row.
            result.skipped_no_subject_or_object += 1
            continue
        for bid in (subj_id, obj_id):
            if bid not in ids_before and bid not in seen_new:
                seen_new.add(bid)
                result.new_beliefs.append(bid)
        if subj_id == obj_id:
            result.skipped_no_subject_or_object += 1
            continue
        if store.get_edge(subj_id, obj_id, relation) is not None:
            result.skipped_duplicate_edges += 1
            continue
        store.insert_edge(Edge(
            src=subj_id, dst=obj_id,
            type=relation, weight=1.0,
            anchor_text=anchor,
        ))
        result.new_edges.append((subj_id, obj_id, relation))
    return result


def _belief_id_from_log(store: MemoryStore, log_id: str) -> str | None:
    """Read the worker-stamped belief id off a single log row, or None
    when the row is absent / unstamped. Stamping shape: a non-empty
    derived_belief_ids list whose first entry is the canonical bid."""
    entry = store.get_ingest_log_entry(log_id)
    if entry is None:
        return None
    ids = entry.get("derived_belief_ids")
    if not isinstance(ids, list) or not ids:
        return None
    head = ids[0]
    return head if isinstance(head, str) and head else None
