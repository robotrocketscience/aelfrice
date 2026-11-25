"""Mechanical (subject, relation, object) triple extraction from prose.

Pure regex over a fixed pattern bank. No POS tagger, no embedding,
no LLM. The one call site is the context-rebuilder's query-enrichment
pass (`rebuild_log.py`), which adds each triple's phrases to the query
it records for a prompt. Nothing stores the triples.

`extract_triples(text)` -> list[Triple] is pure, with no store
side-effects. See docs/design/triple_extractor.md.

Pattern set is intentionally narrow (six relation families). The
roadmap notes a minimal set + a future health metric that surfaces
"X messages parsed; Y triples produced" so users can see when their
corpus needs broader patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# `np_pattern` is a leaf module (no aelfrice imports) so
# `entity_extractor` can share the NP regex without closing a
# store ↔ extractors cycle through this module (#499).
from aelfrice.np_pattern import NOUN_PHRASE_PATTERN, _NP
from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
)

ANCHOR_CONTEXT_TARGET: Final[int] = 80
"""Soft target for anchor_text length — match span plus a little
surrounding prose if the match itself is shorter. Spec calls for
~80 chars; the absolute cap is enforced by `Edge.anchor_text` itself
(ANCHOR_TEXT_MAX_LEN at the dataclass boundary)."""


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
    # The verb phrase this pattern was built from. Nothing reads it now,
    # but the `_PATTERNS` digest pin in `detector_thresholds` covers it.
    template: str


def _build_pattern(
    template: str,
    edge_type: str,
    *,
    swap: bool = False,
) -> _RelationPattern:
    """Compile a template like 'supports' or 'is supported by' into a
    full regex with NP captures around it. The verb phrase tokens are
    space-separated; `\\s+` is inserted between each so multi-space
    inputs still match.
    """
    verb_pattern = r"\s+".join(re.escape(t) for t in template.split())
    full = rf"(?<![\w-]){_np('subject')}\s+{verb_pattern}\s+{_np('object')}"
    regex = re.compile(full, re.IGNORECASE)
    return _RelationPattern(
        regex=regex,
        edge_type=edge_type,
        swap=swap,
        template=template,
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
    """Trim + collapse whitespace, so two triples that disagree only in
    whitespace carry the same phrase."""
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
    each reported. Empty / non-relational input returns an empty list.
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
