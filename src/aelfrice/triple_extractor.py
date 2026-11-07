"""Mechanical (subject, relation, object) triple extraction from prose.

Pure regex over a fixed pattern bank. No POS tagger, no embedding,
no LLM. Reusable by any v1.x ingest caller that has prose at
hand. Current call sites: the v1.2.0 commit-ingest hook
(`hook_commit_ingest.py`) and the context-rebuilder's query-enrichment
pass (`rebuild_log.py`, `extract_triples` only). Not currently
wired into transcript-ingest or `aelf remember`.

Two-piece API per docs/design/triple_extractor.md:

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

from aelfrice.classification_core import INGEST_CLASSIFIER_VERSION, rule_set_hash
from aelfrice.derivation_worker import run_worker
# `np_pattern` is a leaf module (no aelfrice imports) so
# `entity_extractor` can share the NP regex without closing a
# store ↔ extractors cycle through this module (#499).
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
    # The verb phrase this pattern was built from, kept so a bank can be
    # filtered by template without parsing it back out of the regex.
    template: str


def _build_pattern(
    template: str,
    edge_type: str,
    *,
    swap: bool = False,
    exact_verb_case: bool = False,
) -> _RelationPattern:
    """Compile a template like 'supports' or 'is supported by' into a
    full regex with NP captures around it. The verb phrase tokens are
    space-separated; `\\s+` is inserted between each so multi-space
    inputs still match.

    `exact_verb_case` ignores case in the noun phrases only, so the verb
    has to appear as the template spells it. Its left boundary also refuses
    an apostrophe, so `it's` can't start a phrase at `s`.
    """
    verb_pattern = r"\s+".join(re.escape(t) for t in template.split())
    if exact_verb_case:
        full = (
            rf"(?<![\w'\u2019-])(?i:{_np('subject')})\s+{verb_pattern}"
            rf"\s+(?i:{_np('object')})"
        )
        regex = re.compile(full)
    else:
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


# --- The ingest-constrained bank (#1159 §3, #1376) -----------------------
#
# Six of the templates above are single tokens that are also common plural
# nouns or non-relational verbs in ordinary prose. On the ingest path the
# input is a git commit body, which is exactly the register where they
# collide: "Two tests that encoded the old behaviour are removed." yields
# ('Two', TESTS, 'that encoded the old behaviour are'), and `ingest_triples`
# then records *both* phrases as beliefs, so one bogus triple becomes two
# fragment beliefs.
#
# Measured over the 400 most recent commit bodies on `github/main`, the full
# bank fires 82 times and **none of the 82 is a genuine relation**. The six
# templates below account for 67 of them (`tests` 51, `covers` 12,
# `replaces` 3, `follows` 1; `supports` and `extends` never fire at all).
# `benchmarks/triple_precision_1376.py` re-derives that table.
#
# Each dropped template keeps a sibling that cannot collide with a plural
# noun, so the relation stays reachable: `supersedes` for `replaces`,
# `is derived from` / `is based on` for `extends`, `comes after` / `is after`
# / `succeeds` for `follows`, `is supported by` for `supports`. TESTS lost
# both of its single-token forms, so the two passive multi-token forms #1376
# names as the obvious replacement are added here — ingest-side only.
#
# #1874: the verb's case is exact here, because 17 of 40 stored edges came
# from an uppercase edge-type name used as a noun ("the SUPERSEDES writer").
_INGEST_EXCLUDED_TEMPLATES: Final[frozenset[str]] = frozenset({
    "supports",
    "replaces",
    "extends",
    "follows",
    "tests",
    "covers",
})

_INGEST_ADDITIONAL_PATTERNS: Final[tuple[_RelationPattern, ...]] = (
    _build_pattern("is tested by", EDGE_TESTS, swap=True, exact_verb_case=True),
    _build_pattern("is covered by", EDGE_TESTS, swap=True, exact_verb_case=True),
)

_INGEST_PATTERNS: Final[tuple[_RelationPattern, ...]] = tuple(
    _build_pattern(p.template, p.edge_type, swap=p.swap, exact_verb_case=True)
    for p in _PATTERNS
    if p.template not in _INGEST_EXCLUDED_TEMPLATES
) + _INGEST_ADDITIONAL_PATTERNS

# #1874: a phrase made only of these words names nothing, and it has one belief
# id across every commit, so it would collect edges. The ingest path stores
# nothing for a match with such a phrase on either side. The list holds closed
# classes only. A determiner plus a content word, such as "the old", stays:
# telling it from "the parser" takes a grammar check, which this module doesn't
# do. `_is_function_word` decides how a word in a phrase is looked up.
_INGEST_FUNCTION_WORDS: Final[frozenset[str]] = frozenset({
    # The determiners in `np_pattern._DET`, which the noun phrase admits.
    "the", "a", "an", "our", "their", "its", "this", "that", "these", "those",
    "my", "your", "his", "her",
    # Personal pronouns, which stand for something the phrase doesn't name.
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    # Reflexive and possessive pronouns, for the same reason.
    "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "yourselves", "themselves", "mine", "yours", "hers", "ours", "theirs",
    # Relative and interrogative words, which open a clause the match cut off.
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    # Quantifiers, which count something the phrase doesn't name.
    "all", "both", "each", "every", "either", "neither", "some", "any", "none",
    "many", "most", "few", "several", "another", "such", "other", "others",
    "one", "more", "less", "much",
    # Indefinite pronouns, which name no particular thing.
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone",
    # Conjunctions, which join phrases the match cut apart.
    "and", "or", "but", "nor", "yet", "so", "for", "if", "unless", "because",
    "although", "though", "while", "whereas", "whether", "than", "as", "since",
    "until", "till", "once",
    # Prepositions, which need an object the match cut off.
    "of", "to", "in", "on", "at", "by", "with", "from", "about", "into", "onto",
    "over", "under", "after", "before", "between", "among", "through", "during",
    "without", "within", "against", "above", "below", "across", "along",
    "around", "behind", "beside", "besides", "beyond", "despite", "except",
    "inside", "outside", "near", "off", "out", "up", "down", "per", "via",
    "toward", "towards", "upon", "unlike", "versus",
    # Auxiliaries and modals, which need a main verb the match cut off. The
    # stems are what's left of a contraction when a phrase ends at its
    # apostrophe, as in "doesn't".
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "has", "have", "had", "having", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "ought", "isn", "aren", "wasn",
    "weren", "don", "doesn", "didn", "hasn", "haven", "hadn", "won", "wouldn",
    "shouldn", "couldn", "mustn", "shan", "ain",
    # Negators, which need something to negate.
    "not", "no", "never", "cannot",
    # Closed-class adverbs, which qualify something the phrase doesn't name.
    "also", "then", "just", "only", "even", "still", "too", "very", "now",
    "here", "there", "thus", "hence", "however", "therefore", "instead",
    "already", "again", "ever", "rather", "quite", "perhaps", "maybe",
})


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


def _line_spans(text: str) -> list[tuple[int, int]]:
    """`(start, end)` of each line of `text`, terminator included.

    `str.splitlines` breaks on every line separator `\\s` can match, so a
    match inside one span can't join two lines. The offsets index `text`,
    which keeps a match's anchor context free to read past its line.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end))
        start = end
    return spans


def _is_function_word(word: str) -> bool:
    """True when `word` counts as a listed function word (#1874).

    A trailing `-` or `_` is a fragment of the same word, so `it-` counts as
    `it`. A hyphen or underscore inside a word leaves one word, so `one-shot`
    and `all_of` don't count. A word of two or more letters in all capitals
    is a name, such as `IT` or `WHO`. `A` and `I` still count, because they
    read the same at the start of a sentence.
    """
    bare = word.rstrip("-_")
    if len(bare) > 1 and bare.isupper():
        return False
    return bare.lower() in _INGEST_FUNCTION_WORDS


def _has_function_word_endpoint(subject: str, obj: str) -> bool:
    """True when a phrase on either side is only function words (#1874)."""
    return any(
        all(_is_function_word(word) for word in phrase.split())
        for phrase in (subject, obj)
    )


def extract_triples(
    text: str, *, constrain_collision_verbs: bool = False
) -> list[Triple]:
    """Extract (subject, relation, object) triples from `text`.

    Returns the triples in left-to-right order of their match start
    in the input. Overlapping matches from different patterns are
    each reported (downstream `ingest_triples` dedups by edge tuple).
    Empty / non-relational input returns an empty list.

    `constrain_collision_verbs` selects the ingest-constrained bank
    (`_INGEST_PATTERNS`): the six single-token templates that double as
    plural nouns are dropped and TESTS' two passive forms are added. The
    verb's case is exact, each match stays on one line of `text`, and a
    match with a phrase that is only function words on either side is dropped
    (#1874). It defaults to False so the read path is unchanged: operator
    ruling of 2026-08-06 scopes #1376 to the write path, and `rebuild_log`
    calls this on every prompt with behaviour that was never measured. Only
    `hook_commit_ingest` passes True.
    """
    if not text:
        return []
    if constrain_collision_verbs:
        patterns, spans = _INGEST_PATTERNS, _line_spans(text)
    else:
        patterns, spans = _PATTERNS, [(0, len(text))]
    triples: list[Triple] = []
    for pat in patterns:
        for span_start, span_end in spans:
            for m in pat.regex.finditer(text, span_start, span_end):
                subj_raw = m.group("subject")
                obj_raw = m.group("object")
                subj = _normalize_phrase(subj_raw)
                obj = _normalize_phrase(obj_raw)
                if not subj or not obj:
                    continue
                if constrain_collision_verbs and _has_function_word_endpoint(
                    subj, obj,
                ):
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
            classifier_version=INGEST_CLASSIFIER_VERSION,
            rule_set_hash=rule_set_hash(),
        )
        obj_log = store.record_ingest(
            source_kind=INGEST_SOURCE_GIT,
            raw_text=triple.object,
            session_id=session_id,
            ts=ts,
            raw_meta={"call_site": CORROBORATION_SOURCE_COMMIT_INGEST},
            classifier_version=INGEST_CLASSIFIER_VERSION,
            rule_set_hash=rule_set_hash(),
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
