"""Semantic relationship detection over belief pairs (#201).

Stdlib-only detector that classifies near-paraphrase belief pairs by
**negation/quantifier signal divergence**. Complements
`triple_extractor.py` (which catches explicit-prose CONTRADICTS like
"X contradicts Y") and fills the gap that `contradiction.py`'s
docstring acknowledges: the v1.x regex classifier rarely produces
CONTRADICTS edges, so semantic disagreement on near-paraphrase pairs
goes undetected.

This module is the **algorithm**. The audit-only CLI surface
(`aelf doctor relationships`) lives in `cli.py`. The write-path
hook is bench-gated per the #201 ratification and ships behind the
shared candidate-pair pipeline once dedup's #197 corpus benchmark
clears.

## Verdict shape

For each candidate pair the detector emits one of four verdicts:

* `unrelated` — token overlap below the candidate threshold or
  signals do not disagree.
* `contradicts` — high token overlap on the substantive content
  axis AND modality or quantifier signals diverge. Confidence
  scales with overlap and signal strength.
* `refines` — high overlap, signals agree, and one belief is a
  strict token-subset of the other (the longer belief refines the
  shorter). Distinguished from `supersedes` (the dedup module's
  domain) by lack of an explicit replacement signal.
* `supersedes` — reserved for the dedup pass (#197); this module
  does not emit `supersedes` directly.

## Modality + quantifier vocabularies

Per the #201 ratification: stdlib-only, no LLM, no embedding. The
vocabularies are deliberately narrow — false-positive cost on this
audit pass is operator review time, false-negative cost is one
missed contradiction surfaced later by `aelf resolve`.

* **Negation**: `not / no / never / cannot / can't / don't / doesn't
  / didn't / won't / wouldn't / shouldn't / couldn't / isn't / aren't
  / wasn't / weren't / hasn't / haven't / hadn't`. Catches the most
  common negation forms in imperative and declarative prose.
* **Hedge**: `might / may / could / perhaps / maybe / possibly /
  probably / likely`. Soft uncertainty markers; presence weakens
  certainty.
* **Certainty**: `must / always / definitely / certainly / absolutely
  / indeed`. Strong assertion markers.
* **Quantifiers** (axis from -1.0 = never to +1.0 = always): `never
  -> -1.0`, `rarely / seldom -> -0.5`, `sometimes / occasionally ->
  0.0`, `often / usually -> +0.5`, `always -> +1.0`. Compound forms
  (`almost never`, `in most cases`) are deferred per ratification —
  the bench will tell us whether they're worth the regex surface.

## Subject-extraction granularity

The naive overlap test produces false positives on "different aspects
of same subject" pairs (e.g. *"use uv for Python deps"* + *"never use
pip in this project"* both reference Python tooling but aren't
contradicting each other). The detector mitigates this by computing
overlap on **content tokens only** — modality, hedge, certainty,
quantifier, and stopword tokens are subtracted from both sides before
the agreement score. A pair clears the contradiction gate only when
the residual content tokens still overlap heavily AND the signal axes
disagree. The `aelf resolve` surface is the safety net, not the
primary defense — but the residual-content check buys most of the
margin against the obvious false-positive shape.
"""
from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, IO, cast

from aelfrice.dedup import (
    DEFAULT_JACCARD_MIN,
    DEFAULT_MAX_CANDIDATE_PAIRS,
    _jaccard_prefiltered_pairs,
    jaccard,
)
from aelfrice.store import MemoryStore

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
RELATIONSHIP_SECTION: Final[str] = "relationship_detector"
JACCARD_MIN_KEY: Final[str] = "jaccard_min"
CONFIDENCE_MIN_KEY: Final[str] = "confidence_min"
MAX_CANDIDATE_PAIRS_KEY: Final[str] = "max_candidate_pairs"

# --- Defaults (research-line + #201 ratification) ----------------------

DEFAULT_CONFIDENCE_MIN: Final[float] = 0.85
"""High-confidence threshold for write-path emission of CONTRADICTS
edges. Per #201 ratification. The audit surface reports every pair
above the candidate Jaccard but tags rows above this threshold as
`auto_emit` candidates so the write-path flip is mechanical."""

# Reuse dedup's candidate-pair Jaccard prefilter; the relationship
# detector and dedup share the same near-paraphrase candidate pool by
# design (one candidate-generation pass feeds both modules in the
# unified-pass shape ratified by #201).

# --- Verdict types ------------------------------------------------------

VERDICT_UNRELATED: Final[str] = "unrelated"
VERDICT_CONTRADICTS: Final[str] = "contradicts"
VERDICT_REFINES: Final[str] = "refines"
VERDICT_SUPERSEDES: Final[str] = "supersedes"
VERDICTS: Final[frozenset[str]] = frozenset({
    VERDICT_UNRELATED,
    VERDICT_CONTRADICTS,
    VERDICT_REFINES,
    VERDICT_SUPERSEDES,
})

# --- Vocabulary ---------------------------------------------------------

NEGATION_TOKENS: Final[frozenset[str]] = frozenset({
    "not", "no", "never", "cannot",
})
"""Whole-word negation markers. The bm25 tokenizer splits contractions
on the apostrophe (`don't` -> `don` + `t`), so contracted forms are
detected via a separate regex pass rather than the token set —
`_NEGATION_CONTRACTION_RE` catches them on the raw content."""

_NEGATION_CONTRACTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:do|does|did|can|won|would|should|could|is|are|was|were|"
    r"has|have|had)n['’]?t\b",
    re.IGNORECASE,
)
"""Matches `don't`, `cant`, `wouldn't`, `couldnt`, `won't`, etc. Both
straight (`'`) and curly (`’`) apostrophes; the apostrophe itself is
optional so `dont` / `cant` (apostrophe-stripped prose) also matches.
Run on raw content before tokenization."""

HEDGE_TOKENS: Final[frozenset[str]] = frozenset({
    "might", "may", "could",
    "perhaps", "maybe", "possibly",
    "probably", "likely",
})

CERTAINTY_TOKENS: Final[frozenset[str]] = frozenset({
    "must", "always", "definitely",
    "certainly", "absolutely", "indeed",
})

QUANTIFIER_AXIS: Final[dict[str, float]] = {
    "never": -1.0,
    "rarely": -0.5,
    "seldom": -0.5,
    "sometimes": 0.0,
    "occasionally": 0.0,
    "often": 0.5,
    "usually": 0.5,
    "always": 1.0,
}
"""Position on the frequency axis. `always` and `never` are the
strongest contradiction signal; `often` vs `rarely` is medium; `often`
vs `usually` agree. Used additively across all quantifier hits in a
belief, then clamped to [-1.0, 1.0]."""

STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "of", "in", "on", "at", "by", "for", "with", "from", "to", "as",
    "and", "or", "but", "if", "so", "than", "then", "thus",
    "it", "its", "we", "you", "your", "our", "their",
    "i", "me", "my", "he", "she", "they", "them",
})
"""Common stopwords stripped from the residual-content overlap. Kept
narrow — bm25's tokenizer already lowercases and word-splits, this
adds the function-word skim that keeps the residual focused on
substantive content tokens."""

SIGNAL_TOKENS: Final[frozenset[str]] = (
    NEGATION_TOKENS
    | HEDGE_TOKENS
    | CERTAINTY_TOKENS
    | frozenset(QUANTIFIER_AXIS.keys())
)
"""Union of all signal vocabularies — these tokens are subtracted from
both sides before the residual-content overlap check, so the
agreement score reflects substantive prose rather than the modality
itself."""

# --- Signal extraction --------------------------------------------------


@dataclass(frozen=True)
class ModalitySignals:
    """Aggregated modality + quantifier signals for one belief.

    `negation` is True when any negation token appears (one is enough;
    a single `not` flips the polarity of an assertion). `hedge_count`
    and `certainty_count` are raw counts; relative magnitude matters
    more than absolute. `quantifier_axis` is the clamped sum of all
    quantifier-axis hits when `has_quantifier` is True; otherwise the
    belief carries no quantifier information and the pair contributes
    no quantifier-axis disagreement (a missing quantifier is not the
    same as a neutral quantifier — "use uv" is silent on frequency,
    "sometimes use uv" is a frequency claim).
    """
    negation: bool
    hedge_count: int
    certainty_count: int
    has_quantifier: bool
    quantifier_axis: float


def _extract_modality(content: str, tokens: frozenset[str]) -> ModalitySignals:
    """Compute modality signals from a belief.

    `tokens` is the same frozenset the dedup candidate-prefilter uses.
    `content` is the raw belief text — needed because the bm25
    tokenizer splits contractions on the apostrophe, so contracted
    negations (`don't`, `wouldn't`, ...) are caught by a separate
    regex pass over the raw content rather than via the token set.
    """
    negation = (
        bool(tokens & NEGATION_TOKENS)
        or _NEGATION_CONTRACTION_RE.search(content) is not None
    )
    hedge_count = sum(1 for t in tokens if t in HEDGE_TOKENS)
    certainty_count = sum(1 for t in tokens if t in CERTAINTY_TOKENS)
    axis = 0.0
    has_quantifier = False
    for t in tokens:
        if t in QUANTIFIER_AXIS:
            axis += QUANTIFIER_AXIS[t]
            has_quantifier = True
    if axis > 1.0:
        axis = 1.0
    elif axis < -1.0:
        axis = -1.0
    return ModalitySignals(
        negation=negation,
        hedge_count=hedge_count,
        certainty_count=certainty_count,
        has_quantifier=has_quantifier,
        quantifier_axis=axis,
    )


def _residual_content_jaccard(
    tokens_a: frozenset[str],
    tokens_b: frozenset[str],
) -> float:
    """Jaccard over content tokens only (signal + stopword tokens removed).

    Used as the agreement score: high residual overlap means the two
    beliefs are talking about the *same content*, so any modality
    disagreement is a real contradiction signal rather than two
    different assertions about an overlapping subject.
    """
    drop = SIGNAL_TOKENS | STOPWORDS
    residual_a = tokens_a - drop
    residual_b = tokens_b - drop
    return jaccard(residual_a, residual_b)


# --- Verdict logic ------------------------------------------------------


@dataclass(frozen=True)
class RelationshipPair:
    """One classified pair of beliefs.

    `belief_a_id` is the lexicographically smaller id; `belief_b_id`
    the larger. Pair ordering is deterministic so two runs over the
    same store produce the same pair list.

    `verdict` is one of `VERDICTS`. `confidence` is in [0.0, 1.0].
    `auto_emit` is True when `confidence >= confidence_min` — i.e.
    when this row would clear the gate the future write-path hook
    will use (audit-only ship; no edges are inserted).
    """
    belief_a_id: str
    belief_b_id: str
    verdict: str
    confidence: float
    jaccard_score: float
    residual_score: float
    auto_emit: bool


def _signal_disagreement(a: ModalitySignals, b: ModalitySignals) -> float:
    """Compute the disagreement magnitude between two signal sets.

    Returns a value in [0.0, 1.0]:
      - Pure negation flip (one negated, the other not) -> 1.0.
      - Quantifier-axis distance contributes its absolute delta /
        2.0 (so always↔never = 1.0, often↔rarely = 0.5).
      - Hedge-vs-certainty asymmetry contributes a smaller term: if
        one belief hedges (hedge_count > 0, certainty_count == 0) and
        the other asserts (certainty_count > 0, hedge_count == 0),
        that's 0.5 toward disagreement.

    The three terms are aggregated with `max` rather than sum, on the
    grounds that a single strong signal disagreement is a sufficient
    contradiction marker; piling them on doesn't make a pair more
    contradictory.
    """
    if a.negation != b.negation:
        neg_term = 1.0
    else:
        neg_term = 0.0
    if a.has_quantifier and b.has_quantifier:
        q_term = abs(a.quantifier_axis - b.quantifier_axis) / 2.0
    else:
        q_term = 0.0
    a_hedges = a.hedge_count > 0 and a.certainty_count == 0
    b_hedges = b.hedge_count > 0 and b.certainty_count == 0
    a_certain = a.certainty_count > 0 and a.hedge_count == 0
    b_certain = b.certainty_count > 0 and b.hedge_count == 0
    if (a_hedges and b_certain) or (a_certain and b_hedges):
        m_term = 0.5
    else:
        m_term = 0.0
    return max(neg_term, q_term, m_term)


def classify_pair(
    *,
    content_a: str,
    content_b: str,
    tokens_a: frozenset[str],
    tokens_b: frozenset[str],
    jaccard_score: float,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
) -> RelationshipPair:
    """Classify a single candidate belief pair.

    Caller passes pre-tokenized sets (from the dedup candidate-pair
    pipeline) to avoid re-tokenizing. The verdict is derived from:

    1. The residual-content Jaccard (subject-extraction granularity).
    2. The signal-disagreement magnitude between the two beliefs.
    3. Token-subset relationship (for `refines` verdict).

    `confidence_min` is the auto-emit threshold; the returned
    `auto_emit` flag is purely informational on the audit surface.
    """
    residual = _residual_content_jaccard(tokens_a, tokens_b)
    a_signals = _extract_modality(content_a, tokens_a)
    b_signals = _extract_modality(content_b, tokens_b)
    disagreement = _signal_disagreement(a_signals, b_signals)

    if disagreement >= 0.5 and residual >= 0.5:
        confidence = min(1.0, residual * 0.6 + disagreement * 0.4)
        verdict = VERDICT_CONTRADICTS
    elif residual >= 0.7 and disagreement < 0.25:
        if tokens_a < tokens_b or tokens_b < tokens_a:
            verdict = VERDICT_REFINES
            confidence = residual
        else:
            verdict = VERDICT_UNRELATED
            confidence = 0.0
    else:
        verdict = VERDICT_UNRELATED
        confidence = 0.0

    return RelationshipPair(
        belief_a_id="",
        belief_b_id="",
        verdict=verdict,
        confidence=confidence,
        jaccard_score=jaccard_score,
        residual_score=residual,
        auto_emit=verdict == VERDICT_CONTRADICTS and confidence >= confidence_min,
    )


# --- Audit pass ---------------------------------------------------------


@dataclass
class RelationshipAuditReport:
    """Summary of one audit pass over the store.

    `pairs` lists every pair whose verdict is not `unrelated`, sorted
    by descending confidence then `(belief_a_id, belief_b_id)`. Counts
    include `n_candidate_pairs` (raw O(n^2) visited),
    `n_jaccard_above_threshold` (cleared the candidate Jaccard prefilter),
    and per-verdict counts. `truncated` is `True` when the candidate
    pool exceeded `max_candidate_pairs`.
    """
    n_beliefs_scanned: int
    n_candidate_pairs: int
    n_jaccard_above_threshold: int
    n_contradicts: int
    n_refines: int
    n_auto_emit: int
    truncated: bool
    pairs: tuple[RelationshipPair, ...] = field(default_factory=tuple)


def detect_relationships(
    store: MemoryStore,
    *,
    jaccard_min: float = DEFAULT_JACCARD_MIN,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> RelationshipAuditReport:
    """Walk the store, classify candidate pairs, return a report.

    Read-only: no edges are inserted, no beliefs are mutated. The
    write-path hook (insert CONTRADICTS edges for `auto_emit` rows
    and POTENTIALLY_STALE edges for sub-threshold rows) is the
    bench-gated R2 deferred behind the shared dedup corpus benchmark.

    Reuses dedup's `_jaccard_prefiltered_pairs` so the candidate-pair
    pool matches the dedup module exactly. This is the unified-pass
    shape ratified by #201: one prefilter feeds both detectors.
    """
    if not 0.0 <= jaccard_min <= 1.0:
        raise ValueError(
            f"jaccard_min must be in [0.0, 1.0], got {jaccard_min}",
        )
    if not 0.0 <= confidence_min <= 1.0:
        raise ValueError(
            f"confidence_min must be in [0.0, 1.0], got {confidence_min}",
        )
    if max_candidate_pairs < 1:
        raise ValueError(
            f"max_candidate_pairs must be >= 1, got {max_candidate_pairs}",
        )

    beliefs = store.list_beliefs_for_indexing()
    n_beliefs = len(beliefs)
    if n_beliefs < 2:
        return RelationshipAuditReport(
            n_beliefs_scanned=n_beliefs,
            n_candidate_pairs=0,
            n_jaccard_above_threshold=0,
            n_contradicts=0,
            n_refines=0,
            n_auto_emit=0,
            truncated=False,
        )

    candidates, raw_count, truncated = _jaccard_prefiltered_pairs(
        beliefs,
        jaccard_min=jaccard_min,
        max_pairs=max_candidate_pairs,
    )

    pairs: list[RelationshipPair] = []
    for id_a, content_a, id_b, content_b, ta, tb, j_score in candidates:
        partial = classify_pair(
            content_a=content_a,
            content_b=content_b,
            tokens_a=ta,
            tokens_b=tb,
            jaccard_score=j_score,
            confidence_min=confidence_min,
        )
        if partial.verdict == VERDICT_UNRELATED:
            continue
        pairs.append(
            RelationshipPair(
                belief_a_id=id_a,
                belief_b_id=id_b,
                verdict=partial.verdict,
                confidence=partial.confidence,
                jaccard_score=partial.jaccard_score,
                residual_score=partial.residual_score,
                auto_emit=partial.auto_emit,
            )
        )
    pairs.sort(
        key=lambda p: (-p.confidence, p.belief_a_id, p.belief_b_id),
    )

    n_contradicts = sum(1 for p in pairs if p.verdict == VERDICT_CONTRADICTS)
    n_refines = sum(1 for p in pairs if p.verdict == VERDICT_REFINES)
    n_auto = sum(1 for p in pairs if p.auto_emit)
    return RelationshipAuditReport(
        n_beliefs_scanned=n_beliefs,
        n_candidate_pairs=raw_count,
        n_jaccard_above_threshold=len(candidates),
        n_contradicts=n_contradicts,
        n_refines=n_refines,
        n_auto_emit=n_auto,
        truncated=truncated,
        pairs=tuple(pairs),
    )


# --- Config loader ------------------------------------------------------


@dataclass(frozen=True)
class RelationshipDetectorConfig:
    """Resolved `[relationship_detector]` section of `.aelfrice.toml`.

    All fields default to module-level constants; any may be overridden
    by a project-local `.aelfrice.toml`. Malformed values fall back to
    the default with a stderr trace, matching the `[dedup]` /
    `[implicit_feedback]` resolution conventions.
    """
    jaccard_min: float = DEFAULT_JACCARD_MIN
    confidence_min: float = DEFAULT_CONFIDENCE_MIN
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS


def _warn(serr: IO[str], msg: str) -> None:
    print(f"aelfrice relationship_detector: {msg}", file=serr)


def _load_unit_float(
    section: dict[str, Any],
    key: str,
    default: float,
    candidate: Path,
    serr: IO[str],
) -> float:
    obj: Any = section.get(key, default)
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        _warn(
            serr,
            f"ignoring [{RELATIONSHIP_SECTION}] {key} in {candidate} "
            f"(expected float in [0.0, 1.0])",
        )
        return default
    val = float(obj)
    if not 0.0 <= val <= 1.0:
        _warn(
            serr,
            f"ignoring [{RELATIONSHIP_SECTION}] {key} in {candidate} "
            f"(expected float in [0.0, 1.0])",
        )
        return default
    return val


def load_relationship_config(
    start: Path | None = None,
) -> RelationshipDetectorConfig:
    """Walk up from `start` looking for `.aelfrice.toml`.

    Returns the resolved `[relationship_detector]` config. Missing
    file / missing section / malformed TOML / wrong-typed values all
    degrade to defaults with a stderr trace; never raises.
    """
    serr: IO[str] = sys.stderr
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            try:
                raw = candidate.read_bytes()
            except OSError as exc:
                _warn(serr, f"cannot read {candidate}: {exc}")
                return RelationshipDetectorConfig()
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except tomllib.TOMLDecodeError as exc:
                _warn(serr, f"malformed TOML in {candidate}: {exc}")
                return RelationshipDetectorConfig()
            section_obj: Any = parsed.get(RELATIONSHIP_SECTION, {})
            if not isinstance(section_obj, dict):
                return RelationshipDetectorConfig()
            section = cast(dict[str, Any], section_obj)
            j_min = _load_unit_float(
                section, JACCARD_MIN_KEY, DEFAULT_JACCARD_MIN,
                candidate, serr,
            )
            c_min = _load_unit_float(
                section, CONFIDENCE_MIN_KEY, DEFAULT_CONFIDENCE_MIN,
                candidate, serr,
            )
            mp_obj: Any = section.get(
                MAX_CANDIDATE_PAIRS_KEY, DEFAULT_MAX_CANDIDATE_PAIRS,
            )
            if (
                isinstance(mp_obj, bool)
                or not isinstance(mp_obj, int)
                or mp_obj < 1
            ):
                _warn(
                    serr,
                    f"ignoring [{RELATIONSHIP_SECTION}] "
                    f"{MAX_CANDIDATE_PAIRS_KEY} in {candidate} "
                    f"(expected positive int)",
                )
                mp_resolved = DEFAULT_MAX_CANDIDATE_PAIRS
            else:
                mp_resolved = mp_obj
            return RelationshipDetectorConfig(
                jaccard_min=j_min,
                confidence_min=c_min,
                max_candidate_pairs=mp_resolved,
            )
        if current.parent == current:
            break
        current = current.parent
    return RelationshipDetectorConfig()


# --- Audit formatter ----------------------------------------------------


def format_audit_report(report: RelationshipAuditReport) -> str:
    """Render a `RelationshipAuditReport` as plain text.

    Used by `aelf doctor relationships`. The shape mirrors
    `format_audit_report` in `dedup.py` so the doctor surface stays
    consistent.
    """
    lines: list[str] = []
    lines.append("aelf doctor relationships")
    lines.append("=" * 40)
    lines.append(f"Beliefs scanned             : {report.n_beliefs_scanned}")
    lines.append(f"Candidate pairs visited     : {report.n_candidate_pairs}")
    lines.append(
        f"Above Jaccard threshold     : {report.n_jaccard_above_threshold}"
    )
    if report.truncated:
        lines.append(
            f"  (truncated to {DEFAULT_MAX_CANDIDATE_PAIRS} — see "
            f"[relationship_detector] max_candidate_pairs)"
        )
    lines.append(f"Contradicts                 : {report.n_contradicts}")
    lines.append(f"Refines                     : {report.n_refines}")
    lines.append(
        f"Auto-emit (>= confidence_min): {report.n_auto_emit}"
    )
    lines.append("")
    if not report.pairs:
        lines.append(
            "No semantic relationships above the configured thresholds."
        )
        return "\n".join(lines)
    lines.append("Top pairs (verdict, confidence, jaccard, residual):")
    for p in report.pairs[:25]:
        marker = "*" if p.auto_emit else " "
        lines.append(
            f"  {marker} {p.verdict:<12}  c={p.confidence:.3f}  "
            f"j={p.jaccard_score:.3f}  r={p.residual_score:.3f}  "
            f"{p.belief_a_id}  ~  {p.belief_b_id}"
        )
    if len(report.pairs) > 25:
        lines.append(f"  ... ({len(report.pairs) - 25} more)")
    lines.append("")
    lines.append(
        "Audit-only: no CONTRADICTS or POTENTIALLY_STALE edges have "
        "been written. The write-path hook is bench-gated per #201 "
        "ratification."
    )
    return "\n".join(lines)
