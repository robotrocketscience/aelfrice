"""Semantic relationship detector (#201) — audit-only ship.

Stdlib-only port of the research-line `relationship_detector.py`.
For two beliefs that share enough lexical overlap to be plausibly
about the same subject, classify the relationship by **modality**
(negation) + **quantifier axis** (universal / frequent / occasional
/ rare / negative) signal divergence.

Output verdict is one of:

* ``contradicts`` — agreeing residual content tokens (the
  subject/predicate of the belief) but disagreeing modality. Either
  one side negates and the other affirms, or both have quantifiers
  whose axes are far apart (e.g. universal vs negative).
* ``refines`` — agreeing residual content tokens *and* agreeing
  modality. One belief sharpens or scopes the other.
* ``unrelated`` — residual-content overlap below the relatedness
  floor.

This module is the **detector**. The audit-only CLI surface
(``aelf doctor relationships``) lives in ``cli.py``; the write-path
hook that would actually insert ``CONTRADICTS`` edges and the new
``POTENTIALLY_STALE`` edge type for sub-confidence pairs are the
bench-gated R2 deferred behind the corpus benchmark per the #201
ratification. ``SUPERSEDES`` is intentionally not produced here —
that verdict is the dedup module's job (``aelf doctor dedup``).

Detection is regex / token-set heuristics — no LLM, no embedding.

## Design notes

* **Quantifier "missing" is not "neutral".** ``has_quantifier`` is
  tracked separately from ``quantifier_axis``. Without this guard,
  ``"use uv"`` (no quantifier) vs ``"always use uv"`` (axis = +1.0)
  trips the contradiction gate at axis distance 1.0; the first
  belief is silent on frequency, not asserting a neutral one.
* **Negation contractions match against raw content, not tokens.**
  The bm25 tokenizer splits ``don't`` into ``["don", "t"]`` so
  contracted negations are unreachable from the cached frozenset. A
  separate regex on the raw content bridges the gap; both ``'`` and
  ``’`` and apostrophe-stripped forms (``dont``, ``cant``) match.
* **Residual-content overlap (subject-extraction granularity).**
  Agreement is measured on tokens after subtracting modality,
  quantifier, and stopword tokens. Mitigates the spec's flagged
  false-positive shape (``"use uv for python deps"`` vs
  ``"never use pip in this project"`` overlap on ``use`` but not on
  any substantive content; verdict stays ``unrelated``).
"""
from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, IO, cast

from aelfrice.bm25 import tokenize
from aelfrice.dedup import _jaccard_prefiltered_pairs, jaccard
from aelfrice.store import MemoryStore

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
SECTION: Final[str] = "relationship_detector"
JACCARD_KEY: Final[str] = "jaccard_min"
CONFIDENCE_KEY: Final[str] = "confidence_min"
MAX_CANDIDATE_PAIRS_KEY: Final[str] = "max_candidate_pairs"

# --- Verdict labels ----------------------------------------------------

LABEL_CONTRADICTS: Final[str] = "contradicts"
LABEL_REFINES: Final[str] = "refines"
LABEL_UNRELATED: Final[str] = "unrelated"

VERDICT_LABELS: Final[frozenset[str]] = frozenset(
    {LABEL_CONTRADICTS, LABEL_REFINES, LABEL_UNRELATED}
)

# --- Defaults (research-line + #201 ratification) ----------------------

# Pair must share this much Jaccard token overlap to enter the
# classifier. Same shape as dedup's prefilter so the unified-pass can
# reuse the candidate pool.
DEFAULT_JACCARD_MIN: Final[float] = 0.4
# Minimum residual-content Jaccard for the pair to be considered
# plausibly about the same subject. Below this, verdict is
# ``unrelated`` regardless of modality signals.
DEFAULT_RESIDUAL_OVERLAP_MIN: Final[float] = 0.4
# Auto-flag ``contradicts`` only at score >= this floor. Audit
# reports both sides; only the high-confidence half is eligible for
# the deferred write-path hook.
DEFAULT_CONFIDENCE_MIN: Final[float] = 0.5
DEFAULT_MAX_CANDIDATE_PAIRS: Final[int] = 5000

# --- Modality vocabulary -----------------------------------------------

# Token-level negation markers (post-tokenize, post-stopword). We
# also do a separate regex pass on the raw content for contractions.
_NEGATION_TOKENS: Final[frozenset[str]] = frozenset({
    "not", "no", "never", "none", "nothing", "nobody", "nowhere",
    "neither", "nor", "without", "avoid", "refuse", "prohibit",
    "prohibited", "disallow", "disallowed", "forbid", "forbidden",
    "deny", "denied",
})

# Negation contractions: matched on the raw content. Every English
# negation contraction ends in ``n't`` after a stem (``do``, ``wo``,
# ``ca``, ``shouldn``, ``ai``, …), so an apostrophe-form pattern is
# all we need. ``'`` and ``’`` both match. The apostrophe-stripped
# form is enumerated separately to catch typos and naive transcripts.
_CONTRACTION_NEGATION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\w+n['’]t\b|"
    r"\b(?:dont|doesnt|didnt|isnt|arent|wasnt|werent|hasnt|havent|hadnt|"
    r"shouldnt|wouldnt|couldnt|mustnt|mightnt|wont|cant|shant|aint)\b",
    re.IGNORECASE,
)

# --- Quantifier axis ---------------------------------------------------
#
# The axis is a real-valued frequency claim. Two quantifiers conflict
# when their axes are far apart (e.g. universal vs negative). A
# missing quantifier is *not* a neutral 0.0 — see Signals.has_quantifier.

QUANT_AXIS: Final[dict[str, float]] = {
    # universal
    "always": 1.0,
    "every": 1.0,
    "all": 1.0,
    "must": 1.0,
    # frequent
    "usually": 0.6,
    "often": 0.6,
    "mostly": 0.6,
    "frequently": 0.6,
    # occasional
    "sometimes": 0.0,
    "occasionally": 0.0,
    # rare
    "rarely": -0.6,
    "seldom": -0.6,
    # negative
    "never": -1.0,
}

_QUANTIFIER_TOKENS: Final[frozenset[str]] = frozenset(QUANT_AXIS)

# --- Stopwords ---------------------------------------------------------
#
# Excluded from the residual-content overlap check. Modality and
# quantifier tokens are ALSO subtracted so the residual is the
# subject/predicate of the belief.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "with", "by", "and",
    "or", "but", "if", "then", "than", "as", "at", "from", "this",
    "that", "these", "those", "it", "its", "i", "you", "we", "they",
    "he", "she", "him", "her", "them", "us", "me", "my", "your",
    "our", "their", "his", "hers", "do", "does", "did", "have",
    "has", "had", "will", "would", "should", "could", "can",
    "into", "about", "over", "under", "out", "up", "down", "use",
    "used", "using", "uses", "any", "some", "t", "s", "don", "won",
    "doesn", "didn", "isn", "aren", "wasn", "weren", "shouldn",
    "wouldn", "couldn", "hasn", "haven", "hadn", "cant", "wont",
    "dont", "isnt",
} | _NEGATION_TOKENS | _QUANTIFIER_TOKENS)


# --- Signal extraction -------------------------------------------------


@dataclass(frozen=True)
class Signals:
    """Modality + quantifier signals extracted from one belief."""
    tokens: tuple[str, ...]
    residual_content: frozenset[str]
    has_negation: bool
    has_quantifier: bool
    quantifier_axis: float


def _has_contraction_negation(text: str) -> bool:
    """Detect negation contractions on the raw content.

    The bm25 tokenizer drops apostrophes, so ``don't`` becomes
    ``("don", "t")`` and is unreachable from the cached token set.
    """
    return bool(_CONTRACTION_NEGATION_RE.search(text))


def extract_signals(text: str) -> Signals:
    """Tokenise + classify modality and quantifier signals."""
    toks = tokenize(text)
    tok_set = set(toks)
    has_negation = bool(tok_set & _NEGATION_TOKENS) or _has_contraction_negation(text)
    quantifier_token: str | None = None
    for t in toks:
        if t in QUANT_AXIS:
            quantifier_token = t
            break
    has_quantifier = quantifier_token is not None
    quantifier_axis = QUANT_AXIS[quantifier_token] if quantifier_token else 0.0
    residual = frozenset(t for t in tok_set if t not in _STOPWORDS)
    return Signals(
        tokens=tuple(toks),
        residual_content=residual,
        has_negation=has_negation,
        has_quantifier=has_quantifier,
        quantifier_axis=quantifier_axis,
    )


# --- Verdict classifier ------------------------------------------------


@dataclass(frozen=True)
class RelationshipVerdict:
    """Verdict + score for one belief pair."""
    label: str
    score: float
    residual_overlap: float
    rationale: str


def _residual_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def analyze(
    text_a: str,
    text_b: str,
    *,
    residual_overlap_min: float = DEFAULT_RESIDUAL_OVERLAP_MIN,
) -> RelationshipVerdict:
    """Classify the relationship between two belief texts.

    Pure function over the two strings. Read by the audit pass and by
    the bench gate at ``relationship_detector.classify``.
    """
    sa = extract_signals(text_a)
    sb = extract_signals(text_b)
    overlap = _residual_jaccard(sa.residual_content, sb.residual_content)
    if overlap < residual_overlap_min:
        return RelationshipVerdict(
            label=LABEL_UNRELATED,
            score=0.0,
            residual_overlap=overlap,
            rationale="residual_overlap_below_floor",
        )

    n_term = 1.0 if sa.has_negation != sb.has_negation else 0.0
    if sa.has_quantifier and sb.has_quantifier:
        # Axis distance / 2 lands in [0.0, 1.0].
        q_term = abs(sa.quantifier_axis - sb.quantifier_axis) / 2.0
    else:
        # Missing quantifier is silent, not neutral.
        q_term = 0.0
    score = (n_term + q_term) / 2.0
    if score > 0.0:
        rationale = (
            "negation_and_quantifier_disagree"
            if n_term and q_term
            else "negation_disagrees"
            if n_term
            else "quantifier_axes_disagree"
        )
        return RelationshipVerdict(
            label=LABEL_CONTRADICTS,
            score=round(score, 3),
            residual_overlap=overlap,
            rationale=rationale,
        )
    return RelationshipVerdict(
        label=LABEL_REFINES,
        score=0.0,
        residual_overlap=overlap,
        rationale="modality_agrees",
    )


def classify(text_a: str, text_b: str) -> str:
    """Bench-gate entry point: return the verdict label only."""
    return analyze(text_a, text_b).label


# --- Audit report types ------------------------------------------------


@dataclass(frozen=True)
class RelationshipPair:
    """Pair-level result emitted by the audit."""
    belief_a_id: str
    belief_b_id: str
    label: str
    score: float
    residual_overlap: float
    rationale: str


@dataclass
class RelationshipsAuditReport:
    """Summary of one audit pass over the store."""
    n_beliefs_scanned: int
    n_candidate_pairs: int
    n_contradicts_high: int
    n_contradicts_low: int
    n_refines: int
    truncated: bool
    pairs: tuple[RelationshipPair, ...] = field(default_factory=tuple)


# --- Top-level audit entry point ---------------------------------------


def relationships_audit(
    store: MemoryStore,
    *,
    jaccard_min: float = DEFAULT_JACCARD_MIN,
    residual_overlap_min: float = DEFAULT_RESIDUAL_OVERLAP_MIN,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> RelationshipsAuditReport:
    """Walk the store, classify near-pair relationships, return a report.

    Read-only: no edges are inserted, no beliefs are mutated. Reuses
    dedup's Jaccard prefilter (``_jaccard_prefiltered_pairs``) for
    candidate-pair generation — the unified-pass shape from the
    ratification comment is achieved by sharing the prefilter, not
    by sharing the full classifier.

    ``unrelated`` verdicts are dropped from the pair list.

    Raises ``ValueError`` on malformed thresholds.
    """
    for name, val in (
        ("jaccard_min", jaccard_min),
        ("residual_overlap_min", residual_overlap_min),
        ("confidence_min", confidence_min),
    ):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
    if max_candidate_pairs < 1:
        raise ValueError(
            f"max_candidate_pairs must be >= 1, got {max_candidate_pairs}",
        )

    beliefs = store.list_beliefs_for_indexing()
    n_beliefs = len(beliefs)
    if n_beliefs < 2:
        return RelationshipsAuditReport(
            n_beliefs_scanned=n_beliefs,
            n_candidate_pairs=0,
            n_contradicts_high=0,
            n_contradicts_low=0,
            n_refines=0,
            truncated=False,
        )

    candidates, raw_count, truncated = _jaccard_prefiltered_pairs(
        beliefs,
        jaccard_min=jaccard_min,
        max_pairs=max_candidate_pairs,
    )

    pairs: list[RelationshipPair] = []
    for id_a, content_a, id_b, content_b, _ta, _tb, _j in candidates:
        verdict = analyze(
            content_a,
            content_b,
            residual_overlap_min=residual_overlap_min,
        )
        if verdict.label == LABEL_UNRELATED:
            continue
        pairs.append(
            RelationshipPair(
                belief_a_id=id_a,
                belief_b_id=id_b,
                label=verdict.label,
                score=verdict.score,
                residual_overlap=verdict.residual_overlap,
                rationale=verdict.rationale,
            )
        )

    pairs.sort(key=lambda p: (p.belief_a_id, p.belief_b_id))

    n_contra_hi = sum(
        1 for p in pairs
        if p.label == LABEL_CONTRADICTS and p.score >= confidence_min
    )
    n_contra_lo = sum(
        1 for p in pairs
        if p.label == LABEL_CONTRADICTS and p.score < confidence_min
    )
    n_ref = sum(1 for p in pairs if p.label == LABEL_REFINES)

    return RelationshipsAuditReport(
        n_beliefs_scanned=n_beliefs,
        n_candidate_pairs=raw_count,
        n_contradicts_high=n_contra_hi,
        n_contradicts_low=n_contra_lo,
        n_refines=n_ref,
        truncated=truncated,
        pairs=tuple(pairs),
    )


# --- Config loader -----------------------------------------------------


@dataclass(frozen=True)
class RelationshipDetectorConfig:
    """Resolved ``[relationship_detector]`` section of `.aelfrice.toml`."""
    jaccard_min: float = DEFAULT_JACCARD_MIN
    residual_overlap_min: float = DEFAULT_RESIDUAL_OVERLAP_MIN
    confidence_min: float = DEFAULT_CONFIDENCE_MIN
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS


def _load_unit_float(
    section: dict[str, Any],
    key: str,
    default: float,
    candidate: Path,
    serr: IO[str],
) -> float:
    obj: Any = section.get(key, default)
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        print(
            f"aelfrice relationship_detector: ignoring [{SECTION}] {key} in "
            f"{candidate} (expected float in [0.0, 1.0])",
            file=serr,
        )
        return default
    val = float(obj)
    if not 0.0 <= val <= 1.0:
        print(
            f"aelfrice relationship_detector: ignoring [{SECTION}] {key} in "
            f"{candidate} (expected float in [0.0, 1.0])",
            file=serr,
        )
        return default
    return val


def load_relationship_detector_config(
    start: Path | None = None,
) -> RelationshipDetectorConfig:
    """Walk up from `start` looking for `.aelfrice.toml`.

    Missing file / missing section / malformed TOML / wrong-typed
    values all degrade to defaults with a stderr trace; never raises.
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
                print(
                    f"aelfrice relationship_detector: cannot read {candidate}: "
                    f"{exc}",
                    file=serr,
                )
                return RelationshipDetectorConfig()
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    raw.decode("utf-8", errors="replace"),
                )
            except tomllib.TOMLDecodeError as exc:
                print(
                    f"aelfrice relationship_detector: malformed TOML in "
                    f"{candidate}: {exc}",
                    file=serr,
                )
                return RelationshipDetectorConfig()
            section_obj: Any = parsed.get(SECTION, {})
            if not isinstance(section_obj, dict):
                return RelationshipDetectorConfig()
            section = cast(dict[str, Any], section_obj)
            jm = _load_unit_float(
                section, JACCARD_KEY, DEFAULT_JACCARD_MIN,
                candidate, serr,
            )
            cm = _load_unit_float(
                section, CONFIDENCE_KEY, DEFAULT_CONFIDENCE_MIN,
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
                print(
                    f"aelfrice relationship_detector: ignoring [{SECTION}] "
                    f"{MAX_CANDIDATE_PAIRS_KEY} in {candidate} "
                    f"(expected positive int)",
                    file=serr,
                )
                mp_resolved = DEFAULT_MAX_CANDIDATE_PAIRS
            else:
                mp_resolved = mp_obj
            return RelationshipDetectorConfig(
                jaccard_min=jm,
                confidence_min=cm,
                max_candidate_pairs=mp_resolved,
            )
        if current.parent == current:
            break
        current = current.parent
    return RelationshipDetectorConfig()


# --- Report formatter --------------------------------------------------


def format_audit_report(
    report: RelationshipsAuditReport,
    *,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    max_pairs_displayed: int = 25,
) -> str:
    """Render a `RelationshipsAuditReport` as a plain-text block."""
    lines: list[str] = []
    lines.append("aelf doctor relationships")
    lines.append("=" * 40)
    lines.append(f"Beliefs scanned         : {report.n_beliefs_scanned}")
    lines.append(f"Candidate pairs visited : {report.n_candidate_pairs}")
    if report.truncated:
        lines.append(
            f"  (pair list truncated at {DEFAULT_MAX_CANDIDATE_PAIRS} — "
            f"see [relationship_detector] max_candidate_pairs)"
        )
    lines.append(
        f"Contradicts (auto-emit eligible, score >= {confidence_min:.2f}) : "
        f"{report.n_contradicts_high}"
    )
    lines.append(
        f"Contradicts (audit-only, sub-confidence)                    : "
        f"{report.n_contradicts_low}"
    )
    lines.append(
        f"Refines (modality agrees, partial overlap)                  : "
        f"{report.n_refines}"
    )
    lines.append("")
    if not report.pairs:
        lines.append("No related belief pairs above the configured floor.")
        return "\n".join(lines)
    lines.append("Top pairs (label, score, residual overlap):")
    for p in report.pairs[:max_pairs_displayed]:
        lines.append(
            f"  {p.belief_a_id}  ~  {p.belief_b_id}  "
            f"[{p.label}]  s={p.score:.3f}  o={p.residual_overlap:.3f}  "
            f"({p.rationale})"
        )
    if len(report.pairs) > max_pairs_displayed:
        lines.append(
            f"  ... ({len(report.pairs) - max_pairs_displayed} more)"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIDENCE_MIN",
    "DEFAULT_JACCARD_MIN",
    "DEFAULT_MAX_CANDIDATE_PAIRS",
    "DEFAULT_RESIDUAL_OVERLAP_MIN",
    "LABEL_CONTRADICTS",
    "LABEL_REFINES",
    "LABEL_UNRELATED",
    "VERDICT_LABELS",
    "RelationshipDetectorConfig",
    "RelationshipPair",
    "RelationshipVerdict",
    "RelationshipsAuditReport",
    "Signals",
    "analyze",
    "classify",
    "extract_signals",
    "format_audit_report",
    "load_relationship_detector_config",
    "relationships_audit",
]
