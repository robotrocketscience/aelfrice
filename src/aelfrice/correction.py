"""No-LLM correction detector.

Heuristic detector that identifies user corrections / directives by
counting signal-class hits across six categories: imperative-verb
start, always/never absolutist language, negation, emphasis, prior
reference, and declarative override. A text counts
as a correction when at least two distinct signals fire (precision
trade-off: single-signal matches have ~60% precision; the explicit
two-signal threshold trades recall for precision so the explicit
correct-this-belief path covers the gap).

Confidence is the signal count scaled by 0.3, capped at 1.0.

Ported from the previous codebase's correction_detection.py
(experiment-1-V2). Validated at 92% accuracy on the original
correction corpus; that corpus is not part of v1.0 so this module
ships the regex set verbatim and unit tests cover each signal class
in isolation rather than retrying the corpus accuracy claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_IMPERATIVE_RE: re.Pattern[str] = re.compile(
    r"^(use|add|remove|update|follow|convert|make|do|try|run|keep|"
    r"leave|report|copy|stop|always|never|we are|calls|5k)\b"
)

# #225: anchor patterns matched anywhere in the sentence. The original
# `_IMPERATIVE_RE` only matched at the start, so utterances like
# "the previous instruction supersedes …" or "do not amend commits"
# never tripped the imperative gate, leaving correction-class and
# requirement-class candidates one signal short of the
# `CORRECTION_SIGNAL_THRESHOLD` floor. Counting these as distinct
# imperative-class anchors recovers +12pp macro-F1 on the labeled
# correction corpus (lab campaign R0', 2026-04-28). The split into
# three sub-patterns keeps the existing leading-imperative semantics
# intact while letting the new anchors fire from any sentence
# position.
_CORRECTION_ANCHOR_RE: re.Pattern[str] = re.compile(
    r"\b(supersedes|no longer|the previous|instead|actually)\b"
)
_REQUIREMENT_ANCHOR_RE: re.Pattern[str] = re.compile(
    r"\b(must not|cannot|fails if)\b"
)

_DECLARATIVE_RE: re.Pattern[str] = re.compile(
    r"(?:is|are|needs to be|should be|must be) "
    r"(?:the|a|an|\d|only|always)"
)

_ALWAYS_NEVER_TERMS: tuple[str, ...] = (
    "always",
    "never",
    "every time",
    "every single",
    "from now on",
    "permanently",
    "period",
)

# #1159 §5: these were tested with bare substring containment, so `"no "`
# matched inside "pia*no* is" and `"not "` inside "can*not* ". Every term
# list in this module is now compiled into a word-boundary alternation
# (see `_boundary_alternation`), the shape `directive_detector.py` already
# uses.
#
# The trailing spaces that used to stand in for a right-hand boundary are
# gone, but `\b` alone is *not* an equivalent replacement: `\b` matches
# before a hyphen where a space did not, so `\bno\b` fires inside "no-op",
# "no-match" and "not-yet-issued". Measured over the 44,687 active beliefs
# of one live store, plain `\b` on both sides newly fires negation on 465
# beliefs — 249 hyphen compounds (false positives; 214 "no-", 35 "not-")
# and 216 real negations the trailing-space form missed (sentence-final
# "…is not.", "…probably no,", quote-adjacent). Negation therefore takes a
# right bound of `(?![\w-])`, which keeps the 216 and drops the 249.
# Operator ruling of 2026-08-09; `benchmarks/classifier_boundary_1368.py`
# re-derives the split and cross-checks its replica against `_NEGATION_RE`.
#
# This bound is deliberately negation-only. The sibling categories carry
# the same hyphen compounds (72 for always/never, 75 for prior-reference)
# but theirs are semantically live — "always-on" is still an absolutist
# claim and "already-shipped" is still a prior reference, whereas "no-op"
# negates nothing. Widening the exclusion to them would cost real signal.
#
# #1159 §4: "stop" used to appear in three of the categories below
# (`_IMPERATIVE_RE`, `_NEGATION_TERMS`, `_EMPHASIS_TERMS`), so a single
# token cleared a `CORRECTION_SIGNAL_THRESHOLD` that exists to require two
# *independent* signals. "stop" now lives only in `_IMPERATIVE_RE`, where
# an imperative-verb start is what it actually is, and "cannot" — which
# satisfied both `_REQUIREMENT_ANCHOR_RE` and negation's `"not "` — is
# fixed by the word boundary alone (`\bnot\b` does not match inside
# "cannot"). `test_classifier_word_boundaries_1368.py` asserts both.
#
# The categories are *not* fully token-disjoint, and this module does not
# claim they are: "always" and "never" remain in both `_IMPERATIVE_RE`'s
# verb bank and `_ALWAYS_NEVER_TERMS`, so `detect_correction("Always run
# the tests.")` still returns two signals off one token. That overlap
# predates #1159 §4, which named only "stop"; removing it changes what the
# write path admits and is deferred with the other keyword drops. It is
# pinned by a test so the next reader finds it deliberate.
_NEGATION_TERMS: tuple[str, ...] = (
    "do not",
    "don't",
    "dont",
    "not",
    "no more",
    "no",
)

_EMPHASIS_TERMS: tuple[str, ...] = (
    "!",
    "hate",
    "ever again",
    "zero question",
    "100 times",
)

_PRIOR_REF_TERMS: tuple[str, ...] = (
    "we've been",
    "i told you",
    "we discussed",
    "we agreed",
    "already",
    "iirc",
    "we decided",
)


def _boundary_alternation(
    terms: tuple[str, ...], right_boundary: str = r"\b"
) -> re.Pattern[str]:
    """Compile `terms` into one word-boundary alternation.

    Same shape as `directive_detector.py`'s verb pattern: alternatives are
    sorted length-descending so a multi-word phrase matches before its own
    single-word prefix. The boundary is attached only on the sides where
    the term actually starts/ends with a word character — `"!"` carries no
    boundary at all, and `"don't"` gets one on each end.

    `right_boundary` is the assertion placed after a word-final term.
    The default `\\b` is the ordinary word boundary. Negation passes
    `(?![\\w-])` instead, which additionally refuses a following hyphen so
    "no-op" does not read as a negation; see the §5 note above for the
    measurement that scopes that to negation alone.

    Terms must already be lowercase; callers match against lowercased text.
    """
    parts: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.escape(term)
        if term[:1].isalnum() or term[:1] == "_":
            pattern = r"\b" + pattern
        if term[-1:].isalnum() or term[-1:] == "_":
            pattern = pattern + right_boundary
        parts.append(pattern)
    return re.compile("(?:" + "|".join(parts) + ")")


_ALWAYS_NEVER_RE: re.Pattern[str] = _boundary_alternation(_ALWAYS_NEVER_TERMS)
_NEGATION_RE: re.Pattern[str] = _boundary_alternation(
    _NEGATION_TERMS, right_boundary=r"(?![\w-])"
)
_EMPHASIS_RE: re.Pattern[str] = _boundary_alternation(_EMPHASIS_TERMS)
_PRIOR_REF_RE: re.Pattern[str] = _boundary_alternation(_PRIOR_REF_TERMS)

CORRECTION_SIGNAL_THRESHOLD: int = 2
_CONFIDENCE_PER_SIGNAL: float = 0.3


@dataclass
class CorrectionResult:
    """Output of detect_correction.

    Fields:
    - is_correction: True iff signals fired at least CORRECTION_SIGNAL_THRESHOLD
      distinct categories.
    - signals: deduped, deterministically-ordered list of signal categories
      that fired.
    - confidence: signal-count * 0.3, capped at 1.0.
    """

    is_correction: bool
    signals: list[str]
    confidence: float


def detect_correction(text: str) -> CorrectionResult:
    """Score `text` against the six correction-signal categories.

    Categories (in evaluation order, which is also the output order):
        imperative, always_never, negation, emphasis, prior_ref,
        declarative

    Pure function: no I/O, no side effects, deterministic for any input.
    """
    text_lower: str = text.lower().strip()
    signals: list[str] = []

    if (
        _IMPERATIVE_RE.match(text_lower)
        or _CORRECTION_ANCHOR_RE.search(text_lower)
        or _REQUIREMENT_ANCHOR_RE.search(text_lower)
    ):
        signals.append("imperative")

    if _ALWAYS_NEVER_RE.search(text_lower):
        signals.append("always_never")

    if _NEGATION_RE.search(text_lower):
        signals.append("negation")

    if _EMPHASIS_RE.search(text_lower):
        signals.append("emphasis")

    if _PRIOR_REF_RE.search(text_lower):
        signals.append("prior_ref")

    if _DECLARATIVE_RE.search(text_lower):
        signals.append("declarative")

    # #1162 §5: a seventh `directive` category used to fire here off
    # `_DIRECTIVE_TERMS`, a strict subset of
    # `classification_core._REQUIREMENT_KEYWORDS`. Because
    # `classify_sentence` returns `requirement` at step 3, before it ever
    # calls `detect_correction` at step 4, no text that could set
    # `directive` reached this line through the production path. Deleted
    # per operator ruling rather than reordered.

    is_correction: bool = len(signals) >= CORRECTION_SIGNAL_THRESHOLD
    confidence: float = min(1.0, len(signals) * _CONFIDENCE_PER_SIGNAL)
    return CorrectionResult(
        is_correction=is_correction,
        signals=signals,
        confidence=confidence,
    )
