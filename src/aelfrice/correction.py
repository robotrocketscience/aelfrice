"""No-LLM correction detector.

Heuristic detector that identifies user corrections / directives by
counting signal-class hits across seven categories: imperative-verb
start, always/never absolutist language, negation, emphasis, prior
reference, declarative override, and strong directive. A text counts
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

_NEGATION_TERMS: tuple[str, ...] = (
    "do not",
    "don't",
    "dont",
    "stop",
    "not ",
    "no more",
    "no ",
)

_EMPHASIS_TERMS: tuple[str, ...] = (
    "!",
    "hate",
    "stop",
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

_DIRECTIVE_TERMS: tuple[str, ...] = (
    "must",
    "require",
    "mandatory",
    "hard cap",
    "hard rule",
)

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
    """Score `text` against the seven correction-signal categories.

    Categories (in evaluation order, which is also the output order):
        imperative, always_never, negation, emphasis, prior_ref,
        declarative, directive

    Pure function: no I/O, no side effects, deterministic for any input.
    """
    text_lower: str = text.lower().strip()
    signals: list[str] = []

    if _IMPERATIVE_RE.match(text_lower):
        signals.append("imperative")

    if any(term in text_lower for term in _ALWAYS_NEVER_TERMS):
        signals.append("always_never")

    if any(term in text_lower for term in _NEGATION_TERMS):
        signals.append("negation")

    if any(term in text_lower for term in _EMPHASIS_TERMS):
        signals.append("emphasis")

    if any(term in text_lower for term in _PRIOR_REF_TERMS):
        signals.append("prior_ref")

    if _DECLARATIVE_RE.search(text_lower):
        signals.append("declarative")

    if any(term in text_lower for term in _DIRECTIVE_TERMS):
        signals.append("directive")

    is_correction: bool = len(signals) >= CORRECTION_SIGNAL_THRESHOLD
    confidence: float = min(1.0, len(signals) * _CONFIDENCE_PER_SIGNAL)
    return CorrectionResult(
        is_correction=is_correction,
        signals=signals,
        confidence=confidence,
    )
