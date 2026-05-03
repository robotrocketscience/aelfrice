"""Directive detection — H1 split of #199 (#374).

Candidate regex detector against which the bench gate is evaluated.

Per spec [`docs/v2_enforcement.md` § H1](../docs/v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate),
H1 (`process_directive`, TODO lifecycle, escalation) does not start until a
labeled corpus shows ≥80% precision and ≥60% recall on 200 coding prompts.
This module provides the detector under test. It does not auto-create
beliefs, build the TODO lifecycle, or wire into the rebuild path; that work
is gated on this gate passing.

Shape:

    detect_directive(text: str) -> bool

True when the text reads as an imperative directive the user intends as a
durable rule. False for questions, hedged statements, and reported speech.

The verb bank below is the spec's "29 imperatives" reconstructed from the
ratification text (issue #374, `docs/v2_enforcement.md` § H1). The exact
membership is tunable; the gate scores whatever detector is in this module.
"""
from __future__ import annotations

import re

# 29 imperative / deontic markers. Order is irrelevant — alternation in a
# single regex with word boundaries.
_IMPERATIVE_VERBS: tuple[str, ...] = (
    "never",
    "always",
    "must",
    "must not",
    "do not",
    "don't",
    "should",
    "should not",
    "shouldn't",
    "shall",
    "ensure",
    "require",
    "required",
    "requires",
    "avoid",
    "prefer",
    "only",
    "before",
    "after",
    "unless",
    "whenever",
    "need to",
    "needs to",
    "cannot",
    "can't",
    "won't",
    "forbidden",
    "mandatory",
    "prohibited",
)

# Build a single alternation regex. Sort by length desc so multi-word
# phrases match before their single-word prefixes.
_VERB_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in sorted(_IMPERATIVE_VERBS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Hedge markers — when present, the imperative reads as opinion or
# uncertainty rather than a directive.
_HEDGE_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|probably|might|i think|i guess|i wonder|not sure|"
    r"kinda|sort of|sometimes|occasionally|in theory)\b",
    re.IGNORECASE,
)

# Reported-speech / conditional-narration markers. A clause like
# "I never push to main when I'm tired" describes habit, not rule.
_NARRATION_PATTERN = re.compile(
    r"\bi\s+(?:never|always|don't|do not|won't|cannot|can't|must|should)\b"
    r"(?:\s+\w+){1,8}\s+(?:when|while|if|because|since)\b",
    re.IGNORECASE,
)

# Wh-question leaders (always interrogative). Aux-verb leaders (do/does/can/...)
# are only interrogative when paired with a trailing '?', which is handled
# separately in `detect_directive`.
_WH_QUESTION_LEADING = re.compile(
    r"^\s*(?:what|why|how|when|where|who|which|whose|whom)\b",
    re.IGNORECASE,
)


def detect_directive(text: str) -> bool:
    """Return True if `text` reads as a durable imperative directive.

    Filters applied in order:
      1. Empty / whitespace-only → False.
      2. Questions (leading interrogative or trailing '?') → False.
      3. Reported-speech / habitual narration ("I never X when Y") → False.
      4. Hedged statements ("maybe", "I think", …) → False.
      5. Otherwise: True iff any imperative verb appears.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if stripped.endswith("?"):
        return False
    if _WH_QUESTION_LEADING.search(stripped):
        return False
    if _NARRATION_PATTERN.search(stripped):
        return False
    if _HEDGE_PATTERN.search(stripped):
        return False
    return _VERB_PATTERN.search(stripped) is not None
