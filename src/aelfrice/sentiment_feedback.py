"""Implicit sentiment-from-prose feedback (#193, v2.0 opt-in).

Reads each user prompt, regex-matches against twelve positive and
twelve negative sentiment patterns, and emits a `SentimentSignal`.
The signal is distributed equally across the previous turn's
retrieved beliefs via `apply_sentiment_to_pending`, which calls
`feedback.apply_feedback` once per pending belief id.

Design contract (spec: `docs/v2_sentiment_feedback.md`):

  * **Default off.** Opt-in via `[feedback] sentiment_from_prose = true`
    in `.aelfrice.toml`, or `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1`.
    Off by default means existing users see no behavior change.
  * **Inbound only.** Pure regex over text the hook already receives.
    No outbound calls. No LLM. Deterministic.
  * **Length guard.** Prompts longer than `MAX_PROMPT_CHARS` (200) are
    assumed task content, not user feedback. `detect_sentiment` returns
    `None` for these so the regex bank does not match incidental phrases
    in long pastes.
  * **Pattern provenance.** Twelve positive patterns and twelve negative
    patterns ported from the research-line `agentmemory/sentiment_feedback.py`
    per the v2.0 ratification. Two strong-amplifier subsets
    (STRONG_POSITIVE, STRONG_NEGATIVE) escalate `confidence` when the
    matched pattern carries higher signal-to-noise than the base set.
  * **Audit row source.** All apply_feedback calls from this module use
    `source = SENTIMENT_INFERRED_SOURCE` so the audit trail can split
    explicit-user feedback from sentiment-inferred feedback.
  * **Correction-frequency escalator.** `detect_correction_frequency`
    fires a stronger negative signal when >= CORRECTION_FREQ_THRESHOLD
    fraction of the recent N turns were corrections. Stateless beyond
    the caller-supplied window.

Distribution shape: `apply_sentiment_to_pending` distributes the signal
equally across all pending belief ids. Per-rank scaling is explicitly
out of scope (decision ratified 2026-04-29: "matches the research-line
behavior; ranked distribution adds a knob without an evidence-gate").

This module does NOT decide *when* to call into it. The hook layer is
responsible for: (a) checking the config flag, (b) resolving the
"previous turn's retrieved beliefs" set, and (c) calling
`apply_sentiment_to_pending`. That wiring is a separate concern; the
module surface here is pure.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Final, Sequence

from aelfrice.feedback import FeedbackResult, apply_feedback
from aelfrice.store import MemoryStore

# --- Public constants ---------------------------------------------------

MAX_PROMPT_CHARS: Final[int] = 200
"""Prompts longer than this are assumed task content; detector returns None."""

SENTIMENT_INFERRED_SOURCE: Final[str] = "sentiment_inferred"
"""Source string written to feedback_history rows from this module."""

CONFIG_FEEDBACK_SECTION: Final[str] = "feedback"
CONFIG_SENTIMENT_KEY: Final[str] = "sentiment_from_prose"
ENV_SENTIMENT: Final[str] = "AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE"

POSITIVE: Final[str] = "positive"
NEGATIVE: Final[str] = "negative"

BASE_VALENCE: Final[float] = 1.0
"""Magnitude passed to apply_feedback for a base-pattern match."""

AMPLIFIED_VALENCE: Final[float] = 1.5
"""Magnitude for a strong-pattern match (50% boost over base)."""

ESCALATED_NEGATIVE_VALENCE: Final[float] = 2.0
"""Magnitude for a correction-frequency escalation. Doubles the base."""

CORRECTION_FREQ_THRESHOLD: Final[float] = 0.4
"""Fraction of recent turns that must be negative to fire the escalator."""

CORRECTION_FREQ_MIN_TURNS: Final[int] = 5
"""Minimum recent-turn count before the escalator can fire (avoids
firing on a 2-of-2 sample)."""

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


# --- Pattern banks ------------------------------------------------------
# Word-boundary anchored, case-insensitive. Compiled once at import.
# Order matters only insofar as the first match wins; the strong subsets
# are checked separately so a base-set match still wins when the prompt
# does not also hit a strong pattern.

_POSITIVE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("ok_good", r"\bok(ay)?[\s,!.]+good\b"),
    ("yes", r"\byes\b"),
    ("yeah", r"\byeah\b"),
    ("perfect", r"\bperfect\b"),
    ("great", r"\bgreat\b"),
    ("nice", r"\bnice\b"),
    ("thanks", r"\bthanks?\b"),
    ("correct", r"\bcorrect\b"),
    ("right", r"\bthat'?s right\b"),
    ("works", r"\b(it|that) works?\b"),
    ("looks_good", r"\blooks? good\b"),
    ("good_job", r"\bgood (job|work)\b"),
)

_NEGATIVE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("no", r"\bno\b"),
    ("nope", r"\bnope\b"),
    ("wrong", r"\b(that'?s |is )?wrong\b"),
    ("incorrect", r"\bincorrect\b"),
    ("not_what", r"\bnot what i\b"),
    ("fix_it", r"\bfix (it|this|that)\b"),
    ("broken", r"\bbroken\b"),
    ("doesnt_work", r"\bdoes ?n'?t work\b"),
    ("stop", r"\bstop (doing |that)\b"),
    ("i_told_you", r"\bi (already )?told you\b"),
    ("undo", r"\bundo (that|it|this)\b"),
    ("try_again", r"\btry again\b"),
)

_STRONG_POSITIVE: Final[frozenset[str]] = frozenset({
    "perfect", "correct", "right", "good_job",
})

_STRONG_NEGATIVE: Final[frozenset[str]] = frozenset({
    "wrong", "incorrect", "i_told_you", "fix_it", "broken",
})

_COMPILED_POSITIVE: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (name, re.compile(pat, re.IGNORECASE)) for name, pat in _POSITIVE_PATTERNS
)
_COMPILED_NEGATIVE: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (name, re.compile(pat, re.IGNORECASE)) for name, pat in _NEGATIVE_PATTERNS
)


# --- Signal dataclass ---------------------------------------------------


@dataclass(frozen=True)
class SentimentSignal:
    """One sentiment classification of one prompt.

    `sentiment` is POSITIVE or NEGATIVE.
    `valence` is the magnitude+sign passed to `apply_feedback`. Positive
    signals carry positive valence; negative carry negative.
    `confidence` is in [0.0, 1.0] and reflects whether the match came
    from the base or strong subset. Future versions may layer additional
    signals; today it is a two-level enum encoded as 0.6 (base) or 0.9
    (strong-amplified).
    `pattern` is the named pattern that matched. Stable identifier for
    the audit trail; the matched substring itself is captured in
    `matched_text`.
    `matched_text` is the literal substring from the prompt. Used by the
    audit row to record what triggered the classification without
    storing the full prompt.
    """

    sentiment: str
    valence: float
    confidence: float
    pattern: str
    matched_text: str


# --- Detector -----------------------------------------------------------


def detect_sentiment(prompt: str) -> SentimentSignal | None:
    """Classify one prompt. Returns None if no pattern matches or the
    prompt exceeds the length guard.

    Detection order:
      1. Length guard. > MAX_PROMPT_CHARS -> None (assumed task content).
      2. Strong negatives, then strong positives. Strong wins over base
         when both match because the higher-confidence interpretation
         is the safer one to act on.
      3. Base negatives, then base positives. Negatives are checked
         first within each tier on the asymmetric-cost principle: a
         missed correction silently entrenches a wrong belief, while
         a missed positive only delays a small posterior gain.
    """
    if not prompt:
        return None
    if len(prompt) > MAX_PROMPT_CHARS:
        return None

    strong_neg = _first_match(_COMPILED_NEGATIVE, prompt, _STRONG_NEGATIVE)
    if strong_neg is not None:
        return _make_signal(NEGATIVE, strong_neg, strong=True)

    strong_pos = _first_match(_COMPILED_POSITIVE, prompt, _STRONG_POSITIVE)
    if strong_pos is not None:
        return _make_signal(POSITIVE, strong_pos, strong=True)

    base_neg = _first_match(_COMPILED_NEGATIVE, prompt, None)
    if base_neg is not None:
        return _make_signal(NEGATIVE, base_neg, strong=False)

    base_pos = _first_match(_COMPILED_POSITIVE, prompt, None)
    if base_pos is not None:
        return _make_signal(POSITIVE, base_pos, strong=False)

    return None


def _first_match(
    compiled: tuple[tuple[str, re.Pattern[str]], ...],
    prompt: str,
    restrict_to: frozenset[str] | None,
) -> tuple[str, str] | None:
    """Return (pattern_name, matched_text) for the first hit, or None.
    `restrict_to` filters the candidate set; None means consider all.
    """
    for name, pat in compiled:
        if restrict_to is not None and name not in restrict_to:
            continue
        m = pat.search(prompt)
        if m is not None:
            return (name, m.group(0))
    return None


def _make_signal(
    sentiment: str, hit: tuple[str, str], strong: bool
) -> SentimentSignal:
    name, text = hit
    if strong:
        magnitude = AMPLIFIED_VALENCE
        confidence = 0.9
    else:
        magnitude = BASE_VALENCE
        confidence = 0.6
    valence = magnitude if sentiment == POSITIVE else -magnitude
    return SentimentSignal(
        sentiment=sentiment,
        valence=valence,
        confidence=confidence,
        pattern=name,
        matched_text=text,
    )


def classify(prompt: str) -> str:
    """Three-way label adapter: "positive" | "negative" | "neutral".

    Used by the bench-gate harness (#319 / #193) to score against the
    labeled corpus. Returns "neutral" when no pattern matches or the
    prompt fails the length guard.
    """
    signal = detect_sentiment(prompt)
    if signal is None:
        return "neutral"
    return signal.sentiment


def detect_correction_frequency(
    recent_signals: Sequence[SentimentSignal | None],
    *,
    threshold: float = CORRECTION_FREQ_THRESHOLD,
    min_turns: int = CORRECTION_FREQ_MIN_TURNS,
) -> bool:
    """Return True if the recent-signal window indicates the user is
    correcting at or above `threshold` fraction. Returns False when the
    window is shorter than `min_turns` to avoid firing on tiny samples.

    Counts NEGATIVE-classified signals only; None and POSITIVE entries
    contribute to the denominator without raising the rate.
    """
    if len(recent_signals) < min_turns:
        return False
    negatives = sum(
        1 for s in recent_signals if s is not None and s.sentiment == NEGATIVE
    )
    return (negatives / len(recent_signals)) >= threshold


# --- Application --------------------------------------------------------


def apply_sentiment_to_pending(
    store: MemoryStore,
    signal: SentimentSignal,
    pending_belief_ids: Sequence[str],
    *,
    now: str | None = None,
    escalated: bool = False,
) -> list[FeedbackResult]:
    """Distribute one sentiment signal equally across the pending belief
    set. Calls `apply_feedback` once per belief id.

    Returns the list of FeedbackResults. Beliefs that no longer exist
    in the store are skipped silently (not an error: the previous turn
    may have surfaced a belief that has since been deleted).

    `escalated` upgrades a negative signal's magnitude to
    `ESCALATED_NEGATIVE_VALENCE`. Has no effect on positive signals;
    the correction-frequency path only escalates negatives by design.

    `propagate=False` is passed through to `apply_feedback` because
    sentiment signals are not corrective in the contradictor-walk sense:
    they are bulk implicit signals on a retrieval set, not user
    judgments on individual contradicting beliefs.
    """
    if not pending_belief_ids:
        return []

    valence = signal.valence
    if escalated and signal.sentiment == NEGATIVE:
        valence = -ESCALATED_NEGATIVE_VALENCE

    results: list[FeedbackResult] = []
    for bid in pending_belief_ids:
        if store.get_belief(bid) is None:
            continue
        result = apply_feedback(
            store=store,
            belief_id=bid,
            valence=valence,
            source=SENTIMENT_INFERRED_SOURCE,
            now=now,
            propagate=False,
        )
        results.append(result)
    return results


# --- Config -------------------------------------------------------------


def is_enabled(config: dict[str, dict] | None = None) -> bool:
    """Whether sentiment-from-prose is enabled.

    Resolution order:
      1. Env var `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE` if set.
      2. `[feedback] sentiment_from_prose` in the supplied config dict.
      3. Default False.

    The hook layer is the caller; this module does not read disk.
    """
    raw = os.environ.get(ENV_SENTIMENT)
    if raw is not None:
        token = raw.strip().lower()
        if token in _ENV_TRUTHY:
            return True
        if token in _ENV_FALSY:
            return False

    if config is None:
        return False
    section = config.get(CONFIG_FEEDBACK_SECTION)
    if not isinstance(section, dict):
        return False
    value = section.get(CONFIG_SENTIMENT_KEY)
    return bool(value) if isinstance(value, bool) else False
