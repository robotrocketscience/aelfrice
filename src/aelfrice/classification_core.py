"""Leaf-side of `aelfrice.classification`: pure regex/keyword
classifier and source-adjusted Beta priors.

Split out of `classification.py` so `derivation.py` (and other
non-orchestration consumers) can use `classify_sentence` /
`get_source_adjusted_prior` without closing the
classification ↔ derivation ↔ derivation_worker module-import cycle
that scanner.py and the onboard handshake live inside.

Imports here must stay leaf: `aelfrice.correction` (no aelfrice deps)
and `aelfrice.models` (constants + dataclasses) only. Anything that
touches `scanner` / `derivation_worker` belongs in `classification.py`,
not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from aelfrice.correction import detect_correction
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
)

# --- Priors (per Exp 61, restricted to v1.0's 4-type catalog) -----------

TYPE_PRIORS: Final[dict[str, tuple[float, float]]] = {
    BELIEF_REQUIREMENT: (9.0, 0.5),  # 94.7% prior — hard constraints
    BELIEF_CORRECTION: (9.0, 0.5),   # 94.7% — user corrections
    BELIEF_PREFERENCE: (7.0, 1.0),   # 87.5% — user preferences
    BELIEF_FACTUAL: (3.0, 1.0),      # 75.0% — stated facts and analyses
}

# Deflation factor for non-user sources. TYPE_PRIORS are calibrated for
# user-stated content; agent-inferred or document-extracted content
# starts at lower alpha so the feedback loop earns the rest of the
# confidence rather than inheriting it. Without this, scanner-extracted
# beliefs would cluster at 90-95% confidence on day one and Thompson
# sampling would lose discriminative power.
_AGENT_INFERRED_DEFLATION: Final[float] = 0.2
_DEFLATED_ALPHA_FLOOR: Final[float] = 0.5

USER_SOURCE: Final[str] = "user"

# --- Heuristic keyword sets ---------------------------------------------

_REQUIREMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "must",
    "require",
    "mandatory",
    "hard cap",
    "constraint",
    "hard rule",
)

_PREFERENCE_KEYWORDS: Final[tuple[str, ...]] = (
    "prefer",
    "favorite",
    "always use",
    "never use",
    "i like",
    "i hate",
    "i want",
)

_QUESTION_PREFIXES: Final[tuple[str, ...]] = (
    "what ",
    "how ",
    "why ",
    "when ",
    "where ",
    "can ",
    "does ",
    "is there",
    "should ",
    "would ",
    "could ",
)

# Statement-form speculative floats (#1081). Unlike `_is_question` these do
# NOT end in `?` — they are declarative in *form* but tentative in *force*:
# a floated idea or musing ("Maybe we need X", "What if we tried Y") rather
# than an assertion. Passive transcript capture was storing these as
# `factual` beliefs; they read as conversation, not durable knowledge.
#
# Two high-precision arms keep false positives low:
#  - LEADING hedge: the sentence *opens* with a hedge marker. A sentence
#    that starts with "Maybe " / "What if " / "Should we " (no `?`) is almost
#    always a float; a genuine assertion that merely *contains* "maybe"
#    mid-sentence is not matched.
#  - a small set of unambiguous proposal-hedge PHRASES ("we probably need",
#    "maybe we") that signal a floated proposal wherever they appear.
#
# Applied only to the default-factual bucket (see `classify_sentence` step
# 6), so a typed requirement / correction / preference is never dropped for
# being hedged. Bald hedge-free musings ("All averages are not useful.")
# have no deterministic signal and are intentionally out of scope for this
# ingest gate — they are handled by the retrieval / curation lanes
# (#1081 directions C/D), not here. Pure/deterministic: replay-stable.
_FLOAT_LEADING_HEDGES: Final[tuple[str, ...]] = (
    "maybe ",
    "maybe,",
    "perhaps ",
    "perhaps,",
    "what if ",
    "how about ",
    "what about ",
    "i wonder ",
    "i wonder if",
    "wondering if ",
    "not sure ",
    "i guess ",
    "i suppose ",
    "or maybe ",
    "or perhaps ",
    "or should ",
    "or we could ",
    "or we should ",
    "should we ",
    "could we ",
    "shouldn't we ",
    "shouldnt we ",
)

_FLOAT_INTERNAL_HEDGES: Final[tuple[str, ...]] = (
    "maybe we ",
    "maybe there ",
    "maybe it would ",
    "we probably need ",
    "we could probably ",
    "we might want ",
    "we should probably ",
    "not sure if ",
    "not sure whether ",
)

# An internal sentence boundary: ./!/? — optionally followed by a closing
# quote/paren/bracket — then whitespace. The closers matter: a prior
# sentence ending in `."` / `.)` / `?"` must still register as a boundary so
# the #1027 single-sentence-question rule never drops a multi-sentence
# belief that merely closes with a question (its declarative content is real
# and must persist).
_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r'''[.!?]['")\]]*\s''')


# --- Output ---------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Output of classify_sentence.

    Fields:
    - belief_type: one of `factual / correction / preference / requirement`.
      For non-persistable sentences (questions, coordination, meta), still
      returns `factual` but `persist=False`.
    - alpha, beta: Beta-Bernoulli prior, source-adjusted.
    - persist: False for ephemeral content (questions, etc.); True
      otherwise. Caller (scanner / onboarding) is responsible for
      skipping non-persisting sentences before insertion.
    - pending_classification: True when this was the regex-fallback path
      and a future host-LLM pass could refine the type. Always True as of
      v3.8.0; no code path currently flips it to False — the
      polymorphic-host-handshake integration remains unimplemented.
    """

    belief_type: str
    alpha: float
    beta: float
    persist: bool
    pending_classification: bool


# --- Helpers --------------------------------------------------------------


def get_source_adjusted_prior(
    belief_type: str,
    source: str,
) -> tuple[float, float]:
    """Resolve the Beta prior for a (type, source) pair.

    User-sourced content gets the full TYPE_PRIORS value. Non-user
    sources get alpha deflated by `_AGENT_INFERRED_DEFLATION`, with a
    `_DEFLATED_ALPHA_FLOOR` so the deflated alpha never drops below 0.5
    (which would make posterior_mean numerically degenerate at low beta).
    """
    prior = TYPE_PRIORS.get(belief_type)
    if prior is None:
        # Unknown type collapses to factual prior — keeps the function
        # total without surfacing a partial-failure mode to callers.
        prior = TYPE_PRIORS[BELIEF_FACTUAL]
    alpha, beta = prior
    if source != USER_SOURCE:
        alpha = max(_DEFLATED_ALPHA_FLOOR, alpha * _AGENT_INFERRED_DEFLATION)
    return (alpha, beta)


def _is_question(text_lower: str) -> bool:
    """True when the sentence is a question and therefore non-persistable.

    Two arms, both requiring a trailing `?`:
    - wh-prefixed interrogative (covers multi-clause "what … and how …?").
    - #1027: any SINGLE-sentence interrogative ("want me to run it?",
      "which way?") — conversational questions the wh-prefix list misses.
      Restricted to single sentences (no internal `[.!?]\\s` boundary, no
      newline) so a multi-sentence belief that merely closes with a
      question keeps its declarative content. Questions are still LOGGED
      to the transcript (this only blocks belief creation, not logging).
    """
    s = text_lower.strip()
    if not s.endswith("?"):
        return False
    if s.startswith(_QUESTION_PREFIXES):
        return True
    if "\n" in s:
        return False
    return _SENTENCE_BOUNDARY_RE.search(s[:-1]) is None


def _is_speculative_float(text_lower: str) -> bool:
    """True when the sentence is a hedged/hypothetical float (#1081).

    Declarative in form (no trailing `?`, so `_is_question` misses it) but
    tentative in force — a floated idea or musing rather than an assertion.
    Matches only high-precision markers: a LEADING hedge (the sentence opens
    with "maybe …" / "what if …" / "should we …" and friends) or one of a
    small set of unambiguous proposal-hedge phrases ("we probably need …",
    "maybe we …"). Callers apply this only to the default-factual bucket, so
    a typed requirement / correction / preference is never dropped for being
    hedged. Pure and deterministic — re-runs identically on `aelf rebuild`.
    """
    s = text_lower.strip()
    if not s:
        return False
    if s.startswith(_FLOAT_LEADING_HEDGES):
        return True
    return any(phrase in s for phrase in _FLOAT_INTERNAL_HEDGES)


def _has_any(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text_lower for kw in keywords)


# --- Public API -----------------------------------------------------------


def classify_sentence(text: str, source: str) -> ClassificationResult:
    """Synchronous, deterministic classification.

    Pipeline (in evaluation order):
    1. Empty / whitespace-only -> factual, persist=False.
    2. Question form -> factual, persist=False.
    3. User source + requirement keywords -> requirement.
    4. User source + correction-detector positive -> correction.
    5. Preference keywords -> preference.
    6. Speculative float (hedged musing, #1081) -> factual, persist=False.
    7. Default -> factual.

    Always sets pending_classification=True in v1.0; the host-handshake
    path that flips it to False ships at v0.6.0.

    Pure function. No I/O, no third-party deps, deterministic for any
    (text, source) pair.
    """
    text_lower = text.lower().strip()

    # 1. Empty.
    if not text_lower:
        alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
        return ClassificationResult(
            belief_type=BELIEF_FACTUAL,
            alpha=alpha,
            beta=beta,
            persist=False,
            pending_classification=True,
        )

    # 2. Questions don't persist.
    if _is_question(text_lower):
        alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
        return ClassificationResult(
            belief_type=BELIEF_FACTUAL,
            alpha=alpha,
            beta=beta,
            persist=False,
            pending_classification=True,
        )

    # 3. Requirement keywords (must, mandatory, hard rule, ...)
    #    Originally gated on `source == USER_SOURCE` to suppress
    #    document-text false positives. Per #226: that gate caused
    #    every onboard-extracted requirement (source = `doc:…`,
    #    `ast:…`, `git:…`) to mis-classify, producing zero
    #    requirement counts on the labeled corpus. The
    #    source-prior-deflation in `get_source_adjusted_prior`
    #    already lowers alpha for non-user sources, so the
    #    false-positive risk is handled at the scoring layer
    #    rather than by gating classification.
    if _has_any(text_lower, _REQUIREMENT_KEYWORDS):
        alpha, beta = get_source_adjusted_prior(BELIEF_REQUIREMENT, source)
        return ClassificationResult(
            belief_type=BELIEF_REQUIREMENT,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 4. Correction (per the no-LLM detector). Same #226 reasoning
    #    — the user-only gate suppressed onboard-extracted
    #    corrections; deflated alpha at non-user sources handles
    #    the false-positive concern.
    cresult = detect_correction(text)
    if cresult.is_correction:
        alpha, beta = get_source_adjusted_prior(BELIEF_CORRECTION, source)
        return ClassificationResult(
            belief_type=BELIEF_CORRECTION,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 5. Preference keywords (any source).
    if _has_any(text_lower, _PREFERENCE_KEYWORDS):
        alpha, beta = get_source_adjusted_prior(BELIEF_PREFERENCE, source)
        return ClassificationResult(
            belief_type=BELIEF_PREFERENCE,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 6. Speculative float (#1081): a hedged musing that reached the
    #    default-factual bucket (not a typed requirement/correction/
    #    preference). Declarative in form but tentative in force — passive
    #    capture was storing these conversational fragments as beliefs.
    #    persist=False, like questions. Placed last so a hedged sentence
    #    that IS a requirement/correction/preference is kept.
    if _is_speculative_float(text_lower):
        alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
        return ClassificationResult(
            belief_type=BELIEF_FACTUAL,
            alpha=alpha,
            beta=beta,
            persist=False,
            pending_classification=True,
        )

    # 7. Default: factual.
    alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
    return ClassificationResult(
        belief_type=BELIEF_FACTUAL,
        alpha=alpha,
        beta=beta,
        persist=True,
        pending_classification=True,
    )
