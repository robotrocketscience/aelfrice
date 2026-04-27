"""Sentence classification: assign one of the four belief types and a
source-adjusted Beta prior.

v1.0 ships the synchronous regex/keyword fallback only. This is the
single classifier used in onboarding, CI, and any environment without a
host LLM. The polymorphic onboard handshake — where a host LLM
classifies sentences in its own context and returns results — lands in
v0.6.0 alongside the MCP server, when there's actually a host present.

Aelfrice never imports the `anthropic` SDK at any point in v1.0
(pre-commit #7). Classification calls flow through the host's existing
LLM context via the polymorphic protocol, never via aelfrice's own
network calls.

Type priors (alpha, beta) are calibrated for user-sourced content. Non-
user sources (scanner extracts, document text, agent-inferred) get
deflated alpha so the feedback loop earns confidence rather than
inheriting it.

The four belief types correspond exactly to the v1.0 surface in
`models.py` — `factual`, `correction`, `preference`, `requirement`.
The richer agentmemory-v4 type catalog (DECISION, ASSUMPTION, ANALYSIS,
TODO etc.) is collapsed into `factual` here; v1.x can re-expand the
catalog if usage data justifies it.
"""
from __future__ import annotations

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
      and a future host-LLM pass could refine the type. Always True in
      v1.0; the polymorphic-host-handshake path that flips this to False
      lands in v0.6.0.
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
    return text_lower.startswith(_QUESTION_PREFIXES) and text_lower.endswith("?")


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
    6. Default -> factual.

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

    # 3. User-sourced requirement (must, mandatory, hard rule, ...)
    #    Document text uses the same keywords descriptively; restrict
    #    requirement classification to user sources to reduce false
    #    positives during onboarding scans.
    if source == USER_SOURCE and _has_any(text_lower, _REQUIREMENT_KEYWORDS):
        alpha, beta = get_source_adjusted_prior(BELIEF_REQUIREMENT, source)
        return ClassificationResult(
            belief_type=BELIEF_REQUIREMENT,
            alpha=alpha,
            beta=beta,
            persist=True,
            pending_classification=True,
        )

    # 4. User-sourced correction (per the no-LLM detector).
    if source == USER_SOURCE:
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

    # 6. Default: factual.
    alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, source)
    return ClassificationResult(
        belief_type=BELIEF_FACTUAL,
        alpha=alpha,
        beta=beta,
        persist=True,
        pending_classification=True,
    )
