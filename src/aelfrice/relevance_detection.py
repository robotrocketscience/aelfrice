"""Reference-detection layer for #779 close-the-loop relevance signal.

Layer 2 of the umbrella: given a list of injected beliefs and the
assistant's response text from the same UPS-turn cycle, score
``referenced ∈ {0, 1}`` per (event_id, belief_content) pair. The
result drives ``MemoryStore.update_meta_belief(key, SIGNAL_RELEVANCE,
evidence=referenced, ...)`` calls in the sweeper (Layer 3).

Strategies, in spec cost order (#779 § 2):

1. **Exact substring** — belief content normalised (NFC + casefold +
   whitespace collapse) appears verbatim in the agent's response.
   Highest precision, low recall. Ships as the v1 default per the
   2026-05-14 ratification (Q5).
2. *(Out of scope here)* N-gram overlap above threshold — opt-in,
   sub-issue.

PHILOSOPHY (#605, locked ``c06f8d575fad71fb``): pure function over
``(events, response_text) → list[(event_id, referenced)]``. No
embeddings, no LLM judges, no wall-clock state. Same inputs yield
the same output bit-for-bit.
"""
from __future__ import annotations

import unicodedata
from typing import Final

# Marker strategy names; threaded through `.aelfrice.toml` later
# (sub-issue). Kept here so the substring-vs-ngram dispatch surface
# is named once.
STRATEGY_EXACT_SUBSTRING: Final[str] = "exact_substring"
STRATEGY_NGRAM_OVERLAP: Final[str] = "ngram_overlap"

# Minimum normalised-belief length below which substring detection is
# suppressed. Empty / one-character beliefs would match anything;
# preserve the precision bias even at the cost of recall on
# pathologically short content.
_MIN_NORMALIZED_LENGTH: Final[int] = 8


def normalize_text(text: str) -> str:
    """Return the canonical form used for exact-substring comparison.

    Three deterministic steps:
      1. ``unicodedata.normalize('NFC', text)`` — combine separately-
         encoded combining marks so e.g. "café" (composed) and
         "café" (decomposed) match.
      2. ``str.casefold()`` — case-insensitive matching that handles
         locale-specific folds (German ß → ss, etc.) better than
         ``str.lower()``.
      3. Whitespace collapse — every run of ``str.isspace()``
         characters collapses to a single space; leading and trailing
         whitespace is stripped. Captures CRLF / NBSP / tab variations
         the agent might emit when rewriting.

    Pure / stdlib only. Re-running on already-normalised text is a
    fixed-point — running it twice yields the same result.
    """
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.casefold()
    return " ".join(folded.split())


def is_referenced(belief_content: str, response_text: str) -> bool:
    """Return True iff the normalised belief content appears verbatim
    inside the normalised response.

    The two arguments are passed through :func:`normalize_text`
    before the substring check, so callers don't need to pre-normalise.
    Beliefs shorter than :data:`_MIN_NORMALIZED_LENGTH` after
    normalisation are conservatively classified as not-referenced —
    a 1-3 character belief would match almost any response, which
    would shift relevance posteriors on noise.
    """
    normalised_belief = normalize_text(belief_content)
    if len(normalised_belief) < _MIN_NORMALIZED_LENGTH:
        return False
    normalised_response = normalize_text(response_text)
    return normalised_belief in normalised_response


def score_references(
    belief_pairs: list[tuple[int, str]],
    response_text: str,
    *,
    strategy: str = STRATEGY_EXACT_SUBSTRING,
) -> list[tuple[int, int]]:
    """Score every (event_id, belief_content) pair against the response.

    Returns ``[(event_id, referenced), ...]`` where ``referenced`` is
    0 or 1. The list order matches the input — the sweeper iterates
    it in lockstep with the original event list.

    ``strategy`` is reserved for the n-gram opt-in path (sub-issue).
    Anything other than :data:`STRATEGY_EXACT_SUBSTRING` raises
    ``ValueError`` rather than silently picking a fallback.
    """
    if strategy != STRATEGY_EXACT_SUBSTRING:
        raise ValueError(
            f"unsupported detection strategy: {strategy!r}; "
            f"only {STRATEGY_EXACT_SUBSTRING!r} ships in v1"
        )
    normalised_response = normalize_text(response_text)
    out: list[tuple[int, int]] = []
    for event_id, belief_content in belief_pairs:
        normalised_belief = normalize_text(belief_content)
        if len(normalised_belief) < _MIN_NORMALIZED_LENGTH:
            out.append((event_id, 0))
            continue
        referenced = 1 if normalised_belief in normalised_response else 0
        out.append((event_id, referenced))
    return out
