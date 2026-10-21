"""The sentences ingest stores from a turn (#809, #1650).

Split out of ``aelfrice.ingest`` so a module that needs the same split,
such as ``aelfrice.reworded_restatement``, can import it without
importing ingest. ``aelfrice.ingest`` re-exports ``stored_sentences``.
"""
from __future__ import annotations

import re
from typing import Final

from aelfrice.extraction import extract_sentences
from aelfrice.noise_filter import is_transcript_noise

# #809 / #785 § 3: pattern-based subfloor-noise detector with
# length-floor scope.
#
# Three short-fragment classes are filtered at the sentence-level
# ingest boundary: code-fence boundaries (` ```bash`, `` ``` ``),
# header stubs ending in `:` ("Acceptance criteria:", "Open
# questions:"), and markdown bullet stubs (`- run tests`, `* foo`).
# A matched sentence does not become a freestanding belief row; when
# it sits between two full-length-belief sentences in the same turn
# it attaches as `anchor_text` on an intra-turn DERIVED_FROM edge,
# preserving relational meaning without inflating the belief-row
# count. Unanchored matches are silently dropped.
#
# The pattern check is **scoped to short content** by a length cap
# (`_SUBFLOOR_MAX_LEN`). The spec's intent is to filter short
# fragments that carry no semantic claim; long-form content that
# happens to start with `- ` or end in `:` (real prose statements,
# multi-sentence list items that survived `extract_sentences`) is
# load-bearing and must not be dropped. A pattern-only check
# (without the length cap) over-applies and drops long-form prose
# ending in `:` — e.g.,
# "If you look at the way the rebuilder picks beliefs, the order is
# always the same:".
#
# Why 80 chars: matches the spec literal in
# docs/feature-ingest-speaker-gate.md §3. Empirically, sentences of
# this kind that exceed ~80 chars are dominantly real prose; below
# ~80 chars they are dominantly header stubs / fragment markers.
#
# A standalone 80-char length floor (without the pattern check) is
# also wrong — it would drop short legit claims like "The default
# port is 8080." that the test suite encodes as ingest-eligible.
# The combination (pattern AND < 80) catches the load-bearing noise
# class while preserving both short legit claims and long-form
# prose.
#
# Acknowledged residual false-positive risk: a short complete
# sentence ending with `:` ("He said:", "Note:") still drops.
# These are ambiguous in isolation and the cost of preserving them
# is a more elaborate rule (verb-detection, list-following-context)
# that gets fragile fast. Trade-off accepted at empirical scope;
# re-measure if production data surfaces a non-trivial miss rate.
_SUBFLOOR_MAX_LEN: Final[int] = 80
_SUBFLOOR_BULLET_PREFIX = re.compile(r"^[-*+]\s")


def _looks_like_subfloor_noise(sentence: str) -> bool:
    """True when `sentence` matches one of the three short-fragment
    noise patterns (code-fence boundary, `:`-suffix header stub,
    markdown bullet stub) AND its stripped length is below
    `_SUBFLOOR_MAX_LEN`. Long-form content is never noise regardless
    of leading or trailing markers — see module docstring."""
    stripped = sentence.strip()
    if not stripped or len(stripped) >= _SUBFLOOR_MAX_LEN:
        return False
    if stripped.startswith("```"):
        return True
    if stripped.endswith(":"):
        return True
    if _SUBFLOOR_BULLET_PREFIX.match(stripped):
        return True
    return False


def partition_sentences(text: str) -> tuple[list[str], list[list[str]]]:
    """The sentences a turn stores as beliefs, and the clauses between them.

    #809: sentences are split, transcript noise is dropped, and sub-floor
    clauses are separated from full-length belief candidates.
    `subfloor_between[i]` holds the sub-floor clauses (in original order)
    that preceded `full_sentences[i]`; clauses before the first full
    sentence and after the last are unanchored and dropped.
    """
    sentences = [s for s in extract_sentences(text) if not is_transcript_noise(s)]
    full_sentences: list[str] = []
    subfloor_between: list[list[str]] = []
    pending_subfloor: list[str] = []
    for sentence in sentences:
        if _looks_like_subfloor_noise(sentence):
            pending_subfloor.append(sentence)
        else:
            full_sentences.append(sentence)
            subfloor_between.append(pending_subfloor)
            pending_subfloor = []
    return full_sentences, subfloor_between


def stored_sentences(text: str) -> list[str]:
    """The sentences ingest would store as beliefs from ``text`` (#1650).

    The same split and filters a turn goes through, so a caller can ask
    which beliefs typing ``text`` would produce. ``derive()`` can still
    decline one (``classify_sentence(...).persist``).
    """
    return partition_sentences(text)[0]
