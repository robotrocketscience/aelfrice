"""Sentence extraction from conversation text.

Implements the Exp 57/61 dumb extraction pipeline: strip noise, split on boundaries,
discard fragments. No classification, no keyword filtering.
"""
from __future__ import annotations

import re


# Triple-backtick region, including the language tag. Exported rather
# than inlined because the onboard path strips fences too (#1371 §10) and
# the two must agree: a second copy of this pattern is a divergence
# waiting to happen, and the whole defect was that only one path had it.
#
# Anchored to line starts (`(?m)^`), and deliberately so. Unanchored, a
# ``` appearing *inside prose* — a sentence about markdown, an inline code
# span — pairs with the opening delimiter of the next real fence and flips
# parity for the rest of the document: the prose tail is deleted as fence
# interior and the code body is promoted to prose. On the onboard path,
# which walks a repo's own documentation, that is common enough to matter
# — this repo's `docs/user/CONFIG.md` loses 128 of its 248 paragraphs to
# it. Both paths inherit the anchoring, which is the point of sharing one
# object. Up to three leading spaces are allowed, matching CommonMark; a
# closing delimiter is still required, so an unterminated fence matches
# nothing and its content survives, unchanged from before.
#
# The closing delimiter takes only trailing whitespace, never an info
# string. CommonMark allows a language on the *opening* fence alone, so a
# ```` ```python ```` line inside a block is body text — matching it as a
# close ends the span early, leaking the remainder of the block into the
# cleaned output along with the real closing delimiter as stray prose.
CODE_FENCE_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]{0,3}```[^\n]*\n[\s\S]*?^[ \t]{0,3}```[ \t]*\r?$"
)

# Minimum character length for a sentence fragment to be kept.
_MIN_LEN: int = 10


def extract_sentences(text: str) -> list[str]:
    """Extract atomic sentences from conversation text.

    Process:
    1. Strip code blocks (triple-backtick regions)
    2. Strip inline code backticks but keep surrounding text
    3. Strip URLs
    4. Strip markdown formatting (headers, bold, italic, table rows, list markers)
    5. Split on newlines first
    6. Within each line, split on sentence-ending punctuation followed by space
    7. Discard fragments under 10 characters

    Returns list of clean sentences.
    """
    # Step 1: strip code blocks (triple-backtick regions, including language tag)
    cleaned: str = CODE_FENCE_RE.sub(" ", text)

    # Step 2: strip inline code backticks, keep surrounding text
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)

    # Step 3: strip URLs
    cleaned = re.sub(r"https?://\S+", " ", cleaned)

    # Step 4: strip markdown formatting
    # Headers: lines starting with one or more # characters
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Bold: **text** or __text__
    cleaned = re.sub(r"\*{2}([^*]*)\*{2}", r"\1", cleaned)
    # Underscore-bold/italic must be at a true word boundary so
    # snake_case identifiers and file paths like ``auth_service.py``
    # are not mangled. Negative lookbehind/lookahead require non-
    # alphanumeric context outside the underscore (``\b`` does not
    # work because ``_`` is in ``\w``).
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_{2}([^_]*)_{2}(?![A-Za-z0-9])", r"\1", cleaned,
    )
    # Italic: *text* or _text_ (single, not double)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(
        r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])", r"\1", cleaned,
    )
    # Markdown table rows: lines containing | characters
    cleaned = re.sub(r"^\s*\|.*\|\s*$", " ", cleaned, flags=re.MULTILINE)
    # List markers: leading -, *, +, or numbered list items
    cleaned = re.sub(r"^[ \t]*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[ \t]*\d+\.\s+", "", cleaned, flags=re.MULTILINE)

    # Step 5: split on newlines first
    lines: list[str] = cleaned.splitlines()

    sentences: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Step 6: within each line, split on sentence-ending punctuation followed by space
        # Use re.split with a lookbehind so the punctuation stays with the preceding sentence.
        parts: list[str] = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            # Step 7: discard fragments under 10 characters
            if len(part) >= _MIN_LEN:
                sentences.append(part)

    return sentences
