"""Sentence extraction from conversation text.

Implements the Exp 57/61 dumb extraction pipeline: strip noise, split on boundaries,
discard fragments. No classification, no keyword filtering.
"""
from __future__ import annotations

import re


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
    cleaned: str = re.sub(r"```[\s\S]*?```", " ", text)

    # Step 2: strip inline code backticks, keep surrounding text
    cleaned = re.sub(r"`[^`]*`", " ", cleaned)

    # Step 3: strip URLs
    cleaned = re.sub(r"https?://\S+", " ", cleaned)

    # Step 4: strip markdown formatting
    # Headers: lines starting with one or more # characters
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Bold: **text** or __text__
    cleaned = re.sub(r"\*{2}([^*]*)\*{2}", r"\1", cleaned)
    cleaned = re.sub(r"_{2}([^_]*)_{2}", r"\1", cleaned)
    # Italic: *text* or _text_ (single, not double)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
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
