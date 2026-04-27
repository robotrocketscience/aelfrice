"""Noise filter for the synchronous onboard path.

Drops candidate paragraphs that match well-known *non-belief* shapes
before they reach `aelfrice.classification.classify_sentence`. The
intent is signal-to-noise: keep prose that asserts or describes
something specific to the project; drop structural scaffolding,
boilerplate, and stubs that are present in nearly every repository
and contribute nothing distinguishable to retrieval.

Four noise categories, one predicate (`is_noise`) that returns True
when any category fires:

1. **Markdown heading blocks.** A paragraph whose every non-empty line
   starts with `#…#######` (1–6 leading hashes followed by whitespace
   and at least one non-whitespace token). Matches both single-line
   headings (`# Title`) and contiguous heading runs (`# Title\n## Sub`).
   A paragraph that mixes heading and prose is *not* noise.

2. **Checklist blocks.** A paragraph whose every non-empty line begins
   with a markdown task-list marker `- [ ]`, `- [x]`, `* [ ]`, etc.
   Matches both single-line and multi-line checklist runs. A paragraph
   that mixes checklist and prose is not noise.

3. **Three-word fragments.** A paragraph with fewer than four
   whitespace-separated tokens. The existing
   `scanner._MIN_PARAGRAPH_CHARS` threshold (24) catches *short* text;
   this catches *low-content* text — a 24-char paragraph can still be
   only one or two words ("INSTRUCTIONS:", "TODO list incoming").

4. **License-header boilerplate.** A paragraph matching one of seven
   canonical license-preamble signatures: copyright lines (©/(c)),
   the MIT permissive grant, the Apache 2.0 cite, the BSD
   redistribution clause, GPL/LGPL banner, and the bare phrases
   "MIT License" / "All rights reserved." Conservative — false
   negatives (some boilerplate slips through) are preferred to false
   positives (real prose mentioning copyright dropped).

This module is intentionally pure-stdlib and pure-function. No state,
no I/O. The patterns are compiled once at import time. Adding a new
category is a one-function-and-one-test change.

Wired into `aelfrice.scanner.scan_repo`: candidates that match
`is_noise` are skipped before classification and counted in
`ScanResult.skipped_noise`. The original three counters
(`inserted`, `skipped_existing`, `skipped_non_persisting`) are
unchanged in meaning.
"""
from __future__ import annotations

import re
from typing import Final

# Markdown heading line: 1–6 leading hashes, whitespace, then at least
# one non-whitespace character. Matches `# Title`, `### Sub`, but not
# `#tag` (no space) or `# ` (heading with no body).
_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^\s*#{1,6}\s+\S")

# Markdown task-list marker: `- [ ]`, `- [x]`, `* [X]`, `+ [ ]`.
# Whitespace inside the brackets is `[ ]`; checked-off uses `[x]` /
# `[X]`. The trailing `\s` requires at least one space after the
# bracket (so `- [x]done` is not a task list — it's prose with weird
# punctuation).
_CHECKLIST_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*+]\s*\[[ xX]\]\s"
)

# Below this many whitespace-separated tokens, treat as a fragment
# (slogan, label, single-symbol stub). 4 means "you need at least
# subject + verb + object + one more" worth of words.
_MIN_WORDS: Final[int] = 4

# License-header signatures. Tuned conservatively: each pattern picks
# out a phrase that is unmistakably part of an OSI license preamble,
# not a casual mention. Compile-once for speed in onboard hot path.
_LICENSE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bcopyright\s*(\([cC]\)|©)", re.IGNORECASE),
    re.compile(
        r"permission is hereby granted,?\s+free of charge",
        re.IGNORECASE,
    ),
    re.compile(r"licensed under the apache license", re.IGNORECASE),
    re.compile(
        r"redistribution and use in source and binary",
        re.IGNORECASE,
    ),
    re.compile(r"gnu (lesser )?general public license", re.IGNORECASE),
    re.compile(r"^\s*MIT License\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\ball rights reserved\b\.?\s*$", re.IGNORECASE),
)


def is_noise(text: str) -> bool:
    """Return True if `text` should be dropped before classification.

    Empty or whitespace-only text is noise. Otherwise: short-circuit
    on the four categories below; first match wins.
    """
    if not text or not text.strip():
        return True
    if is_three_word_fragment(text):
        return True
    if is_heading_block(text):
        return True
    if is_checklist_block(text):
        return True
    if is_license_boilerplate(text):
        return True
    return False


def is_three_word_fragment(text: str) -> bool:
    """True when `text` has fewer than `_MIN_WORDS` (4) whitespace tokens."""
    return len(text.split()) < _MIN_WORDS


def is_heading_block(text: str) -> bool:
    """True when every non-empty line in `text` is a markdown heading.

    A paragraph that mixes heading and prose returns False — only
    pure heading runs are filtered.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_HEADING_RE.match(ln) is not None for ln in lines)


def is_checklist_block(text: str) -> bool:
    """True when every non-empty line in `text` is a checklist item.

    A paragraph that mixes a task-list line with prose returns False.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_CHECKLIST_RE.match(ln) is not None for ln in lines)


def is_license_boilerplate(text: str) -> bool:
    """True when `text` matches any of the seven license signatures.

    Conservative: only well-known license-preamble phrases match.
    Casual mentions of copyright in prose ("copyright disputes are
    a frequent issue in this domain") will not match because none
    of the seven patterns picks up a bare word.
    """
    return any(p.search(text) is not None for p in _LICENSE_PATTERNS)
