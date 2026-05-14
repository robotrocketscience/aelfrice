"""Pure-regex entity extractor for the v1.3.0 L2.5 retrieval tier.

Reads a string and returns a list of `Entity` records — one per match,
in left-to-right order of match start. No store side-effects; no I/O;
no third-party deps. Mirrors the shape of `triple_extractor.py` so
callers can drop one in next to the other without learning a second
API.

Pattern bank (nine kinds, declared order = priority):

    1. file_path (POSIX)
    2. file_path (Windows, drive-letter)
    3. url
    4. error_code
    5. version
    6. branch
    7. identifier (dotted Python)
    8. identifier (snake_case / CamelCase)
    9. noun_phrase

Earlier-listed kinds win on overlap: a span matched as `file_path` is
removed from the input before later kinds run over it, so
`aelfrice/retrieval.py` produces one `file_path` entity, not three
(`aelfrice/retrieval.py`, `aelfrice.retrieval`, `retrieval`). The
overlap policy mirrors `triple_extractor`'s consumed-interval list.

`noun_phrase` runs LAST and is the most permissive. The reused
`triple_extractor._NP` pattern catches multi-word phrases the
structured kinds miss. Sentence-prose noise from this pattern is the
dominant source of index bloat — the per-call `max_entities` cap
exists to bound that.

Spec: docs/design/entity_index.md § Pattern bank.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from aelfrice.np_pattern import NOUN_PHRASE_PATTERN

# --- Per-kind regexes ----------------------------------------------------

# All patterns are precompiled at import time. The hot path is
# `_iter_kind` which runs `pat.finditer(text)` per kind, so the only
# work paid per call is the regex engine itself.

# POSIX-ish file path. At least one `/`, with a trailing dotted
# extension. Negative lookbehind / lookahead block adjacent word
# characters and `/`s so we don't capture a substring of a longer
# path or URL fragment.
_FILE_PATH_POSIX: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w/])(?:[\w.-]+/)+[\w.-]+\.[\w]+(?![\w/])",
)

# Drive-letter Windows path. Documented in the spec as optional at
# v1.3.0 (the agent rarely emits these on macOS/Linux); we still
# include the pattern so cross-platform corpora don't silently miss
# them.
_FILE_PATH_WIN: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])[A-Za-z]:\\(?:[\w. -]+\\)+[\w.-]+\.[\w]+",
)

# HTTP(S) URL up to the first whitespace / closing bracket / quote.
# Conservative on the right boundary so trailing prose punctuation
# (`see https://x.y/z.`) does not leak into the entity.
_URL: Final[re.Pattern[str]] = re.compile(
    r"\bhttps?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)

# Three error-code families combined into one pattern for a single
# regex pass:
#   - `HTTP 503` / `HTTPS 200` (status codes with a digit triple).
#   - `E404` / `E1001` (E-prefixed Unix-style codes, 3-5 digits).
#   - The named Python exception classes the agent's tracebacks emit,
#     plus `sqlite3.<Class>` for the v1.x SQLite call sites.
_ERROR_CODE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:HTTPS?\s+\d{3}"
    r"|E[0-9]{3,5}"
    r"|(?:OSError|ValueError|RuntimeError|TypeError|KeyError"
    r"|IndexError|AttributeError|FileNotFoundError|PermissionError"
    r"|TimeoutError|sqlite3\.[A-Z][a-zA-Z]+))\b",
)

# Semver-shaped: `v1.2.0`, `1.2.0a0`, `1.2.0-rc1`. The optional `v`
# is the standard product-copy prefix (`release/v*` tags require it).
# Restrict the trailing optional segment to ASCII alphanumerics +
# `.` + `-` so adjacent prose words don't get stitched in.
_VERSION: Final[re.Pattern[str]] = re.compile(
    r"\bv?\d+\.\d+\.\d+(?:[.-][A-Za-z0-9.-]+)?\b",
)

# Conventional branch prefix `<type>/<rest>` per the project commit-
# message rules. The type list mirrors `.githooks/check-commit-msg`.
_BRANCH: Final[re.Pattern[str]] = re.compile(
    r"\b(?:feat|fix|docs|refactor|chore|exp|test|ci|build"
    r"|style|perf|gate|audit|release|revert)/[A-Za-z0-9._/-]+\b",
)

# Dotted Python identifier — `aelfrice.retrieval`, `store.list_locked_beliefs`.
# Disambiguated from sentence prose by requiring at least one dot AND
# a lowercase head (so casual "Mr. Smith" or "Inc." doesn't match).
_IDENT_DOTTED: Final[re.Pattern[str]] = re.compile(
    r"\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b",
)

# snake_case (`session_id`, `apply_feedback`) or CamelCase
# (`MemoryStore`, `RetrievalCache`) with at least two parts. The two-
# part requirement keeps single English words (`apply`, `session`)
# out — those are noun_phrase candidates.
_IDENT_SNAKE_OR_CAMEL: Final[re.Pattern[str]] = re.compile(
    r"\b(?:[a-z]+_[a-z0-9_]+|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b",
)

# Reused noun-phrase pattern from triple_extractor. Up to 5 word
# tokens, optionally led by an article / possessive. We DON'T anchor
# on word boundaries because the spec puts noun_phrase LAST and
# requires it to skip already-consumed spans, which the consumed-
# interval list handles separately.
_NOUN_PHRASE: Final[re.Pattern[str]] = re.compile(NOUN_PHRASE_PATTERN)


# Kind constants. Exposed so callers (tests, future ranker) can refer
# to them without stringly-typing.
KIND_FILE_PATH: Final[str] = "file_path"
KIND_URL: Final[str] = "url"
KIND_ERROR_CODE: Final[str] = "error_code"
KIND_VERSION: Final[str] = "version"
KIND_BRANCH: Final[str] = "branch"
KIND_IDENTIFIER: Final[str] = "identifier"
KIND_NOUN_PHRASE: Final[str] = "noun_phrase"

KINDS: Final[frozenset[str]] = frozenset({
    KIND_FILE_PATH,
    KIND_URL,
    KIND_ERROR_CODE,
    KIND_VERSION,
    KIND_BRANCH,
    KIND_IDENTIFIER,
    KIND_NOUN_PHRASE,
})


# Ordered (kind, regex) tuples. Earlier wins on overlap. file_path
# runs before url so a path inside a URL still surfaces only as the
# URL; in practice URL appears earlier here so URL wins on URL-
# embedded paths. The ordering is the design — see spec § Pattern
# bank notes.
_PATTERN_BANK: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (KIND_FILE_PATH, _FILE_PATH_POSIX),
    (KIND_FILE_PATH, _FILE_PATH_WIN),
    (KIND_URL, _URL),
    (KIND_ERROR_CODE, _ERROR_CODE),
    (KIND_VERSION, _VERSION),
    (KIND_BRANCH, _BRANCH),
    (KIND_IDENTIFIER, _IDENT_DOTTED),
    (KIND_IDENTIFIER, _IDENT_SNAKE_OR_CAMEL),
    (KIND_NOUN_PHRASE, _NOUN_PHRASE),
)


DEFAULT_MAX_ENTITIES: Final[int] = 64
"""Hard ceiling per `extract_entities` call. Past this, overflow
matches are dropped on the floor and the caller (or telemetry) sees
the truncation only via the returned list length being == cap. Sized
to protect against pathological inputs (license blobs, JSON pasted
into a transcript)."""


@dataclass(frozen=True)
class Entity:
    """One extracted entity with its source span.

    `raw` is the literal matched substring; `lower` is `raw.lower()`
    used as the index key. `kind` is one of the KIND_* constants.
    `span_start` / `span_end` are byte offsets into the source text
    used by the future highlighting / span-aware ranker (currently
    written for free since the regex `Match` exposes them).
    """
    raw: str
    lower: str
    kind: str
    span_start: int
    span_end: int


def _overlaps(start: int, end: int, consumed: list[tuple[int, int]]) -> bool:
    """Return True if [start, end) intersects any [a, b) interval in
    `consumed`. Linear scan — `consumed` is small in practice (one
    entry per accepted match, capped at DEFAULT_MAX_ENTITIES).
    """
    for a, b in consumed:
        if start < b and a < end:
            return True
    return False


def extract_entities(
    text: str,
    *,
    max_entities: int = DEFAULT_MAX_ENTITIES,
) -> list[Entity]:
    """Return entities in left-to-right order of match start.

    Pure regex over `text`. No side effects, no I/O, no exceptions
    (an empty / non-string-shaped input returns []). Earlier kinds in
    `_PATTERN_BANK` win on overlap — once a span is accepted, later
    kinds skip any match that intersects it.

    `max_entities` is a hard ceiling per call. Past the cap, overflow
    matches are dropped silently. The cap defaults to 64 per the
    docs/design/entity_index.md § Refresh strategy back-of-envelope.
    """
    if not text or max_entities <= 0:
        return []

    out: list[Entity] = []
    consumed: list[tuple[int, int]] = []

    for kind, pat in _PATTERN_BANK:
        if len(out) >= max_entities:
            break
        for m in pat.finditer(text):
            if len(out) >= max_entities:
                break
            start, end = m.start(), m.end()
            if start >= end:
                # Zero-width match — skip. The noun-phrase pattern's
                # leading determiner is optional, but the body is at
                # least one token, so a zero-width match here would
                # indicate a regex bug. Defensive guard.
                continue
            if _overlaps(start, end, consumed):
                continue
            raw = m.group(0)
            # Strip trailing whitespace defensively — _NOUN_PHRASE
            # can capture trailing prose tokens whose final char is a
            # newline split. `raw.rstrip()` keeps the span coherent
            # without re-running the regex.
            stripped = raw.rstrip()
            if not stripped:
                continue
            adj_end = start + len(stripped)
            consumed.append((start, adj_end))
            out.append(Entity(
                raw=stripped,
                lower=stripped.lower(),
                kind=kind,
                span_start=start,
                span_end=adj_end,
            ))

    out.sort(key=lambda e: e.span_start)
    return out
