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

Wired into `aelfrice.scanner.scan_repo`: candidates that match
`is_noise` are skipped before classification and counted in
`ScanResult.skipped_noise`.

## Power-user configuration

The user-facing surface is a single TOML file: `.aelfrice.toml` at the
project root (or any ancestor directory). One file, one schema, no
regex required to use it correctly.

```toml
# .aelfrice.toml
[noise]
# Turn off any of the four built-in categories. Subset of:
# headings | checklists | fragments | license
disable = ["fragments"]

# Override the default fragment threshold (defaults to 4).
min_words = 3

# Drop paragraphs containing any of these whole words.
# Match is case-insensitive and word-bounded — "jso" drops paragraphs
# containing the standalone word "jso", but NOT "json" or "jso-files".
exclude_words = ["jso", "DRAFT", "WIP"]

# Drop paragraphs containing any of these substrings, anywhere in the
# text. Match is case-insensitive but otherwise literal — punctuation
# and whitespace inside the phrase are matched verbatim.
exclude_phrases = ["Last updated:", "TODO:", "FIXME"]
```

`scan_repo` discovers `.aelfrice.toml` by walking up from `root` to
the filesystem root. The first `.aelfrice.toml` found wins. If none
is found, the default config ships (all four sub-predicates on,
`min_words = 4`, no excludes). Malformed TOML, unknown keys, or
wrong-typed values degrade silently to the default or skip the bad
entry; the failure is traced to `stderr` so onboard remains
non-blocking.

The `NoiseConfig` dataclass is also exposed for library use (tests,
programmatic callers): `is_noise(text, config)` and
`scan_repo(store, root, noise_config=...)`. The TOML file is the
*recommended* surface; the dataclass is the implementation.
"""
from __future__ import annotations

import re
import sys
import tomllib
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Final, Mapping

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

# Default below this many whitespace-separated tokens: treat as a
# fragment (slogan, label, single-symbol stub). 4 means "you need at
# least subject + verb + object + one more" worth of words.
DEFAULT_MIN_WORDS: Final[int] = 4

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

# Recognised category names for the `disable` field of [noise].
_VALID_DISABLE_TOKENS: Final[frozenset[str]] = frozenset({
    "headings", "checklists", "fragments", "license",
})


# ---------------------------------------------------------------------------
# Transcript-noise filter — compiled once at module load
# ---------------------------------------------------------------------------
#
# Five categories of sentences that appear in transcript turns but carry
# zero belief content. Each category is documented below and tested in
# tests/test_noise_filter.py.
#
# 1. Shell-command shape: starts with a recognised shell prefix at the
#    leftmost position (case-sensitive). Covers `cd /`, `git `, `gh `,
#    `uv run`, `pytest`, `python `. The leading space in `git ` and
#    `gh ` is intentional — it distinguishes the command from prose that
#    starts with a word that merely contains the token (e.g. "ghosts").
#
# 2. Tool-call rendering glyph: ⏺ (U+23FA). Emitted at the start
#    of tool-call narration lines by some transcript surfaces.
#
# 3. Pseudo-XML worktree/task tags: `<worktree`, `<output-file`,
#    `<task-`, `<summary>Background`. These are structural delimiters
#    injected by orchestration layers; they are not prose beliefs.
#
# 4. Single-word progress emits: matches `^[A-Z][a-z]+ing\.$` — a lone
#    capitalised gerund followed by a full stop. Examples: "Polling.",
#    "Running.", "Waiting." Note that "Standing by." does NOT match
#    this pattern (two words); it is caught by category 5.
#
# 5. Agent ack emits: short one-line acknowledgements that convey no
#    project-specific knowledge. Pattern allows the bare keyword or the
#    keyword followed by up to 40 characters. Examples: "Yes.",
#    "Standing by.", "Polling for results.", "Nothing to report.",
#    "Ready when you are.", "No changes needed."

_TRANSCRIPT_SHELL_PREFIXES: Final[tuple[str, ...]] = (
    "cd /",
    "git ",
    "gh ",
    "uv run",
    "pytest",
    "python ",
)

# U+23FA — tool-call rendering glyph emitted by some transcript surfaces.
_TRANSCRIPT_GLYPH_PREFIX: Final[str] = "⏺"

_TRANSCRIPT_XML_PREFIXES: Final[tuple[str, ...]] = (
    "<worktree",
    "<output-file",
    "<task-",
    "<summary>Background",
)

# Single-word capitalised gerund followed by a full stop: "Polling.", "Running."
_TRANSCRIPT_PROGRESS_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z][a-z]+ing\.$"
)

# Agent ack emit: bare keyword or keyword + optional short trailing text.
_TRANSCRIPT_ACK_RE: Final[re.Pattern[str]] = re.compile(
    r"^(Yes|No|Standing by|Ready|Nothing|Polling)( .{0,40})?\.?$"
)

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration surface for the noise filter.

    All defaults match the v1.0.1 ship behaviour. A power user
    overrides via a `.aelfrice.toml` file at the project root (or any
    ancestor); see module docstring for the schema. The dataclass
    itself is exposed for tests and library use.

    Disabling a sub-predicate (`drop_<category> = False`) means the
    category never fires. Setting `min_words = 0` disables the
    fragment check (no paragraph has fewer than zero tokens).

    `exclude_words` is a tuple of strings. Each word is compiled to a
    case-insensitive word-boundary regex at construction
    (`\\bword\\b`), so "jso" drops paragraphs containing the standalone
    token "jso" but NOT "json" or "jso-files". Empty tuple by default.

    `exclude_phrases` is a tuple of strings. Each phrase is matched
    case-insensitively as a literal substring anywhere in the text;
    punctuation and whitespace inside the phrase are matched verbatim.
    Empty tuple by default.

    No regex is exposed at the user surface. If you need true regex
    semantics, extend the module — that is a deliberate v1.0.1
    constraint, not an oversight.
    """
    drop_headings: bool = True
    drop_checklists: bool = True
    drop_fragments: bool = True
    drop_license: bool = True
    min_words: int = DEFAULT_MIN_WORDS
    exclude_words: tuple[str, ...] = ()
    exclude_phrases: tuple[str, ...] = ()
    # Internal pre-compiled forms. Populated in __post_init__ so the
    # hot path (is_noise) does no work per call.
    _word_patterns: tuple[re.Pattern[str], ...] = field(
        default=(), init=False, repr=False, compare=False,
    )
    _phrase_lowers: tuple[str, ...] = field(
        default=(), init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        compiled = tuple(
            re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
            for w in self.exclude_words
            if w
        )
        lowers = tuple(p.lower() for p in self.exclude_phrases if p)
        # frozen=True forbids normal assignment; use object.__setattr__.
        object.__setattr__(self, "_word_patterns", compiled)
        object.__setattr__(self, "_phrase_lowers", lowers)

    @classmethod
    def default(cls) -> "NoiseConfig":
        """Return the canonical v1.0.1 configuration."""
        return cls()

    @classmethod
    def from_mapping(
        cls,
        section: Mapping[str, Any],
        *,
        stderr: IO[str] | None = None,
    ) -> "NoiseConfig":
        """Build from a parsed `[noise]` table.

        Tolerant of unknown keys (forward-compat) and wrong-typed
        values (skip-and-warn). Non-string entries in `exclude_words`
        or `exclude_phrases` are skipped individually; the rest of
        the config still loads.
        """
        serr: IO[str] = stderr if stderr is not None else sys.stderr

        disabled = _parse_string_list(
            section, "disable", stderr=serr,
        )
        disabled_lower = {s.lower() for s in disabled}

        min_words_raw = section.get("min_words", DEFAULT_MIN_WORDS)
        if not isinstance(min_words_raw, int) or isinstance(
            min_words_raw, bool
        ):
            print(
                f"aelfrice noise_filter: ignoring [noise] min_words "
                f"(expected int, got {type(min_words_raw).__name__})",
                file=serr,
            )
            min_words = DEFAULT_MIN_WORDS
        else:
            min_words = max(0, min_words_raw)

        words = _parse_string_list(
            section, "exclude_words", stderr=serr,
        )
        phrases = _parse_string_list(
            section, "exclude_phrases", stderr=serr,
        )

        return cls(
            drop_headings="headings" not in disabled_lower,
            drop_checklists="checklists" not in disabled_lower,
            drop_fragments="fragments" not in disabled_lower,
            drop_license="license" not in disabled_lower,
            min_words=min_words,
            exclude_words=tuple(words),
            exclude_phrases=tuple(phrases),
        )

    @classmethod
    def from_toml_path(
        cls,
        path: Path,
        *,
        stderr: IO[str] | None = None,
    ) -> "NoiseConfig":
        """Load a NoiseConfig from a specific `.aelfrice.toml` path.

        Reads the file, parses TOML, looks for the `[noise]` table.
        Returns the default config if the file is missing, unreadable,
        malformed, or has no `[noise]` table. All failures trace to
        `stderr` (defaults to `sys.stderr`); none raise.
        """
        serr: IO[str] = stderr if stderr is not None else sys.stderr
        try:
            raw = path.read_bytes()
        except (OSError, PermissionError) as exc:
            print(
                f"aelfrice noise_filter: cannot read {path}: {exc}",
                file=serr,
            )
            return cls.default()
        try:
            parsed = tomllib.loads(raw.decode("utf-8", errors="replace"))
        except tomllib.TOMLDecodeError as exc:
            print(
                f"aelfrice noise_filter: malformed TOML in {path}: {exc}",
                file=serr,
            )
            return cls.default()
        section = parsed.get("noise", {})
        if not isinstance(section, dict):
            print(
                f"aelfrice noise_filter: [noise] in {path} is not a table",
                file=serr,
            )
            return cls.default()
        return cls.from_mapping(section, stderr=serr)

    @classmethod
    def discover(
        cls,
        start: Path | None = None,
        *,
        stderr: IO[str] | None = None,
    ) -> "NoiseConfig":
        """Walk up from `start` looking for `.aelfrice.toml`.

        First file found wins. If none is found before reaching the
        filesystem root, the default config is returned. `start=None`
        uses the current working directory.
        """
        current = (start if start is not None else Path.cwd()).resolve()
        seen: set[Path] = set()
        while current not in seen:
            seen.add(current)
            candidate = current / CONFIG_FILENAME
            if candidate.is_file():
                return cls.from_toml_path(candidate, stderr=stderr)
            if current.parent == current:
                break
            current = current.parent
        return cls.default()


_DEFAULT_CONFIG: Final[NoiseConfig] = NoiseConfig.default()


def _resolve(config: NoiseConfig | None) -> NoiseConfig:
    return config if config is not None else _DEFAULT_CONFIG


def _parse_string_list(
    section: Mapping[str, Any],
    key: str,
    *,
    stderr: IO[str],
) -> list[str]:
    """Pull a list-of-strings field from a TOML section. Tolerant: a
    missing key returns []; a non-list value warns and returns [];
    non-string entries inside the list are dropped with a warning."""
    raw = section.get(key, [])
    if not isinstance(raw, list):
        print(
            f"aelfrice noise_filter: ignoring [noise] {key} "
            f"(expected list, got {type(raw).__name__})",
            file=stderr,
        )
        return []
    out: list[str] = []
    for item in raw:  # type: ignore[union-attr]
        if isinstance(item, str):
            out.append(item)
        else:
            print(
                f"aelfrice noise_filter: skipping non-string entry "
                f"in [noise] {key}",
                file=stderr,
            )
    return out


def is_noise(text: str, config: NoiseConfig | None = None) -> bool:
    """Return True if `text` should be dropped before classification.

    Empty or whitespace-only text is noise unconditionally
    (cannot be opted out via config). Otherwise: short-circuit on
    the four categories under `config`; first match wins. User
    `exclude_words` and `exclude_phrases` run last.
    """
    if not text or not text.strip():
        return True
    cfg = _resolve(config)
    if cfg.drop_headings and is_heading_block(text, cfg):
        return True
    if cfg.drop_checklists and is_checklist_block(text, cfg):
        return True
    if cfg.drop_fragments and is_three_word_fragment(text, cfg):
        return True
    if cfg.drop_license and is_license_boilerplate(text, cfg):
        return True
    if cfg._word_patterns and any(
        p.search(text) is not None for p in cfg._word_patterns
    ):
        return True
    if cfg._phrase_lowers:
        text_lower = text.lower()
        if any(phr in text_lower for phr in cfg._phrase_lowers):
            return True
    return False


def is_three_word_fragment(
    text: str, config: NoiseConfig | None = None,
) -> bool:
    """True when `text` has fewer than `config.min_words` whitespace tokens."""
    cfg = _resolve(config)
    return len(text.split()) < cfg.min_words


def is_heading_block(
    text: str, config: NoiseConfig | None = None,
) -> bool:
    """True when every non-empty line in `text` is a markdown heading.

    A paragraph that mixes heading and prose returns False — only
    pure heading runs are filtered. The `config` argument is currently
    only consulted for forward-compatibility; the heading regex itself
    is fixed by the markdown spec.
    """
    _ = config
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_HEADING_RE.match(ln) is not None for ln in lines)


def is_checklist_block(
    text: str, config: NoiseConfig | None = None,
) -> bool:
    """True when every non-empty line in `text` is a checklist item.

    A paragraph that mixes a task-list line with prose returns False.
    `config` reserved for forward-compatibility.
    """
    _ = config
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_CHECKLIST_RE.match(ln) is not None for ln in lines)


def is_license_boilerplate(
    text: str, config: NoiseConfig | None = None,
) -> bool:
    """True when `text` matches any of the seven license signatures.

    Conservative: only well-known license-preamble phrases match.
    Casual mentions of copyright in prose ("copyright disputes are
    a frequent issue in this domain") will not match because none
    of the seven patterns picks up a bare word.
    `config` reserved for forward-compatibility.
    """
    _ = config
    return any(p.search(text) is not None for p in _LICENSE_PATTERNS)


def is_transcript_noise(sentence: str) -> bool:
    """Return True if `sentence` is transcript scaffolding, not a belief.

    Checks five categories in order; first match returns True:

    1. **Shell-command shape** — starts with a recognised shell prefix
       (`cd /`, `git `, `gh `, `uv run`, `pytest`, `python `).
       Match is case-sensitive and position-anchored at index 0.
    2. **Tool-call rendering glyph** — starts with ⏺ (U+23FA).
    3. **Pseudo-XML structural tags** — starts with `<worktree`,
       `<output-file`, `<task-`, or `<summary>Background`.
    4. **Single-word progress emit** — matches `^[A-Z][a-z]+ing\\.$`
       (a lone capitalised gerund and a full stop, nothing else).
    5. **Agent ack emit** — matches
       `^(Yes|No|Standing by|Ready|Nothing|Polling)( .{0,40})?\\.*$`;
       covers bare keywords and short trailing phrases up to 40 chars.

    All patterns are case-sensitive as written. Empty or whitespace-only
    strings return False (they are handled upstream by `is_noise`).
    """
    if not sentence or not sentence.strip():
        return False

    # Category 1: shell-command shape
    for prefix in _TRANSCRIPT_SHELL_PREFIXES:
        if sentence.startswith(prefix):
            return True

    # Category 2: tool-call rendering glyph (U+23FA)
    if sentence.startswith(_TRANSCRIPT_GLYPH_PREFIX):
        return True

    # Category 3: pseudo-XML structural tags
    for prefix in _TRANSCRIPT_XML_PREFIXES:
        if sentence.startswith(prefix):
            return True

    # Category 4: single-word progress emit
    if _TRANSCRIPT_PROGRESS_RE.match(sentence) is not None:
        return True

    # Category 5: agent ack emit
    if _TRANSCRIPT_ACK_RE.match(sentence) is not None:
        return True

    return False


# Punctuation characters stripped during N-gram tokenisation. We remove
# everything that is not a word character or whitespace so that
# "features," and "features" are treated as the same token.
_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Used by `similarity_to_reference` for both the query text and the
    reference document so that tokenisation is symmetric.
    """
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _ngrams(tokens: list[str], n: int) -> frozenset[tuple[str, ...]]:
    """Return the set of unique N-gram tuples from *tokens*.

    A text shorter than *n* tokens produces an empty set, which
    results in a Jaccard score of 0.0 (disjoint — no shared N-grams).
    """
    if len(tokens) < n:
        return frozenset()
    return frozenset(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def similarity_to_reference(
    text: str,
    reference_path: Path,
    *,
    n: int = 3,
    threshold: float = 0.6,
) -> tuple[bool, float, str | None]:
    """N-gram Jaccard similarity gate against a reference document.

    Tokenises both ``text`` and the file at ``reference_path`` into
    normalised N-gram windows (default n=3 words, lowercased, with
    punctuation stripped). Computes Jaccard similarity over the
    N-gram sets.

    Returns ``(over_threshold, score, matched_passage_excerpt)``.

    * ``over_threshold`` — True when ``score >= threshold``.
    * ``score`` — Jaccard similarity in [0.0, 1.0].
    * ``matched_passage_excerpt`` — a short excerpt of the best
      matching window from ``text`` when ``over_threshold`` is True,
      else None.

    **Failure mode (important):** this is a *surface-similarity* gate,
    not a *plagiarism* gate. It catches near-verbatim paraphrase —
    re-orderings, minor substitutions, mild expansions of the same
    surface tokens. It does **not** catch semantic paraphrase (content
    reworded into entirely different vocabulary). False positives are
    expected for short technical phrases that legitimately share many
    tokens with the reference (e.g. repeated API names, standard error
    messages). Tune ``threshold`` upward to reduce false positives at
    the cost of sensitivity.

    Mirrors the shape of ``is_noise`` / ``is_three_word_fragment``:
    pure function, no side effects, stdlib only, deterministic across
    re-runs.

    Callers that need only the boolean can write::

        flagged, _, _ = similarity_to_reference(text, ref)

    The ``reference_path`` is read fresh on every call so that the
    caller controls caching. For high-volume use, pre-read the file and
    call ``_ngrams(_tokenise(content), n)`` once, then reuse the
    frozenset.
    """
    ref_text = reference_path.read_text(encoding="utf-8", errors="replace")
    ref_tokens = _tokenise(ref_text)
    ref_set = _ngrams(ref_tokens, n)

    text_tokens = _tokenise(text)
    text_set = _ngrams(text_tokens, n)

    if not text_set and not ref_set:
        # Both sides have fewer than n tokens — no N-grams possible.
        # Treat as perfectly similar (both are empty / trivially the same).
        return (True, 1.0, text.strip()[:120] or None) if text.strip() else (False, 0.0, None)

    union = text_set | ref_set
    if not union:
        return (False, 0.0, None)

    intersection = text_set & ref_set
    score = len(intersection) / len(union)

    over = score >= threshold
    excerpt: str | None = None
    if over:
        # Find the contiguous window of tokens in *text* that contributes
        # the most matching N-grams, then reconstruct a short excerpt.
        best_start = 0
        best_count = 0
        window = n * 5  # ~5x the gram width: enough context without being verbose
        for i in range(max(1, len(text_tokens) - window + 1)):
            chunk = frozenset(
                tuple(text_tokens[j : j + n])
                for j in range(i, min(i + window, len(text_tokens) - n + 1))
            )
            count = len(chunk & ref_set)
            if count > best_count:
                best_count = count
                best_start = i
        end = min(best_start + window, len(text_tokens))
        excerpt = " ".join(text_tokens[best_start:end])[:200]

    return (over, score, excerpt)
