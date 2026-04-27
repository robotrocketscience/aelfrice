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
