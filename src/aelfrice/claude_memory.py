"""Parser and slot-equality comparison for the claude-memory file format.

The upstream auto-memory writes a MEMORY.md index file at
``~/.claude/projects/<encoded-path>/memory/MEMORY.md`` and individual
per-memory ``.md`` files in the same directory. This module provides:

1. ``derive_memory_dir(project_path)`` — converts an absolute project
   path to the corresponding claude-memory directory using the same
   path-encoding the upstream tool applies: strip the leading slash,
   replace every remaining ``/`` with ``-``.

2. ``parse_memory_bullets(text)`` — parses the MEMORY.md bullet list
   into a flat list of :class:`MemoryBullet` namedtuples without any
   filesystem I/O.

3. ``extract_slot(text)`` — extracts a deterministic ``(subject,
   predicate)`` slot from a belief/bullet string via regex. Returns
   ``None`` when no slot can be reliably identified (callers skip those
   rows for comparison).

4. ``SlotRow`` — a namedtuple that carries the extracted slot fields
   plus the raw text and its source path, returned by both the bullet
   parser and the aelfrice locked-belief extractor.

5. ``compare_slots`` — takes two iterables of :class:`SlotRow` and
   returns the four-bucket :class:`ComparisonResult` used by
   ``aelf audit-claude-memory``.

The slot-extraction regex is intentionally conservative: a "slot" is a
two-word ``Subject Predicate`` prefix (optionally followed by
``[IsA|HasA|…]``) at the start of a sentence-like fragment. Bullets that
do not match are placed in the bucket they were found in (aelfrice-only
or claude-memory-only) rather than being compared, consistent with the
locked PHILOSOPHY decision to prefer deterministic stdlib gates over
fuzzy heuristics.

No filesystem state is mutated here. This module is safe for #938
(``aelf:search`` contradiction-flag column) to import without pulling
in any CLI surface.
"""
from __future__ import annotations

import re
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Final, NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MemoryBullet(NamedTuple):
    """A single bullet parsed from a MEMORY.md index file.

    :param link_text: the human-readable label from ``[label](path)``
    :param link_target: the relative path from ``[label](path)``
    :param description: the text that follows `` — `` (may be empty)
    :param source_path: absolute path of the MEMORY.md file this came from
    """
    link_text: str
    link_target: str
    description: str
    source_path: str


class SlotRow(NamedTuple):
    """A belief or bullet with an extracted (subject, predicate) slot.

    When ``slot_subject`` or ``slot_predicate`` is an empty string the slot
    could not be extracted; :func:`compare_slots` skips those rows.

    :param slot_subject: first word(s) extracted as the subject
    :param slot_predicate: following word(s) extracted as the predicate
    :param slot_value: rest of the content after the predicate (the "value")
    :param raw_text: original full text of the belief or bullet description
    :param source: label identifying where this row came from
      (belief id for aelfrice rows; MEMORY.md link-text for claude rows)
    :param source_path: file path (DB path or MEMORY.md path)
    """
    slot_subject: str
    slot_predicate: str
    slot_value: str
    raw_text: str
    source: str
    source_path: str


class ComparisonResult(NamedTuple):
    """Four-bucket result from :func:`compare_slots`.

    Each list contains :class:`SlotRow` pairs or single rows:

    :param duplicates: list of ``(aelfrice_row, claude_row)`` pairs where
      slot subject+predicate AND slot value match exactly (case-insensitive)
    :param contradictions: list of ``(aelfrice_row, claude_row)`` pairs where
      slot subject+predicate match but values differ
    :param aelfrice_only: :class:`SlotRow` items whose slot has no match in
      the claude-memory set
    :param claude_only: :class:`SlotRow` items whose slot has no match in
      the aelfrice set
    """
    duplicates: list[tuple[SlotRow, SlotRow]]
    contradictions: list[tuple[SlotRow, SlotRow]]
    aelfrice_only: list[SlotRow]
    claude_only: list[SlotRow]


class MemoryFile(NamedTuple):
    """A parsed per-memory ``.md`` file from the claude-memory store (#985).

    The upstream auto-memory writes one fact per file with a YAML-ish
    frontmatter block:

    .. code-block:: text

        ---
        name: gitea-token-trap
        description: Bash tool uses a frozen zshrc snapshot
        metadata:
          type: feedback
        ---

        <the fact body>

    :param name: the ``name:`` slug (stable identity across edits); empty
      string when the file has no ``name`` key.
    :param description: the one-line ``description:`` value; may be empty.
    :param memory_type: ``metadata.type`` (``user`` / ``feedback`` /
      ``project`` / ``reference``); empty string when absent or unparseable.
    :param body: the markdown body after the closing ``---`` fence, stripped.
    """
    name: str
    description: str
    memory_type: str
    body: str


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

# Matches one or more leading slashes to strip from the absolute path.
_LEADING_SLASH_RE = re.compile(r"^/+")


def derive_memory_dir(project_path: str | Path) -> Path:
    """Return the claude-memory directory for ``project_path``.

    The upstream tool encodes the absolute project directory as a filesystem
    path by stripping the leading ``/`` and replacing every remaining ``/``
    with ``-``.  For example ``/Users/alice/projects/myapp`` becomes
    ``-Users-alice-projects-myapp`` and the full memory directory is
    ``~/.claude/projects/-Users-alice-projects-myapp/memory/``.

    This function replicates that encoding without touching the filesystem.
    """
    abs_path = str(Path(project_path).resolve())
    encoded = _LEADING_SLASH_RE.sub("", abs_path).replace("/", "-")
    return Path.home() / ".claude" / "projects" / f"-{encoded}" / "memory"


def is_memory_index(path: str | Path) -> bool:
    """True when ``path`` is the MEMORY.md index rather than a fact file.

    The index is a list of pointers, not a fact; the #985 write-through
    mirror skips it (each individual fact file is mirrored instead).
    """
    return Path(path).name == "MEMORY.md"


def is_memory_fact_path(path: str | Path) -> bool:
    """True when ``path`` is a per-memory fact ``.md`` file in a claude-memory
    store: ``.../.claude/projects/<encoded>/memory/<name>.md``.

    Structural match on the path shape rather than the cwd-derived directory
    so the #985 mirror works under git worktrees (where ``cwd`` encodes a
    different project path than the one the memory directory is keyed on).
    The ``MEMORY.md`` index is excluded — only individual fact files mirror.
    """
    p = Path(path)
    if p.suffix != ".md" or p.name == "MEMORY.md":
        return False
    parent = p.parent
    if parent.name != "memory":
        return False
    projects = parent.parent.parent
    return projects.name == "projects" and projects.parent.name == ".claude"


# ---------------------------------------------------------------------------
# Write-through mirror flag (#985) — default OFF / opt-in
# ---------------------------------------------------------------------------

_CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
_MEMORY_SECTION: Final[str] = "memory"
_MIRROR_TOML_KEY: Final[str] = "mirror_claude_memory"
ENV_MIRROR_CLAUDE_MEMORY: Final[str] = "AELFRICE_MIRROR_CLAUDE_MEMORY"

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
# Mirrors retrieval._ENV_FALSY: an explicitly-empty value is *not* falsy —
# it falls through to None (no override) so a lower-precedence source decides.
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_mirror_override() -> bool | None:
    """True/False when AELFRICE_MIRROR_CLAUDE_MEMORY is set to a recognised
    truthy/falsy value, else None (so a lower-precedence source decides)."""
    raw = os.environ.get(ENV_MIRROR_CLAUDE_MEMORY)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_FALSY:
        return False
    if norm in _ENV_TRUTHY:
        return True
    return None


def _read_mirror_toml(start: Path | None = None) -> bool | None:
    """Walk up from ``start`` for a ``.aelfrice.toml`` with
    ``[memory] mirror_claude_memory``. Returns the bool when found, else
    None. Tolerant: malformed TOML or a wrong-typed value returns None and
    traces to stderr without raising — the default wins. Mirrors
    ``retrieval._read_toml_flag_for`` semantics for the ``[memory]`` section.
    """
    current = (start if start is not None else Path.cwd()).resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / _CONFIG_FILENAME
        if candidate.is_file():
            try:
                parsed: dict[str, Any] = tomllib.loads(
                    candidate.read_bytes().decode("utf-8", errors="replace"),
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(
                    f"aelfrice claude-memory: cannot read {candidate}: {exc}",
                    file=sys.stderr,
                )
                return None
            section_obj: Any = parsed.get(_MEMORY_SECTION, {})
            if not isinstance(section_obj, dict):
                return None
            if _MIRROR_TOML_KEY not in section_obj:  # type: ignore[operator]
                return None
            value: Any = section_obj[_MIRROR_TOML_KEY]  # type: ignore[index]
            if isinstance(value, bool):
                return value
            print(
                f"aelfrice claude-memory: ignoring [{_MEMORY_SECTION}] "
                f"{_MIRROR_TOML_KEY} in {candidate} (expected bool)",
                file=sys.stderr,
            )
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def is_mirror_enabled(
    explicit: bool | None = None,
    *,
    start: Path | None = None,
) -> bool:
    """Resolve the #985 claude-memory write-through mirror flag.

    Precedence (first decisive wins):
      1. ``AELFRICE_MIRROR_CLAUDE_MEMORY`` env var (truthy/falsy normalised).
      2. Explicit ``explicit`` kwarg from the caller.
      3. ``[memory] mirror_claude_memory`` in ``.aelfrice.toml``.
      4. Default: **False** — the mirror is opt-in, consistent with the
         narrow-surface PHILOSOPHY (#605) and the opt-in-by-default precedent
         set by ADR 0003 decision 4 (#973); ratified for this mirror under
         #985. With the mirror off, ``/aelf:audit-claude-memory`` remains the
         bridge.
    """
    env = _env_mirror_override()
    if env is not None:
        return env
    if explicit is not None:
        return explicit
    toml_value = _read_mirror_toml(start)
    if toml_value is not None:
        return toml_value
    return False


# ---------------------------------------------------------------------------
# Per-memory file frontmatter parser (#985)
# ---------------------------------------------------------------------------

# A frontmatter block is delimited by a ``---`` fence on its own line at the
# very start of the file and a closing ``---`` fence. We parse the keys we
# need with stdlib regex only — no YAML dependency, consistent with the
# locked PHILOSOPHY decision (#605) to prefer deterministic stdlib gates.
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\n(?P<fm>.*?\n)---[ \t]*\n(?P<body>.*)\Z",
    re.DOTALL,
)
# Top-level ``key: value`` (no leading indentation). Used for name/description.
_FM_TOP_KEY_RE = re.compile(
    r"^(?P<key>[A-Za-z0-9_-]+):[ \t]*(?P<val>.*?)[ \t]*$",
)
# ``type:`` nested one level under ``metadata:``. The upstream format places
# it as an indented child of ``metadata:``; accept any leading indentation so
# we tolerate 2- or 4-space styles.
_FM_TYPE_RE = re.compile(
    r"^[ \t]+type:[ \t]*(?P<val>[A-Za-z_]+)[ \t]*$",
)

# The four documented metadata.type values. Unknown values parse to "" so the
# derive() mapping falls back to the conservative agent-inferred prior.
_KNOWN_MEMORY_TYPES = frozenset({"user", "feedback", "project", "reference"})


def _strip_quotes(value: str) -> str:
    """Remove a single matching pair of surrounding quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_memory_file(text: str) -> MemoryFile | None:
    """Parse a per-memory ``.md`` file into a :class:`MemoryFile`.

    Returns ``None`` when the text has no parseable frontmatter fence or
    when the body is empty after stripping — callers skip those files
    rather than minting a content-free belief.

    The parser is intentionally conservative and line-oriented:

    * ``name`` and ``description`` are read from top-level (unindented)
      ``key: value`` lines inside the frontmatter block.
    * ``metadata.type`` is read from the first indented ``type:`` line whose
      value is one of the four documented types; any other value yields an
      empty ``memory_type``.

    No filesystem I/O. Pure function of the input text.
    """
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return None
    fm = m.group("fm")
    body = m.group("body").strip()
    if not body:
        return None

    name = ""
    description = ""
    memory_type = ""
    for line in fm.splitlines():
        type_m = _FM_TYPE_RE.match(line)
        if type_m is not None:
            val = type_m.group("val")
            if val in _KNOWN_MEMORY_TYPES:
                memory_type = val
            continue
        top_m = _FM_TOP_KEY_RE.match(line)
        if top_m is None:
            continue
        key = top_m.group("key")
        if key == "name" and not name:
            name = _strip_quotes(top_m.group("val").strip())
        elif key == "description" and not description:
            description = _strip_quotes(top_m.group("val").strip())

    return MemoryFile(
        name=name,
        description=description,
        memory_type=memory_type,
        body=body,
    )


# ---------------------------------------------------------------------------
# MEMORY.md bullet parser
# ---------------------------------------------------------------------------

# Matches a MEMORY.md bullet of the form:
#   - [link text](path) — description
# The description part (after ` — `) is optional.
_BULLET_RE = re.compile(
    r"^\s*-\s+"                   # list marker
    r"\[(?P<text>[^\]]+)\]"       # [link text]
    r"\((?P<path>[^)]+)\)"        # (path)
    r"(?:\s+[—–-]+\s+(?P<desc>.+))?$",  # optional ` — description`
    re.UNICODE,
)


def parse_memory_bullets(text: str, source_path: str = "") -> list[MemoryBullet]:
    """Parse MEMORY.md content into a list of :class:`MemoryBullet`.

    Only lines that match the ``- [text](path)`` pattern are returned.
    Heading lines, blank lines, and non-bullet lines are silently skipped.
    The description is the text after `` — `` (em-dash, en-dash, or ASCII
    hyphen-minus); it may be empty when the bullet has no description.

    Args:
        text: raw content of a MEMORY.md file (newline-separated).
        source_path: path of the MEMORY.md file, stored on each bullet for
            downstream reporting; does not affect parsing.

    Returns:
        List of :class:`MemoryBullet`, one per matched bullet line.
    """
    bullets: list[MemoryBullet] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if m is None:
            continue
        bullets.append(MemoryBullet(
            link_text=m.group("text").strip(),
            link_target=m.group("path").strip(),
            description=(m.group("desc") or "").strip(),
            source_path=source_path,
        ))
    return bullets


# ---------------------------------------------------------------------------
# Slot extraction
# ---------------------------------------------------------------------------

# A "slot" is a (subject, predicate) pair that can be extracted
# deterministically from a short assertion string.  The regex looks for a
# two-or-three capitalised token run at the start of the string that looks
# like "Subject Predicate …".
#
# Pattern rationale (conservative):
#   - subject: one or two capitalised tokens (e.g. "Foo" or "Foo Bar")
#   - predicate: one capitalised or lowercase token that follows
#   - value: everything after the predicate
#
# When the string starts with a lowercased word (common for bare description
# text) we still try the two-token split but record both tokens as-is.
# If fewer than two whitespace-separated tokens exist, we return None.
_SLOT_RE = re.compile(
    r"""
    ^
    (?P<subject>
        [A-Za-z][A-Za-z0-9_-]*          # first word
        (?:\s+[A-Za-z][A-Za-z0-9_-]*)?  # optional second word
    )
    \s+
    (?P<predicate>[A-Za-z][A-Za-z0-9_-]*)   # predicate token
    (?:\s+(?P<value>.+))?                    # rest = value
    $
    """,
    re.VERBOSE | re.DOTALL,
)


def extract_slot(text: str) -> tuple[str, str, str] | None:
    """Extract a ``(subject, predicate, value)`` slot from ``text``.

    Returns ``None`` when the text cannot be reliably tokenised into a
    subject + predicate pair (e.g. single-word strings, empty input).
    The slot fields are stripped but otherwise returned verbatim (no
    stemming, no lowercasing) so that callers can apply case-insensitive
    comparison themselves.
    """
    if not text or not text.strip():
        return None
    m = _SLOT_RE.match(text.strip())
    if m is None:
        return None
    subject = m.group("subject").strip()
    predicate = m.group("predicate").strip()
    value = (m.group("value") or "").strip()
    if not subject or not predicate:
        return None
    return (subject, predicate, value)


def slot_row_from_belief(
    belief_id: str,
    content: str,
    db_path: str,
) -> SlotRow | None:
    """Build a :class:`SlotRow` from an aelfrice belief, or ``None`` if the
    slot cannot be extracted.

    Args:
        belief_id: the belief's ``id`` field (used as ``source``).
        content: the belief's ``content`` field.
        db_path: path to the aelfrice DB (used as ``source_path``).
    """
    slot = extract_slot(content)
    if slot is None:
        return None
    subject, predicate, value = slot
    return SlotRow(
        slot_subject=subject,
        slot_predicate=predicate,
        slot_value=value,
        raw_text=content,
        source=belief_id,
        source_path=db_path,
    )


def slot_row_from_bullet(bullet: MemoryBullet) -> SlotRow | None:
    """Build a :class:`SlotRow` from a :class:`MemoryBullet`, or ``None`` if
    the slot cannot be extracted.

    The text used for slot extraction is the bullet's ``description`` when
    non-empty; otherwise the ``link_text`` is tried as a fallback.
    """
    text = bullet.description if bullet.description else bullet.link_text
    slot = extract_slot(text)
    if slot is None:
        return None
    subject, predicate, value = slot
    return SlotRow(
        slot_subject=subject,
        slot_predicate=predicate,
        slot_value=value,
        raw_text=text,
        source=bullet.link_text,
        source_path=bullet.source_path,
    )


# ---------------------------------------------------------------------------
# Four-bucket comparison
# ---------------------------------------------------------------------------


def _slot_key(row: SlotRow) -> str:
    """Normalised ``subject␟predicate`` key for slot matching."""
    return (row.slot_subject.lower().strip() + "\x1f"
            + row.slot_predicate.lower().strip())


def compare_slots(
    aelfrice_rows: list[SlotRow],
    claude_rows: list[SlotRow],
) -> ComparisonResult:
    """Compare two lists of :class:`SlotRow` and return the four buckets.

    Rows without an extractable slot (empty ``slot_subject`` or
    ``slot_predicate``) are treated as unmatched and placed in the
    respective ``*_only`` bucket.  The caller is responsible for ensuring
    that only rows with non-empty slots are passed; the function is
    defensive and handles them gracefully.

    Matching is case-insensitive on the ``subject``+``predicate`` key.
    Value comparison is also case-insensitive and stripped for
    duplicate/contradiction classification.

    Args:
        aelfrice_rows: slot rows extracted from locked aelfrice beliefs.
        claude_rows: slot rows extracted from claude-memory bullets.

    Returns:
        :class:`ComparisonResult` with four lists populated.
    """
    # Index claude rows by normalised slot key → list (multiple bullets can
    # share the same slot key; we match against the first found).
    claude_index: dict[str, list[SlotRow]] = {}
    for row in claude_rows:
        if not row.slot_subject or not row.slot_predicate:
            continue
        key = _slot_key(row)
        claude_index.setdefault(key, []).append(row)

    duplicates: list[tuple[SlotRow, SlotRow]] = []
    contradictions: list[tuple[SlotRow, SlotRow]] = []
    aelfrice_only: list[SlotRow] = []
    matched_claude_keys: set[str] = set()

    for arow in aelfrice_rows:
        if not arow.slot_subject or not arow.slot_predicate:
            aelfrice_only.append(arow)
            continue
        key = _slot_key(arow)
        if key not in claude_index:
            aelfrice_only.append(arow)
            continue
        # Match against the first claude row with this key.
        crow = claude_index[key][0]
        matched_claude_keys.add(key)
        aval = arow.slot_value.lower().strip()
        cval = crow.slot_value.lower().strip()
        if aval == cval:
            duplicates.append((arow, crow))
        else:
            contradictions.append((arow, crow))

    # Any claude rows whose key was never matched → claude-only.
    claude_only: list[SlotRow] = []
    for row in claude_rows:
        if not row.slot_subject or not row.slot_predicate:
            claude_only.append(row)
            continue
        key = _slot_key(row)
        if key not in matched_claude_keys:
            claude_only.append(row)

    return ComparisonResult(
        duplicates=duplicates,
        contradictions=contradictions,
        aelfrice_only=aelfrice_only,
        claude_only=claude_only,
    )
