"""Belief categories — keyword-triggered rule grouping (#1126, v4 opt-in).

A *category* groups beliefs (repo-rules, git-workflow, prose-and-docs, …)
and binds them to an *activation trigger*. When the trigger fires, the
category's member beliefs are surfaced as a distinct rule block, so the
right rules land in context at the right moment — something a static
`CLAUDE.md` / `AGENTS.md` cannot do (it is always-on and unconditional).

Design contract (spec: `docs/design/belief_categories.md`, umbrella #1126):

  * **Default off.** Opt-in via `[belief_categories] enabled = true` in
    `.aelfrice.toml`, or `AELFRICE_BELIEF_CATEGORIES=1`. Off by default
    means existing users see no behavior change.
  * **Advisory injection, not enforcement.** v1 surfaces a category's
    rules into context on the `UserPromptSubmit` lane. It never blocks a
    tool call — the enforcement triad (#199) rejected hook-level hard
    enforcement, and every aelfrice hook is fail-open / exit-0. A hard
    `deny` lane is explicitly out of scope for v1.
  * **Deterministic matching.** Triggers are *user-declared* literal
    phrases / globs, matched by precompiled `re` (case-insensitive,
    word-boundary) and `fnmatch`. No embeddings, no model call, no
    `random`/wall-clock (#605). Same (prompt, categories) → same match.
  * **Membership is a join, not a belief column.** A belief may belong to
    several categories; the `Belief` dataclass is untouched. Category rows
    (with their trigger config) and the `belief_categories` M2M join live
    in their own tables — see `store.py`.

This module is pure: it owns the dataclasses, the `trigger_json`
(de)serialization, the deterministic matchers, and the config gate. It
does not touch disk or the store. The store layer stamps `created_at` and
persists; the hook layer checks `is_enabled`, loads categories, calls
`match_prompt`, and formats the block. Those wirings are separate concerns.

v1 wires only the **prompt** lane (keyword + always-on) into the hook. The
**command** and **file** trigger lanes are parsed, stored, and matched
here (and unit-tested), but the `PreToolUse` wiring that consumes them is
a documented follow-up — see the umbrella issue's non-goals.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from typing import Final, cast

# --- Public constants ---------------------------------------------------

CONFIG_SECTION: Final[str] = "belief_categories"
CONFIG_ENABLED_KEY: Final[str] = "enabled"
ENV_ENABLED: Final[str] = "AELFRICE_BELIEF_CATEGORIES"

# `default_lock` is an advisory hint for `aelf lock --category` / the
# `aelf category` UI: it says how members of this category are *typically*
# locked, not a hard constraint (the user always chooses per belief).
DEFAULT_LOCK_NONE: Final[str] = "none"
DEFAULT_LOCK_LOCKED: Final[str] = "locked"
DEFAULT_LOCK_ADVISORY: Final[str] = "advisory"
DEFAULT_LOCK_VALUES: Final[frozenset[str]] = frozenset(
    {DEFAULT_LOCK_NONE, DEFAULT_LOCK_LOCKED, DEFAULT_LOCK_ADVISORY}
)

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

# A category name is a short kebab-ish slug: lowercase letters, digits,
# hyphens, underscores. Keeps names URL/CLI/JSON-safe and predictable.
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


# --- Trigger ------------------------------------------------------------


@dataclass(frozen=True)
class CategoryTrigger:
    """The activation config for one category.

    All three lanes are stored; v1 wires only ``keywords`` (+ the
    category's ``always_on`` flag) into the UserPromptSubmit hook.

      * ``keywords``   — literal phrases matched against the user prompt,
                         case-insensitive, on word boundaries. Internal
                         whitespace in a phrase matches any run of
                         whitespace (``"commit and push"`` matches
                         ``"commit  and   push"``).
      * ``tool_globs`` — ``fnmatch`` globs matched against a tool-call
                         command string (e.g. ``"git push*"``).
      * ``file_globs`` — ``fnmatch`` globs matched against touched paths
                         (e.g. ``"tests/**"``).
    """

    keywords: tuple[str, ...] = ()
    tool_globs: tuple[str, ...] = ()
    file_globs: tuple[str, ...] = ()

    def to_json(self) -> str:
        """Canonical JSON for the ``categories.trigger_json`` column.

        Keys are sorted and lists are stored verbatim (order-preserving)
        so the serialization is stable and byte-reproducible.
        """
        return json.dumps(
            {
                "keywords": list(self.keywords),
                "tool_globs": list(self.tool_globs),
                "file_globs": list(self.file_globs),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "CategoryTrigger":
        """Parse a ``trigger_json`` blob. Tolerant: any malformed / wrong-
        typed field degrades to an empty lane rather than raising, so a
        hand-edited or forward-version row can never break the hook.
        """
        if not raw:
            return cls()
        try:
            parsed: object = json.loads(raw)
        except (ValueError, TypeError):
            return cls()
        if not isinstance(parsed, dict):
            return cls()
        obj = cast("dict[str, object]", parsed)
        return cls(
            keywords=_str_tuple(obj.get("keywords")),
            tool_globs=_str_tuple(obj.get("tool_globs")),
            file_globs=_str_tuple(obj.get("file_globs")),
        )

    def is_empty(self) -> bool:
        """True when no lane carries any matcher."""
        return not (self.keywords or self.tool_globs or self.file_globs)


def _str_tuple(value: object) -> tuple[str, ...]:
    """Coerce an arbitrary JSON value into a tuple of non-empty strings.

    Non-list, non-string, and empty entries are dropped. Deterministic:
    preserves input order, no sorting (the caller chose the order).
    """
    if not isinstance(value, list):
        return ()
    items = cast("list[object]", value)
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item)
    return tuple(out)


# --- Category -----------------------------------------------------------


@dataclass(frozen=True)
class Category:
    """One belief category: a name, an activation policy, and a lock hint.

    ``created_at`` is an ISO-8601 UTC string stamped by the store on
    insert; the module never reads the clock itself.
    """

    name: str
    always_on: bool = False
    trigger: CategoryTrigger = field(default_factory=CategoryTrigger)
    default_lock: str = DEFAULT_LOCK_NONE
    created_at: str = ""


def normalize_name(name: str) -> str:
    """Return a validated, canonical category name (trimmed, lowercased).

    Raises ``ValueError`` on an empty or malformed name so a bad name is
    rejected at the CLI/store boundary rather than silently stored.
    """
    slug = (name or "").strip().lower()
    if not _NAME_RE.match(slug):
        raise ValueError(
            f"invalid category name {name!r}: must be lowercase "
            "alphanumeric with '-'/'_' (e.g. 'git-workflow')"
        )
    return slug


def normalize_default_lock(value: str | None) -> str:
    """Validate the ``default_lock`` hint, defaulting to 'none'."""
    if value is None:
        return DEFAULT_LOCK_NONE
    token = value.strip().lower()
    if token not in DEFAULT_LOCK_VALUES:
        raise ValueError(
            f"invalid default_lock {value!r}: must be one of "
            f"{sorted(DEFAULT_LOCK_VALUES)}"
        )
    return token


# --- Matching -----------------------------------------------------------


def _compile_keywords(keywords: tuple[str, ...]) -> re.Pattern[str] | None:
    """Compile a case-insensitive, word-boundary alternation of literal
    phrases. Internal whitespace in a phrase matches any whitespace run.

    Returns ``None`` when there are no keywords, so callers can cheaply
    skip. Deterministic: the pattern is a pure function of the inputs.
    """
    parts: list[str] = []
    for kw in keywords:
        token = kw.strip()
        if not token:
            continue
        # Escape the literal phrase, then relax escaped internal spaces to
        # match any whitespace run so prompt spacing variance still hits.
        escaped = re.escape(token)
        escaped = re.sub(r"(?:\\ )+", r"\\s+", escaped)
        parts.append(escaped)
    if not parts:
        return None
    # \b anchors on both sides so 'push' does not match 'pushover'. For a
    # phrase whose edges are non-word (rare), \b still behaves correctly
    # against surrounding whitespace/punctuation.
    pattern = r"\b(?:" + "|".join(parts) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def keyword_hit(prompt: str, trigger: CategoryTrigger) -> bool:
    """True when any of the trigger's keyword phrases appears in ``prompt``."""
    if not prompt:
        return False
    pat = _compile_keywords(trigger.keywords)
    return bool(pat and pat.search(prompt))


def _glob_hit(candidates: tuple[str, ...], values: list[str]) -> bool:
    """True when any value matches any ``fnmatch`` glob, case-insensitively
    on every platform.

    ``fnmatch.fnmatch`` case-folds via ``os.path.normcase``, which is a
    no-op on POSIX — so it is case-*sensitive* on Linux (where CI runs)
    and case-insensitive on macOS/Windows. That platform split breaks the
    determinism contract (#605), so we lower-case both sides explicitly
    and use ``fnmatch.fnmatchcase`` (which never consults ``normcase``).
    """
    for value in values:
        v = value.lower()
        for pattern in candidates:
            if pattern and fnmatch.fnmatchcase(v, pattern.lower()):
                return True
    return False


def command_hit(command: str, trigger: CategoryTrigger) -> bool:
    """True when a tool-call command string matches any ``tool_globs``
    entry. (Stored + tested; the PreToolUse wiring is a follow-up.)"""
    if not command or not trigger.tool_globs:
        return False
    return _glob_hit(trigger.tool_globs, [command])


def paths_hit(paths: list[str], trigger: CategoryTrigger) -> bool:
    """True when any touched path matches any ``file_globs`` entry.
    (Stored + tested; the PreToolUse wiring is a follow-up.)"""
    if not paths or not trigger.file_globs:
        return False
    return _glob_hit(trigger.file_globs, paths)


def match_prompt(prompt: str, categories: list[Category]) -> list[Category]:
    """Return the categories activated for ``prompt`` — the v1 lane.

    A category fires when it is ``always_on`` OR any of its keyword
    phrases appears in the prompt. Results are de-duplicated by name and
    returned in a deterministic order: name ASC. Determinism matters —
    the injected block must be byte-identical for identical inputs.
    """
    fired: dict[str, Category] = {}
    for cat in categories:
        if cat.always_on or keyword_hit(prompt, cat.trigger):
            fired[cat.name] = cat
    return [fired[name] for name in sorted(fired)]


# --- Config gate --------------------------------------------------------


def is_enabled(config: dict[str, object] | None = None) -> bool:
    """Whether the belief-categories injection lane is enabled.

    Resolution order (mirrors ``sentiment_feedback.is_enabled``):
      1. Env var ``AELFRICE_BELIEF_CATEGORIES`` if set to a known token.
      2. ``[belief_categories] enabled`` in the supplied config dict.
      3. Default False.

    The hook layer loads the TOML and passes the dict in; this module
    does not read disk.
    """
    raw = os.environ.get(ENV_ENABLED)
    if raw is not None:
        token = raw.strip().lower()
        if token in _ENV_TRUTHY:
            return True
        if token in _ENV_FALSY:
            return False

    if config is None:
        return False
    section = config.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        return False
    sect = cast("dict[str, object]", section)
    value = sect.get(CONFIG_ENABLED_KEY)
    return value if isinstance(value, bool) else False


# --- Seed set -----------------------------------------------------------
#
# The starter taxonomy from the umbrella spec. NOT auto-installed — a user
# opts in explicitly via `aelf category init`, which upserts these
# (idempotently). Provided here as the single source so the CLI and the
# docs/tests agree on the seed triggers.

SEED_CATEGORIES: Final[tuple[Category, ...]] = (
    Category(
        name="repo-rules",
        always_on=True,
        trigger=CategoryTrigger(),
        default_lock=DEFAULT_LOCK_LOCKED,
    ),
    Category(
        name="git-workflow",
        always_on=False,
        trigger=CategoryTrigger(
            # Literal-phrase matching (deterministic) trades recall for no
            # false positives; the #1126 R&D measured ~39% miss on natural
            # phrasings, so this list covers the common surface forms
            # (commit/committed, merge/merged, push, ship, land, PR,
            # github, …). It will never be exhaustive — that is the
            # deterministic-matching trade-off, not a bug.
            keywords=(
                "commit",
                "committed",
                "commit and push",
                "push",
                "git push",
                "pull request",
                "open a pr",
                "pr",
                "rebase",
                "merge",
                "merged",
                "release",
                "ship",
                "ship it",
                "land",
                "github",
                "tag",
            ),
        ),
        default_lock=DEFAULT_LOCK_LOCKED,
    ),
    Category(
        name="secrets-and-safety",
        always_on=True,
        trigger=CategoryTrigger(
            keywords=("push", "publish", "upload", "secret", "token", "credential"),
            tool_globs=("git push*", "*git push*"),
        ),
        default_lock=DEFAULT_LOCK_LOCKED,
    ),
    Category(
        name="prose-and-docs",
        always_on=False,
        trigger=CategoryTrigger(
            keywords=(
                "write",
                "draft",
                "readme",
                "changelog",
                "docs",
                "documentation",
                "blog",
                "prose",
            ),
        ),
        default_lock=DEFAULT_LOCK_ADVISORY,
    ),
    Category(
        name="testing",
        always_on=False,
        trigger=CategoryTrigger(
            keywords=("test", "tests", "pytest"),
            file_globs=("tests/**", "test_*.py", "*_test.py"),
        ),
        default_lock=DEFAULT_LOCK_NONE,
    ),
)
