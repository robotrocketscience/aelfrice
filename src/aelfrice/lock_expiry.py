"""Parse and render the time-boxed-lock window (#1314).

Two rules shape everything here.

**A window is resolved to an absolute UTC instant at write time.** The
store never holds a relative expression like `7d`, because a relative
expression has no fixed meaning once written — it would silently
re-anchor to whenever it was next read. `--for 7d` is a way of *saying*
an instant, not a thing to store.

**A month is not 30 days.** `1mo` from January 31 is February 28, not
March 2. Calendar units use calendar arithmetic with day-of-month
clamping; only `d` and `w` are fixed-length. Getting this wrong is the
kind of error that surfaces once a year, on the day it matters.

Stdlib-only, and it imports nothing from `aelfrice`, so the CLI, the
hooks and the store can all reach it without an import cycle.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Final

__all__ = [
    "FOREVER",
    "LockExpiryError",
    "extract_stated_window",
    "stated_window_attaches_to_memory",
    "stated_window_is_ambiguous",
    "format_remaining",
    "parse_for",
    "parse_until",
]

# The explicit synonym for "no expiry", so a caller can spell out the
# permanent case rather than expressing it as the absence of a flag.
FOREVER: Final[str] = "forever"

# `<N><unit>`. `mo` is months and is checked before `m` could be — there
# is deliberately no bare `m`, because it reads as either minutes or
# months and this grammar has no use for a window that short.
_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<n>\d+)(?P<unit>d|w|mo|y)$", re.IGNORECASE,
)

_UNIT_NAMES: Final[str] = "d (days), w (weeks), mo (months), y (years)"


class LockExpiryError(ValueError):
    """A `--for` / `--until` value that cannot be resolved to an instant."""


def _add_months(start: datetime, months: int) -> datetime:
    """`start` shifted by `months`, clamping the day into the target month.

    January 31 + 1mo is February 28 (or 29), not March 2. Python has no
    stdlib calendar-month arithmetic, so this is hand-rolled rather than
    approximated as 30 days.
    """
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    # Day 1 of the *next* month minus one day is the last day of this
    # one, which avoids hard-coding month lengths or a leap-year rule.
    if month == 12:
        first_of_next = datetime(year + 1, 1, 1, tzinfo=start.tzinfo)
    else:
        first_of_next = datetime(year, month + 1, 1, tzinfo=start.tzinfo)
    last_day = (first_of_next - timedelta(days=1)).day
    return start.replace(year=year, month=month, day=min(start.day, last_day))


def parse_for(spec: str, *, now: datetime) -> str | None:
    """Resolve a `--for` window to an absolute ISO-8601 UTC instant.

    Returns None for `forever` — the permanent lock, which is stored as
    a NULL expiry rather than as a far-future timestamp, so it stays
    indistinguishable from a pre-#1314 lock.

    Raises `LockExpiryError` on an unparseable spec or a zero window. A
    zero window is rejected rather than treated as "expire immediately":
    `--for 0d` is far more likely to be a scripting mistake than a
    request to write a lock and sweep it away on the same open.
    """
    text = spec.strip()
    if text.lower() == FOREVER:
        return None
    match = _DURATION_RE.match(text)
    if match is None:
        raise LockExpiryError(
            f"cannot parse --for {spec!r}; expected <N><unit> where unit is "
            f"{_UNIT_NAMES}, or {FOREVER!r}"
        )
    # `_DURATION_RE` bounds the count's *shape* (`\d+`) but not its length,
    # and CPython refuses to convert a decimal string longer than
    # `sys.get_int_max_str_digits()` (4300 by default), raising a bare
    # `ValueError`. That is the same contract escape as the arithmetic
    # below, one statement earlier, so it is wrapped the same way rather
    # than left to reach a caller that only catches `LockExpiryError`.
    try:
        count = int(match.group("n"))
    except ValueError as exc:
        raise LockExpiryError(
            f"--for {spec!r} has too many digits to read as a count; "
            f"use a smaller window or {FOREVER!r}"
        ) from exc
    if count == 0:
        raise LockExpiryError(
            f"--for {spec!r} is a zero-length window; use {FOREVER!r} for a "
            "permanent lock, or omit the flag"
        )
    unit = match.group("unit").lower()
    base = now.astimezone(timezone.utc)
    # A spec can match `_DURATION_RE` and still be unrepresentable: the
    # result lands past `datetime.max`, or the count is too large to
    # convert to a C int at all. Left unhandled, the calendar units
    # surface a bare `ValueError` ("year N is out of range") and the
    # fixed-length ones an `OverflowError` — neither of which is the
    # documented contract, and `cli._cmd_lock` deliberately catches
    # `LockExpiryError` and nothing else so that a malformed window
    # fails without writing a lock the user then has to undo. Every
    # caller resolving a window through here (`cli.py:1950-1962`,
    # `mcp_server.py:367-373`) is written to that one exception.
    try:
        if unit == "d":
            expires = base + timedelta(days=count)
        elif unit == "w":
            expires = base + timedelta(weeks=count)
        elif unit == "mo":
            expires = _add_months(base, count)
        else:
            expires = _add_months(base, count * 12)
    except LockExpiryError:  # pragma: no cover - defensive; see below
        # `LockExpiryError` subclasses `ValueError`, so it would be
        # caught and re-wrapped by the clause below. Nothing in the
        # arithmetic raises it today; re-raising unchanged keeps that
        # true if something ever does.
        raise
    except (OverflowError, ValueError) as exc:
        raise LockExpiryError(
            f"--for {spec!r} resolves past the largest representable date; "
            f"use a smaller window or {FOREVER!r}"
        ) from exc
    return expires.isoformat()


def parse_until(spec: str, *, now: datetime) -> str:
    """Resolve a `--until` value to an absolute ISO-8601 UTC instant.

    Accepts a bare `YYYY-MM-DD` (midnight UTC) or a full ISO timestamp.
    A naive timestamp is read as UTC rather than as local time, so the
    value written matches the value typed regardless of the machine's
    zone.

    A past value raises rather than being silently accepted and swept on
    the next open. The user asked for a window; a window that has
    already closed is a mistake worth naming, not a lock to quietly
    discard.
    """
    text = spec.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LockExpiryError(
            f"cannot parse --until {spec!r}; expected YYYY-MM-DD or an "
            f"ISO-8601 timestamp ({exc})"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= now.astimezone(timezone.utc):
        raise LockExpiryError(
            f"--until {spec!r} resolves to {parsed.isoformat()}, which is not "
            "in the future; a lock cannot be created already expired"
        )
    return parsed.isoformat()


def format_remaining(expires_at: str | None, *, now: datetime) -> str:
    """Render the window left on a lock, for `aelf locked`.

    `—` for a permanent lock. Otherwise the two coarsest non-zero units
    (`6d 4h`, `3h 12m`, `45m`), because a lock listing is scanned, not
    read, and a full duration is noise at that width.

    `expired` is never returned on the live path: the open-time sweep
    has already flipped anything due, so a listed lock is by
    construction still in its window. A past value here therefore means
    a caller passed one directly, and `0m` is the honest rendering of
    "no window left" without claiming a state the sweep would have
    removed.
    """
    if expires_at is None:
        return "—"
    try:
        target = datetime.fromisoformat(expires_at)
    except ValueError:
        # A malformed stored value must not take down a listing.
        return "?"
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    delta = target.astimezone(timezone.utc) - now.astimezone(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "0m"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = [(days, "d"), (hours, "h"), (minutes, "m")]
    shown = [f"{value}{suffix}" for value, suffix in parts if value]
    if not shown:
        return "0m"
    return " ".join(shown[:2])


# --- natural-language window extraction (#1315) --------------------------
#
# Maps a window the user SPELLED OUT to a `--for` spec. It never infers
# one: "remember this" has no window and returns None, because inferring
# an expiry the user did not state is an explicit non-goal of #1315 — a
# guessed window expires their lock on a date they never agreed to.
#
# Deliberately small. Every pattern requires an explicit unit word, so
# the ambiguous cases the issue names ("for the trip", "until I'm back")
# do not match and the caller refuses-and-asks rather than guessing.
_NUMBER_WORDS: Final[dict[str, int]] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Unit word -> the `parse_for` unit it resolves to. Plurals are handled
# by the `s?` in the pattern rather than by separate entries.
_UNIT_WORDS: Final[dict[str, str]] = {
    "day": "d", "week": "w", "month": "mo", "year": "y",
}

_STATED_WINDOW_RE: Final[re.Pattern[str]] = re.compile(
    r"\bfor\s+(?:the\s+)?(?:next\s+)?"
    r"(?P<count>\d+|" + "|".join(_NUMBER_WORDS) + r")\s+"
    r"(?P<unit>" + "|".join(_UNIT_WORDS) + r")s?\b",
    re.IGNORECASE,
)

# "for the next week" with no count word — the count is implied by
# "next". Kept as its own pattern rather than making the count optional
# in the one above, because an optional count would also match a bare
# "for week" and there is no such English.
_NEXT_UNIT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bfor\s+the\s+next\s+(?P<unit>" + "|".join(_UNIT_WORDS) + r")\b",
    re.IGNORECASE,
)

# Units `parse_for` has no spec for. A window stated in one of them is
# still a window the user stated; it just cannot be resolved (#1440).
_SUB_DAY_UNIT_WORDS: Final[tuple[str, ...]] = ("second", "minute", "hour")

# The sub-day units that are also English adjectives. Only these need the
# clause-boundary guard below; "hour" has no adjective reading.
_ADJECTIVAL_UNIT_WORDS: Final[tuple[str, ...]] = ("second", "minute")

# Counts above the `_NUMBER_WORDS` ceiling that stand alone in front of
# the unit, which is where this vocabulary reads them: "twenty days".
# A compound ("twenty-five") is covered by the optional suffix below,
# since the count's *value* is never read — an unusable window resolves
# to None whatever the number is.
_LARGE_NUMBER_WORDS: Final[tuple[str, ...]] = (
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety",
)

_LARGE_NUMBER_SUFFIX: Final[str] = (
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
)

# Counts that name a quantity without fixing it. Ordered longest-first
# so "a couple of days" is read as one quantifier and not as "a couple"
# followed by a unit word "of".
#
# The article is NOT baked into an entry. It used to be ("a couple of",
# "a number of"), and `_UNUSABLE_WINDOW_RE`'s shared `(?:the\s+)?
# (?:next\s+)?` prefix eats the article slot, so those spellings were
# unreachable in "for the next couple of days" while "few" — listed
# unarticled — worked there. The article gets its own optional slot
# instead, which is what makes the articled and unarticled forms behave
# identically (#1440).
_QUANTIFIER_WORDS: Final[tuple[str, ...]] = (
    "number of", "couple of", "couple", "several",
    "numerous", "many", "some", "few",
)

_QUANTIFIER: Final[str] = (
    r"(?:a\s+)?(?:" + "|".join(_QUANTIFIER_WORDS) + r")"
)

# Scale words, which never occupy that leading slot: English writes "two
# hundred days" and "a dozen days", never "hundred days". Listed beside
# `_LARGE_NUMBER_WORDS` they were unreachable in every sentence that
# actually uses them, so they carry their own optional multiplier
# instead (#1440).
#
# The multiplier slot takes a quantifier as well as a number ("a couple
# hundred days"), the scale word may be partitive ("dozens of days"),
# and an `and`-joined tail is read after it ("two hundred and fifty
# days"). None of the three collides with the resolving patterns: the
# count and unit sweep in `test_no_two_patterns_claim_the_same_window`
# is the evidence, and it covers all three spellings.
_SCALE_NUMBER_WORDS: Final[tuple[str, ...]] = ("hundred", "thousand", "dozen")

_SCALED_COUNT: Final[str] = (
    r"(?:(?:"
    + _QUANTIFIER
    + r"|(?:" + "|".join(_LARGE_NUMBER_WORDS) + r")" + _LARGE_NUMBER_SUFFIX
    + r"|\d+|"
    + "|".join(_NUMBER_WORDS)
    + r")\s+)?(?:"
    + "|".join(_SCALE_NUMBER_WORDS)
    + r")s?(?:\s+of)?"
    + r"(?:\s+and\s+(?:(?:" + "|".join(_LARGE_NUMBER_WORDS) + r")"
    + _LARGE_NUMBER_SUFFIX + r"|\d+|" + "|".join(_NUMBER_WORDS) + r"))?"
)

# A window stated OUTSIDE the count/unit vocabulary above. It resolves to
# nothing, but it is stated, and the module's rule for a stated-but-
# unusable window is the zero-length one's: record it, do not drop it, so
# that a second window beside it makes the text ambiguous rather than
# silently resolving to the survivor (#1440).
#
# Deliberately still unit-anchored. Widening this to any `for <noun>`
# would make "keep this for the trip" a stated window and turn every
# sentence pairing it with a real window into a refusal, which is a
# recall cost on the supported units — the one thing this must not have.
_UNUSABLE_WINDOW_RE: Final[re.Pattern[str]] = re.compile(
    r"\bfor\s+(?:"
    r"(?:the\s+)?(?:next\s+)?(?:"
    # An unreadable count with any unit word: "a few days", "twenty
    # days", "two hundred days", "2-3 days".
    r"(?:"
    + _QUANTIFIER
    + r"|(?:" + "|".join(_LARGE_NUMBER_WORDS) + r")" + _LARGE_NUMBER_SUFFIX
    + r"|" + _SCALED_COUNT
    + r"|\d+\s*[-–—]\s*\d+"
    r")\s+(?:" + "|".join((*_UNIT_WORDS, *_SUB_DAY_UNIT_WORDS)) + r")s?"
    # A readable count with an unresolvable unit: "30 minutes",
    # "two hours".
    #
    # `second` and `minute` are also English adjectives, and the
    # indefinite article is the form that collides: "for a second
    # opinion" and "for a minute detail" are not durations. Left
    # unguarded they read as a stated window, so a sentence pairing one
    # with a real window refuses instead of resolving it — the recall
    # cost on supported units this pattern's own contract forbids. When
    # the count is `a`/`an` and the unit is one of those two, the phrase
    # must therefore end its clause: end of string, punctuation, or a
    # conjunction. "for a second," and "for a second." still state a
    # window; "for a second opinion" no longer does. A numeral or a
    # spelled count ("two seconds") is never adjectival and stays
    # unguarded, as does "an hour", which has no adjective reading.
    r"|(?:a|an)\s+(?:" + "|".join(_ADJECTIVAL_UNIT_WORDS) + r")s?"
    r"(?=$|[^\w\s]|\s+(?:and|or|but|then|so)\b)"
    r"|(?:\d+|" + "|".join(w for w in _NUMBER_WORDS if w not in ("a", "an"))
    + r")\s+(?:" + "|".join(_SUB_DAY_UNIT_WORDS) + r")s?"
    r"|(?:a|an)\s+(?:"
    + "|".join(u for u in _SUB_DAY_UNIT_WORDS
               if u not in _ADJECTIVAL_UNIT_WORDS) + r")s?"
    r")"
    # The count implied by "next", with an unresolvable unit: "for the
    # next hour". Same grammar as `_NEXT_UNIT_RE`, one class of unit
    # further out, so it is spelled the same way that pattern spells it
    # — `the` and `next` both required, no plural — rather than sharing
    # the optional prefix above, which would also admit a bare "for
    # hour" and "for the hour". There is no such English, and a stated
    # window is the one thing this module must not invent.
    r"|the\s+next\s+(?:" + "|".join(_SUB_DAY_UNIT_WORDS) + r")"
    r")\b",
    re.IGNORECASE,
)


def _stated_windows(text: str) -> list[str | None]:
    """Every window `text` states, in the order it states them.

    Both patterns are scanned and the results merged by position, so
    that "first" means *first in the sentence* rather than *first
    pattern tried*. Scanning only one of them is how a sentence that
    names two windows can look unambiguous: `_STATED_WINDOW_RE` cannot
    see a bare "for the next week", so a text naming that plus a counted
    window used to report a single window and resolve to the wrong one.

    A zero-length window appears as `None` — stated, but unusable, since
    `parse_for` rejects it. It stays in the list rather than being
    dropped so that it still counts as *a* stated window: "for 0 days,
    then for a week" names two things and must refuse, not silently
    resolve to the survivor.

    A window stated outside the count/unit vocabulary ("for 30 minutes",
    "for the next hour", "for a few days", "for twenty days", "for two
    hundred days") is the same situation and gets the same `None`
    (#1440). Before that it was not recorded at all, so "for 30 minutes
    for a week" reported one window and resolved to the one stated
    *second* — the opposite of the documented rule.
    """
    return [spec for _, spec in _stated_windows_with_positions(text)]


def _stated_windows_with_positions(text: str) -> list[tuple[int, str | None]]:
    """`_stated_windows`, keeping each match's start offset.

    Split out for `stated_window_attaches_to_memory`, which needs to know
    *where* the first window sits in order to ask what governs it. The
    ordering contract lives here so both callers share one definition of
    "first".
    """
    found: list[tuple[int, str | None]] = []
    for match in _STATED_WINDOW_RE.finditer(text):
        raw = match.group("count").lower()
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        spec = (
            None
            if count == 0
            else f"{count}{_UNIT_WORDS[match.group('unit').lower()]}"
        )
        found.append((match.start(), spec))
    for match in _NEXT_UNIT_RE.finditer(text):
        found.append((match.start(), f"1{_UNIT_WORDS[match.group('unit').lower()]}"))
    # All three patterns read the same `\bfor`, but no two of them can
    # match at the same offset: the resolving pair splits on count word
    # versus unit word, and the unusable pattern's count slots and unit
    # slots are each disjoint from theirs. So position alone is a total
    # order over the matches, and no window is recorded twice — which
    # matters, because a double-count would read a single stated window
    # as an ambiguity and refuse. That disjointness is a property of the
    # three patterns, not of this loop, and it is pinned as one by
    # `test_no_two_patterns_claim_the_same_window` rather than defended
    # here by a dedupe no input reaches.
    #
    # This is a DELIBERATE deviation from #1440 AC1 as written ("dedupe
    # on match start"). The criterion's intent — no window counted twice
    # — is met; its literal mechanism is not implemented, because a
    # branch no input reaches is a coverage claim the tests cannot back,
    # and the sweep test fails on any future widening that would make it
    # reachable. Recorded here so the deviation is not silent (#1440).
    for match in _UNUSABLE_WINDOW_RE.finditer(text):
        found.append((match.start(), None))
    found.sort(key=lambda item: item[0])
    return found


def extract_stated_window(text: str) -> str | None:
    """Return the `--for` spec the text states, or None if it states none.

    `"prioritise this for the next week"` -> `"1w"`.
    `"keep this for two months"`          -> `"2mo"`.
    `"remember this"`                     -> `None`.
    `"keep this for the trip"`            -> `None`.

    None means *no window was stated*, not *no window is wanted*. The
    caller must not substitute a default: #1315's non-goals forbid
    inferring an expiry the user did not state, and the failure mode of
    guessing is a lock that expires on a date the user never agreed to.

    Only the first window stated is returned, counting both spellings.
    A sentence naming two different windows is ambiguous, and the caller
    is expected to refuse rather than pick one — see
    `stated_window_is_ambiguous`.
    """
    if not text:
        return None
    windows = _stated_windows(text)
    if not windows:
        return None
    # None here means the first thing stated was a zero-length window,
    # which `parse_for` rejects; surfacing None keeps that refusal in one
    # place rather than raising from a detector that must stay total.
    return windows[0]


# A memory verb plus a **self-referential object**. Both halves are
# required, and the object is what makes it work: "keep CI logs for 30
# days" has the verb but its object is the logs, so the 30 days is a
# property of the logs and not of how long to remember the rule. Only
# "keep *this* for 30 days" attaches the window to the memory.
_MEMORY_ANCHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:remember|recall|keep|retain|memori[sz]e|prioriti[sz]e|"
    r"hold\s+on\s+to|lock|store|bear\s+in\s+mind)\b"
    r"(?:\s+(?:that|about|onto))?"
    r"\s+(?:this|that|it|these|those|the\s+above|the\s+following)\b",
    re.IGNORECASE,
)

# How far after the anchor the window may sit and still be its object.
# One short intervening phrase, not a clause.
_ANCHOR_WINDOW_MAX_GAP: Final[int] = 40

# A window on the far side of one of these is in a different clause, so
# the anchor before it is not governing it. An en/em dash joins clauses as
# surely as a semicolon; the ASCII hyphen is left out because it is also
# the compound-word character, and is caught as a bare token below instead.
_CLAUSE_BREAKS: Final[frozenset[str]] = frozenset(".!?;:\n–—")

# Words that open a new predicate. A window after one of them is the
# object of *that* predicate, not of the memory verb: "always remember
# this and cache the index for two weeks" says how long to cache, not how
# long to remember. Punctuation-only gating cannot see this — the leak
# needs no comma at all, so adding one to `_CLAUSE_BREAKS` closes exactly
# one of its spellings and leaves `and`/`but`/`then`/`while`/`so` open.
_GAP_CONNECTIVES: Final[frozenset[str]] = frozenset({
    "-", "also", "after", "although", "and", "as", "because", "before",
    "but", "however", "if", "meanwhile", "or", "plus", "since", "so",
    "then", "though", "unless", "when", "whenever", "while", "whilst",
    "yet",
})


def _gap_opens_a_new_predicate(gap: str) -> bool:
    """True when `gap` is not a bare continuation of the memory clause.

    Split out so the anchor rule reads as one question. Tokens are
    stripped of surrounding punctuation but not split on it, so a
    hyphenated compound (`build-time`) is one token and does not collide
    with the clause-joining bare `-`.
    """
    if set(gap) & _CLAUSE_BREAKS:
        return True
    words = {word.strip(",.'\"()").lower() for word in gap.split()}
    return bool(words & _GAP_CONNECTIVES)


def stated_window_attaches_to_memory(text: str) -> bool:
    """True when the first stated window is governed by a memory verb.

    #1315's extractor cannot otherwise tell *"remember this for two
    weeks"* from *"the rule is: do Y for two weeks"* — both are
    directives, and both state a countable window. The second is a
    subject-matter duration, and proposing `--for` from it produces a
    lock that expires on a date the user never agreed to, which is the
    failure this module's docstring names.

    Measured before this gate existed: on a 44,679-belief live store the
    directive-window arm fired 9 times and **0** of the 9 stated a
    retention window — `Blocked for 9 days`, `traveling for a week`,
    `Results available for 29 days`. Realized attachment precision was
    0/9 locally and 0/90 across other stores, so the gate is not a
    refinement, it is the difference between the suffix being right
    sometimes and never.

    Ratified by the operator 2026-08-06 over the alternatives of shipping
    the suffix as-is or dropping it: narrow the extractor, and accept the
    recall cost.

    Deliberately structural rather than a keyword test. The anchor needs
    the verb **and** a self-referential object, because the verb alone is
    what admits `keep CI logs for 30 days`. The object noun, if any, is
    left to the gap rather than absorbed into the anchor — an anchor that
    swallowed trailing words would swallow the window too and never
    match. The window must then follow within `_ANCHOR_WINDOW_MAX_GAP`
    characters, with the gap between them opening no new predicate, so an
    anchor early in a paragraph cannot govern a duration in a later
    sentence *or* in a later clause of the same sentence. Length and
    punctuation alone are not enough for the second of those: prefixing a
    memory clause to a rejected sentence is otherwise all it takes to
    license its subject-matter window — see `_gap_opens_a_new_predicate`.
    """
    if not text:
        return False
    windows = _stated_windows_with_positions(text)
    if not windows:
        return False
    position = windows[0][0]
    for match in _MEMORY_ANCHOR_RE.finditer(text):
        end = match.end()
        if end > position:
            break
        gap = text[end:position]
        if len(gap) <= _ANCHOR_WINDOW_MAX_GAP and not _gap_opens_a_new_predicate(
            gap
        ):
            return True
    return False


def stated_window_is_ambiguous(text: str) -> bool:
    """True when the text states more than one distinct window.

    "for two days, actually for a week" names two, and picking either is
    a guess. The caller refuses and asks instead of proposing a lock the
    user has to notice is wrong.
    """
    if not text:
        return False
    return len(set(_stated_windows(text))) > 1
