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
    # The two patterns cannot match the same span — one requires a count
    # word where the other requires a unit word — so position alone is a
    # total order over the matches.
    found.sort(key=lambda item: item[0])
    return [spec for _, spec in found]


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


def stated_window_is_ambiguous(text: str) -> bool:
    """True when the text states more than one distinct window.

    "for two days, actually for a week" names two, and picking either is
    a guess. The caller refuses and asks instead of proposing a lock the
    user has to notice is wrong.
    """
    if not text:
        return False
    return len(set(_stated_windows(text))) > 1
