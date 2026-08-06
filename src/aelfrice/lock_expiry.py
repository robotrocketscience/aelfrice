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
