"""#1396: an unrepresentable `--for` window must honour `parse_for`'s contract.

`parse_for` documents `LockExpiryError` as the failure mode, and every caller is
written to exactly that — `cli._cmd_lock` catches it and nothing else, on purpose,
so a malformed window fails *before* `_open_store()` rather than after writing a
lock the user has to notice and undo. A spec that matches `_DURATION_RE` but lands
past `datetime.max` used to escape as a bare `ValueError` or an `OverflowError`.

Two things make these assertions non-vacuous:

* **Neither single exception type covers both families.** `LockExpiryError`
  subclasses `ValueError`, so for the calendar units `pytest.raises(ValueError)`
  passes on the unfixed code *and* the fixed code and pins nothing; for the
  fixed-length units the unfixed error is an `OverflowError`, which is not a
  `ValueError`, so the mirror-image expectation misses the calendar half. The
  type is therefore asserted exactly, and the chained `__cause__` is asserted to
  be the original arithmetic error — which proves the escape was wrapped rather
  than the raise site merely moved.
* **The boundary is anchor-dependent.** The first failing count differs with the
  `now` passed in, so #1396's table (binary-searched on a different 2026 anchor)
  does not reproduce on an arbitrary one — three of its five values resolve fine
  from 2026-01-01. `_ANCHOR` is pinned here and the counts below are searched
  against *that*, so the parametrisation cannot rot into testing nothing.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.lock_expiry import FOREVER, LockExpiryError, parse_for

# Every count below is relative to this instant. Do not change it without
# re-deriving the table.
_ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)

# (spec, the arithmetic error that must end up chained as __cause__).
# The two calendar units overflow inside `_add_months`' `datetime(...)`
# construction and raise ValueError; the two fixed-length units overflow in
# `timedelta` arithmetic and raise OverflowError. Covering only one family
# would leave the other escape live.
FIRST_UNREPRESENTABLE = [
    ("2912443d", OverflowError),
    ("416064w", OverflowError),
    ("95687mo", ValueError),
    ("7974y", ValueError),
    # Too large to convert to a C int at all — a different OverflowError
    # from the "date value out of range" one above.
    ("99999999999999999999d", OverflowError),
]

# One count below each boundary. These must still resolve: a fix that clamped
# or rejected broadly would satisfy every assertion above and quietly break
# every long-but-legal window.
LAST_REPRESENTABLE = ["2912442d", "416063w", "95686mo", "7973y"]


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the DB away from the developer's real repo-local store."""
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.mark.parametrize(("spec", "cause_type"), FIRST_UNREPRESENTABLE)
def test_unrepresentable_window_raises_lock_expiry_error(
    spec: str, cause_type: type[BaseException]
) -> None:
    """AC1/AC4: the documented exception, for every unit, both escape families."""
    with pytest.raises(LockExpiryError) as excinfo:
        parse_for(spec, now=_ANCHOR)

    exc = excinfo.value
    # Exact type, not just isinstance: `LockExpiryError` IS a `ValueError`, so
    # an isinstance check against ValueError cannot tell fixed from unfixed.
    assert type(exc) is LockExpiryError
    assert not isinstance(exc, OverflowError)
    # The original error is chained, so the escape was wrapped rather than the
    # failure being relocated or swallowed.
    assert isinstance(exc.__cause__, cause_type)
    assert spec in str(exc)


@pytest.mark.parametrize("spec", LAST_REPRESENTABLE)
def test_the_largest_legal_window_still_resolves(spec: str) -> None:
    """The fix must not clamp legal windows — one count below each boundary."""
    result = parse_for(spec, now=_ANCHOR)
    assert result is not None
    assert result.startswith("9999-")


def test_ordinary_windows_are_untouched() -> None:
    """A guard against a broad rejection passing the boundary tests."""
    assert parse_for("7d", now=_ANCHOR).startswith("2026-01-08")
    assert parse_for("6mo", now=_ANCHOR).startswith("2026-07-01")
    assert parse_for("5000y", now=_ANCHOR).startswith("7026-01-01")
    assert parse_for(FOREVER, now=_ANCHOR) is None


def test_the_boundary_table_is_actually_the_boundary() -> None:
    """Fixture adequacy: each 'first unrepresentable' count must be minimal.

    If a future change moved the limit, the parametrised counts could all sit
    far past it and still pass while no longer testing the *edge*. Asserting
    that count-1 resolves and count does not keeps the table honest — and it is
    what catches the anchor drift described in the module docstring.
    """
    import re

    for spec, _cause in FIRST_UNREPRESENTABLE:
        m = re.match(r"^(\d+)(d|w|mo|y)$", spec)
        assert m is not None
        count, unit = int(m.group(1)), m.group(2)
        if count > 10**12:  # the C-int case has no meaningful predecessor
            continue
        assert parse_for(f"{count - 1}{unit}", now=_ANCHOR) is not None, spec
        with pytest.raises(LockExpiryError):
            parse_for(f"{count}{unit}", now=_ANCHOR)


def test_a_count_with_too_many_digits_is_wrapped_too() -> None:
    """AC1: the escape one statement above the arithmetic, not just inside it.

    `_DURATION_RE` bounds the count's shape (`\\d+`) but not its length, and
    CPython refuses to convert a decimal string longer than
    `sys.get_int_max_str_digits()`, raising a bare `ValueError` — from the
    `int()` call, which sits before any window arithmetic. Wrapping only the
    arithmetic leaves this family escaping to `cli._cmd_lock` as a traceback,
    which is the failure mode AC1 and AC2 name.

    The limit is process-settable (`PYTHONINTMAXSTRDIGITS` moves it), so it is
    pinned here to CPython's own minimum rather than assumed, and the spec is
    sized against the pinned value. Without pinning, a raised limit would let
    `int()` succeed and the assertion would silently start exercising the
    `timedelta` overflow instead — a different path that is already covered.
    """
    original = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)  # the smallest limit CPython accepts
    try:
        spec = "9" * 641 + "d"
        with pytest.raises(LockExpiryError) as excinfo:
            parse_for(spec, now=_ANCHOR)
    finally:
        sys.set_int_max_str_digits(original)

    exc = excinfo.value
    assert type(exc) is LockExpiryError
    # The bare ValueError from `int()`, chained rather than replaced — and
    # distinguishable from the arithmetic overflow by its own message.
    assert isinstance(exc.__cause__, ValueError)
    assert "digits" in str(exc.__cause__)


def test_zero_and_unparseable_still_raise_their_own_messages() -> None:
    """The new wrapper must not swallow the pre-existing rejections."""
    with pytest.raises(LockExpiryError, match="zero-length window"):
        parse_for("0d", now=_ANCHOR)
    with pytest.raises(LockExpiryError, match="cannot parse"):
        parse_for("7 fortnights", now=_ANCHOR)


def test_cli_lock_exits_non_zero_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC2: `aelf lock --for 999999y` is a message and exit 1, not a traceback.

    `_cmd_lock` resolves the window before `_open_store()`, so this path never
    touches a store — the assertion is about the exception contract reaching the
    CLI's `except LockExpiryError`, which is the whole reason the wrap matters.
    """
    rc = main(["lock", "a statement that will never be written", "--for", "999999y"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "aelf lock:" in captured.err
    assert "999999y" in captured.err
