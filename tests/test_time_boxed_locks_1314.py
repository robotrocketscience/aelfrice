"""#1314: a lock can carry a window, and the window actually closes.

The mechanism is a column plus an idempotent sweep at store open. The
sweep exists instead of a `now`-aware lock predicate because `lock_level`
is read in roughly fifteen places, and `list_speculative_beliefs` is the
*complement* of `list_locked_beliefs` — a predicate applied to one and
not the other drops an expired lock out of both tiers at once and makes
it invisible to L0 and L1 alike. That specific defect is what
`test_expired_lock_appears_in_both_tiers_correctly` pins, in both
directions, because asserting only the absence would pass just as well
with the belief deleted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.lock_expiry import (
    LockExpiryError,
    format_remaining,
    parse_for,
    parse_until,
)
from aelfrice.models import (
    FEEDBACK_SOURCE_LOCK_EXPIRE,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the dotdir and DB away from the developer's real store.

    The live store is repo-local, so an unpinned test opens it, sweeps
    *its* locks, and passes on CI while corrupting local state. Pinned
    per-test, not per-session.
    """
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))
    monkeypatch.delenv("AELF_AUTOLOCK_CORRECTIONS", raising=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mk(
    store: MemoryStore,
    bid: str,
    *,
    expires_at: str | None,
    lock: str = LOCK_USER,
) -> None:
    stamp = _now().isoformat()
    store.insert_belief(Belief(
        id=bid,
        content=f"belief body for {bid}",
        content_hash=f"hash-{bid}",
        alpha=2.0,
        beta=1.0,
        type="fact",
        lock_level=lock,
        locked_at=stamp if lock == LOCK_USER else None,
        created_at=stamp,
        last_retrieved_at=None,
        origin="user_stated",
        lock_expires_at=expires_at,
    ))


def _past() -> str:
    return (_now() - timedelta(days=1)).isoformat()


def _future(days: int = 3) -> str:
    return (_now() + timedelta(days=days)).isoformat()


# --- migration ----------------------------------------------------------

def test_existing_locks_migrate_to_no_expiry(tmp_path: Path) -> None:
    """A pre-#1314 store's locks are permanent and stay that way.

    The whole no-silent-loss claim rests on this: a user who upgrades
    must not find a lock swept away because the new column defaulted to
    something other than NULL.
    """
    db = str(tmp_path / "legacy.db")
    store = MemoryStore(db)
    _mk(store, "legacy-a", expires_at=None)
    _mk(store, "legacy-b", expires_at=None)
    before = [b.id for b in store.list_locked_beliefs()]
    store.close()

    store = MemoryStore(db)
    try:
        assert [b.id for b in store.list_locked_beliefs()] == before
        assert all(b.lock_expires_at is None for b in store.list_locked_beliefs())
    finally:
        store.close()


# --- the sweep ----------------------------------------------------------

def test_sweep_flips_only_due_rows(tmp_path: Path) -> None:
    db = str(tmp_path / "sweep.db")
    store = MemoryStore(db)
    _mk(store, "due", expires_at=_past())
    _mk(store, "future", expires_at=_future())
    _mk(store, "permanent", expires_at=None)
    store.close()

    store = MemoryStore(db)
    try:
        assert sorted(b.id for b in store.list_locked_beliefs()) == [
            "future", "permanent",
        ]
        assert store.get_belief("due").lock_level == LOCK_NONE
    finally:
        store.close()


def test_expired_lock_appears_in_both_tiers_correctly(
    tmp_path: Path,
) -> None:
    """The defect the sweep design exists to prevent.

    Asserted in both directions on purpose. "Absent from the locked
    list" alone is satisfied by a belief that was deleted, or by one that
    fell out of every tier — which is exactly the bug a `now`-aware
    predicate on one side of the complement would have produced.
    """
    db = str(tmp_path / "tiers.db")
    store = MemoryStore(db)
    _mk(store, "expired", expires_at=_past())
    store.close()

    store = MemoryStore(db)
    try:
        assert "expired" not in {b.id for b in store.list_locked_beliefs()}
        assert "expired" in {b.id for b in store.list_speculative_beliefs()}
    finally:
        store.close()


def test_sweep_preserves_origin_expiry_and_locked_at(tmp_path: Path) -> None:
    """Only `lock_level` moves.

    `origin` stays because the belief *was* user-asserted and only its
    injection privilege expired — rewriting it would reproduce the
    autolock origin-laundering defect. The expiry and `locked_at` stay
    because after the flip they are the only record of *why* this belief
    is no longer locked.
    """
    db = str(tmp_path / "preserve.db")
    store = MemoryStore(db)
    _mk(store, "kept", expires_at=_past())
    store.close()

    store = MemoryStore(db)
    try:
        b = store.get_belief("kept")
        assert b.lock_level == LOCK_NONE
        assert b.origin == "user_stated"
        assert b.lock_expires_at is not None
        assert b.locked_at is not None
    finally:
        store.close()


def test_sweep_is_idempotent_and_writes_one_audit_row_per_flip(
    tmp_path: Path,
) -> None:
    db = str(tmp_path / "idem.db")
    store = MemoryStore(db)
    _mk(store, "a", expires_at=_past())
    _mk(store, "b", expires_at=_past())
    store.close()

    store = MemoryStore(db)
    try:
        rows = [
            e for e in store.list_feedback_events()
            if e.source == FEEDBACK_SOURCE_LOCK_EXPIRE
        ]
        assert sorted(e.belief_id for e in rows) == ["a", "b"]
        assert store.last_lock_sweep() is not None
        assert store.last_lock_sweep()[1] == 2
        # Re-running inside the same open must flip nothing further.
        assert store.sweep_expired_locks() == 0
    finally:
        store.close()

    store = MemoryStore(db)
    try:
        rows = [
            e for e in store.list_feedback_events()
            if e.source == FEEDBACK_SOURCE_LOCK_EXPIRE
        ]
        assert len(rows) == 2, "a second open re-audited an already-swept lock"
    finally:
        store.close()


def test_sweep_ignores_beliefs_that_were_never_locked(tmp_path: Path) -> None:
    """An expiry on an unlocked belief is inert.

    `lock_expires_at` is retained through a flip, so unlocked rows
    carrying a past expiry are the normal steady state. Without the
    `lock_level = 'user'` term the sweep would re-audit every one of them
    on every open, forever.
    """
    db = str(tmp_path / "unlocked.db")
    store = MemoryStore(db)
    _mk(store, "never-locked", expires_at=_past(), lock=LOCK_NONE)
    store.close()

    store = MemoryStore(db)
    try:
        assert store.last_lock_sweep() is None
        assert not [
            e for e in store.list_feedback_events()
            if e.source == FEEDBACK_SOURCE_LOCK_EXPIRE
        ]
    finally:
        store.close()


def test_list_expiring_locks_excludes_already_swept(tmp_path: Path) -> None:
    db = str(tmp_path / "expiring.db")
    store = MemoryStore(db)
    _mk(store, "soon", expires_at=_future(2))
    _mk(store, "later", expires_at=_future(30))
    _mk(store, "gone", expires_at=_past())
    store.close()

    store = MemoryStore(db)
    try:
        horizon = (_now() + timedelta(days=7)).isoformat()
        assert [b.id for b in store.list_expiring_locks(before=horizon)] == [
            "soon",
        ]
    finally:
        store.close()


# --- parsing ------------------------------------------------------------

@pytest.mark.parametrize(
    ("spec", "expected_delta_days"),
    [("1d", 1), ("14d", 14), ("1w", 7), ("2w", 14)],
)
def test_parse_for_fixed_length_units(
    spec: str, expected_delta_days: int,
) -> None:
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    got = datetime.fromisoformat(parse_for(spec, now=now))
    assert (got - now).days == expected_delta_days


def test_parse_for_uses_calendar_arithmetic_not_thirty_days() -> None:
    """1mo from January 31 is February 28, not March 2.

    Approximating a month as 30 days is the error that surfaces once a
    year, on the day it matters.
    """
    now = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
    assert parse_for("1mo", now=now).startswith("2026-02-28")
    leap = datetime(2024, 2, 29, 12, 0, tzinfo=timezone.utc)
    assert parse_for("1y", now=leap).startswith("2025-02-28")


def test_parse_for_forever_is_no_expiry() -> None:
    """`forever` is NULL, not a far-future timestamp.

    Stored as a sentinel date it would be indistinguishable from a real
    window in every listing and eventually expire on its own.
    """
    assert parse_for("forever", now=_now()) is None
    assert parse_for("FOREVER", now=_now()) is None


@pytest.mark.parametrize(
    "spec", ["0d", "7", "7m", "abc", "-1d", "7 d", "", "d7"],
)
def test_parse_for_rejects(spec: str) -> None:
    with pytest.raises(LockExpiryError):
        parse_for(spec, now=_now())


def test_parse_until_accepts_bare_date_as_midnight_utc() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_until("2026-08-10", now=now) == "2026-08-10T00:00:00+00:00"


def test_parse_until_reads_a_naive_timestamp_as_utc() -> None:
    """Not as local time — otherwise the value written depends on where
    the machine is, and two users typing the same thing get different
    windows."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_until("2026-08-10T06:30:00", now=now).startswith(
        "2026-08-10T06:30:00+00:00",
    )


@pytest.mark.parametrize("spec", ["2020-01-01", "not-a-date", ""])
def test_parse_until_rejects(spec: str) -> None:
    with pytest.raises(LockExpiryError):
        parse_until(spec, now=_now())


def test_format_remaining_renders_two_coarsest_units() -> None:
    now = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
    assert format_remaining(None, now=now) == "—"
    assert format_remaining("2026-02-06T16:00:00+00:00", now=now) == "6d 4h"
    assert format_remaining("2026-01-31T15:12:00+00:00", now=now) == "3h 12m"
    assert format_remaining("garbage", now=now) == "?"


# --- CLI ----------------------------------------------------------------
#
# Driven in-process through `main(argv, out=...)`, the convention the
# other CLI tests use. Not a subprocess: this needs no process isolation,
# and a subprocess here would be one more blocking call for the
# termination policy to bound.

def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch,
         tmp_path: Path) -> tuple[int, str]:
    import io

    from aelfrice.cli import main

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("AELF_SESSION_ID", "test-1314")
    buf = io.StringIO()
    return main(argv, out=buf), buf.getvalue()


def test_cli_lock_for_sets_a_window_and_locked_shows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, out = _run(
        ["lock", "the router holds the reserved address", "--for", "7d"],
        monkeypatch, tmp_path,
    )
    assert rc == 0
    assert "window: expires" in out
    _, listing = _run(["locked"], monkeypatch, tmp_path)
    assert "[6d" in listing or "[7d" in listing, listing


def test_cli_bare_relock_clears_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one genuinely ambiguous case, decided on the issue."""
    text = "the power feed on that machine is unreliable"
    _run(["lock", text, "--for", "7d"], monkeypatch, tmp_path)
    _, out = _run(["lock", text], monkeypatch, tmp_path)
    assert "expiry cleared" in out, out
    _, listing = _run(["locked"], monkeypatch, tmp_path)
    assert "[—]" in listing, listing


def test_cli_relock_with_for_refreshes_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "a second session owns the network config right now"
    _run(["lock", text, "--for", "1d"], monkeypatch, tmp_path)
    _run(["lock", text, "--for", "3w"], monkeypatch, tmp_path)
    store = MemoryStore(str(tmp_path / "cli.db"))
    try:
        expiry = store.list_locked_beliefs()[0].lock_expires_at
        assert expiry is not None
        # 3w out, not the 1d the first call set.
        assert datetime.fromisoformat(expiry) - _now() > timedelta(days=14)
    finally:
        store.close()


def test_cli_rejects_a_bad_window_without_writing_a_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parse runs before the store opens.

    Otherwise a typo leaves a *permanent* lock behind — the exact
    outcome the user was trying to avoid by passing a window.
    """
    rc, _ = _run(
        ["lock", "some statement about the build", "--for", "7m"],
        monkeypatch, tmp_path,
    )
    assert rc == 1
    _, listing = _run(["locked"], monkeypatch, tmp_path)
    assert "no locked beliefs" in listing


def test_cli_rejects_a_past_until(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, _ = _run(
        ["lock", "some statement about the build", "--until", "2020-01-01"],
        monkeypatch, tmp_path,
    )
    assert rc == 1
    _, listing = _run(["locked"], monkeypatch, tmp_path)
    assert "no locked beliefs" in listing


def test_cli_for_and_until_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(
            ["lock", "x y z", "--for", "7d", "--until", "2030-01-01"],
            monkeypatch, tmp_path,
        )
    assert excinfo.value.code != 0


def test_cli_unlock_clears_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "the scheduled job failed to fire last night"
    _run(["lock", text, "--for", "7d"], monkeypatch, tmp_path)
    store = MemoryStore(str(tmp_path / "cli.db"))
    try:
        bid = store.list_locked_beliefs()[0].id
    finally:
        store.close()

    assert _run(["unlock", bid], monkeypatch, tmp_path)[0] == 0

    store = MemoryStore(str(tmp_path / "cli.db"))
    try:
        b = store.get_belief(bid)
        assert b is not None
        assert b.lock_level == LOCK_NONE
        assert b.lock_expires_at is None
    finally:
        store.close()
