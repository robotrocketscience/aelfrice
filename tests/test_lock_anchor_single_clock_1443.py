"""#1443 — the persisted lock anchor is the anchor the window resolved from.

`_cmd_lock` used to read the clock twice: once for `parse_for`/`parse_until`
before `_open_store()`, and again for `locked_at` after it. Opening the
store is a WRITE — DDL, any pending migration, and the #1314 open-time
expiry sweep — so the gap is real and not a constant, and the pair the row
ends up holding is `(locked_at, lock_expires_at)` where the expiry was
resolved from an *earlier* instant than the anchor beside it.

Nothing user-visible was wrong at `--for 1w`. What was wrong is that the
identity a later expiry audit, replay probe or checker would assume —
`lock_expires_at == parse_for(spec, now=locked_at)` — was false by a
variable margin, and nothing said so either way. These tests are that
missing statement.

Asserted against `parse_for` itself, not against a hand-built expected
timestamp: an expected value computed in the test would re-implement the
calendar arithmetic and could agree with a wrong anchor.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.lock_expiry import parse_for, parse_until
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the DB away from the developer's repo-local store.

    An unpinned run would open the live store, sweep *its* locks and
    write a lock into it.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("AELF_SESSION_ID", "test-1443")
    monkeypatch.delenv("AELF_AUTOLOCK_CORRECTIONS", raising=False)


def _run(argv: list[str]) -> int:
    from aelfrice.cli import main

    return main(argv, out=io.StringIO())


def _only_lock(tmp_path: Path) -> tuple[str, str]:
    """The `(locked_at, lock_expires_at)` pair the CLI actually wrote."""
    store = MemoryStore(str(tmp_path / "cli.db"))
    try:
        locks = store.list_locked_beliefs()
        assert len(locks) == 1, locks
        locked_at = locks[0].locked_at
        expires_at = locks[0].lock_expires_at
    finally:
        store.close()
    assert locked_at is not None
    assert expires_at is not None
    return locked_at, expires_at


def _as_dt(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).astimezone(timezone.utc)


def _delay_store_open_past_a_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `_open_store()` cross a whole-second boundary.

    Store-open latency is what separated the two clock reads, and on a
    fast machine it is usually sub-second — which under a truncating
    stamp can leave both reads inside the same second and let a
    two-read implementation pass by luck. Waiting to the next boundary
    makes the second read land in a later second deterministically, so
    this test fails on the two-read shape every run rather than most
    runs. It does not simulate the defect: the wait is inside
    `_open_store`, exactly where the real latency is.
    """
    from aelfrice import cli

    real_open = cli._open_store

    def _slow_open() -> object:
        now = time.time()
        time.sleep(1.0 - (now % 1.0) + 0.01)
        return real_open()

    monkeypatch.setattr(cli, "_open_store", _slow_open)


def test_the_persisted_expiry_resolves_from_the_persisted_anchor(
    tmp_path: Path,
) -> None:
    """Hypothesis: `lock_expires_at == parse_for(spec, now=locked_at)`,
    exactly.

    An equality, not a tolerance: the point of the invariant is that a
    checker can recompute the expiry from the stored anchor and get the
    stored value back. A tolerance would pass on the two-read shape too,
    which is precisely the state that made the identity untrustworthy.
    """
    assert _run(["lock", "the deploy key lives in the vault", "--for", "1w"]) == 0
    locked_at, expires_at = _only_lock(tmp_path)
    assert expires_at == parse_for("1w", now=_as_dt(locked_at))


def test_store_open_latency_does_not_move_the_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: the identity survives a slow `_open_store()`, because
    the clock is read once and threaded through both writes.

    Falsifiable by restoring the second read: `locked_at` is then stamped
    from an instant at least a second after the one the window resolved
    from, and the recomputed expiry misses the stored one by that gap.
    """
    _delay_store_open_past_a_second(monkeypatch)
    assert _run(["lock", "the runbook lives beside the chart", "--for", "3d"]) == 0
    locked_at, expires_at = _only_lock(tmp_path)
    assert expires_at == parse_for("3d", now=_as_dt(locked_at))
    # ... and the anchor is the one the window was resolved from, not one
    # a second later: 3d apart to the microsecond.
    assert _as_dt(expires_at) - _as_dt(locked_at) == timedelta(days=3)


def test_the_until_path_takes_the_same_single_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: `--until` reads the clock on the same line and gets
    the same treatment (#1443 AC4).

    Stated plainly: this arm does NOT go red on the two-read shape, and
    it is not offered as if it did. `parse_until` resolves the *typed*
    instant and consults `now` only to refuse a past value, so its result
    is anchor-independent and a second clock read cannot move it. What
    the audit pins is the other half of the pair — that the `locked_at`
    written beside it is an instant the stored expiry is still in the
    future of, so an expiry audit reading the row gets a coherent
    interval either way. The distinguishing assertions are the two
    `--for` tests above.
    """
    _delay_store_open_past_a_second(monkeypatch)
    target = (datetime.now(timezone.utc) + timedelta(days=2)).replace(
        microsecond=0,
    )
    spec = target.isoformat()
    assert _run(["lock", "the freeze ends after the audit", "--until", spec]) == 0
    locked_at, expires_at = _only_lock(tmp_path)
    assert expires_at == parse_until(spec, now=_as_dt(locked_at))


def test_a_malformed_window_still_fails_before_the_store_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: moving the clock read did not move the validation.

    #1314's ordering property is that a bad `--for` never writes a
    permanent lock the user has to notice and undo. Pinned by making
    `_open_store` fail outright: if the parse still runs first, the
    command exits 1 without ever calling it.
    """
    from aelfrice import cli

    opened: list[bool] = []

    def _explode() -> object:
        opened.append(True)
        raise AssertionError("the store was opened before the window parsed")

    monkeypatch.setattr(cli, "_open_store", _explode)
    assert _run(["lock", "some statement about the build", "--for", "7m"]) == 1
    assert opened == []
