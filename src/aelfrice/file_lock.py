"""Portable advisory file locking (#1329).

`fcntl` is Unix-only. Importing it at module scope, as `session_ring` and
`auto_install` both did, makes every `aelf` command die at import on
Windows — including `aelf --help` and `aelf doctor`, because `cli` imports
`auto_install` eagerly at module load. The failure is total and has nothing
to do with locking: the process never reaches argument parsing.

This module is the single place that knows how a host takes an advisory
lock. Three operations, matching the two shapes the callers actually need:

* :func:`lock_exclusive` — blocking, used by the ring writers.
* :func:`try_lock_exclusive` — non-blocking, used by the `auto_install`
  merge (another process holding it means "they will finish the merge, so
  skip") and by `session_ring._flock_until`'s poll loop.
* :func:`unlock` — best-effort release.

## Contention is reported one way on every host

`fcntl.flock(..., LOCK_NB)` raises `BlockingIOError` on contention;
`msvcrt.locking(..., LK_NBLCK)` raises a plain `OSError` with `EACCES`,
which is *not* a `BlockingIOError` and would sail straight past
`auto_install`'s `except BlockingIOError`. So :func:`try_lock_exclusive`
normalises: contention always raises `BlockingIOError`, whatever the
backend. `BlockingIOError` is an `OSError` subclass, so the callers that
discriminate on `errno in (EACCES, EAGAIN)` keep working unchanged.

## The no-backend fallback is a no-op, and that is a real limitation

If neither backend imports, the lock operations do nothing and report
success. They do **not** raise: `auto_install` only catches
`BlockingIOError`, so raising here would convert "no advisory locking" into
a crash at CLI entry, which is the bug this module exists to fix.

A no-op lock is single-process-safe, not concurrency-safe. Two `aelf`
processes racing the same settings.json on such a host can interleave a
read-modify-write. :data:`HAVE_ADVISORY_LOCKS` is exported so `doctor` and
the docs can say so out loud rather than let a user infer serialisation
that is not there.

Windows note: `msvcrt.locking` locks a byte range relative to the current
file position, not the whole file, so every operation here seeks to 0,
acts on :data:`_LOCK_REGION_BYTES`, and restores the caller's position.
The lock file is a dedicated sibling that nothing else reads, so the
region is arbitrary — it only has to be *the same* region every time.
`LK_LOCK` also retries for ~10s and then fails rather than blocking
indefinitely the way `flock(LOCK_EX)` does; a caller that must not proceed
unlocked should pass an explicit timeout instead of relying on the
blocking form.
"""
from __future__ import annotations

import errno
import os
from typing import Any, Final

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows / by masking
    _fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None  # type: ignore[assignment]

_LOCK_REGION_BYTES: Final[int] = 1
"""Bytes locked on the Windows backend. One is enough — the lock file is a
dedicated sibling and the region only has to be consistent across calls."""

_CONTENTION_ERRNOS: Final[frozenset[int]] = frozenset(
    {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK}
)
"""Errnos that mean "someone else holds it", as opposed to "this host or
filesystem cannot lock at all". `EDEADLK` is included because Windows
reports a self-deadlock rather than a plain refusal when the calling
process already owns the region."""


def _have_advisory_locks() -> bool:
    return _fcntl is not None or _msvcrt is not None


HAVE_ADVISORY_LOCKS: Final[bool] = _have_advisory_locks()
"""False when neither backend imported — locking degrades to a no-op and
concurrent `aelf` processes are not serialised. See the module docstring."""


def _windows_region(fd: int, mode: int) -> None:
    """Apply `mode` to the fixed region, restoring the file position.

    `msvcrt.locking` is position-relative, so an unbalanced seek here would
    lock one region and unlock a different one — the failure mode is a
    permanently held lock, which is worse than no lock at all.
    """
    msvcrt: Any = _msvcrt
    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, mode, _LOCK_REGION_BYTES)
    finally:
        os.lseek(fd, saved, os.SEEK_SET)


def lock_exclusive(fd: int) -> None:
    """Take an exclusive lock on `fd`, blocking until it is available.

    Raises `OSError` when the host has a backend but the filesystem cannot
    lock — callers treat that as the documented unlocked fall-through.
    Does nothing when no backend is available.
    """
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
    elif _msvcrt is not None:
        _windows_region(fd, _msvcrt.LK_LOCK)


def try_lock_exclusive(fd: int) -> None:
    """Take an exclusive lock on `fd` without blocking.

    Raises `BlockingIOError` if another process holds it — on every
    backend, see the module docstring. Any other `OSError` propagates
    unchanged and means locking is unsupported here, which callers
    distinguish from contention. Does nothing when no backend is
    available, so a caller on such a host proceeds as if it won the lock.
    """
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return
    if _msvcrt is not None:
        try:
            _windows_region(fd, _msvcrt.LK_NBLCK)
        except BlockingIOError:
            raise
        except OSError as exc:
            if exc.errno in _CONTENTION_ERRNOS:
                raise BlockingIOError(
                    exc.errno, "advisory lock held by another process"
                ) from exc
            raise


def unlock(fd: int) -> None:
    """Release the lock on `fd`.

    Best-effort by contract at every call site — closing the descriptor
    releases the lock regardless — but errors are propagated rather than
    swallowed here so the callers keep making that decision themselves.
    """
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
    elif _msvcrt is not None:
        _windows_region(fd, _msvcrt.LK_UNLCK)


__all__ = [
    "HAVE_ADVISORY_LOCKS",
    "lock_exclusive",
    "try_lock_exclusive",
    "unlock",
]
