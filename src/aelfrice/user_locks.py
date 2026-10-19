"""User-scope locks in the L0 lock set (#1681, PR2 of 3).

`aelf lock --user` writes a lock to the user-scope store
(`db_paths.user_db_path()`), which every repository and worktree shares.
This module is the read side: it puts those locks into the same L0 set
the repository's locks form, so they are injected with them and count
against the same budget, relevance floor, and payload ceiling.

Three rules, each deterministic and with no embeddings:

* **Read-only, never create.** The user store is opened `mode=ro`, and
  only when the file exists. A missing store contributes nothing and
  leaves no file behind. The database file of an existing store is never
  written, though a read-only open lets SQLite create the `-wal` and
  `-shm` files beside it. A store that exists but can't be opened or read
  contributes nothing and prints one stderr line per process.
* **The repository copy wins.** A user lock that near-duplicates a
  repository lock, by `dedup.near_duplicate_locks_among` (the test
  `find_near_duplicate_locks` applies: Jaccard >= 0.8 and Levenshtein
  ratio >= 0.85), is dropped.
* **Order.** Repository locks first, in the repository store's order,
  then the surviving user locks in the user store's order. Both stores
  order locks `locked_at DESC, id ASC`, which is also the order
  `aelf locked` lists the two scopes in.

A read-only handle skips the open-time expiry sweep (#1314), so a user
lock whose `lock_expires_at` has passed is still `lock_level = 'user'` on
disk until a writable open sweeps it. It is dropped here with the
comparison the sweep itself makes, so an expired user lock is never
injected.

A user lock is not a row in the repository store, so a write keyed by
belief id there (an exposure row, an injection event, a deferred-feedback
row) must skip it. `is_user_scope_lock` is how those writers tell.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

from aelfrice.models import LOCK_USER, Belief

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

#: Instance attribute set on a `Belief` read from the user-scope store.
#: Not a dataclass field, so it changes no equality, `asdict`, or
#: serialized form; it only lets a repository-store writer skip the row.
USER_SCOPE_ATTR: Final[str] = "_aelfrice_lock_scope"
USER_SCOPE_VALUE: Final[str] = "user"

# Paths already reported as unreadable in this process, so a hook that
# reads the L0 set more than once per fire warns once.
_WARNED: set[str] = set()

# Locks last read from each user-store path, keyed by the file signature
# they were read at. One hook fire can build the L0 set more than once
# (the first prompt's `<locked>` section and its retrieval; the compact
# SessionStart's baseline and its rebuild), and the store is opened once.
#
# The signature is the size, mtime, and inode of the database file and of
# its `-wal` sidecar. It is only as fresh as the filesystem's mtime
# resolution: a write that lands in the same timestamp tick and leaves
# both sizes unchanged can be missed. That staleness lasts at most one
# process. The hooks are one-shot processes, and nothing long-lived
# calls `read_user_locks`, so a missed write shows up on the next fire.
#
# At most `_CACHE_MAX` paths are kept, oldest dropped first. One process
# normally reads one user store; the bound keeps a process that is handed
# many `AELFRICE_USER_DB` values from holding every one.
_FileSig = tuple[int, int, int]
_CACHE: dict[str, tuple[tuple[_FileSig | None, _FileSig | None], list[Belief]]] = {}
_CACHE_MAX: Final[int] = 4

#: How long the read-only user-store connection waits on another
#: connection's lock, in milliseconds. A hook reads this store on every
#: prompt and must not stall on it: a store some other process holds
#: under an exclusive lock is treated as unreadable after this long, and
#: the repository locks are injected alone. The repository store keeps
#: its 5000 ms default; it is the store the hook writes.
USER_STORE_BUSY_TIMEOUT_MS: Final[int] = 150


def is_user_scope_lock(b: object) -> bool:
    """Whether `b` was read from the user-scope lock store."""
    return getattr(b, USER_SCOPE_ATTR, None) == USER_SCOPE_VALUE


def _same_file(a: Path, b: str) -> bool:
    """Whether the user store path and the repository store path match.

    Mirrors the `aelf lock --user` same-file guard: if
    `AELFRICE_USER_DB` names the repository store, its locks are already
    in the L0 set as repository locks and are not read a second time.
    """
    if b == ":memory:":
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


#: The end of the unreadable-store warning: what the caller does instead.
FALLBACK_INJECT: Final[str] = "injecting repository locks only"
FALLBACK_LIST: Final[str] = "listing repository locks only"


def _warn_once(
    path: Path,
    exc: BaseException,
    stderr: IO[str] | None,
    fallback: str = FALLBACK_INJECT,
) -> None:
    key = str(path)
    if key in _WARNED:
        return
    _WARNED.add(key)
    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    print(
        f"aelfrice: cannot read the user lock store at {path}: "
        f"{first_line}; {fallback}.",
        file=stderr if stderr is not None else sys.stderr,
    )


def _file_sig(p: Path) -> _FileSig | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _store_sig(path: Path) -> tuple[_FileSig | None, _FileSig | None]:
    # An empty `-wal` holds no pages, so it reads the same as no `-wal`.
    # The read-only open itself leaves an empty one behind; without this
    # the first read would invalidate its own cache entry.
    wal = _file_sig(Path(f"{path}-wal"))
    if wal is not None and wal[1] == 0:
        wal = None
    return (_file_sig(path), wal)


def open_user_store_for_read() -> MemoryStore | None:
    """The user store opened read-only, or None when it doesn't exist.

    For commands that look a user lock up by id (`aelf show`,
    `aelf feedback`). Never creates the file. Raises what the read-only
    open raises when the file exists but can't be opened.
    """
    from aelfrice.db_paths import user_db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    path = user_db_path()
    if str(path) == ":memory:" or not path.is_file():
        return None
    return MemoryStore(str(path), project_context_default="", read_only=True)


def read_user_locks(
    repo_db_path: str | None = None,
    *,
    now: str | None = None,
    stderr: IO[str] | None = None,
    fallback: str = FALLBACK_INJECT,
) -> list[Belief]:
    """Active, unexpired user-scope locks, or [] when there are none.

    Never creates the user store. Returns [] when it is missing, when it
    is the repository store (`repo_db_path`), or when it can't be opened
    or read; the last case prints one stderr line per process, which
    ends with `fallback` (what the caller does instead).

    `now` is the ISO-8601 UTC instant the expiry filter compares against;
    it defaults to the wall clock, read once.

    The store is opened once per process for as long as the file is
    unchanged (`_CACHE`); the expiry filter runs on every call. "Unchanged"
    means the same size, mtime, and inode for the database and its
    `-wal`, so a write in the same mtime tick that keeps both sizes can be
    missed for the rest of this process. The hooks are one-shot processes,
    so the next fire reads it.
    """
    from aelfrice.db_paths import user_db_path  # noqa: PLC0415
    from aelfrice.store import (  # noqa: PLC0415
        MemoryStore,
        ReadOnlyStoreUnavailable,
    )

    path = user_db_path()
    if str(path) == ":memory:" or not path.is_file():
        return []
    if repo_db_path is not None and _same_file(path, repo_db_path):
        return []
    key = str(path)
    sig = _store_sig(path)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == sig:
        rows = cached[1]
    else:
        try:
            store = MemoryStore(
                str(path), project_context_default="", read_only=True,
                busy_timeout_ms=USER_STORE_BUSY_TIMEOUT_MS,
            )
            try:
                rows = [
                    b for b in store.list_locked_beliefs()
                    if b.lock_level == LOCK_USER
                ]
            finally:
                store.close()
        except (sqlite3.Error, OSError, ReadOnlyStoreUnavailable) as exc:
            # Cached as "no user locks" for this signature, so a process
            # that finds the store locked or corrupt pays the timeout and
            # warns once, not on every read. A change to the file moves
            # the signature and the next read tries again.
            rows = []
            _warn_once(path, exc, stderr, fallback)
        for b in rows:
            setattr(b, USER_SCOPE_ATTR, USER_SCOPE_VALUE)
        _CACHE.pop(key, None)
        _CACHE[key] = (sig, rows)
        while len(_CACHE) > _CACHE_MAX:
            del _CACHE[next(iter(_CACHE))]
    ts = now if now is not None else datetime.now(timezone.utc).isoformat()
    # The #1314 sweep's predicate, inverted: due when at or before now.
    return [
        b for b in rows
        if b.lock_expires_at is None or b.lock_expires_at > ts
    ]


def merge_lock_sets(
    repo_locks: Sequence[Belief], user_locks: Sequence[Belief],
) -> list[Belief]:
    """The L0 set: repository locks, then user locks that aren't duplicates.

    A user lock is dropped when it near-duplicates any repository lock,
    which covers the same statement locked in both scopes (same text,
    same id). Order: `repo_locks` as given, then the surviving
    `user_locks` as given. With no user locks this returns a copy of
    `repo_locks`, unchanged.
    """
    out: list[Belief] = list(repo_locks)
    if not user_locks:
        return out
    from aelfrice.dedup import near_duplicate_locks_among  # noqa: PLC0415

    repo_pairs = sorted(((b.id, b.content) for b in repo_locks), key=lambda t: t[0])
    repo_ids = {b.id for b in repo_locks}
    for u in user_locks:
        if u.id in repo_ids:
            continue
        if near_duplicate_locks_among(repo_pairs, u.content):
            continue
        out.append(u)
    return out


def l0_lock_set(
    store: MemoryStore,
    *,
    user_locks: Sequence[Belief] | None = None,
    fallback: str = FALLBACK_INJECT,
) -> list[Belief]:
    """Every lock an injection path injects: the store's, then the user's.

    `user_locks=None` reads the user-scope store; pass a list (or `()`)
    to use locks the caller has already read. `fallback` ends the
    warning an unreadable user store prints.
    """
    if user_locks is None:
        user_locks = read_user_locks(store.db_path, fallback=fallback)
    return merge_lock_sets(store.list_locked_beliefs(), user_locks)


def repo_locks_duplicating_user_locks(
    repo_locks: Sequence[tuple[str, str]],
    user_locks: Sequence[tuple[str, str]],
) -> list[tuple[str, str, list[str]]]:
    """Repository locks that a user lock near-duplicates (#1681).

    Each lock is `(id, content)`. Returns `(repo_id, repo_content,
    user_ids)` for every repository lock that shares its id with a user
    lock or that a user lock near-duplicates, by the test the L0 merge
    applies (`near_duplicate_locks_among`). These are the repository
    locks injected in place of a user lock; `aelf doctor` lists them so
    the user can unlock them by hand. Nothing is merged or changed here.

    Order: `repo_locks` as given; within a row, user ids in
    `user_locks` order.
    """
    if not repo_locks or not user_locks:
        return []
    from aelfrice.dedup import near_duplicate_locks_among  # noqa: PLC0415

    # Same direction as `merge_lock_sets`: each user lock's text against
    # the repository locks, so this lists exactly the pairs that make the
    # merge drop a user lock.
    repo_pairs = sorted(repo_locks, key=lambda t: t[0])
    repo_ids = {rid for rid, _ in repo_locks}
    by_repo: dict[str, list[str]] = {}
    for uid, ucontent in user_locks:
        matched = {p.belief_a_id for p in near_duplicate_locks_among(repo_pairs, ucontent)}
        if uid in repo_ids:
            matched.add(uid)
        for rid in matched:
            by_repo.setdefault(rid, []).append(uid)
    return [
        (rid, rcontent, by_repo[rid])
        for rid, rcontent in repo_locks if rid in by_repo
    ]

