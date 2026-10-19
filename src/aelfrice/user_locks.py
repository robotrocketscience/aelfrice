"""User-scope locks in the L0 lock set (#1681, PR2 of 3).

`aelf lock --user` writes a lock to the user-scope store
(`db_paths.user_db_path()`), which every repository and worktree shares.
This module is the read side: it puts those locks into the same L0 set
the repository's locks form, so they are injected with them and count
against the same budget, relevance floor, and payload ceiling.

Three rules, each deterministic and with no embeddings:

* **Read-only, never create.** The user store is opened `mode=ro`, and
  only when the file exists. A missing store contributes nothing and
  leaves no file behind. A store that exists but can't be opened or read
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


def _warn_once(path: Path, exc: BaseException, stderr: IO[str] | None) -> None:
    key = str(path)
    if key in _WARNED:
        return
    _WARNED.add(key)
    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    print(
        f"aelfrice: cannot read the user lock store at {path}: "
        f"{first_line}; injecting repository locks only.",
        file=stderr if stderr is not None else sys.stderr,
    )


def read_user_locks(
    repo_db_path: str | None = None,
    *,
    now: str | None = None,
    stderr: IO[str] | None = None,
) -> list[Belief]:
    """Active, unexpired user-scope locks, or [] when there are none.

    Never creates the user store. Returns [] when it is missing, when it
    is the repository store (`repo_db_path`), or when it can't be opened
    or read; the last case prints one stderr line per process.

    `now` is the ISO-8601 UTC instant the expiry filter compares against;
    it defaults to the wall clock, read once.
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
    try:
        store = MemoryStore(str(path), project_context_default="", read_only=True)
        try:
            rows = store.list_locked_beliefs()
        finally:
            store.close()
    except (sqlite3.Error, OSError, ReadOnlyStoreUnavailable) as exc:
        _warn_once(path, exc, stderr)
        return []
    ts = now if now is not None else datetime.now(timezone.utc).isoformat()
    out: list[Belief] = []
    for b in rows:
        if b.lock_level != LOCK_USER:
            continue
        # The #1314 sweep's predicate, inverted: due when at or before now.
        if b.lock_expires_at is not None and b.lock_expires_at <= ts:
            continue
        setattr(b, USER_SCOPE_ATTR, USER_SCOPE_VALUE)
        out.append(b)
    return out


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
) -> list[Belief]:
    """Every lock an injection path injects: the store's, then the user's.

    `user_locks=None` reads the user-scope store; pass a list (or `()`)
    to use locks the caller has already read.
    """
    if user_locks is None:
        user_locks = read_user_locks(store.db_path)
    return merge_lock_sets(store.list_locked_beliefs(), user_locks)

