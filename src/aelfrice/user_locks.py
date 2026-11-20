"""User-scope locks in the L0 lock set (#1681, PR2 of 3).

`aelf lock --user` writes a lock to the user-scope store
(`db_paths.user_db_path()`), which every repository and worktree shares.
This module is the read side: it puts those locks into the same L0 set
the repository's locks form, so they are injected with them and count
against the same budget, relevance floor, and payload ceiling, up to
their share of the budget and of the hook's room
(`DEFAULT_USER_LOCK_SHARE`, #1944).

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
from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
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
    open raises when the file exists but can't be opened, and the
    `OSError` of a path it can't check (`db_paths.store_file_present`).
    """
    from aelfrice.db_paths import (  # noqa: PLC0415
        store_file_present,
        user_db_path,
    )
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    path = user_db_path()
    if str(path) == ":memory:" or not store_file_present(path):
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
    from aelfrice.db_paths import (  # noqa: PLC0415
        store_file_present,
        user_db_path,
    )
    from aelfrice.store import (  # noqa: PLC0415
        MemoryStore,
        ReadOnlyStoreUnavailable,
    )

    path = user_db_path()
    if str(path) == ":memory:":
        return []
    try:
        present = store_file_present(path)
    except OSError as exc:
        _warn_once(path, exc, stderr, fallback)
        return []
    if not present:
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


DEFAULT_USER_LOCK_SHARE: Final[float] = 0.35
"""Share of the hook's room, and of the retrieval budget, user locks may claim (#1944).

User locks are injected in every repository. Without a bound of their
own they took a whole hook fire: on a 5.4.0 install whose user store held
70 locks, 98.5% of the rows a prompt injected were locks, and the median
block sat at the 9,500-character payload limit. This share is what they
may claim when anything else wants the room: at most this fraction of
the hook envelope's room (`HOOK_PAYLOAD_CHAR_LIMIT` less what the fire's
other blocks take), and of the retrieval token budget. A user lock past
the share is shown only in room nothing else wants, so it never displaces
a repository lock or a retrieval hit, and the block names any it can't
show in its last line.

0.35 is measured. On copies of 24 repository stores (2026-11-20), the
repository's own locks took at most 3,005 rendered characters in 23 of
them, and 8,835 in the last. The prompt hook's retrieval budget,
`DEFAULT_HOOK_TOKEN_BUDGET` of 1,500 tokens, is 6,000 characters at 4
characters a token, which leaves 3,500 characters of the 9,500, or
36.8%. The same floor follows from the repository side: 3,005 characters
of locks plus the 3,000-character relevance floor
(`RELEVANCE_BUDGET_FLOOR_FRACTION` of that budget) leave 3,495, or
36.8%. 0.35 is the 5% step below both. The 24th store's locks take 93%
of the room alone, so no share changes what it shows.

`[retrieval] user_lock_share` in `.aelfrice.toml` and
`AELFRICE_USER_LOCK_SHARE` set it; see `resolve_user_lock_share`. A share
of 1.0 is the old behavior, with user locks bounded by the room alone.
"""

ENV_USER_LOCK_SHARE: Final[str] = "AELFRICE_USER_LOCK_SHARE"
USER_LOCK_SHARE_FLAG: Final[str] = "user_lock_share"


def _share_or_none(value: float, source: str, serr: IO[str]) -> float | None:
    if 0.0 <= value <= 1.0:
        return value
    print(
        f"aelfrice: ignoring {source}={value!r} (must be from 0 to 1)",
        file=serr,
    )
    return None


class _SharePin:
    """Where a fire resolves the share from, and the value once it has."""

    __slots__ = ("start", "stderr", "value")

    def __init__(self, start: Path | None, stderr: IO[str] | None) -> None:
        self.start = start
        self.stderr = stderr
        self.value: float | None = None


_PINNED_SHARE: ContextVar[_SharePin | None] = ContextVar(
    "aelfrice_user_lock_share", default=None,
)


def pin_user_lock_share(
    start: Path | None, stderr: IO[str] | None,
) -> Token[_SharePin | None]:
    """Resolve the share from `start` with `stderr` for every read until unpinned.

    A hook fire pins the payload's cwd and its own stderr, so every read
    in the fire (the retrieval charge and the envelope's cut) resolves
    from the project the session is in and gets one value, with any
    warning printed once, on the hook's stream. The first read resolves
    it; a fire that reads none never loads the configuration, so a
    gate-skipped prompt stays off the retrieval import path. Pass the
    token to `unpin_user_lock_share`.
    """
    return _PINNED_SHARE.set(_SharePin(start, stderr))


def unpin_user_lock_share(token: Token[_SharePin | None]) -> None:
    _PINNED_SHARE.reset(token)


def resolve_user_lock_share(
    *, start: Path | None = None, stderr: IO[str] | None = None,
) -> float:
    """The user-lock share, a fraction from 0 to 1 (#1944).

    Inside `pin_user_lock_share`, the pinned `start` and `stderr` are
    used, and the first read's value is kept. Otherwise the
    precedence is the retrieval budget's: `AELFRICE_USER_LOCK_SHARE`,
    then `[retrieval] user_lock_share` in the nearest `.aelfrice.toml`
    from `start`, then `DEFAULT_USER_LOCK_SHARE`. A value that isn't a
    number, or is outside 0 to 1, prints one line to `stderr` and falls
    through to the next tier.
    """
    pin = _PINNED_SHARE.get()
    if pin is not None:
        if pin.value is None:
            pin.value = _resolve_share(start=pin.start, stderr=pin.stderr)
        return pin.value
    return _resolve_share(start=start, stderr=stderr)


def _resolve_share(*, start: Path | None, stderr: IO[str] | None) -> float:
    from aelfrice.retrieval import (  # noqa: PLC0415
        RETRIEVAL_SECTION,
        _read_toml_float_for,  # pyright: ignore[reportPrivateUsage]
    )

    serr = stderr if stderr is not None else sys.stderr
    raw = os.environ.get(ENV_USER_LOCK_SHARE)
    if raw is not None and raw.strip():
        try:
            env = _share_or_none(float(raw), ENV_USER_LOCK_SHARE, serr)
        except ValueError:
            print(
                f"aelfrice: ignoring {ENV_USER_LOCK_SHARE}={raw!r} "
                "(expected a number)",
                file=serr,
            )
            env = None
        if env is not None:
            return env
    toml_value = _read_toml_float_for(USER_LOCK_SHARE_FLAG, start)
    if toml_value is not None:
        toml_share = _share_or_none(
            toml_value, f"[{RETRIEVAL_SECTION}] {USER_LOCK_SHARE_FLAG}", serr,
        )
        if toml_share is not None:
            return toml_share
    return DEFAULT_USER_LOCK_SHARE


def fit_user_lock_share(
    costs: Sequence[tuple[str, int]], room: int,
) -> tuple[list[str], list[str]]:
    """Split user locks into those the share holds and those past it (#1944).

    `costs` is `(id, cost)` in render order; `room` is the share, in the
    same unit. First fit: each lock stays when it fits beside the ones
    kept before it, else it is past the share, so one long lock doesn't
    push out every shorter one after it. Returns `(kept, past)`, each in
    render order.
    """
    kept: list[str] = []
    past: list[str] = []
    used = 0
    for bid, cost in costs:
        if used + cost <= room:
            used += cost
            kept.append(bid)
        else:
            past.append(bid)
    return kept, past


def l0_charge(
    locked: Sequence[Belief], cost: Callable[[Belief], int], budget: int,
) -> int:
    """What the L0 set charges against a retrieval budget (#1944).

    Repository locks are charged in full, as before. User locks are
    charged up to `resolve_user_lock_share()` of `budget` together, so a
    large user store can't reduce the room for retrieval hits to the
    relevance floor. Every lock is still returned and rendered; this
    changes only what the packer reserves for them. With no user lock in
    `locked`, it is the plain sum and reads no configuration.
    """
    repo = 0
    user = 0
    for b in locked:
        if is_user_scope_lock(b):
            user += cost(b)
        else:
            repo += cost(b)
    if not user:
        return repo
    return repo + min(user, int(budget * resolve_user_lock_share()))


def user_lock_render_chars(b: Belief) -> int:
    """Characters one user lock costs in a hook block, newline included.

    The estimate `aelf doctor` uses (#1944): the `<belief>` element with
    its `scope="user"` marker and escaped content, or, for a
    reference-tier lock, its one-line `ref` manifest entry. The hook
    measures the block it renders instead, which also counts a `seen`
    pointer to the lock.
    """
    from aelfrice.hook_payload import escape_for_hook_block  # noqa: PLC0415
    from aelfrice.render_cost import (  # noqa: PLC0415
        MANIFEST_LINE_WRAPPER_CHARS,
        belief_line_chars,
    )
    from aelfrice.retrieval import (  # noqa: PLC0415
        is_reference_lock,
        lock_manifest_line,
    )

    if is_reference_lock(b):
        return len(escape_for_hook_block(lock_manifest_line(b))) + (
            MANIFEST_LINE_WRAPPER_CHARS
        )
    return belief_line_chars(
        len(escape_for_hook_block(b.content)), user_scope=True,
    )


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

