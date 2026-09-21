"""One `.aelfrice.toml` discovery walk, shared by every config reader.

Discovery — "which `.aelfrice.toml` applies here?" — is
section-independent: the walk finds the file, and the section/key lookup
that follows is the caller's own business. That is why this module holds
the walk and nothing else. Every caller keeps its own section, its own
keys, its own precedence and its own defaults; converting a caller
changes *how many stat calls it makes*, never what it resolves.

Stdlib-only, deliberately. The memo this module carries was born inside
`aelfrice.retrieval` (#1289 / PR #1298), but `retrieval` is ~4,600 lines
and importing it is the wrong dependency direction for a module like
`cadence` or `noise_filter` that wants six lines of config. Nothing here
imports from `aelfrice`, so there is no cycle to reason about and no
import cost beyond `pathlib`.

Staleness semantics, stated rather than left implicit: the memo lives
for the duration of one `config_discovery_scope` — one retrieval, one
hook turn — and is discarded at the end of it. A `.aelfrice.toml`
created, deleted or moved *between* two operations is picked up by the
next one exactly as if no memo existed. Only a change made *during* a
single operation is missed, which no caller can observe. A
process-lifetime cache was explicitly rejected: it would make a config
file created after the first call invisible until restart.

Outside a scope every call walks, so a direct caller that never opts in
keeps its original behaviour.

Because every reader funnels through :func:`discover_config`, the bound
on how far that walk may climb (#1582) lives here too, and every reader
inherits it without knowing about it. Adding a config reader that walks
for itself would opt out of the bound silently, which is why
``tests/test_config_discovery_shared.py`` pins the set of modules that
may name a config filename at all.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final

__all__ = [
    "CONFIG_FILENAME",
    "WORKTREE_MARKER",
    "config_discovery_scope",
    "discover_config",
]

# The project-config filename every reader walks up looking for.
CONFIG_FILENAME: Final[str] = ".aelfrice.toml"

# The directory entry that marks the root of a git work tree, and so the
# top of a project. A directory in an ordinary clone; a *file* in a
# linked work tree or a submodule, which is why the probe below is
# `exists()` rather than `is_dir()` — bounding ordinary clones only
# would let every `git worktree` checkout keep walking.
WORKTREE_MARKER: Final[str] = ".git"

# A ContextVar rather than a plain dict so concurrent operations cannot
# see each other's memo. The two concurrency primitives differ and the
# difference is worth stating: a `threading.Thread` started inside a
# scope begins with a *fresh* context, so it walks — correct, if
# slightly wasteful. An `asyncio` task created inside a scope *copies*
# the context, so it keeps this dict and goes on using it after the
# scope exits, which is a stale read of one operation's age. Nothing in
# `aelfrice` creates a thread or a task — grep `src/aelfrice` for
# `asyncio`, `threading.Thread`, `concurrent.futures`: no hits — so the
# asyncio case is latent, not live. Anything that introduces one must
# enter its own scope rather than inherit this one.
_CONFIG_DISCOVERY_MEMO: ContextVar[dict[Path, Path | None] | None] = ContextVar(
    "aelfrice_config_discovery_memo",
    default=None,
)

# Memo key standing for "the caller passed no `start`", i.e. resolve
# from cwd. Not a real path, and cannot collide with one: every other
# key is an absolute resolved directory.
_CWD_KEY: Final[Path] = Path("\x00cwd")


def _home_dir() -> Path | None:
    """The current user's home directory, resolved, or None.

    Never raises. `Path.home()` reads `$HOME` on POSIX and falls back to
    the password database, which can raise `RuntimeError` when the user
    has no entry; `resolve()` can raise `OSError` on a hostile path. In
    either case the home bound simply does not apply, and the work-tree
    bound still does.
    """
    try:
        return Path.home().resolve()
    except (RuntimeError, OSError):
        return None


@contextmanager
def config_discovery_scope() -> Iterator[None]:
    """Memoize `.aelfrice.toml` discovery for the duration of the block.

    Entering is what turns the memo on; outside a scope every caller
    walks, preserving the original behaviour for direct callers. Nesting
    is safe — an inner scope reuses the outer memo rather than shadowing
    it, so a hook turn wrapping four retrievals costs one walk, not
    five.
    """
    if _CONFIG_DISCOVERY_MEMO.get() is not None:
        yield
        return
    token = _CONFIG_DISCOVERY_MEMO.set({})
    try:
        yield
    finally:
        _CONFIG_DISCOVERY_MEMO.reset(token)


def discover_config(start: Path | None = None) -> Path | None:
    """Return the nearest in-project `.aelfrice.toml` at or above `start`.

    Returns None when there is none. `start=None` means "from the
    current working directory". Inside a `config_discovery_scope` the
    result is memoized per resolved start directory, so N readers cost
    one walk instead of N.

    **The walk is bounded and cannot leave the project (#1582).**
    Ascending from `start`, each directory is examined in this order,
    and the first rule that fires ends the walk:

    1. The directory *is* the user's home directory — stop, and do not
       examine it. `$HOME/.aelfrice.toml` is therefore never read. The
       rule bounds the ancestors of `start`, not the machine: it fires
       only when `$HOME` is one of them, so it says nothing about a
       start outside `$HOME`.
    2. The directory holds a `.aelfrice.toml` — that file is the answer.
    3. The directory holds `.git`, so it is a git work-tree root — stop.
       Configuration at the work-tree root is honoured, because rule 2
       is checked first; configuration above it is not.
    4. The directory is the filesystem root — stop.

    Rule 1 is what makes `docs/user/CONFIG.md`'s "there is no global
    configuration and no per-user configuration" true. Rule 3 is what
    makes discovery a property of the project rather than of where on
    the machine the project happens to be checked out: the same repo
    resolves the same configuration under `$HOME`, under `/tmp`, and on
    a CI runner.

    Outside a git work tree there is no project marker, so rule 3
    cannot fire and the walk runs to whichever of rules 1 and 4 it
    meets first. For a `start` under `$HOME` that is rule 1. For a
    `start` outside `$HOME` it is rule 4, and the walk crosses every
    intermediate directory to the filesystem root — including
    directories at or above the level `$HOME` sits at, which rule 1
    does not cover because it matches `$HOME` itself and nothing else.
    The user's own `$HOME/.aelfrice.toml` stays unreachable either way.
    Ascending is the deliberate choice over "examine `start` only",
    which would silently stop honouring a config at the top of a
    non-git project directory that a caller reaches from a
    subdirectory.

    `AELFRICE_DB` is not a discovery input. It names the store to open,
    not the project the configuration belongs to, and letting it move
    the walk would reintroduce exactly the bug this bound removes — a
    resolver whose answer depends on ambient environment rather than on
    the tree being worked in. To use configuration that lives outside
    the project, set the per-key `AELFRICE_*` environment variable,
    which wins over TOML in every resolver's precedence.

    Distinct `start` directories are distinct memo keys and each costs
    its own walk. That is not a defect: a caller that deliberately
    resolves config from a *different* directory (the hook resolving the
    agent's payload cwd rather than the hook process's incidental cwd)
    is asking a different question and must get the answer to it.

    **Invariant: no caller may `os.chdir` inside a scope.** `start=None`
    binds to the cwd at the scope's *first* such call and is not
    re-read afterwards, so a chdir mid-scope would resolve config from
    the old directory. This is a deliberate trade, not an oversight:
    re-reading means `Path.cwd().resolve()` per call, and `resolve()` is
    O(path depth) in `lstat` — measured at 7 `lstat` for a depth-8 cwd,
    against 26 `start=None` calls in one `retrieve()`, i.e. ~182 extra
    syscalls per retrieval, the same cost class #1289 removed. The
    invariant holds today: the only `os.chdir` in `src/aelfrice` is
    `project_warm._warm_store`, which runs in its own CLI process,
    restores cwd in a `finally`, and is never inside a scope.
    """
    memo = _CONFIG_DISCOVERY_MEMO.get()
    if memo is not None and start is None and _CWD_KEY in memo:
        # Resolve the key before `Path.cwd().resolve()`, which is itself
        # a syscall pair — the default `start=None` is what nearly every
        # reader passes, so it is the case worth short-circuiting.
        return memo[_CWD_KEY]
    base = (start if start is not None else Path.cwd()).resolve()
    if memo is not None and base in memo:
        return memo[base]
    home = _home_dir()
    located: Path | None = None
    current = base
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        if current == home:
            # Rule 1: stop *before* probing, so the per-user config is
            # unreachable rather than merely last.
            break
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            located = candidate
            break
        if (current / WORKTREE_MARKER).exists():
            # Rule 3. Probed only on a miss, so the common hit path
            # pays nothing for the bound; and the bound shortens far
            # more walks than it lengthens.
            break
        if current.parent == current:
            break
        current = current.parent
    if memo is not None:
        memo[base] = located
        if start is None:
            memo[_CWD_KEY] = located
    return located
