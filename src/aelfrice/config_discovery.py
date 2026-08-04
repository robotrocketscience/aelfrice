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
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final

__all__ = [
    "CONFIG_FILENAME",
    "config_discovery_scope",
    "discover_config",
]

# The project-config filename every reader walks up looking for.
CONFIG_FILENAME: Final[str] = ".aelfrice.toml"

# A ContextVar rather than a plain dict so concurrent operations in
# threads or async tasks cannot see each other's memo. A thread started
# inside a scope begins with a fresh context, so it walks — correct, if
# slightly wasteful.
_CONFIG_DISCOVERY_MEMO: ContextVar[dict[Path, Path | None] | None] = ContextVar(
    "aelfrice_config_discovery_memo",
    default=None,
)

# Memo key standing for "the caller passed no `start`", i.e. resolve
# from cwd. Not a real path, and cannot collide with one: every other
# key is an absolute resolved directory.
_CWD_KEY: Final[Path] = Path("\x00cwd")


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
    """Return the nearest `.aelfrice.toml` at or above `start`, else None.

    `start=None` means "from the current working directory". Inside a
    `config_discovery_scope` the result is memoized per resolved start
    directory, so N readers cost one walk instead of N.

    Distinct `start` directories are distinct memo keys and each costs
    its own walk. That is not a defect: a caller that deliberately
    resolves config from a *different* directory (the hook resolving the
    agent's payload cwd rather than the hook process's incidental cwd)
    is asking a different question and must get the answer to it.
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
    located: Path | None = None
    current = base
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            located = candidate
            break
        if current.parent == current:
            break
        current = current.parent
    if memo is not None:
        memo[base] = located
        if start is None:
            memo[_CWD_KEY] = located
    return located
