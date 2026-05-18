"""Shared DB path resolution.

Extracted out of `aelfrice.cli` so feature modules
(`context_rebuilder`, `hook_tail`, `hook`, `hook_commit_ingest`,
`hook_search_tool`, `project_warm`, `mcp_server`, `telemetry`,
`transcript_logger`) can resolve the canonical DB path without
importing from `cli`. CLI was the historical home for these helpers
but is the project's top of stack — feature modules importing from it
closes 14+ module-import cycles flagged by CodeQL (#499 Cluster C).

`cli.py` re-exports the symbols here for backward compatibility with
tests and external callers that already use `aelfrice.cli.db_path`.

Imports here must stay leaf-side: `aelfrice.store` (which itself only
imports `models` + `ulid`) is the only intra-package dep.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

from aelfrice.store import MemoryStore

DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"


def _git_common_dir() -> Path | None:
    """Absolute path of cwd's git-common-dir, or None when not in a repo.

    Two worktrees of one repo share a --git-common-dir, so resolving
    against this gives them a single shared DB. Returns None when cwd
    is outside any git work-tree, when the `git` binary is missing, or
    when the rev-parse call fails for any reason — callers fall back to
    the home-dir path.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    return Path(raw).resolve()


def db_path() -> Path:
    """Resolve the DB path.

    Resolution order:
    1. $AELFRICE_DB (explicit override; honoured even inside a git repo).
    2. <git-common-dir>/aelfrice/memory.db when cwd is in a git work-tree.
    3. ~/.aelfrice/memory.db (legacy global fallback for non-git dirs).

    The DB stays under .git/, which git does not track — the brain
    graph never crosses the git boundary.
    """
    override = os.environ.get("AELFRICE_DB")
    if override:
        return Path(override)
    git_dir = _git_common_dir()
    if git_dir is not None:
        return git_dir / "aelfrice" / DEFAULT_DB_FILENAME
    return DEFAULT_DB_DIR / DEFAULT_DB_FILENAME


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(str(p))


# v3.2 #858 active project context resolver.
PROJECT_CONTEXT_ENV: Final[str] = "AELFRICE_PROJECT_CONTEXT"
"""Env var name read by `active_project_context()`. Stable public name;
callers may set this per-shell to scope retrieval to a named within-repo
context. Empty / unset means "cross-context — no retrieval filter
applied", which is the pre-#858 default behaviour."""


def active_project_context() -> str:
    """Resolve the active within-repo project-context tag.

    Returns the value of `$AELFRICE_PROJECT_CONTEXT` after stripping
    surrounding whitespace; empty string when unset or whitespace-only.

    The empty-string return value is the "no filter" marker: callers
    (today, the UserPromptSubmit hook) treat it as "show every belief,
    regardless of its stored project_context". A non-empty value tells
    the filter to drop project-scope beliefs whose stored
    project_context is neither '' nor an exact match.

    Distinct from `db_path()` (which picks WHICH DB to read). Two
    worktrees of the same repo share one DB via --git-common-dir; this
    resolver is what lets those two worktrees see DIFFERENT slices of
    the shared DB based on the active context.

    Empty-only-on-unset semantics deliberately omit a `.aelfrice/context`
    state-file fallback at this commit. State-file discovery is a
    follow-up surface (an `aelf context set <name>` CLI subcommand
    writes the file; the resolver consults env var first, file second).
    Env var alone is sufficient for interactive agent sessions and
    CI pipelines that set context at startup.
    """
    raw = os.environ.get(PROJECT_CONTEXT_ENV, "")
    return raw.strip()
