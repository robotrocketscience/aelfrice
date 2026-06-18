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

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Final

from aelfrice.store import MemoryStore

# #970 repo-store on-disk layout: db_path() places the store at
# <git-common-dir>/aelfrice/memory.db, so the parent dir name is this.
_REPO_STORE_PARENT_DIRNAME: Final[str] = "aelfrice"

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


def _identity_from_git_common_dir(git_dir: Path) -> str:
    """Build a stable repo-identity token from a git-common-dir (#970).

    Format: ``<repo-root-basename>-<8 hex>``. git-common-dir is
    ``<root>/.git`` (shared across a repo's worktrees), so its parent's
    name is the repo root basename — included for human legibility in
    `aelf` output and migrate provenance. The 8-hex BLAKE2b digest of the
    absolute git-common-dir disambiguates two same-named repos.

    `git_dir` is resolved to an absolute path before hashing so the same
    physical repo yields one identity regardless of how the path reached
    here. `repo_identity()` already passes a resolved path, but `migrate`
    can receive a relative `--from` legacy path; without this a relative
    source would digest a different string than the in-repo `repo_identity`
    and break cross-tool consistency.
    """
    git_dir = git_dir.resolve()
    root = git_dir.parent
    basename = root.name or "repo"
    digest = hashlib.blake2b(str(git_dir).encode("utf-8"), digest_size=4).hexdigest()
    return f"{basename}-{digest}"


def repo_identity_from_db_path(p: Path) -> str:
    """Repo identity for a store at `p`, derived from its on-disk layout.

    The repo store lives at ``<git-common-dir>/aelfrice/memory.db``
    (`db_path()`), so the git-common-dir is `p.parent.parent` when the
    parent dir is the `aelfrice` subdir. Returns '' for the home-dir
    fallback (`~/.aelfrice/memory.db`), an in-memory DB, or any path that
    does not match the repo-store layout — those stores carry no repo
    identity, so their rows stay cross-context. Reuses the already-resolved
    path instead of re-forking `git`, so it adds no subprocess cost to the
    store-open hot path.
    """
    if str(p) == ":memory:":
        return ""
    if p.parent.name != _REPO_STORE_PARENT_DIRNAME:
        return ""
    return _identity_from_git_common_dir(p.parent.parent)


def repo_identity() -> str:
    """Stable repo identity for the cwd's git repo, or '' outside one.

    Reuses the git-common-dir `db_path()` keys on, so two worktrees of one
    repo share an identity. This is the value a user exports as
    ``AELFRICE_PROJECT_CONTEXT`` to activate project-context retrieval
    scoping for the current repo (the column is populated and migrate-safe
    regardless; the resolver default stays env-driven per #970). Forks
    `git` once; prefer `repo_identity_from_db_path()` when a resolved DB
    path is already in hand.
    """
    git_dir = _git_common_dir()
    if git_dir is None:
        return ""
    return _identity_from_git_common_dir(git_dir)


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(str(p), project_context_default=repo_identity_from_db_path(p))


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

    Per ADR 0003 (#970) the stored project_context convention is repo
    identity (see `repo_identity`). Scoping is opt-in: this resolver
    stays env-driven, so the default (unset) is still "no filter". To
    activate per-repo scoping, export
    ``AELFRICE_PROJECT_CONTEXT="$(python -c 'from aelfrice.db_paths import
    repo_identity; print(repo_identity())')"`` (or set it to the repo
    identity by any means). The column is populated and migrate-safe
    regardless of whether the filter is active.

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
