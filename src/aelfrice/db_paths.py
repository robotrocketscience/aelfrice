"""Shared DB path resolution.

Extracted out of `aelfrice.cli` so feature modules
(`context_rebuilder`, `hook_tail`, `hook`, `hook_commit_ingest`,
`hook_search_tool`, `project_warm`, `telemetry`,
`transcript_logger`) can resolve the canonical DB path without
importing from `cli`. CLI was the historical home for these helpers
but is the project's top of stack — feature modules importing from it
closes 14+ module-import cycles flagged by CodeQL (#499 Cluster C).

`cli.py` re-exports the symbols here for backward compatibility with
tests and external callers that already use `aelfrice.cli.db_path`.

Imports here must stay leaf-side: `aelfrice.store` (which imports
`models`, `meta_beliefs`, and `ulid`) is the only intra-package dep.
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
    is outside any git work-tree, when the `git` binary is missing,
    when the rev-parse call fails for any reason, or when its output is
    not valid UTF-8 — callers fall back to the home-dir path.

    `errors="strict"` here, unlike the `replace` used at the other git
    boundaries (#1441). Those degrade to a path that does not exist and
    the caller already handles that; this one is *built on*. `db_path()`
    appends to it and `_open_store()` runs `mkdir(parents=True)`, so a
    mojibake substitution would silently create a real second store
    under a garbled directory instead of falling back. A decode failure
    is a fallback, not a rename.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=5,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
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


#: Sidecar holding the durable repo identity, beside the repo store at
#: ``<git-common-dir>/aelfrice/``. It lives on the filesystem rather than
#: in `schema_meta` because the identity is needed to *construct* the
#: store (it is `project_context_default`), and reading it from inside the
#: store it parameterises is circular. `local_scope_id` is also not this
#: value by design — `store.py` records it as federation provenance
#: ("which DB a row came from"), a different lifecycle.
_IDENTITY_SIDECAR_NAME: Final[str] = "identity"


def _durable_identity(
    store_dir: Path, path_derived: str, *, create: bool = False,
) -> str:
    """Resolve the repo identity that both hosts agree on (#1415).

    The identity used to be a digest of the absolute git-common-dir. Native
    Windows and WSL spell one physical repository differently
    (``C:\\repo\\.git`` against ``/mnt/c/repo/.git``), so they minted two
    identities for the same store while reading and writing the same
    ``memory.db`` — splitting one repo's provenance in two.

    The first host to reach this writes its answer to a sidecar in the
    directory the store already shares, and every later host reads it. That
    seed is deliberately `path_derived`, **not** a fresh UUID: on the host
    that creates the file the identity is byte-identical to what it was
    before, so rows already stamped with it stay reachable and no migration
    is needed. The other host's spelling is appended as an alias, which is
    what makes the split *recorded* rather than merely resolved.

    Fails soft to `path_derived` on any I/O error — a read-only checkout or
    an unwritable ``.git`` must not break store open, and degrading to the
    old behaviour is exactly as correct as it was before.

    `create=False` (the default) makes this a pure read: no sidecar is
    seeded and no alias is appended. Only a caller that is opening *its
    own* store passes `create=True`. `migrate` resolves the identity of a
    legacy source it deliberately opens ``mode=ro``, and a dry run must
    leave that repository byte-identical; a resolver that mkdir'd and wrote
    into whatever path it was handed would write into another repo's
    ``.git`` just for being named on a `--from`.
    """
    sidecar = store_dir / _IDENTITY_SIDECAR_NAME
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except OSError:
        raw = ""

    lines = [ln.strip() for ln in raw.splitlines()]
    entries = [ln for ln in lines if ln and not ln.startswith("#")]

    if entries:
        canonical, aliases = entries[0], entries[1:]
        if create and path_derived != canonical and path_derived not in aliases:
            # Record this host's spelling. Once per host, not per open.
            try:
                with sidecar.open("a", encoding="utf-8") as fh:
                    fh.write(f"{path_derived}\n")
            except OSError:
                pass
        return canonical

    if not create:
        return path_derived

    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive create: two hosts racing the first open must not both
        # believe they set the canonical value. The loser re-reads below.
        with sidecar.open("x", encoding="utf-8") as fh:
            fh.write(
                "# aelfrice repo identity (#1415). First line is canonical;\n"
                "# later lines are equivalent spellings seen on other hosts.\n"
                f"{path_derived}\n"
            )
        return path_derived
    except FileExistsError:
        try:
            for ln in sidecar.read_text(encoding="utf-8").splitlines():
                stripped = ln.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
        except OSError:
            pass
    except OSError:
        pass
    return path_derived


def repo_identity_from_db_path(p: Path, *, create: bool = False) -> str:
    """Repo identity for a store at `p`, derived from its on-disk layout.

    The repo store lives at ``<git-common-dir>/aelfrice/memory.db``
    (`db_path()`), so the git-common-dir is `p.parent.parent` when the
    parent dir is the `aelfrice` subdir. Returns '' for the home-dir
    fallback (`~/.aelfrice/memory.db`), an in-memory DB, or any path that
    does not match the repo-store layout — those stores carry no repo
    identity, so their rows stay cross-context. Reuses the already-resolved
    path instead of re-forking `git`, so it adds no subprocess cost to the
    store-open hot path.

    Consults the `identity` sidecar beside the store, which is what makes
    one repository's identity survive two host spellings of its path
    (#1415). That read is non-mutating by default: `create=True` — which
    also seeds the sidecar and records this host's spelling — is for a
    caller opening its own store, not for one naming somebody else's, and
    `migrate` names the legacy source it opens read-only.
    """
    if str(p) == ":memory:":
        return ""
    if p.parent.name != _REPO_STORE_PARENT_DIRNAME:
        return ""
    return _durable_identity(
        p.parent, _identity_from_git_common_dir(p.parent.parent), create=create,
    )


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
    return _durable_identity(
        git_dir / _REPO_STORE_PARENT_DIRNAME,
        _identity_from_git_common_dir(git_dir),
        create=True,
    )


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(
        str(p),
        # This process is opening its own store, so it is the caller
        # entitled to seed the identity sidecar (#1415).
        project_context_default=repo_identity_from_db_path(p, create=True),
    )


def open_store_for_read() -> MemoryStore:
    """Open the store for a command whose contract is observational.

    #1416. `aelf search` against a readable-but-not-writable store died
    in `MemoryStore.__init__` with `sqlite3.OperationalError: attempt to
    write a readonly database` — a *store open is a write* here (DDL,
    migrations, the scope-id mint, and since #1314 the expired-lock
    sweep), so a read command never reached retrieval. That is the
    everyday shape of a Codex workspace-write session: the workspace is
    writable, `.git/` — where the repo store lives — is not.

    The writable open is attempted first, and **only** a permission
    failure falls back to `mode=ro`. Not because trying twice is elegant,
    but because the two handles are not interchangeable: the read-only
    one runs no migration, so a store written by an older binary is read
    at whatever shape it has, and no expired lock has been swept, so
    `aelf locked` lists windows the writable path would already have
    dropped. Preferring `mode=ro` unconditionally would impose that
    degraded semantics on every user whose store is perfectly writable.
    The failed attempt writes nothing — it failed *because* it could not.

    Raises `ReadOnlyStoreUnavailable` when the fallback cannot be opened
    either; callers turn that into a message rather than a traceback.
    """
    import sqlite3

    from aelfrice.store import is_readonly_open_failure

    p = db_path()
    # Deliberately no `create=True`: this function exists for the case
    # where `.git/` is not writable, and an observational command has no
    # business seeding an identity sidecar there (#1415).
    ident = repo_identity_from_db_path(p)
    if str(p) != ":memory:":
        try:
            _ensure_parent_dir(p)
        except OSError:
            # A non-writable parent-of-parent is itself the condition this
            # function exists for; let the open decide.
            pass
    try:
        return MemoryStore(str(p), project_context_default=ident)
    except sqlite3.DatabaseError as exc:
        if not is_readonly_open_failure(exc):
            raise
    return MemoryStore(str(p), project_context_default=ident, read_only=True)


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
