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
`models`, `meta_beliefs`, and `ulid`) and the stdlib-only
`aelfrice.config_discovery` are the only intra-package deps.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final, cast

from aelfrice.config_discovery import WORKTREE_MARKER, discover_config
from aelfrice.store import MemoryStore

# #970 repo-store on-disk layout: db_path() places the store at
# <git-common-dir>/aelfrice/memory.db, so the parent dir name is this.
_REPO_STORE_PARENT_DIRNAME: Final[str] = "aelfrice"

DEFAULT_DB_DIR: Final[Path] = Path.home() / ".aelfrice"
DEFAULT_DB_FILENAME: Final[str] = "memory.db"


# #1734: one prompt-hook fire asked for the git-common-dir 5 times, and
# each `git rev-parse` fork cost about 44 ms. The key is the cwd plus the
# environment variables that choose which repository git discovers. A
# process that changes its cwd (`project_warm` does) gets a fresh lookup.
# Other inputs, such as `PATH` or git's config variables, are left out:
# aelfrice never changes them mid-process.
_GIT_DISCOVERY_ENV: Final[tuple[str, ...]] = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
)
_git_common_dir_cache: dict[tuple[str | None, ...], Path | None] = {}


def clear_git_common_dir_cache() -> None:
    """Forget every cached git-common-dir lookup.

    For callers that change what git would discover without changing the
    cwd or the environment, such as `git init` in a directory that was
    already looked up. The test suite calls it before each test.
    """
    _git_common_dir_cache.clear()


def _git_common_dir(start: Path | None = None) -> Path | None:
    """Absolute path of cwd's git-common-dir, or None when not in a repo.

    `start` asks about another directory instead of the cwd (#1796). The
    cache key holds the directory asked about, so an answer for `start`
    is never reused for the cwd or the other way round.

    Cached per process, keyed on the cwd and git's discovery environment
    variables (#1734). A `None` result is cached too, so the fallback is
    the same on every call, with one exception: a timed-out lookup is not
    cached, because a loaded machine can answer the next call. When the
    cwd itself can't be read, nothing is cached and the lookup runs as
    before.
    """
    try:
        where = os.getcwd() if start is None else os.path.abspath(start)
        key = (where, *(os.environ.get(k) for k in _GIT_DISCOVERY_ENV))
    except (OSError, ValueError):
        key = None
    if key is not None and key in _git_common_dir_cache:
        return _git_common_dir_cache[key]
    try:
        # The cwd lookup takes no argument, so a test that replaces it
        # with a zero-argument stub keeps working.
        found = (
            _lookup_git_common_dir()
            if start is None
            else _lookup_git_common_dir(start)
        )
    except subprocess.TimeoutExpired:
        return None
    if key is not None:
        _git_common_dir_cache[key] = found
    return found


def _lookup_git_common_dir(start: Path | None = None) -> Path | None:
    """Run `git rev-parse --git-common-dir` for the cwd; None on failure.

    `start`, when given, is the directory git runs in instead of the cwd.
    A `start` that does not exist is an `OSError` and returns None.

    A timeout raises `subprocess.TimeoutExpired` instead, so the cache in
    `_git_common_dir()` can tell a transient failure from an answer.

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
            cwd=start,
        )
    except (
        FileNotFoundError,
        OSError,
        # `UnicodeDecodeError` is the strict decode above. Its parent,
        # `ValueError`, is also what a NUL byte in `start` raises.
        ValueError,
    ):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    return Path(raw).resolve()


def main_checkout_path(path: Path) -> Path:
    """``path`` with a linked worktree's root swapped for its main checkout.

    The host keeps a worktree session's memory under the main checkout's
    project directory: on one machine 17 worktree project directories
    existed and none had a ``memory/`` folder (#1657). So the memory for
    ``<main>/.claude/worktrees/N/sub`` lives under ``<main>/sub``'s
    encoding. A plain checkout, a submodule (whose ``.git`` file has no
    ``commondir``), and a path outside any repo come back unchanged.

    Reads the ``.git`` file and the worktree's ``commondir`` rather than
    running ``git``, so it costs a few ``stat`` calls. Never raises. The
    common dir is resolved, so a main checkout whose ``.git`` is itself a
    symlink maps to the symlink's target project.
    """
    try:
        for root in (path, *path.parents):
            dot_git = root / ".git"
            if dot_git.is_dir():
                return path
            if not dot_git.is_file():
                continue
            text = dot_git.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return path
            gitdir = (root / text[len("gitdir:"):].strip()).resolve()
            commondir_file = gitdir / "commondir"
            if not commondir_file.is_file():
                return path
            common = (gitdir / commondir_file.read_text(encoding="utf-8").strip()).resolve()
            if common.name != ".git":
                return path
            return common.parent / path.relative_to(root)
    except (OSError, ValueError):
        return path
    return path


#: The project-config table that selects a store for a folder (#1796).
STORE_MARKER_SECTION: Final[str] = "store"

#: The key in that table naming the store.
STORE_MARKER_KEY: Final[str] = "path"

#: Prefix of a `[store] path` that names a store in the git common dir.
GIT_STORE_PREFIX: Final[str] = "git:"

#: Directory under `<git-common-dir>/aelfrice/` holding the named stores.
GIT_STORES_DIRNAME: Final[str] = "stores"

#: Suffix a `[store] path` may not end in. Every marker names the
#: directory that holds `memory.db`, so each store owns its folder: the
#: identity file, the turn log, and the other files kept beside a store
#: are keyed on that folder, and two store files in one folder would
#: share them.
STORE_FILE_SUFFIX: Final[str] = ".db"

# A store name after `git:`. A letter or digit first, so `.`, `..` and
# hidden names are out; no separator, so the name stays one directory
# under `stores/`; nothing a Windows path refuses.
_GIT_STORE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# A `store` table header in any spelling: `[store]`, `[store.x]`, an
# array header, or a quoted `["store"]`. Used only when the file fails to
# parse, to tell a broken marker from a broken file that never held one.
# A leading byte order mark is skipped.
_STORE_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"""^\ufeff?\s*\[\[?\s*(["']?)store\1\s*[\].]""", re.MULTILINE,
)

# A top-level `store` key: a dotted `store.path = ...` or an inline
# `store = {...}`, bare or quoted. Only lines before the first table
# header count, since the same text under another table is that table's
# key and declares no store table.
_STORE_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"""^\ufeff?\s*(["']?)store\1\s*[.=]""",
)

# A whole line that is a table header, which ends the top level of a TOML
# file: one or more bare or quoted keys joined by dots, in `[...]` or
# `[[...]]`, then an optional comment. A line that only starts with `[`,
# such as a row of a multi-line array or a line inside a string, does not
# end it.
_KEY_PART: Final[str] = r"""(?:"[^"]*"|'[^']*'|[A-Za-z0-9_-]+)"""
_ANY_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    rf"^\ufeff?\s*\[\[?\s*{_KEY_PART}(?:\s*\.\s*{_KEY_PART})*\s*\]\]?\s*(?:#.*)?$",
)

_BOM: Final[str] = "\ufeff"

#: The rules `resolve_store()` reports, in resolution order.
STORE_RULE_ENV: Final[str] = "env"
STORE_RULE_MARKER: Final[str] = "marker"
STORE_RULE_GIT: Final[str] = "git"
STORE_RULE_GLOBAL: Final[str] = "global"

#: Most project-config files `_find_store_marker` reads in one walk.
#: A real tree holds a handful; the bound only stops a pathological one.
_MARKER_WALK_LIMIT: Final[int] = 64


class StoreMarkerError(ValueError):
    """A `[store]` marker that names no usable store (#1796).

    Raised instead of falling back. A folder whose marker can't be read
    would otherwise write to the repository's shared store, which is the
    store the marker exists to keep it out of.
    """


@dataclass(frozen=True)
class StoreSelection:
    """Which store applies, and the rule that chose it (#1796).

    `marker` is the project-config file holding the `[store]` table, set
    only when `rule` is `STORE_RULE_MARKER`. `git_store` is the name
    after `git:` when the marker names a shared store, and `owned_store`
    is True when the marker names the repository store or the global
    store; both decide the identity the store stamps (`store_identity`).
    """

    path: Path
    rule: str
    marker: Path | None = None
    git_store: str | None = None
    owned_store: bool = False

    def describe(self) -> str:
        """The rule for `aelf status` and `aelf doctor`, in one phrase."""
        if self.rule == STORE_RULE_ENV:
            return f"{STORE_RULE_ENV} (AELFRICE_DB)"
        if self.marker is not None:
            return f"{self.rule} ({self.marker})"
        return self.rule


def _declares_store_table(raw: str) -> bool:
    """True when the text of an unparsable config declares a `store`
    table: a header anywhere, or a top-level dotted or inline key.

    The file can't be parsed, so this reads lines, and it errs toward
    the error: a line inside a string that looks like a `store` table
    counts as one. Any line ending (LF, CRLF, or CR) is accepted.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if _STORE_HEADER_RE.search(raw):
        return True
    for line in raw.split("\n"):
        if _ANY_HEADER_RE.match(line):
            return False
        if _STORE_KEY_RE.match(line):
            return True
    return False


def _read_store_marker(config: Path) -> str | None:
    """The `[store] path` value in `config`, or None when it has no table.

    Raises `StoreMarkerError` for a `[store]` table with no usable
    `path`, and for a file that fails to parse but has a `[store]`
    header. A file that fails to parse and has no such header held no
    marker, so it is None, as it was before markers existed.
    """
    import tomllib

    try:
        raw = config.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    try:
        parsed: dict[str, Any] = tomllib.loads(raw)
    except (tomllib.TOMLDecodeError, RecursionError) as exc:
        if _declares_store_table(raw):
            hint = (
                "; the file starts with a UTF-8 byte order mark, which "
                "TOML does not allow, so save it without one"
                if raw.startswith(_BOM) else ""
            )
            raise StoreMarkerError(
                f"{config}: cannot parse the [{STORE_MARKER_SECTION}] "
                f"marker: {exc}{hint}",
            ) from exc
        return None
    if STORE_MARKER_SECTION not in parsed:
        return None
    section: Any = parsed[STORE_MARKER_SECTION]
    if not isinstance(section, dict):
        raise StoreMarkerError(
            f"{config}: [{STORE_MARKER_SECTION}] must be a table",
        )
    value: Any = cast("dict[str, Any]", section).get(STORE_MARKER_KEY)
    if not isinstance(value, str) or not value.strip():
        raise StoreMarkerError(
            f"{config}: [{STORE_MARKER_SECTION}] needs a non-empty "
            f"{STORE_MARKER_KEY} string",
        )
    if "\x00" in value:
        raise StoreMarkerError(
            f"{config}: [{STORE_MARKER_SECTION}] {STORE_MARKER_KEY} "
            "contains a NUL byte",
        )
    return value


def _marker_store_path(
    config: Path, value: str, start: Path | None,
) -> Path:
    """The store file a `[store] path` value names.

    `git:<name>` is `<git-common-dir>/aelfrice/stores/<name>/memory.db`,
    one store for every worktree. Any other value is a filesystem path:
    absolute, `~`-prefixed, or relative to the folder holding `config`,
    so a relative store is per worktree. The value names the directory
    that holds `memory.db`; one ending in `.db` is an error, so no two
    stores share a folder. The rule reads the text only, so the answer
    does not change once the directory exists.
    """
    if value.startswith(GIT_STORE_PREFIX):
        name = value[len(GIT_STORE_PREFIX):]
        if not _GIT_STORE_NAME_RE.fullmatch(name):
            raise StoreMarkerError(
                f"{config}: [{STORE_MARKER_SECTION}] {STORE_MARKER_KEY} = "
                f"{value!r}: the name after {GIT_STORE_PREFIX!r} must start "
                "with a letter or digit and hold only letters, digits, "
                "'.', '_' and '-'",
            )
        git_dir = _git_common_dir() if start is None else _git_common_dir(start)
        if git_dir is None:
            raise StoreMarkerError(
                f"{config}: [{STORE_MARKER_SECTION}] {STORE_MARKER_KEY} = "
                f"{value!r} needs a git repository, and this folder is "
                "not in one",
            )
        return (
            git_dir / _REPO_STORE_PARENT_DIRNAME / GIT_STORES_DIRNAME
            / name / DEFAULT_DB_FILENAME
        )
    try:
        target = Path(value).expanduser()
    except RuntimeError as exc:
        # `~name` for a user that doesn't exist, or no home directory.
        raise StoreMarkerError(
            f"{config}: [{STORE_MARKER_SECTION}] {STORE_MARKER_KEY} = "
            f"{value!r}: cannot expand the home directory: {exc}",
        ) from exc
    if target.name.endswith(STORE_FILE_SUFFIX):
        raise StoreMarkerError(
            f"{config}: [{STORE_MARKER_SECTION}] {STORE_MARKER_KEY} = "
            f"{value!r} names a file; name the directory that holds "
            f"{DEFAULT_DB_FILENAME} instead",
        )
    if not target.is_absolute():
        target = config.parent / target
    return target / DEFAULT_DB_FILENAME


def _in_work_tree(folder: Path) -> bool:
    """True when `folder` or an ancestor up to the home directory is a
    git work-tree root.

    The home directory itself counts, so a repository rooted at `$HOME`
    is a work tree; nothing above it is examined. Only directory entries
    are probed, never a config file, so this is no second discovery walk
    (#1582).
    """
    try:
        home: Path | None = Path.home().resolve()
    except (RuntimeError, OSError):
        home = None
    current = folder
    for _ in range(len(folder.parts)):
        if (current / WORKTREE_MARKER).exists():
            return True
        if current == home:
            return False
        if current.parent == current:
            return False
        current = current.parent
    return False


def _find_store_marker(start: Path | None) -> tuple[Path, str] | None:
    """The nearest project-config file holding a `[store]` table, and its value.

    A nearer file without the table, or one that can't be read or
    parsed, is passed over, so an unrelated config in a subfolder can't
    send a marked folder's beliefs to the shared store. Each step is a
    fresh `discover_config()` call from the folder above the last file,
    so the walk keeps that function's bounds: it never examines the home
    directory, and it stops at the git work-tree root. Outside a work
    tree there is no root to stop at, so only the nearest file counts.
    """
    try:
        config = discover_config(start)
    except (OSError, ValueError):
        return None
    for _ in range(_MARKER_WALK_LIMIT):
        if config is None:
            return None
        value = _read_store_marker(config)
        if value is not None:
            return config, value
        folder = config.parent
        if (folder / WORKTREE_MARKER).exists() or not _in_work_tree(folder):
            return None
        try:
            config = discover_config(folder.parent)
        except (OSError, ValueError):
            return None
    return None


def _same_dir(a: Path, b: Path) -> bool:
    """True when `a` and `b` are one directory.

    Compared by file identity when both exist, so a symlink or a
    different letter case on a case-insensitive filesystem still
    matches; by resolved path otherwise.
    """
    try:
        if a.is_dir() and b.is_dir():
            return os.path.samefile(a, b)
        return a.resolve() == b.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _same_file(a: Path, b: Path) -> bool:
    """True when store paths `a` and `b` name one store: the same file
    name in the same directory (`_same_dir`)."""
    return a.name == b.name and _same_dir(a.parent, b.parent)


def _is_git_dir(path: Path) -> bool:
    """True when `path` looks like a git directory: a `HEAD` file that
    holds a ref or an object id, and `objects` and `refs` directories.
    Cheap, and needs no git call."""
    try:
        if not ((path / "objects").is_dir() and (path / "refs").is_dir()):
            return False
        head = (path / "HEAD").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return False
    head = head.strip()
    return head.startswith("ref: ") or bool(re.fullmatch(r"[0-9a-f]{40,64}", head))


def _is_global_store(path: Path) -> bool:
    """True when `path` is the global store, `~/.aelfrice/memory.db`."""
    return _same_file(path, DEFAULT_DB_DIR / DEFAULT_DB_FILENAME)


def _names_owned_store(path: Path) -> bool:
    """True when a marker's store `path` is a store another rule owns:
    any repository's store, the global store, or the user lock store.

    Such a marker must keep that store's identity, or its first open
    would seed the store's sidecar with the marker folder's identity and
    relabel the rows every other opener stamps. A repository store is
    `<git-dir>/aelfrice/memory.db`, recognized by its git directory
    (`_is_git_dir`), so the check needs no git call and covers another
    repository's store too. The path is resolved first, and the
    directory compared by identity, so a symlink or a different letter
    case still counts.
    """
    if _is_global_store(path) or _same_file(path, user_db_path()):
        return True
    if path.name != DEFAULT_DB_FILENAME:
        return False
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    git_dir = resolved.parent.parent
    return _is_git_dir(git_dir) and _same_dir(
        resolved.parent, git_dir / _REPO_STORE_PARENT_DIRNAME,
    )


def resolve_store(start: Path | None = None) -> StoreSelection:
    """Resolve the store, and the rule that chose it.

    Resolution order:
    1. $AELFRICE_DB (explicit override; honoured even inside a git repo).
    2. The nearest project-config file holding a `[store]` table (#1796),
       found by `_find_store_marker()`, so never above the git
       work-tree root or in the home directory. A nearer file without
       the table is passed over, unlike the other tables, so a nested
       config can't send a marked folder's beliefs to the shared store.
    3. <git-common-dir>/aelfrice/memory.db when cwd is in a git work-tree.
    4. ~/.aelfrice/memory.db (legacy global fallback for non-git dirs).

    `start` resolves for that directory instead of the cwd, for a caller
    that knows the directory the agent is working in.

    Raises `StoreMarkerError` when the nearest marker names no usable
    store. A config file that can't be found or read is no marker.
    """
    return _resolve(start, start)


def resolve_store_in_work_tree(folder: Path, root: Path) -> StoreSelection:
    """`resolve_store(folder)` for a `folder` the caller knows is inside
    the git work tree rooted at `root`, and in no nested repository.

    Git is asked about `root` instead of `folder`, and that answer is
    cached, so resolving every folder a commit changed costs one git
    call rather than one per folder.
    """
    return _resolve(folder, root)


def owning_folder(root: Path, rel: str) -> Path:
    """The deepest folder of `rel`'s parents, walking down from `root`,
    that is a real directory of this work tree.

    The walk stops before a folder that is missing (a deleted file's
    folder), a symlink (whose target may be outside the repository or in
    another one), the root of a nested repository or submodule, or a
    folder it can't read. The store then resolves from a folder that
    belongs to `root`'s own work tree, so the result is safe to pass to
    `resolve_store_in_work_tree`.
    """
    current = root
    for part in Path(rel).parts[:-1]:
        candidate = current / part
        try:
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or (candidate / WORKTREE_MARKER).exists()
            ):
                break
        except OSError:
            # Python 3.12 and 3.13 raise for a folder without search
            # permission, where 3.14 returns False.
            break
        current = candidate
    return current


def _resolve(start: Path | None, git_start: Path | None) -> StoreSelection:
    """`resolve_store()`, with git lookups made from `git_start`."""
    override = os.environ.get("AELFRICE_DB")
    if override:
        return StoreSelection(Path(override), STORE_RULE_ENV)
    found = _find_store_marker(start)
    if found is not None:
        config, value = found
        path = _marker_store_path(config, value, git_start)
        if value.startswith(GIT_STORE_PREFIX):
            return StoreSelection(
                path, STORE_RULE_MARKER, config,
                git_store=value[len(GIT_STORE_PREFIX):],
            )
        return StoreSelection(
            path, STORE_RULE_MARKER, config,
            owned_store=_names_owned_store(path),
        )
    git_dir = (
        _git_common_dir() if git_start is None else _git_common_dir(git_start)
    )
    if git_dir is not None:
        return StoreSelection(
            git_dir / _REPO_STORE_PARENT_DIRNAME / DEFAULT_DB_FILENAME,
            STORE_RULE_GIT,
        )
    return StoreSelection(DEFAULT_DB_DIR / DEFAULT_DB_FILENAME, STORE_RULE_GLOBAL)


def repository_store_path(root: Path) -> Path | None:
    """The store the `git` rule names for the work tree at `root`.

    That is `<git-common-dir>/aelfrice/memory.db`, or None when `root` is
    in no git repository. Neither `AELFRICE_DB` nor a `[store]` marker is
    read, so this names the repository store even where either would
    choose another. `aelf store split` copies out of this store (#1796).
    """
    git_dir = _git_common_dir(root)
    if git_dir is None:
        return None
    return git_dir / _REPO_STORE_PARENT_DIRNAME / DEFAULT_DB_FILENAME


def same_store_file(a: Path, b: Path) -> bool:
    """True when store paths `a` and `b` name one store file.

    Two files that both exist are compared by file identity, so a store
    file that is a symlink to the other matches. Otherwise the names must
    match, and the directories are compared by identity when both exist,
    so a symlinked directory or a different letter case on a
    case-insensitive filesystem still matches.
    """
    if os.path.isfile(a) and os.path.isfile(b):
        try:
            return os.path.samefile(a, b)
        except OSError:
            pass
    return _same_file(a, b)


def db_path(start: Path | None = None) -> Path:
    """Resolve the DB path. `resolve_store()` holds the resolution order.

    The repository store stays under .git/, which git does not track,
    so the brain graph never crosses the git boundary. A `[store]`
    marker can place a store inside the work tree; keeping that out of
    git is the user's call.
    """
    return resolve_store(start).path


#: Environment variable that names the user-scope lock store (#1681).
USER_DB_ENV: Final[str] = "AELFRICE_USER_DB"

#: Subdirectory of `DEFAULT_DB_DIR` that holds the user-scope store.
USER_STORE_DIRNAME: Final[str] = "user"


def user_db_path() -> Path:
    """Resolve the path of the user-scope lock store (#1681).

    A user lock is a rule about how you work, not about one codebase, so
    it lives in one store that every repository and every worktree
    shares, rather than in the repository store `db_path()` names.

    Resolution order:
    1. $AELFRICE_USER_DB (explicit override).
    2. ~/.aelfrice/user/memory.db.

    `AELFRICE_DB` is deliberately not consulted. It names the repository
    store, and a pin set to scope one project must not also move the
    store every project shares. The store has the same schema as the
    repository store, so lock tiers, expiry, and ids work unchanged.

    Resolving the path creates nothing. Only a write
    (`open_user_store()`) creates the parent directory, so a read of a
    user store that was never written finds no file and leaves none.

    `DEFAULT_DB_DIR` is read at call time, not bound here, so a test can
    repoint it.
    """
    override = os.environ.get(USER_DB_ENV)
    if override:
        return Path(override)
    return DEFAULT_DB_DIR / USER_STORE_DIRNAME / DEFAULT_DB_FILENAME


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_user_store() -> MemoryStore:
    """Open the user-scope lock store for writing, creating it if needed.

    The only call that creates the store or its parent directory. The
    store carries no repository identity (`project_context_default` is
    empty), because its locks belong to no one repository.

    Every repository writes this one file, so writers from different
    repositories can run at the same time. The protection is the one a
    repository store already relies on when two worktrees share it:
    SQLite's own file locking, the WAL journal, `busy_timeout=5000`, and
    the `SQLITE_SCHEMA` retry on the open-time DDL (#1310).
    """
    p = user_db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(str(p), project_context_default="")


def store_file_present(path: Path | str) -> bool:
    """Whether a store file is at `path`.

    Only a path that is not there (`FileNotFoundError`, or a parent that
    is a file, `NotADirectoryError`) is missing. Any other `OSError`,
    such as a parent directory without search permission, is raised: the
    store may be there, so the caller reports it as unreadable rather
    than as absent. `Path.exists()` and `Path.is_file()` can't be used
    for this, because Python 3.14 returns False for those errors and
    3.12 and 3.13 raise them.
    """
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def open_user_store_if_present() -> MemoryStore | None:
    """Open the user-scope store when it exists, else return None.

    For commands that read the user store or change a lock already in
    it. It never creates the file or its directory: a missing store
    returns None before anything is opened. A path that can't be
    checked, such as one under a directory without search permission,
    raises its `OSError` (`store_file_present`), so the
    caller reports the store as unreadable rather than as absent.

    An existing store is opened the way `open_store_for_read()` opens
    the repository store: the writable open first, so expired locks are
    swept and migrations run, then `mode=ro` only when that open is
    refused for lack of write access.

    The existence check and the open are two steps. A store deleted
    between them is recreated empty by the writable open; nothing in
    aelfrice deletes the user store, so only an outside `rm` racing this
    call can reach that.
    """
    import sqlite3

    from aelfrice.store import is_readonly_open_failure

    p = user_db_path()
    if str(p) == ":memory:" or not store_file_present(p):
        return None
    try:
        return MemoryStore(str(p), project_context_default="")
    except sqlite3.DatabaseError as exc:
        if not is_readonly_open_failure(exc):
            raise
    return MemoryStore(str(p), project_context_default="", read_only=True)


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


def _identity_from_marker_dir(marker_dir: Path) -> str:
    """Identity for a store a `[store]` marker selects (#1796).

    The recipe of `_identity_from_git_common_dir`, applied to the folder
    holding the marker: its basename, then the 8-hex BLAKE2b digest of
    its resolved absolute path. Two marker folders in one repository get
    two identities, and the same folder gets the same one every time.
    """
    marker_dir = marker_dir.resolve()
    basename = marker_dir.name or "project"
    digest = hashlib.blake2b(
        str(marker_dir).encode("utf-8"), digest_size=4,
    ).hexdigest()
    return f"{basename}-{digest}"


def _identity_from_git_store(git_dir: Path, name: str) -> str:
    """Identity for a shared `git:<name>` store (#1796).

    Format: ``<repo-root-basename>-<name>-<8 hex>``, the digest taken over
    the resolved git-common-dir and the name. Every worktree of the repo
    computes the same value, whichever one opens the store first, and two
    named stores in one repository get two identities.
    """
    git_dir = git_dir.resolve()
    basename = git_dir.parent.name or "repo"
    digest = hashlib.blake2b(
        f"{git_dir}\x00{name}".encode("utf-8"), digest_size=4,
    ).hexdigest()
    return f"{basename}-{name}-{digest}"


#: Sidecar holding the durable repo identity, beside the repo store at
#: ``<git-common-dir>/aelfrice/``. It lives on the filesystem rather than
#: in `schema_meta` because the identity is needed to *construct* the
#: store (it is `project_context_default`), and reading it from inside the
#: store it parameterises is circular. `local_scope_id` is also not this
#: value by design — `store.py` records it as federation provenance
#: ("which DB a row came from"), a different lifecycle.
_IDENTITY_SIDECAR_NAME: Final[str] = "identity"

_IDENTITY_SIDECAR_HEADER: Final[str] = (
    "# aelfrice repo identity (#1415). First line is canonical;\n"
    "# later lines are equivalent spellings seen on other hosts.\n"
)


def _first_sidecar_entry(sidecar: Path) -> str | None:
    """First non-comment, non-blank line of `sidecar`, or None if it has none."""
    try:
        raw = sidecar.read_text(encoding="utf-8")
    except OSError:
        return None
    for ln in raw.splitlines():
        stripped = ln.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def _seed_identity_sidecar(store_dir: Path, sidecar: Path, path_derived: str) -> str:
    """Place a complete sidecar, and return the identity now in force.

    The value is written to a temporary file in the same directory and
    moved into place, so nothing that kills this process can leave a
    partial `identity` behind. Creating the file first and flushing the
    value at close could: a kill in between left a zero-byte sidecar,
    which every later open reads as "no entry" — permanently reinstating
    the #1415 split with no self-heal and no diagnostic.

    `os.link` is what keeps the move exclusive. A concurrent first open
    that already placed its answer wins, and this call returns *that*
    value rather than overwriting it. A sidecar that exists but holds no
    entry is degenerate — the zero-byte file above, or a comment-only one
    — and is replaced: that replacement is the self-heal. Filesystems
    with no hard-link support (FAT, some network shares) take the same
    replace path, losing the race guard but never the completeness.

    Raises OSError if the directory or the temporary file cannot be
    written; the caller degrades to the path-derived identity. The
    temporary file is unlinked on every path out of here, so a failed
    seed strands nothing in the store directory either.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(
        "w", encoding="utf-8", dir=store_dir, prefix=".identity.", delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle as fh:
            fh.write(f"{_IDENTITY_SIDECAR_HEADER}{path_derived}\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, sidecar)
            return path_derived
        except OSError:
            # The seed lost a race, or the filesystem has no hard links.
            # Both mean "someone else may already have named this repo",
            # and the answer to that is the read two lines down, not an
            # error: whatever is on disk wins, and if nothing is, the
            # `os.replace` below seeds it anyway. Raising here would make
            # a second concurrent open fail rather than agree.
            pass
        existing = _first_sidecar_entry(sidecar)
        if existing is not None:
            return existing
        os.replace(tmp, sidecar)
        return path_derived
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


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
            # A last line with no terminator — what a torn write of the
            # seed leaves — must not be appended to in place: the splice
            # would read back as one token, and as the canonical identity.
            terminator = "" if raw.endswith("\n") else "\n"
            try:
                with sidecar.open("a", encoding="utf-8") as fh:
                    fh.write(f"{terminator}{path_derived}\n")
            except OSError:
                # The alias line is a record of which spellings have been
                # seen, read by nobody but a human debugging an identity
                # mismatch. `canonical` is already resolved and returned
                # below, so failing to append costs a diagnostic and
                # nothing else — a read-only checkout must not lose its
                # identity over it.
                pass
        return canonical

    if not create:
        return path_derived

    try:
        return _seed_identity_sidecar(store_dir, sidecar, path_derived)
    except OSError:
        return path_derived


def repo_identity_from_db_path(p: Path, *, create: bool = False) -> str:
    """Repo identity for a store at `p`, derived from its on-disk layout.

    The repo store lives at ``<git-common-dir>/aelfrice/memory.db``
    (`db_path()`), so the git-common-dir is `p.parent.parent` when the
    parent dir is the `aelfrice` subdir. Returns '' for the home-dir
    fallback (`~/.aelfrice/memory.db`), an in-memory DB, or any path that
    does not match the repo-store layout and has no sidecar (below) —
    those stores carry no repo identity, so their rows stay cross-context. Reuses the already-resolved
    path instead of re-forking `git`, so it adds no subprocess cost to the
    store-open hot path.

    Consults the `identity` sidecar beside the store, which is what makes
    one repository's identity survive two host spellings of its path
    (#1415). That read is non-mutating by default: `create=True` — which
    also seeds the sidecar and records this host's spelling — is for a
    caller opening its own store, not for one naming somebody else's, and
    `migrate` names the legacy source it opens read-only.

    A store outside that layout has an identity only when an `identity`
    sidecar sits beside it, which is what a `[store]` marker writes on
    first open (#1796). The sidecar travels with the store directory, so
    a copied store keeps its identity wherever it is opened from. This
    branch only reads: there is no path-derived value to seed.
    """
    if str(p) == ":memory:":
        return ""
    if p.parent.name != _REPO_STORE_PARENT_DIRNAME:
        # The global store never carries an identity, even if a sidecar
        # was written beside it.
        if _is_global_store(p):
            return ""
        return _first_sidecar_entry(p.parent / _IDENTITY_SIDECAR_NAME) or ""
    return _durable_identity(
        p.parent, _identity_from_git_common_dir(p.parent.parent), create=create,
    )


def store_identity(selection: StoreSelection, *, create: bool = False) -> str:
    """Repo identity for the store `selection` names.

    A marker-selected store keeps its identity in the `identity` sidecar
    beside the store; a sidecar already there wins, so a copied store
    keeps the identity it was created with (#1796). The seed depends on
    what the marker names:

    - a `git:<name>` store: the git common dir and the name
      (`_identity_from_git_store`), the same from every worktree;
    - the repository store or the global store: that store's own
      identity, so the marker can't relabel the rows every other opener
      stamps;
    - any other store: the marker folder (`_identity_from_marker_dir`).

    Every other store goes through `repo_identity_from_db_path`.
    `create` has the meaning it has there.
    """
    marker = selection.marker
    path = selection.path
    if (
        selection.rule != STORE_RULE_MARKER
        or marker is None
        or str(path) == ":memory:"
    ):
        return repo_identity_from_db_path(path, create=create)
    if selection.owned_store:
        resolved = path.resolve()
        if _is_git_dir(resolved.parent.parent):
            # The repository store under its canonical spelling, so a
            # differently-cased path still derives the repository identity.
            resolved = (
                resolved.parent.parent / _REPO_STORE_PARENT_DIRNAME
                / DEFAULT_DB_FILENAME
            )
        return repo_identity_from_db_path(resolved, create=create)
    if selection.git_store is not None:
        # <git-common-dir>/aelfrice/stores/<name>/memory.db
        git_dir = path.parent.parent.parent.parent
        seed = _identity_from_git_store(git_dir, selection.git_store)
    else:
        seed = _identity_from_marker_dir(marker.parent)
    return _durable_identity(path.parent, seed, create=create)


def repo_identity() -> str:
    """Stable repo identity for the cwd's git repo, or '' outside one.

    Reuses the git-common-dir `db_path()` keys on, so two worktrees of one
    repo share an identity. This is the value a user exports as
    ``AELFRICE_PROJECT_CONTEXT`` to activate project-context retrieval
    scoping for the current repo (the column is populated and migrate-safe
    regardless; the resolver default stays env-driven per #970). Forks
    `git` once on a cache miss in `_git_common_dir()` (#1734); prefer
    `repo_identity_from_db_path()` when a resolved DB path is already in
    hand.

    When a `[store]` marker selects the store, this is that store's
    identity (`store_identity`) instead, so the exported value matches
    the rows the marker store stamps (#1796).
    """
    selection = resolve_store()
    if selection.rule == STORE_RULE_MARKER:
        return store_identity(selection, create=True)
    git_dir = _git_common_dir()
    if git_dir is None:
        return ""
    return _durable_identity(
        git_dir / _REPO_STORE_PARENT_DIRNAME,
        _identity_from_git_common_dir(git_dir),
        create=True,
    )


def _open_store() -> MemoryStore:
    selection = resolve_store()
    p = selection.path
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(
        str(p),
        # This process is opening its own store, so it is the caller
        # entitled to seed the identity sidecar (#1415).
        project_context_default=store_identity(selection, create=True),
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

    #1571 amends what #1416's matrix may assert about this. That matrix
    asked that a routed read command perform no preliminary writable
    open *and* emit no failed-write warning. **Neither half survives
    contact with the writable-first order**, and the order is not
    moving: see the paragraph above.

    The first half is refuted unconditionally. This function *always*
    attempts `MemoryStore(...)` without `read_only=True` before it can
    know whether the fallback is needed — the preliminary writable open
    is the policy, not an accident of it, so no routed command can
    satisfy that half on any store.

    The second half is refuted on one shape. On a store whose directory
    is frozen but whose WAL sidecars are live *and whose schema needs no
    write to open*, the writable attempt **succeeds**, the `mode=ro`
    handle is never taken, `aelf search` goes on to attempt its `.bm25f`
    sidecar write, and that write fails with a non-fatal warning. The
    schema qualifier is load-bearing and was missed on the first pass:
    an outdated schema makes the open-time DDL fire and fail, which
    *does* reach the fallback. `tests/test_readonly_store_1416.py`
    constructs exactly that state by dropping `exploration_events` while
    holding the sidecars live, so the two shapes are both real and are
    distinguished by whether initialisation needs to write.

    So the criterion is narrowed to what it can truthfully assert: **the
    `mode=ro` fallback handle emits no failed-write warning.** The
    preliminary writable attempt is retained and expected. Where that
    attempt succeeds, a sidecar write is the designed behaviour and the
    warning is accurate — a write genuinely was tried and genuinely
    failed.

    Making that write conditional on directory writability was
    considered and not done: it would silence an accurate message at the
    cost of a writability probe on a hot path, and nobody has measured
    what that costs.
    """
    import sqlite3

    from aelfrice.store import is_readonly_open_failure

    selection = resolve_store()
    p = selection.path
    # Deliberately no `create=True`: this function exists for the case
    # where `.git/` is not writable, and an observational command has no
    # business seeding an identity sidecar there (#1415).
    ident = store_identity(selection)
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
