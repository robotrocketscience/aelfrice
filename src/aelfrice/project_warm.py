"""Preload a project's belief store on cwd-change events.

Wired to a Claude Code `CwdChanged` hook (out of scope for this module,
lives in HOME repo). The hook fires `aelf project-warm <path>`; this
module resolves the path to a project root, opens the project's
SQLite store, and touches the indexes a future `aelf search` will read.
The "warming" is real but modest: the SQLite page cache and the OS
file cache are populated, so the next process pays only the second-hit
cost. There is no shared in-process cache between CLI invocations —
that's a fact about `aelf`'s process model, not a missing feature.

A sentinel file at `~/.aelfrice/projects/<id>/.last_warm` debounces
repeat calls; subsequent invocations within `debounce_seconds` are
no-ops and return `WarmResult.DEBOUNCED`.

**Sentinel keying:** the sentinel `<id>` is derived from the repo's
`git-common-dir` (via `git rev-parse --git-common-dir`), not from the
worktree path. Two worktrees of the same repo share one git-common-dir
and therefore share one sentinel, matching the v1.1.0 design principle
that worktrees share a single belief store. `ProjectRef.root` remains
the worktree working directory so that `_warm_store` can `os.chdir` to
the right place for `db_path()`.

Resolution order for `resolve_project_root(path)`:

1. `git rev-parse --git-common-dir` from `path`. Two worktrees of the
   same repo share a git-common-dir, so both map to the same `id` and
   share one debounce sentinel. `ProjectRef.root` is still the
   worktree's working directory.
2. First ancestor of `path` containing a `.aelfrice/projects/<id>/`
   directory. Lets non-git workspaces (research notebooks, scratch
   dirs) opt in by hand-creating that layout.
3. None — the caller's contract is to silently no-op for unknown
   paths.

Deny-glob defaults catch the common false-positive cases:
`/tmp/**`, `/var/folders/**`, `~/Downloads/**`, `~/Desktop/**`.
Override via `~/.aelfrice/config.json`'s `project_warm.deny_globs`
key (list of strings; `~` expanded at load time).
"""
from __future__ import annotations

import enum
import fnmatch
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast


_AELFRICE_HOME_DIRNAME: Final[str] = ".aelfrice"
_PROJECTS_SUBDIR: Final[str] = "projects"
_SENTINEL_FILENAME: Final[str] = ".last_warm"
_CONFIG_FILENAME: Final[str] = "config.json"
_PROJECT_ID_LEN: Final[int] = 12

_GIT_TIMEOUT_SECONDS: Final[float] = 5.0

DEFAULT_DEBOUNCE_SECONDS: Final[int] = 60

# Path globs that should never be warmed even if a `.git/` happens to
# exist there. `~` is expanded against `Path.home()` at match time so
# the same defaults work across machines.
DEFAULT_DENY_GLOBS: Final[tuple[str, ...]] = (
    "/tmp/**",
    "/var/folders/**",
    "~/Downloads/**",
    "~/Desktop/**",
)


class WarmResult(str, enum.Enum):
    """Outcome of `warm_project`. String values stay readable in logs."""

    WARMED = "warmed"
    DEBOUNCED = "debounced"
    UNKNOWN_PROJECT = "unknown_project"
    DENIED_PATH = "denied_path"


@dataclass(frozen=True)
class ProjectRef:
    """Resolved project identity. `id` is path-derived and stable."""

    root: Path
    id: str


@dataclass(frozen=True)
class WarmConfig:
    """Resolved warming config. Built from `~/.aelfrice/config.json`."""

    deny_globs: tuple[str, ...] = field(default=DEFAULT_DENY_GLOBS)
    aelfrice_home: Path = field(default_factory=lambda: Path.home() / _AELFRICE_HOME_DIRNAME)


def _project_id(root: Path) -> str:
    """Stable short id for a project root.

    sha256 of the absolute root path, truncated. The full hash is too
    long to embed in shell paths and we don't need cryptographic
    collision resistance — we just want a directory-safe slug that
    survives path renames cleanly within one machine.
    """
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return digest[:_PROJECT_ID_LEN]


def _git_resolve(path: Path) -> tuple[Path, Path] | None:
    """Return (worktree_root, git_common_dir) for `path`, or None.

    `worktree_root` is `git rev-parse --show-toplevel` — the working
    directory for this worktree. `git_common_dir` is the shared git
    object store root (identical across all worktrees of one repo and
    matches what `cli._git_common_dir()` returns). Using git-common-dir
    as the sentinel key means all worktrees of one repo share a single
    debounce sentinel, matching the v1.1.0 design.
    """
    if not path.is_dir():
        return None
    try:
        result = subprocess.run(
            [
                "git", "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
            ],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    worktree_root = Path(lines[0]).resolve()
    common_dir = Path(lines[1]).resolve()
    return worktree_root, common_dir


def _ancestor_with_project_layout(
    path: Path, aelfrice_home: Path,
) -> Path | None:
    """Walk parents looking for a `.aelfrice/projects/<id>/` directory.

    Returns the first ancestor (including `path` itself) that has a
    sentinel directory under `~/.aelfrice/projects/<id>/`, where `<id>`
    is the project id of that ancestor. This is the opt-in path for
    non-git workspaces — the user creates the project dir once, then
    every subsequent cwd into a child path warms it.
    """
    projects_dir = aelfrice_home / _PROJECTS_SUBDIR
    if not projects_dir.is_dir():
        return None
    candidate: Path = path if path.is_dir() else path.parent
    candidate = candidate.resolve()
    seen: set[Path] = set()
    while candidate not in seen:
        seen.add(candidate)
        cid = _project_id(candidate)
        if (projects_dir / cid).is_dir():
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent
    return None


def resolve_project_root(
    path: Path, *, aelfrice_home: Path | None = None,
) -> ProjectRef | None:
    """Resolve a path to a known project root, or None.

    See module docstring for the resolution rules. `aelfrice_home`
    overrides the default `~/.aelfrice/` for tests.

    For git repos/worktrees: `ProjectRef.root` is the worktree working
    directory (used by `_warm_store` for `os.chdir`), while
    `ProjectRef.id` is keyed off the git-common-dir so all worktrees of
    one repo share the same sentinel.
    """
    home = aelfrice_home if aelfrice_home is not None else Path.home() / _AELFRICE_HOME_DIRNAME
    git_result = _git_resolve(path)
    if git_result is not None:
        worktree_root, common_dir = git_result
        return ProjectRef(root=worktree_root, id=_project_id(common_dir))
    root = _ancestor_with_project_layout(path, home)
    if root is None:
        return None
    return ProjectRef(root=root, id=_project_id(root))


def _expand_glob(pattern: str) -> str:
    """Expand a leading `~` against `Path.home()`.

    `fnmatch` does not handle `~` itself, so we resolve it before the
    match. Patterns without a leading `~` pass through unchanged.
    """
    if pattern.startswith("~"):
        return str(Path(pattern).expanduser())
    return pattern


def _is_denied(root: Path, deny_globs: tuple[str, ...]) -> bool:
    target = str(root)
    for raw in deny_globs:
        pattern = _expand_glob(raw)
        if fnmatch.fnmatch(target, pattern):
            return True
        # `**` in fnmatch only matches a single segment; treat `<prefix>/**`
        # as "prefix or any descendant".
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if target == prefix or target.startswith(prefix + "/"):
                return True
    return False


def load_config(
    *, aelfrice_home: Path | None = None,
) -> WarmConfig:
    """Read warming config from `<aelfrice_home>/config.json`.

    Schema (all keys optional):

        {
          "project_warm": {
            "deny_globs": ["/tmp/**", "~/scratch/**"]
          }
        }

    Missing file or invalid JSON falls back to `DEFAULT_DENY_GLOBS`.
    Unknown keys are ignored — config is forward-compatible by design.
    """
    home = aelfrice_home if aelfrice_home is not None else Path.home() / _AELFRICE_HOME_DIRNAME
    path = home / _CONFIG_FILENAME
    deny_globs: tuple[str, ...] = DEFAULT_DENY_GLOBS
    if path.is_file():
        raw_obj: Any
        try:
            raw_obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_obj = None
        if isinstance(raw_obj, dict):
            raw_dict = cast(dict[str, Any], raw_obj)
            section_obj: Any = raw_dict.get("project_warm")
            if isinstance(section_obj, dict):
                section_dict = cast(dict[str, Any], section_obj)
                globs_obj: Any = section_dict.get("deny_globs")
                if isinstance(globs_obj, list):
                    raw_list = cast(list[Any], globs_obj)
                    str_globs: list[str] = [
                        g for g in raw_list if isinstance(g, str)
                    ]
                    if len(str_globs) == len(raw_list):
                        deny_globs = tuple(str_globs)
    return WarmConfig(deny_globs=deny_globs, aelfrice_home=home)


def _sentinel_path(project_id: str, aelfrice_home: Path) -> Path:
    return aelfrice_home / _PROJECTS_SUBDIR / project_id / _SENTINEL_FILENAME


def _read_sentinel(path: Path) -> float | None:
    """Return the unix-ts in `path`, or None when missing/unparseable."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, FileNotFoundError):
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _write_sentinel(path: Path, ts: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{ts:.6f}\n", encoding="utf-8")


def _warm_store(project_root: Path) -> None:
    """Open the project DB and touch the indexes a search would read.

    Cheap: a `count_beliefs` plus `list_locked_beliefs` is enough to
    pull the relevant pages into the SQLite page cache and the OS file
    cache. We close the connection immediately — there is no
    in-process state to keep alive across CLI invocations.

    Imports are local to keep the module light when the warming side is
    never invoked (e.g., on `aelf --version`).
    """
    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    # `db_path()` reads cwd-derived state; project_root is the right cwd.
    import os  # noqa: PLC0415

    saved_cwd = os.getcwd()
    try:
        os.chdir(str(project_root))
        path = db_path()
    finally:
        os.chdir(saved_cwd)
    if not path.exists():
        # Nothing to warm yet — first onboard hasn't run. Touching the
        # parent dir is harmless but a no-op for performance, so skip.
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(path))
    try:
        _ = store.count_beliefs()
        _ = store.list_locked_beliefs()
    finally:
        store.close()


def warm_project(
    ref: ProjectRef,
    *,
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS,
    config: WarmConfig | None = None,
    now: float | None = None,
) -> WarmResult:
    """Warm `ref`'s project store. Idempotent + debounced.

    Returns:
        - `WarmResult.DENIED_PATH` when `ref.root` is on the deny list.
        - `WarmResult.DEBOUNCED` when the sentinel was last written
          fewer than `debounce_seconds` ago.
        - `WarmResult.WARMED` after a successful warm + sentinel
          rewrite.

    `config` defaults to `load_config()` (read from disk). `now`
    defaults to `time.time()` and is wired up so tests can drive the
    debounce window without sleeping.
    """
    cfg = config if config is not None else load_config()
    if _is_denied(ref.root, cfg.deny_globs):
        return WarmResult.DENIED_PATH
    sentinel = _sentinel_path(ref.id, cfg.aelfrice_home)
    current = now if now is not None else time.time()
    last = _read_sentinel(sentinel)
    if last is not None and (current - last) < debounce_seconds:
        return WarmResult.DEBOUNCED
    _warm_store(ref.root)
    _write_sentinel(sentinel, current)
    return WarmResult.WARMED


def warm_path(
    path: Path,
    *,
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS,
    config: WarmConfig | None = None,
    now: float | None = None,
) -> WarmResult:
    """Convenience wrapper: resolve + warm in one call.

    Returns `WarmResult.UNKNOWN_PROJECT` when no project root could be
    resolved from `path`. Otherwise delegates to `warm_project`.
    """
    cfg = config if config is not None else load_config()
    ref = resolve_project_root(path, aelfrice_home=cfg.aelfrice_home)
    if ref is None:
        return WarmResult.UNKNOWN_PROJECT
    return warm_project(
        ref,
        debounce_seconds=debounce_seconds,
        config=cfg,
        now=now,
    )
