"""Live-store checks shared by the census runner and `copy_store` (#1887).

A path counts as live when its real path (symlinks followed) is:

  * inside a `.git` folder, in any letter case;
  * under `~/.aelfrice`, in any letter case, for `$HOME` or the account home;
  * the folder of the store `$AELFRICE_DB` or `$AELFRICE_USER_DB` names, or
    a file directly in it;
  * a `[store]` marker store (#1796): a folder that a `.aelfrice.toml` in
    one of its ancestors, or in the working directory or one of its
    ancestors, names with `[store] path`.

Comparison is case-insensitive, because macOS and Windows file systems
are. A `.aelfrice.toml` that can't be read or parsed on the way up counts
as naming the folder, so an unreadable marker refuses rather than passes.
A marker elsewhere that names a store by absolute path is seen only when
the working directory is under it. Messages never carry a path.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

MARKER_FILENAME = ".aelfrice.toml"
GIT_STORE_PREFIX = "git:"
SQLITE_SIDECARS = ("", "-wal", "-shm")


def real(path: Path) -> Path:
    """The absolute real path, symlinks followed, missing tails allowed."""
    return Path(os.path.realpath(os.path.abspath(path)))


def _key(path: Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in real(path).parts)


def inside(path: Path, root: Path) -> bool:
    """True when path is root or below it, compared case-insensitively."""
    p, r = _key(path), _key(root)
    return p[:len(r)] == r


def account_homes() -> list[Path]:
    homes = [Path.home()]
    try:
        import pwd
        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, AttributeError):
        pass
    return homes


def _marker_target(config: Path) -> Path | None | bool:
    """The store folder a marker names, None for no marker, True if unreadable."""
    try:
        data: dict[str, Any] = tomllib.loads(config.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        return True
    section: Any = data.get("store")
    if not isinstance(section, dict):
        return None
    value: Any = section.get("path")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(value, str) or not value.strip():
        return True
    if value.startswith(GIT_STORE_PREFIX):
        return None  # under a .git folder: the .git rule covers it
    try:
        target = Path(value).expanduser()
    except RuntimeError:
        return True
    return target if target.is_absolute() else config.parent / target


def _marker_targets(start: Path) -> list[Path | bool]:
    out: list[Path | bool] = []
    cur = real(start)
    for d in (cur, *cur.parents):
        cfg = d / MARKER_FILENAME
        if cfg.is_file():
            t = _marker_target(cfg)
            if t is not None:
                out.append(t)
    return out


def live_reason(path: Path) -> str | None:
    """Why `path` is a live store location, or None when it is not."""
    rp = real(path)
    if ".git" in _key(rp):
        return "inside a .git folder"
    for home in account_homes():
        if inside(rp, home / ".aelfrice"):
            return "under ~/.aelfrice"
    for var in ("AELFRICE_DB", "AELFRICE_USER_DB"):
        val = os.environ.get(var)
        # The named store's folder and the files directly in it; not the
        # whole tree below, since the folder may be a home directory.
        if val:
            env_dir = _key(Path(val).parent)
            if _key(rp) == env_dir or _key(rp.parent) == env_dir:
                return f"the store {var} names"
    folder = rp if rp.is_dir() else rp.parent
    for t in _marker_targets(folder) + _marker_targets(Path.cwd()):
        if t is True:
            return "under a .aelfrice.toml that can't be read"
        if isinstance(t, Path) and inside(rp, t):
            return "a [store] marker store"
    return None


def store_live_reason(db_file: Path) -> str | None:
    """live_reason for a db file and each of its -wal and -shm files."""
    for suffix in SQLITE_SIDECARS:
        why = live_reason(db_file.with_name(db_file.name + suffix))
        if why:
            return why
    return None
