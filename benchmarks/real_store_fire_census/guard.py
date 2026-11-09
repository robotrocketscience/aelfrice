"""Live-store checks for the census runner and `copy_store` (#1887).

These checks prevent accidents, alongside the copy manifest
(`copy_token`) that every file the runner reads must match. They refuse a
live location before the manifest is read, and they compare file
identities, `(st_dev, st_ino)` via `os.stat`, rather than
path strings, so a firmlink such as `/System/Volumes/Data/...`, a symlink,
another letter case or a `..` path can't hide one.

A path is live when the path, or any folder above it (taken both as given
and with symlinks resolved):

  * is named `.git` in any letter case, or is a git directory (holds a
    `HEAD` file and an `objects` folder);
  * is the same folder as `~/.aelfrice`, for `$HOME` or the account home;
  * is the folder of the store `$AELFRICE_DB` or `$AELFRICE_USER_DB` names
    (the path itself or its parent, not the whole tree below, since that
    folder may be a home directory), or the path is that db file;
  * is a `[store]` marker store (#1796): the folder a `.aelfrice.toml` above
    the path, or above the working directory, names with `[store] path`. A
    `.aelfrice.toml` on the way that can't be read counts as naming it.

A file with more than one hard link is refused too, since another of its
names may sit in a live store. Messages never carry a path.
"""
from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from typing import Any

MARKER_FILENAME = ".aelfrice.toml"
GIT_STORE_PREFIX = "git:"
SQLITE_SIDECARS = ("", "-wal", "-shm")

Ident = tuple[int, int]


def ident(path: Path) -> Ident | None:
    """(st_dev, st_ino) of path with symlinks followed, or None."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return st.st_dev, st.st_ino


def same(a: Path, b: Path) -> bool:
    """os.path.samefile, False when either path is missing."""
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def chain(path: Path) -> list[Path]:
    """The path and its parents, as given and with symlinks resolved."""
    given = Path(os.path.abspath(path))
    resolved = Path(os.path.realpath(given))
    out: list[Path] = []
    for p in (given, *given.parents, resolved, *resolved.parents):
        if p not in out:
            out.append(p)
    return out


def inside(path: Path, root: Path) -> bool:
    """True when path is root or below it, by file identity."""
    r = ident(root)
    return r is not None and any(ident(p) == r for p in chain(path))


def account_homes() -> list[Path]:
    homes = [Path.home()]
    try:
        import pwd
        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, AttributeError):
        # No pwd module (Windows) or no passwd entry for this uid: $HOME is
        # then the only home to check, so there is nothing to add.
        pass
    return homes


def _is_git_dir(p: Path) -> bool:
    return p.name.lower() == ".git" or ((p / "HEAD").is_file() and (p / "objects").is_dir())


def _marker_target(config: Path) -> Path | None | bool:
    """The store folder a marker names, None for no marker, True if unreadable."""
    try:
        data: dict[str, Any] = tomllib.loads(config.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        return True
    section: Any = data.get("store")
    if section is None:
        return None
    if not isinstance(section, dict):
        return True
    store: dict[str, Any] = dict(section)  # pyright: ignore[reportUnknownArgumentType]
    value: Any = store.get("path")
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
    for d in chain(start):
        cfg = d / MARKER_FILENAME
        if os.path.lexists(cfg):
            t = _marker_target(cfg)
            if t is not None:
                out.append(t)
    return out


def live_reason(path: Path) -> str | None:
    """Why `path` is a live store location, or None when it is not."""
    links = chain(path)
    if any(_is_git_dir(p) for p in links):
        return "inside a .git folder"
    idents = {i for i in (ident(p) for p in links) if i is not None}
    for home in account_homes():
        dot = ident(home / ".aelfrice")
        if dot is not None and dot in idents:
            return "under ~/.aelfrice"
    for var in ("AELFRICE_DB", "AELFRICE_USER_DB"):
        val = os.environ.get(var)
        if not val:
            continue
        env_dir = ident(Path(val).parent)
        own = [ident(links[0]), ident(links[0].parent)]
        if env_dir is not None and env_dir in own:
            return f"the store {var} names"
        if same(links[0], Path(val)):
            return f"the store {var} names"
    for t in _marker_targets(links[0]) + _marker_targets(Path.cwd()):
        if t is True:
            return "under a .aelfrice.toml that can't be read"
        if isinstance(t, Path):
            target = ident(t)
            if target is not None and target in idents:
                return "a [store] marker store"
    try:
        st = os.stat(links[0], follow_symlinks=False)
    except OSError:
        return None
    if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
        return "a file with another hard link"
    return None


def store_live_reason(db_file: Path) -> str | None:
    """live_reason for a db file and each of its -wal and -shm files."""
    for suffix in SQLITE_SIDECARS:
        why = live_reason(db_file.with_name(db_file.name + suffix))
        if why:
            return why
    return None
