"""The copy token: proof that `copy_store` made a folder (#1887).

`copy_store` writes `.fire-census-copy` at the top of each copy it makes.
The file holds a random token and, for every `memory.db` in the copy, its
path relative to the copy and its `st_dev`, `st_ino`, `st_size`,
`st_mtime_ns` and `st_nlink`. It holds no source path.

The runner reads a store only when the nearest token file at or above the
store's folder records that db, and the db's `lstat` still matches: same
device and inode, same size and mtime, a regular file with one link. A
live store has no token, a token copied next to a live db names another
inode, and a hard link to a live file has two links, so all three are
refused whatever their path. The runner then opens the db through a
descriptor and checks the inode again, so a swap after this check is
refused too.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any

TOKEN_FILENAME = ".fire-census-copy"
TOKEN_VERSION = 1
TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
STAMP_KEYS = ("dev", "ino", "size", "mtime_ns", "nlink")
MAX_TOKEN_BYTES = 1 << 20


class TokenError(Exception):
    """A store folder has no valid copy token. The message carries no path."""


def stamp_of(st: os.stat_result) -> dict[str, int]:
    return {"dev": st.st_dev, "ino": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "nlink": st.st_nlink}


def _db_files(root: Path) -> list[Path]:
    """Every regular memory.db under root, relative paths; refuses symlinks."""
    out: list[Path] = []
    for dirpath, dirnames, filenames, dirfd in os.fwalk(root, follow_symlinks=False):
        for name in [*dirnames, *filenames]:
            if stat.S_ISLNK(os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode):
                raise TokenError("the copy holds a symlink")
        if "memory.db" in filenames:
            out.append(Path(dirpath, "memory.db").relative_to(root))
    return sorted(out)


def write_token(root: Path) -> dict[str, int]:
    """Stamp every memory.db under root and write the token file there.

    Refuses a db that is not a regular single-link file or whose -wal still
    holds frames, and an existing token file. Returns integer counts.
    """
    stamps: dict[str, dict[str, int]] = {}
    for rel in _db_files(root):
        db = root / rel
        st = os.stat(db, follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise TokenError("a db in the copy is not a single-link regular file")
        wal = db.with_name(db.name + "-wal")
        if os.path.lexists(wal) and os.stat(wal, follow_symlinks=False).st_size > 0:
            raise TokenError("a db in the copy has a non-empty -wal file")
        stamps[rel.as_posix()] = stamp_of(st)
    doc = {"version": TOKEN_VERSION, "token": secrets.token_hex(16), "dbs": stamps}
    fd = os.open(root / TOKEN_FILENAME,
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return {"token_dbs": len(stamps)}


def _read_token(path: Path) -> dict[str, Any] | None:
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_TOKEN_BYTES:
        raise TokenError("the copy token is not a regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        try:
            doc: Any = json.load(f)
        except ValueError:
            raise TokenError("the copy token can't be parsed") from None
    if (not isinstance(doc, dict) or doc.get("version") != TOKEN_VERSION  # pyright: ignore[reportUnknownMemberType]
            or not isinstance(doc.get("token"), str)  # pyright: ignore[reportUnknownMemberType]
            or not TOKEN_RE.match(str(doc.get("token")))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            or not isinstance(doc.get("dbs"), dict)):  # pyright: ignore[reportUnknownMemberType]
        raise TokenError("the copy token is malformed")
    return dict(doc)  # pyright: ignore[reportUnknownArgumentType]


def find_token(folder: Path) -> tuple[Path, dict[str, Any]]:
    """The nearest token at or above folder (the path as given, not resolved)."""
    start = Path(os.path.abspath(folder))
    for d in (start, *start.parents):
        doc = _read_token(d / TOKEN_FILENAME)
        if doc is not None:
            return d, doc
    raise TokenError("no copy token at or above it; make the copy with copy_store")


def verify_folder(folder: Path) -> Path:
    """Raise TokenError unless a valid token sits at or above folder."""
    return find_token(folder)[0]


def verify_db(db_file: Path) -> tuple[int, int]:
    """(st_dev, st_ino) of a db its copy token records; raises TokenError."""
    db = Path(os.path.abspath(db_file))
    root, doc = find_token(db.parent)
    rel = db.relative_to(root).as_posix()
    raw: Any = doc["dbs"].get(rel)
    if not isinstance(raw, dict):
        raise TokenError("the copy token doesn't record this db")
    rec: dict[str, Any] = dict(raw)  # pyright: ignore[reportUnknownArgumentType]
    if any(type(rec.get(k)) is not int for k in STAMP_KEYS):
        raise TokenError("the copy token doesn't record this db")
    try:
        st = os.stat(db, follow_symlinks=False)
    except FileNotFoundError:
        raise TokenError("the db the copy token records is missing") from None
    if not stat.S_ISREG(st.st_mode):
        raise TokenError("the db is not a regular file")
    if st.st_nlink != 1:
        raise TokenError("the db has another hard link")
    if stamp_of(st) != {k: rec[k] for k in STAMP_KEYS}:
        raise TokenError("the db is not the file the copy token recorded")
    for suffix in ("-wal", "-shm"):
        side = db.with_name(db.name + suffix)
        if os.path.lexists(side) and stat.S_ISLNK(os.stat(side, follow_symlinks=False).st_mode):
            raise TokenError("a -wal or -shm beside the db is a symlink")
    return st.st_dev, st.st_ino
