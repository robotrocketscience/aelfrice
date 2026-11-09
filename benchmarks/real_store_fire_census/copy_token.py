"""The copy token: the manifest of a copy `copy_store` made (#1887).

`copy_store` writes `.fire-census-copy` last, into the temp folder it
created and filled, and only that call writes one (`_write_token`). The
file holds a random token, the device and inode of the copy folder, the
copy's folders, and, for every file it copied, the path inside the copy
and its `st_dev`, `st_ino`, `st_nlink`, size and `st_mtime_ns`. It holds no
source path.

The runner reads a file of a copy only through a `Gate`, which loads the
manifests of the copies in the run. A file the gate reads must be listed,
must still match its entry, and is opened with `O_NOFOLLOW` and checked
again with `fstat`. A file that is in the folder but not in the manifest,
a listed file that changed or went missing, and a folder listing that
differs from the manifest all raise `ManifestMismatch`, and the runner exits
with status 2. `GatedPath` is a `Path` whose reads go through the gate, so
probe code that calls `is_file`, `open`, `read_text`, `glob` or `iterdir`
on a context path reads only what the manifest lists.

The token guards against accidents: a live store, a stale path, a symlink
or a file changed after the copy. It doesn't defend against a user who
writes a token by hand.
"""
from __future__ import annotations

import fnmatch
import io
import json
import os
import re
import secrets
import stat
from collections.abc import Generator
from pathlib import Path
from typing import IO, Any

from . import guard

TOKEN_FILENAME = ".fire-census-copy"
TOKEN_VERSION = 2
TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
STAMP_KEYS = ("dev", "ino", "nlink", "size", "mtime_ns")
MAX_TOKEN_BYTES = 64 << 20
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)

Stamp = dict[str, int]


class TokenError(Exception):
    """A folder has no valid copy token. The message carries no path."""


class ManifestMismatch(Exception):
    """A file of a copy doesn't match the copy's manifest. No path in the message."""


def stamp_of(st: os.stat_result) -> Stamp:
    return {"dev": st.st_dev, "ino": st.st_ino, "nlink": st.st_nlink,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _walk(root: Path) -> tuple[list[str], list[str], bool]:
    """(files, dirs, has_symlink) under root as posix relpaths; the token is left out."""
    files: list[str] = []
    dirs: list[str] = []
    link = False
    for dirpath, dirnames, filenames, dirfd in os.fwalk(root, follow_symlinks=False):
        here = Path(dirpath).relative_to(root)
        for name in dirnames:
            if stat.S_ISLNK(os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode):
                link = True
            else:
                dirs.append((here / name).as_posix())
        for name in filenames:
            mode = os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode
            rel = (here / name).as_posix()
            if stat.S_ISLNK(mode):
                link = True
            elif rel != TOKEN_FILENAME:
                files.append(rel)
    return sorted(files), sorted(dirs), link


def _write_token(root: Path, copied: set[str],  # pyright: ignore[reportUnusedFunction]
                 root_ident: tuple[int, int]) -> dict[str, int]:
    """Write the manifest into root, the temp folder copy_store just filled.

    Refuses a root the guard flags as live, a root that holds a symlink or
    any file not in `copied`, a file with more than one link, and a
    non-empty `-wal`. `root_ident` is the copy folder the entries will be
    moved into. Returns integer counts.
    """
    if guard.live_reason(root):
        raise TokenError("the folder is a live store location")
    files, dirs, link = _walk(root)
    if link:
        raise TokenError("the folder holds a symlink")
    if set(files) - copied:
        raise TokenError("the folder holds a file the copy helper didn't copy")
    entries: dict[str, Stamp] = {}
    for rel in files:
        st = os.stat(root / rel, follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise TokenError("a file in the copy is not a single-link regular file")
        if rel.endswith("-wal") and st.st_size > 0:
            raise TokenError("a db in the copy has a non-empty -wal file")
        entries[rel] = stamp_of(st)
    doc = {"version": TOKEN_VERSION, "token": secrets.token_hex(16),
           "root": {"dev": root_ident[0], "ino": root_ident[1]},
           "dirs": dirs, "files": entries}
    fd = os.open(root / TOKEN_FILENAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return {"token_files": len(entries)}


def _is_int(v: Any) -> bool:
    return type(v) is int


def _is_rel(rel: Any) -> bool:
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def _parse(raw: bytes) -> tuple[tuple[int, int], frozenset[str], dict[str, Stamp]]:
    try:
        doc: Any = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise TokenError("the copy token can't be parsed") from None
    if not isinstance(doc, dict):
        raise TokenError("the copy token is malformed")
    d: dict[str, Any] = dict(doc)  # pyright: ignore[reportUnknownArgumentType]
    root: Any = d.get("root")
    dirs: Any = d.get("dirs")
    files: Any = d.get("files")
    if (d.get("version") != TOKEN_VERSION or not isinstance(d.get("token"), str)
            or not TOKEN_RE.match(str(d.get("token")))
            or not isinstance(root, dict) or not isinstance(dirs, list)
            or not isinstance(files, dict)):
        raise TokenError("the copy token is malformed")
    r: dict[str, Any] = dict(root)  # pyright: ignore[reportUnknownArgumentType]
    if set(r) != {"dev", "ino"} or not all(_is_int(v) for v in r.values()):
        raise TokenError("the copy token is malformed")
    dir_list: list[Any] = list(dirs)  # pyright: ignore[reportUnknownArgumentType]
    if not all(_is_rel(x) for x in dir_list):
        raise TokenError("the copy token is malformed")
    out: dict[str, Stamp] = {}
    file_map: dict[Any, Any] = dict(files)  # pyright: ignore[reportUnknownArgumentType]
    for rel, rec in file_map.items():
        if not _is_rel(rel) or not isinstance(rec, dict):
            raise TokenError("the copy token is malformed")
        e: dict[str, Any] = dict(rec)  # pyright: ignore[reportUnknownArgumentType]
        if set(e) != set(STAMP_KEYS) or not all(_is_int(e[k]) for k in STAMP_KEYS):
            raise TokenError("the copy token is malformed")
        out[str(rel)] = {k: int(e[k]) for k in STAMP_KEYS}
    return (int(r["dev"]), int(r["ino"])), frozenset(str(x) for x in dir_list), out


class Manifest:
    """One copy: its folder, the folders and files it lists, and their stamps."""

    def __init__(self, root: Path, dirs: frozenset[str], files: dict[str, Stamp]) -> None:
        self.root = os.path.abspath(root)
        self.dirs = dirs
        self.files = files
        children: dict[str, set[str]] = {"": set()}
        for rel in (*dirs, *files):
            parent, _, name = rel.rpartition("/")
            children.setdefault(parent, set()).add(name)
        for rel in dirs:
            children.setdefault(rel, set())
        self.children = {k: frozenset(v) for k, v in children.items()}


def load(folder: Path) -> Manifest:
    """The manifest whose token sits in folder itself. Raises TokenError."""
    root = Path(os.path.abspath(folder))
    path = root / TOKEN_FILENAME
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise TokenError("no copy token in the folder; make the copy with copy_store") from None
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_TOKEN_BYTES:
        raise TokenError("the copy token is not a regular file")
    fd = os.open(path, os.O_RDONLY | NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        raw = f.read(MAX_TOKEN_BYTES + 1)
    ident, dirs, files = _parse(raw)
    try:
        dfd = os.open(root, os.O_RDONLY | DIRECTORY | NOFOLLOW)
    except OSError:
        raise TokenError("the copy folder can't be opened") from None
    try:
        dst = os.fstat(dfd)
    finally:
        os.close(dfd)
    if (dst.st_dev, dst.st_ino) != ident:
        raise TokenError("the copy token belongs to another folder")
    return Manifest(root, dirs, files)


def find_root(folder: Path) -> Path:
    """The nearest folder at or above folder that holds a copy token."""
    start = Path(os.path.abspath(folder))
    for d in (start, *start.parents):
        if os.path.lexists(d / TOKEN_FILENAME):
            return d
    raise TokenError("no copy token at or above it; make the copy with copy_store")


class Gate:
    """Every read of a copy's files goes through here."""

    def __init__(self) -> None:
        self._manifests: dict[str, Manifest] = {}

    def add(self, m: Manifest) -> None:
        self._manifests.setdefault(m.root, m)

    def _lookup(self, p: Path | str) -> tuple[Manifest | None, str]:
        a = os.path.abspath(p)
        best: Manifest | None = None
        for root, m in self._manifests.items():
            if (a == root or a.startswith(root.rstrip(os.sep) + os.sep)) and (
                    best is None or len(root) > len(best.root)):
                best = m
        if best is None:
            return None, ""
        rel = os.path.relpath(a, best.root)
        return best, "" if rel == "." else Path(rel).as_posix()

    def file_stat(self, p: Path | str) -> os.stat_result | None:
        """lstat of a listed file that still matches; None for one that isn't there."""
        m, rel = self._lookup(p)
        listed = m is not None and rel in m.files
        if not listed:
            if m is not None and (rel == "" or rel in m.dirs or rel == TOKEN_FILENAME):
                return None
            if os.path.lexists(p):
                raise ManifestMismatch("a file the copy token doesn't list")
            return None
        assert m is not None
        try:
            st = os.stat(p, follow_symlinks=False)
        except FileNotFoundError:
            raise ManifestMismatch("a file the copy token lists is missing") from None
        if not stat.S_ISREG(st.st_mode) or stamp_of(st) != m.files[rel]:
            raise ManifestMismatch("a file doesn't match its copy token entry")
        return st

    def is_dir(self, p: Path | str) -> bool:
        m, rel = self._lookup(p)
        if m is not None and (rel == "" or rel in m.dirs):
            try:
                st = os.stat(p, follow_symlinks=False)
            except FileNotFoundError:
                raise ManifestMismatch("a folder the copy token lists is missing") from None
            if not stat.S_ISDIR(st.st_mode):
                raise ManifestMismatch("a folder doesn't match its copy token")
            return True
        if m is not None and rel in m.files:
            return False
        if os.path.lexists(p):
            raise ManifestMismatch("a folder the copy token doesn't list")
        return False

    def listdir(self, p: Path | str) -> list[str]:
        if not self.is_dir(p):
            raise NotADirectoryError("not a listed folder")
        m, rel = self._lookup(p)
        assert m is not None
        want = m.children.get(rel, frozenset())
        have = frozenset(os.listdir(p)) - ({TOKEN_FILENAME} if rel == "" else set())
        if have != want:
            raise ManifestMismatch("a folder's entries differ from its copy token")
        return sorted(want)

    def open_fd(self, p: Path | str) -> int:
        st = self.file_stat(p)
        if st is None:
            raise FileNotFoundError("not a listed file")
        fd = os.open(p, os.O_RDONLY | NOFOLLOW)
        try:
            fst = os.fstat(fd)
        except BaseException:
            os.close(fd)
            raise
        if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            os.close(fd)
            raise ManifestMismatch("a file changed while it was opened")
        return fd

    def identity(self, p: Path | str) -> tuple[int, int]:
        """(st_dev, st_ino) of a listed db; its -wal and -shm must be listed or absent."""
        st = self.file_stat(p)
        if st is None:
            raise ManifestMismatch("the db is not in the copy token")
        for suffix in ("-wal", "-shm"):
            self.file_stat(str(p) + suffix)
        return st.st_dev, st.st_ino


_WRITE_MODES = frozenset("wax+")


class GatedPath(Path):
    """A Path whose reads go through a Gate and whose writes are refused."""

    def __init__(self, *args: Any, gate: Gate) -> None:  # pyright: ignore[reportInconsistentConstructor]
        super().__init__(*args)
        self.gate = gate

    def with_segments(self, *pathsegments: Any) -> GatedPath:
        return type(self)(*pathsegments, gate=self.gate)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        return self.gate.file_stat(self) is not None

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self.gate.is_dir(self)

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        return self.is_file() or self.is_dir()

    def is_symlink(self) -> bool:
        return False

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        st = self.gate.file_stat(self)
        if st is not None:
            return st
        if self.gate.is_dir(self):
            return os.stat(self, follow_symlinks=False)
        raise FileNotFoundError("not a listed file")

    def lstat(self) -> os.stat_result:
        return self.stat()

    def open(self, mode: str = "r",  # pyright: ignore[reportIncompatibleMethodOverride]
             buffering: int = -1, encoding: str | None = None,
             errors: str | None = None, newline: str | None = None) -> IO[Any]:
        if _WRITE_MODES & set(mode):
            raise PermissionError("census copies are read only")
        return io.open(self.gate.open_fd(self), mode, buffering, encoding, errors, newline)

    def read_text(self, encoding: str | None = None, errors: str | None = None,
                  newline: str | None = None) -> str:
        with self.open("r", encoding=encoding, errors=errors, newline=newline) as f:
            return str(f.read())

    def read_bytes(self) -> bytes:
        with self.open("rb") as f:
            return bytes(f.read())

    def iterdir(self) -> Generator[GatedPath, None, None]:
        for name in self.gate.listdir(self):
            yield self / name

    def glob(self, pattern: str, *, case_sensitive: bool | None = None,
             recurse_symlinks: bool = False) -> Generator[GatedPath, None, None]:
        if "/" in pattern or "**" in pattern:
            raise ValueError("a gated glob takes one path component")
        for name in self.gate.listdir(self):
            if fnmatch.fnmatchcase(name, pattern):
                yield self / name

    def _refuse(self, *a: Any, **k: Any) -> Any:
        raise PermissionError("census copies are read only")

    rglob = walk = write_text = write_bytes = touch = mkdir = unlink = _refuse  # pyright: ignore[reportAssignmentType]
    rename = replace = rmdir = chmod = symlink_to = hardlink_to = _refuse  # pyright: ignore[reportAssignmentType]
