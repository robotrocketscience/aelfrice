"""Byte-copy one aelfrice store folder for the fire census (#1887).

Usage:

    uv run python -m benchmarks.real_store_fire_census.copy_store SRC DEST [--dry-run]

SRC is a store folder, such as `<repo>/.git/aelfrice`, or `~/.aelfrice`,
whose global-log files the census reads even when it holds no
`memory.db`. DEST must not exist yet, and must not be in this repository,
in SRC, or in a live store location (`guard.live_reason`).

The helper never opens SRC with sqlite and never writes under it. It walks
SRC with `os.fwalk` and opens every file with `O_NOFOLLOW`, so it follows no
symlink, and it refuses a SRC that holds one. It reads every file except
`-shm` (an index sqlite rebuilds) as plain bytes, then reads each
`memory.db` and `-wal` a second time and compares the bytes and the size,
mtime and inode. When anything differs, a writer was active, and the
attempt is thrown away. After 20 such attempts it exits 1 and asks you to
close the sessions using the store. Any other error ends the run at once.

It creates DEST itself and works in a fresh `mkdtemp` folder inside it, so
it never deletes a path it didn't create. In the copy only, it checkpoints
each `-wal` into its db, runs `PRAGMA quick_check`, and then writes the
copy token (`copy_token`), which the census runner requires. Output is
counts only, and messages carry no path. Exit codes: 0 on success, 1 when
no attempt got a quiet, sound copy, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from pathlib import Path

from . import copy_token, guard
from .core import has_unfolded_wal

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTEMPTS = 20
RETRY_PAUSE_S = 0.1
CHUNK = 1 << 20
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

DbState = dict[str, tuple[int, int, int, str]]


class BusyError(Exception):
    """A db file changed while it was read."""


class UnsafeSourceError(Exception):
    """The source holds a symlink or something that isn't a file or folder."""


def _walk(src: Path) -> tuple[list[Path], list[Path], bool]:
    """(files, dirs, has_symlink) under src, relative paths, no link followed."""
    files: list[Path] = []
    dirs: list[Path] = []
    link = False
    for dirpath, dirnames, filenames, dirfd in os.fwalk(src, follow_symlinks=False):
        here = Path(dirpath).relative_to(src)
        for name in dirnames:
            if stat.S_ISLNK(os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode):
                link = True
            else:
                dirs.append(here / name)
        for name in filenames:
            mode = os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                link = True
            elif stat.S_ISREG(mode):
                files.append(here / name)
    return sorted(files), sorted(dirs), link


def _is_db_file(rel: Path) -> bool:
    return rel.name in ("memory.db", "memory.db-wal")


def _read(src: Path, rel: Path, out: Path | None) -> tuple[tuple[int, int, int], str]:
    """Read one source file with O_NOFOLLOW; copy it to out when given.

    Returns ((size, mtime_ns, ino) from fstat, sha256 of the bytes read).
    """
    fd = os.open(src / rel, os.O_RDONLY | NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafeSourceError("not a regular file")
        h = hashlib.sha256()
        dst = None
        if out is not None:
            dst = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
        try:
            while True:
                buf = os.read(fd, CHUNK)
                if not buf:
                    break
                h.update(buf)
                if dst is not None:
                    os.write(dst, buf)
        finally:
            if dst is not None:
                os.close(dst)
        return (st.st_size, st.st_mtime_ns, st.st_ino), h.hexdigest()
    finally:
        os.close(fd)


def _db_state(src: Path, files: list[Path], tmp: Path | None) -> DbState:
    out: DbState = {}
    for rel in files:
        if _is_db_file(rel):
            try:
                meta, digest = _read(src, rel, tmp / rel if tmp is not None else None)
            except FileNotFoundError:
                raise BusyError("a db file vanished while it was read") from None
            out[rel.as_posix()] = (*meta, digest)
    return out


def _copy_once(src: Path, tmp: Path) -> None:
    """One attempt into tmp. Raises BusyError when a writer changed a db file."""
    files, dirs, link = _walk(src)
    if link:
        raise UnsafeSourceError("the source holds a symlink")
    for d in dirs:
        (tmp / d).mkdir(parents=True, exist_ok=True)
    for rel in files:
        if not rel.name.endswith("-shm") and not _is_db_file(rel):
            _read(src, rel, tmp / rel)
    first = _db_state(src, files, tmp)
    files_again, _dirs, link = _walk(src)
    if link:
        raise UnsafeSourceError("the source holds a symlink")
    if [f for f in files_again if _is_db_file(f)] != [f for f in files if _is_db_file(f)]:
        raise BusyError("db files appeared or vanished")
    if _db_state(src, files, None) != first:
        raise BusyError("a db file changed while it was read")


def _fold_and_check(db_file: Path) -> bool:
    """Checkpoint the copy's WAL into its db file, then quick_check it."""
    conn = sqlite3.connect(str(db_file), timeout=5.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        row = conn.execute("PRAGMA quick_check").fetchone()
    finally:
        conn.close()
    return row is not None and row[0] == "ok" and not has_unfolded_wal(db_file)


def copy_store(src: Path, dest: Path) -> dict[str, int]:
    """Copy src into dest, which this call creates. Returns integer counts.

    quick_check_ok is 0 when every attempt met a writer or the copy wasn't
    sound; dest is then removed again, since this call made it.
    """
    report = {"attempts": 0, "quick_check_ok": 0, "dbs": 0, "busy": 0}
    os.mkdir(dest)
    try:
        for attempt in range(1, ATTEMPTS + 1):
            report["attempts"] = attempt
            tmp = Path(tempfile.mkdtemp(dir=dest, prefix=".attempt-"))
            try:
                _copy_once(src, tmp)
            except BusyError:
                report["busy"] += 1
                shutil.rmtree(tmp)
                time.sleep(RETRY_PAUSE_S)
                continue
            except BaseException:
                shutil.rmtree(tmp)
                raise
            dbs = sorted(p for p in tmp.rglob("memory.db") if p.is_file())
            sound = all(_fold_and_check(db) for db in dbs)
            if not sound:
                shutil.rmtree(tmp)
                break
            for entry in sorted(tmp.iterdir()):
                os.rename(entry, dest / entry.name)
            os.rmdir(tmp)
            report.update(copy_token.write_token(dest))
            report["quick_check_ok"] = 1
            report["dbs"] = len(dbs)
            break
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    if not report["quick_check_ok"]:
        os.rmdir(dest)
        return report
    files, _dirs, _link = _walk(dest)
    report["files"] = len(files)
    report["bytes"] = sum((dest / f).stat().st_size for f in files)
    return report


def _usage_problem(src: Path, dest: Path) -> str:
    if not src.is_dir() or src.is_symlink():
        return "SRC is not a folder"
    if _walk(src)[2]:
        return "SRC holds a symlink; copy its target by hand"
    if os.path.lexists(dest):
        return "DEST already exists"
    parent = Path(os.path.abspath(dest)).parent
    if not parent.is_dir():
        return "DEST's folder doesn't exist"
    if guard.inside(parent, REPO_ROOT):
        return "DEST is inside the repository"
    if guard.inside(parent, src):
        return "DEST is inside SRC"
    why = guard.live_reason(parent)
    if why:
        return f"DEST is in a live store location ({why})"
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.real_store_fire_census.copy_store")
    ap.add_argument("src", type=Path, help="store folder to copy")
    ap.add_argument("dest", type=Path, help="new folder for the copy, outside this repo")
    ap.add_argument("--dry-run", action="store_true", help="count what would be copied")
    a = ap.parse_args(argv)
    src: Path = a.src
    dest: Path = a.dest
    problem = _usage_problem(src, dest)
    if problem:
        print(f"copy_store: {problem}", file=sys.stderr)
        return 2
    if a.dry_run:
        files, _dirs, _link = _walk(src)
        size = sum(os.stat(src / f, follow_symlinks=False).st_size for f in files)
        print(json.dumps({"dry_run": True, "files": len(files), "bytes": size}, sort_keys=True))
        return 0
    try:
        report = copy_store(src, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"copy_store: failed with {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    if not report["quick_check_ok"]:
        if report["busy"] == report["attempts"]:
            print("copy_store: the store kept changing while it was read; close the sessions "
                  "using it and run copy_store again", file=sys.stderr)
        else:
            print("copy_store: the copied db failed PRAGMA quick_check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
