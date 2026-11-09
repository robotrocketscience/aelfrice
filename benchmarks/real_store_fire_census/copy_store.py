"""Byte-copy one aelfrice store folder for the fire census (#1887).

Usage:

    uv run python -m benchmarks.real_store_fire_census.copy_store SRC DEST [--dry-run]

SRC is a store folder, such as `<repo>/.git/aelfrice`, or a copy source
such as `~/.aelfrice` whose global-log files the census reads even when it
holds no `memory.db`. DEST must not exist yet, and must not be in this
repository, in SRC, or in a live store location (`guard.live_reason`).

The helper never writes under SRC. It first reads
each file under SRC as plain bytes into a temporary folder beside DEST:
`memory.db`, `memory.db-wal`, the JSONL sidecars and any `stores/<name>/`
named stores. It leaves `-shm` files out, because a `-shm` file is an
index that sqlite rebuilds from the WAL, and a copy of a live one can be
stale. It records the size, modification time and inode of every
`memory.db` and `-wal` before and after the read, and when any of them
changed, a writer was active, so it discards the copy and tries again.
When 20 byte copies in a row meet a writer, it copies each db through
sqlite's backup API instead, from a `mode=ro` connection opened only while
the source's `-wal` and `-shm` exist, so the read creates no file beside
the store; like any reader, that connection updates the `-shm` index the
writer keeps, and it never writes the db or the `-wal`. Then, in the copy
only, it checkpoints each `-wal` into its db file and
runs `PRAGMA quick_check`, and renames the folder to DEST. sqlite removes
the copy's `-wal` when that connection closes, which lets the census read
the copy with `immutable=1`.

SRC is refused when it, or anything under it, is a symlink: following one
could read or name a file outside the store. Output is counts only, and
messages carry no path. Exit codes: 0 on success, 1 when no attempt got a
quiet, sound copy, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from . import guard
from .core import has_unfolded_wal

REPO_ROOT = Path(__file__).resolve().parents[2]
BYTE_ATTEMPTS = 20
BACKUP_ATTEMPTS = 30
RETRY_PAUSE_S = 0.02
CHUNK = 1 << 20

Stamp = dict[str, tuple[int, int, int] | None]


def _tree(src: Path) -> tuple[list[Path], list[Path], bool]:
    """(files, dirs, has_symlink) under src, relative paths, no link followed."""
    files: list[Path] = []
    dirs: list[Path] = []
    link = src.is_symlink()
    for root, dnames, fnames in os.walk(src, followlinks=False):
        r = Path(root)
        for d in dnames:
            if (r / d).is_symlink():
                link = True
            dirs.append((r / d).relative_to(src))
        for f in fnames:
            if (r / f).is_symlink():
                link = True
            files.append((r / f).relative_to(src))
    return sorted(files), sorted(dirs), link


def _is_db_file(rel: Path) -> bool:
    return rel.name == "memory.db" or rel.name == "memory.db-wal"


def _stamp(src: Path, files: list[Path]) -> Stamp:
    """size, mtime_ns and inode of every db and -wal file (None when missing)."""
    names = {f for f in files if _is_db_file(f)}
    names |= {f.with_name("memory.db-wal") for f in files if f.name == "memory.db"}
    out: Stamp = {}
    for rel in sorted(names):
        try:
            st = os.stat(src / rel, follow_symlinks=False)
            out[str(rel)] = (st.st_size, st.st_mtime_ns, st.st_ino)
        except FileNotFoundError:
            out[str(rel)] = None
    return out


def _copy_bytes(src: Path, dst: Path) -> None:
    """Read src as plain bytes into a new dst; src is opened read-only."""
    with src.open("rb") as reader, dst.open("xb") as writer:
        while True:
            buf = reader.read(CHUNK)
            if not buf:
                break
            writer.write(buf)


def _copy_once(src: Path, tmp: Path) -> bool:
    """One copy attempt into tmp. False when a writer changed a db file meanwhile.

    Sidecars are copied first; the db files last and back to back, so the
    window the size, mtime and inode check has to cover is as short as it
    can be.
    """
    files, dirs, _link = _tree(src)
    tmp.mkdir(parents=True)
    for d in dirs:
        (tmp / d).mkdir(parents=True, exist_ok=True)
    for rel in files:
        if not rel.name.endswith("-shm") and not _is_db_file(rel):
            _copy_bytes(src / rel, tmp / rel)
    before = _stamp(src, files)
    for rel in sorted(before):
        if before[rel] is None:
            continue
        try:
            _copy_bytes(src / rel, tmp / rel)
        except FileNotFoundError:
            return False  # the writer checkpointed and removed a -wal mid-copy
    return _stamp(src, files) == before


def _backup_db(src_db: Path, dst_db: Path) -> bool:
    """Copy a db in use through sqlite's backup API, from a `mode=ro` connection.

    Only while the source's `-wal` exists, so the read creates no file
    beside it. The connection never writes the db or the `-wal`; as any
    reader does, it updates the `-shm` index the writer keeps.
    """
    if not (src_db.with_name(src_db.name + "-wal").is_file()
            and src_db.with_name(src_db.name + "-shm").is_file()):
        return False
    s = sqlite3.connect(src_db.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    try:
        d = sqlite3.connect(str(dst_db))
        try:
            s.backup(d)
        finally:
            d.close()
    finally:
        s.close()
    return True


def _backup_once(src: Path, tmp: Path) -> bool:
    """A copy attempt that reads the dbs through the backup API."""
    files, dirs, _link = _tree(src)
    tmp.mkdir(parents=True)
    for d in dirs:
        (tmp / d).mkdir(parents=True, exist_ok=True)
    for rel in files:
        if not rel.name.endswith("-shm") and not _is_db_file(rel):
            _copy_bytes(src / rel, tmp / rel)
    for rel in files:
        if rel.name == "memory.db" and not _backup_db(src / rel, tmp / rel):
            return False
    return True


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
    """Copy src to dest. Returns integer counts; quick_check_ok is 0 on failure.

    Byte copies come first, since they never open the source. When the
    store never holds still for one, the backup API takes over.
    """
    report = {"attempts": 0, "quick_check_ok": 0, "dbs": 0, "busy": 0, "errors": 0,
              "via_backup": 0}
    tmp = dest.with_name(dest.name + ".partial")
    for attempt in range(1, BYTE_ATTEMPTS + BACKUP_ATTEMPTS + 1):
        report["attempts"] = attempt
        backup = attempt > BYTE_ATTEMPTS
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            if not (_backup_once if backup else _copy_once)(src, tmp):
                report["busy"] += 1
                continue
            dbs = sorted(p for p in tmp.rglob("memory.db") if p.is_file())
            if all(_fold_and_check(db) for db in dbs):
                os.replace(tmp, dest)
                report["quick_check_ok"] = 1
                report["dbs"] = len(dbs)
                report["via_backup"] = int(backup)
                break
            report["errors"] += 1
        except (OSError, shutil.Error, sqlite3.Error):
            report["errors"] += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if not report["quick_check_ok"]:
                time.sleep(RETRY_PAUSE_S)
    if report["quick_check_ok"]:
        files, _dirs, _link = _tree(dest)
        report["files"] = len(files)
        report["bytes"] = sum((dest / f).stat().st_size for f in files)
    return report


def _usage_problem(src: Path, dest: Path) -> str:
    if not src.is_dir() or src.is_symlink():
        return "SRC is not a folder"
    if _tree(src)[2]:
        return "SRC holds a symlink; copy its target by hand"
    if dest.exists() or dest.is_symlink():
        return "DEST already exists"
    if guard.inside(dest, REPO_ROOT):
        return "DEST is inside the repository"
    if guard.inside(dest, src):
        return "DEST is inside SRC"
    why = guard.live_reason(dest)
    if why:
        return f"DEST is in a live store location ({why})"
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.real_store_fire_census.copy_store")
    ap.add_argument("src", type=Path, help="store folder that holds memory.db")
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
        files, _dirs, _link = _tree(src)
        size = 0
        for f in files:
            try:
                size += (src / f).stat().st_size
            except OSError:
                pass
        print(json.dumps({"dry_run": True, "files": len(files), "bytes": size}, sort_keys=True))
        return 0
    try:
        report = copy_store(src, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"copy_store: failed with {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    if not report["quick_check_ok"]:
        print("copy_store: no attempt got a quiet, sound copy; retry when the store is idle",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
