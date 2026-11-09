"""Byte-copy one aelfrice store folder for the fire census (#1887).

Usage:

    uv run python -m benchmarks.real_store_fire_census.copy_store SRC DEST [--dry-run]

SRC is a store folder: the folder that holds `memory.db`, such as
`<repo>/.git/aelfrice`. DEST must not exist yet and must sit outside this
repository. Every file under SRC is copied byte for byte, including
`memory.db-wal`, `memory.db-shm`, the JSONL sidecars and any
`stores/<name>/` named stores below it. Then, in the copy only, each
`memory.db` is opened read-write, its `-wal` frames are checkpointed into
the db file, and it has to pass `PRAGMA quick_check`. sqlite removes the
copy's `-wal` and `-shm` when that connection closes, which is what lets
the census read the copy with `immutable=1`. When a writer was active
during the copy and the check fails, the copy is redone, up to three
times.

This is the only part of the census that reads a live store, and it only
reads SRC. The census runner reads the copy. Output is counts only.
Exit codes: 0 on success, 1 when the copy never passed the check, 2 on a
usage error.
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

from .core import has_unfolded_wal

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTEMPTS = 3
RETRY_PAUSE_S = 0.5


def _tree_size(src: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for root, _dirs, names in os.walk(src):
        for n in names:
            files += 1
            try:
                size += (Path(root) / n).stat().st_size
            except OSError:
                pass
    return files, size


def _fold_and_check(db_file: Path) -> bool:
    """Checkpoint the copy's WAL into its db file, then quick_check it.

    Runs on the copy only. Returns False when the copy is not a sound db.
    """
    try:
        conn = sqlite3.connect(str(db_file), timeout=5.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            row = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
    return row is not None and row[0] == "ok" and not has_unfolded_wal(db_file)


def _copied_dbs(dest: Path) -> list[Path]:
    return sorted(p for p in dest.rglob("memory.db") if p.is_file())


def copy_store(src: Path, dest: Path) -> dict[str, int]:
    """Copy src to dest and check the copied db. Returns integer counts."""
    report = {"attempts": 0, "quick_check_ok": 0, "dbs": 0}
    for attempt in range(1, ATTEMPTS + 1):
        report["attempts"] = attempt
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, symlinks=True)
        dbs = _copied_dbs(dest)
        if all(_fold_and_check(db) for db in dbs):
            report["quick_check_ok"] = 1
            report["dbs"] = len(dbs)
            break
        time.sleep(RETRY_PAUSE_S)
    report["files"], report["bytes"] = _tree_size(dest)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.real_store_fire_census.copy_store")
    ap.add_argument("src", type=Path, help="store folder that holds memory.db")
    ap.add_argument("dest", type=Path, help="new folder for the copy, outside this repo")
    ap.add_argument("--dry-run", action="store_true", help="count what would be copied")
    a = ap.parse_args(argv)
    src: Path = a.src
    dest: Path = a.dest
    problem = ""
    if not (src / "memory.db").is_file():
        problem = "SRC has no memory.db"
    elif dest.exists():
        problem = "DEST already exists"
    elif dest.resolve() == REPO_ROOT or REPO_ROOT in dest.resolve().parents:
        problem = "DEST is inside the repository"
    elif src.resolve() in dest.resolve().parents:
        problem = "DEST is inside SRC"
    if problem:
        print(f"copy_store: {problem}", file=sys.stderr)
        return 2
    if a.dry_run:
        files, size = _tree_size(src)
        print(json.dumps({"dry_run": True, "files": files, "bytes": size}, sort_keys=True))
        return 0
    report = copy_store(src, dest)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["quick_check_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
