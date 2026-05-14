"""One-shot migration from the v1.0 global DB to the v1.1.0 per-project store.

v1.0 ships a single global DB at ~/.aelfrice/memory.db. v1.1.0 (#88)
introduces per-project resolution that defaults to .git/aelfrice/
memory.db inside git repos. Existing users have all of their
accumulated beliefs in the legacy global store; they need a way to
copy the project-relevant subset into each project's new in-repo
store without losing the global DB itself.

`aelf migrate`:
- reads the legacy DB read-only (path defaults to ~/.aelfrice/memory.db)
- writes into the active project's resolved DB path
- by default applies a path-mention filter (beliefs whose content
  references the absolute project root)
- dry-run by default; --apply commits writes
- --all opts into copying every belief regardless of project relevance
- never deletes from the source DB

In-place migration (#593) reuses the same `migrate()` core: rename
the legacy per-project DB to a `.pre-v1x.bak` sibling, then run
`migrate(legacy=bak, target=original, copy_all=True, apply=True)`.
Used by `aelf doctor` to silently bring pre-v1.x per-project DBs
forward to the modern schema. Backup-first means the operation is
recoverable even if the migration crashes mid-write.

Out of scope for v1.1.0: feedback_history copy (audit-log fidelity
across stores), in-place delete from legacy, --pattern filter,
interactive selection. All v1.2.0+.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aelfrice.models import ORIGIN_UNKNOWN, Belief, Edge
from aelfrice.store import MemoryStore

LEGACY_DB_NAME: Final[str] = "memory.db"
LEGACY_DB_DIR_RELATIVE_TO_HOME: Final[str] = ".aelfrice"


def default_legacy_db_path() -> Path:
    """Where the v1.0 global DB lives by default."""
    return Path.home() / LEGACY_DB_DIR_RELATIVE_TO_HOME / LEGACY_DB_NAME


@dataclass(frozen=True)
class MigrateCounts:
    legacy_beliefs: int
    legacy_edges: int
    matched_beliefs: int
    inserted_beliefs: int
    skipped_existing_beliefs: int
    inserted_edges: int
    skipped_orphan_edges: int


@dataclass(frozen=True)
class MigrateReport:
    """Result of a migrate run. `applied=False` for dry-run."""

    legacy_path: Path
    target_path: Path
    counts: MigrateCounts
    applied: bool


def _belongs_to_project(content: str, project_root: Path) -> bool:
    """Heuristic: belief content references the absolute project root.

    v1.1.0 heuristic — narrow on purpose. Misses beliefs that mention
    the project obliquely (relative paths, project name only). Users
    who want a wider net pass --all to skip the filter entirely.
    """
    abs_root = str(project_root.absolute())
    return abs_root in content


def _read_legacy_beliefs(conn: sqlite3.Connection) -> list[Belief]:
    cur = conn.execute("SELECT * FROM beliefs")
    rows = cur.fetchall()
    has_origin = bool(rows) and "origin" in rows[0].keys()
    out: list[Belief] = []
    for row in rows:
        out.append(Belief(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            type=row["type"],
            lock_level=row["lock_level"],
            locked_at=row["locked_at"],
            created_at=row["created_at"],
            last_retrieved_at=row["last_retrieved_at"],
            origin=row["origin"] if has_origin else ORIGIN_UNKNOWN,
        ))
    return out


def _read_legacy_edges(conn: sqlite3.Connection) -> list[Edge]:
    cur = conn.execute("SELECT * FROM edges")
    return [
        Edge(
            src=row["src"],
            dst=row["dst"],
            type=row["type"],
            weight=float(row["weight"]),
        )
        for row in cur.fetchall()
    ]


def migrate(
    *,
    legacy_path: Path,
    target_path: Path,
    project_root: Path,
    apply: bool,
    copy_all: bool,
) -> MigrateReport:
    """Copy beliefs (and incident edges) from `legacy_path` into `target_path`.

    Pure read on `legacy_path`; opens it via `file:...?mode=ro` URI so
    accidental writes are rejected at the SQLite layer.

    `project_root` is the active project's root; used by the default
    filter to decide which legacy beliefs are project-relevant.

    `apply=False` (dry-run) returns the counts without writing.
    `copy_all=True` skips the project-filter and treats every legacy
    belief as a candidate.

    Raises `FileNotFoundError` if `legacy_path` doesn't exist.
    Raises `ValueError` if `legacy_path == target_path`.
    """
    if legacy_path.resolve() == target_path.resolve():
        raise ValueError(
            f"legacy and target are the same DB: {legacy_path} — refusing to migrate"
        )
    if not legacy_path.exists():
        raise FileNotFoundError(legacy_path)

    # Read-only connection to the legacy DB. The mode=ro URI rejects
    # writes at the SQLite layer.
    legacy_uri = f"file:{legacy_path}?mode=ro"
    legacy_conn = sqlite3.connect(legacy_uri, uri=True)
    legacy_conn.row_factory = sqlite3.Row
    try:
        legacy_beliefs = _read_legacy_beliefs(legacy_conn)
        legacy_edges = _read_legacy_edges(legacy_conn)
    finally:
        legacy_conn.close()

    if copy_all:
        matched = legacy_beliefs
    else:
        matched = [
            b for b in legacy_beliefs
            if _belongs_to_project(b.content, project_root)
        ]
    matched_ids: set[str] = {b.id for b in matched}

    inserted_beliefs = 0
    skipped_existing = 0
    inserted_edges = 0
    skipped_orphan_edges = 0

    if apply:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = MemoryStore(str(target_path))
        try:
            for b in matched:
                if target.get_belief(b.id) is not None:
                    skipped_existing += 1
                    continue
                target.insert_belief(b)
                inserted_beliefs += 1

            # Copy edges only when both endpoints landed in the target.
            # Edges that reference an unmatched legacy belief are
            # skipped — they'd create orphan edges in the target.
            target_existing_ids = {
                b.id for b in matched if target.get_belief(b.id) is not None
            }
            valid_endpoints = target_existing_ids | matched_ids
            for e in legacy_edges:
                if e.src in valid_endpoints and e.dst in valid_endpoints:
                    if target.get_edge(e.src, e.dst, e.type) is not None:
                        continue
                    target.insert_edge(e)
                    inserted_edges += 1
                else:
                    skipped_orphan_edges += 1
        finally:
            target.close()
    else:
        # Dry-run: count what would happen without touching target.
        target_exists = target_path.exists()
        if target_exists:
            target = MemoryStore(str(target_path))
            try:
                for b in matched:
                    if target.get_belief(b.id) is not None:
                        skipped_existing += 1
                    else:
                        inserted_beliefs += 1
            finally:
                target.close()
        else:
            inserted_beliefs = len(matched)
        # Edges-would-copy = edges with both endpoints in matched_ids
        # (deduplicated against existing target edges only when applying).
        for e in legacy_edges:
            if e.src in matched_ids and e.dst in matched_ids:
                inserted_edges += 1
            else:
                skipped_orphan_edges += 1

    counts = MigrateCounts(
        legacy_beliefs=len(legacy_beliefs),
        legacy_edges=len(legacy_edges),
        matched_beliefs=len(matched),
        inserted_beliefs=inserted_beliefs,
        skipped_existing_beliefs=skipped_existing,
        inserted_edges=inserted_edges,
        skipped_orphan_edges=skipped_orphan_edges,
    )
    return MigrateReport(
        legacy_path=legacy_path,
        target_path=target_path,
        counts=counts,
        applied=apply,
    )


IN_PLACE_BACKUP_SUFFIX: Final[str] = ".pre-v1x.bak"


@dataclass(frozen=True)
class InPlaceMigrateReport:
    """Result of an in-place migration of a pre-v1.x per-project DB.

    `db_path`     — path of the now-modern-schema DB (same as input).
    `backup_path` — sibling preserving the original pre-v1.x file.
    `counts`      — copy counts from the underlying `migrate()` call.
    `duration_ms` — wall-clock time for the full operation.
    """

    db_path: Path
    backup_path: Path
    counts: MigrateCounts
    duration_ms: int


def migrate_in_place(
    db_path: Path,
    *,
    backup_suffix: str = IN_PLACE_BACKUP_SUFFIX,
) -> InPlaceMigrateReport:
    """Migrate a pre-v1.x per-project DB in place, preserving a backup.

    Steps:
      1. Validate `db_path` exists and the backup target is free.
      2. Rename `db_path` → `db_path + backup_suffix` (atomic on POSIX).
      3. Read every belief + edge from the backup via the existing
         `migrate()` core (`copy_all=True`, `apply=True`). The target
         is `db_path`, freshly created with the modern schema by
         `MemoryStore`.
      4. Beliefs missing the `origin` column land as `ORIGIN_UNKNOWN`
         (handled by `_read_legacy_beliefs`).

    Raises:
      FileNotFoundError — `db_path` doesn't exist.
      FileExistsError   — backup target already exists; refuses to
                          overwrite to keep prior backups recoverable.

    Used by `aelf doctor` (#593) to silently bring legacy per-project
    DBs forward. Operator-direct invocation is also valid for one-off
    recovery (any caller that owns a legacy per-project DB path).
    """
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    backup_path = db_path.with_name(db_path.name + backup_suffix)
    if backup_path.exists():
        raise FileExistsError(
            f"backup target already exists: {backup_path} — "
            f"refusing to overwrite a prior backup"
        )

    start_ns = time.monotonic_ns()

    # Atomic rename — POSIX guarantees no partial state.
    db_path.rename(backup_path)

    # migrate() requires a project_root, but with copy_all=True the
    # filter is bypassed; any path works.
    report = migrate(
        legacy_path=backup_path,
        target_path=db_path,
        project_root=Path("/"),
        apply=True,
        copy_all=True,
    )

    duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000

    return InPlaceMigrateReport(
        db_path=db_path,
        backup_path=backup_path,
        counts=report.counts,
        duration_ms=duration_ms,
    )
