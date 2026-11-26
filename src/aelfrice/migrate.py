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

import dataclasses
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aelfrice.db_paths import repo_identity_from_db_path
from aelfrice.models import (
    BELIEF_SCOPE_PROJECT,
    CORROBORATION_SOURCE_CONSOLIDATION_MIGRATION,
    CORROBORATION_SOURCE_TYPES,
    CORROBORATION_SPEAKER_USER,
    CORROBORATION_SPEAKERS,
    DEFAULT_LOCK_TIER,
    LOCK_TIERS,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_STATED,
    RETENTION_CLASSES,
    RETENTION_UNKNOWN,
    Belief,
    Edge,
)
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
    # #1911: corroboration rows of the inserted beliefs. An applied run
    # counts the rows that landed in the target, a dry run the rows it
    # would copy. A row whose source type or speaker this version does not
    # know is skipped and counted, and the rest of the migrate goes on.
    copied_corroborations: int = 0
    skipped_unknown_corroborations: int = 0
    # #1911: retention_class values of the inserted beliefs, and lock_tier
    # values of the inserted locked beliefs, that this version does not
    # know. Each one is written as `unknown` or the default lock tier, as
    # before #1911, so the belief still copies. An unlocked belief gets
    # the default tier too, but it is not counted.
    normalized_unknown_values: int = 0


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


_BELIEF_FIELDS: Final[tuple[str, ...]] = tuple(
    f.name for f in dataclasses.fields(Belief)
)


def _read_legacy_beliefs(
    conn: sqlite3.Connection, *, source_identity: str = "",
) -> tuple[list[Belief], dict[str, int]]:
    """Read beliefs from a source DB, preserving project_context provenance.

    Returns the beliefs and, per belief id, how many of its
    retention_class and lock_tier values this version does not know.
    Those values are replaced with `unknown` and the default lock tier,
    so `insert_belief` accepts the row (#1911).

    #970: a migrate must not collapse every row to project_context=''. An
    already-stamped source value is carried over verbatim. A source row
    still at '' (pre-#970 data) is stamped with `source_identity` — the
    repo identity of the *source* store — when it is project-scope and not
    user-locked, so the merged store can still tell which repo each belief
    came from. Locked/global rows stay '' (cross-context). `source_identity`
    is '' for a home-dir / non-repo source, which leaves rows '' as before.
    """
    cur = conn.execute("SELECT * FROM beliefs")
    rows = cur.fetchall()
    keys = set(rows[0].keys()) if rows else set()
    has_origin = "origin" in keys
    has_scope = "scope" in keys
    has_project_context = "project_context" in keys
    out: list[Belief] = []
    normalized: dict[str, int] = {}
    for row in rows:
        scope = row["scope"] if has_scope else BELIEF_SCOPE_PROJECT
        lock_level = row["lock_level"]
        pc = row["project_context"] if has_project_context else ""
        if (
            pc == ""
            and source_identity
            and scope == BELIEF_SCOPE_PROJECT
            and lock_level != LOCK_USER
        ):
            pc = source_identity
        # #1135: the v1.2 origin catch-up (promotion_path.md § 1) used
        # to be re-applied by the target store on every open; it is a
        # marker-gated one-shot now, and the target's marker is already
        # stamped by the time legacy rows land. Apply the same mapping
        # during conversion instead, so migrate output matches the
        # backfill: a still-'unknown' locked row was user-stated.
        # #2013: a still-'unknown' correction row keeps 'unknown',
        # because its type does not show that the user made it.
        origin = row["origin"] if has_origin else ORIGIN_UNKNOWN
        if origin == ORIGIN_UNKNOWN and lock_level == LOCK_USER:
            origin = ORIGIN_USER_STATED
        # #1911: every Belief field the source row has comes across. A
        # column an older source schema lacks takes the dataclass default.
        fields = {name: row[name] for name in _BELIEF_FIELDS if name in keys}
        fields.update(
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            origin=origin,
            scope=scope,
            project_context=pc,
        )
        # A source written by a newer version, or edited by hand, can hold
        # a value `insert_belief` refuses. Normalize it so the migrate
        # copies the belief instead of stopping at this row.
        n = 0
        if fields.get("retention_class", RETENTION_UNKNOWN) not in RETENTION_CLASSES:
            fields["retention_class"] = RETENTION_UNKNOWN
            n += 1
        if fields.get("lock_tier", DEFAULT_LOCK_TIER) not in LOCK_TIERS:
            fields["lock_tier"] = DEFAULT_LOCK_TIER
            # A tier takes effect only on a locked belief, so only there
            # does the replaced value change anything.
            if lock_level == LOCK_USER:
                n += 1
        if n:
            normalized[row["id"]] = n
        out.append(Belief(**fields))
    return out, normalized


def _read_legacy_edges(conn: sqlite3.Connection) -> list[Edge]:
    cur = conn.execute("SELECT * FROM edges")
    rows = cur.fetchall()
    has_anchor = bool(rows) and "anchor_text" in rows[0].keys()
    return [
        Edge(
            src=row["src"],
            dst=row["dst"],
            type=row["type"],
            weight=float(row["weight"]),
            anchor_text=row["anchor_text"] if has_anchor else None,
        )
        for row in rows
    ]


@dataclass(frozen=True)
class _CorroborationRow:
    belief_id: str
    ingested_at: str
    source_type: str
    session_id: str | None
    source_path_hash: str | None
    speaker: str | None


def _read_legacy_corroborations(
    conn: sqlite3.Connection,
) -> list[_CorroborationRow]:
    """Read `belief_corroborations` rows, oldest first (#1911).

    A source older than the table (v1.5.0, #190) has none. A source older
    than the `speaker` column reads every row with speaker None.
    `corroboration_count` is computed from these rows, so copying them is
    how the count reaches the target.
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'belief_corroborations'"
    ).fetchone()
    if has_table is None:
        return []
    cur = conn.execute("SELECT * FROM belief_corroborations ORDER BY id")
    rows = cur.fetchall()
    has_speaker = bool(rows) and "speaker" in rows[0].keys()
    return [
        _CorroborationRow(
            belief_id=row["belief_id"],
            ingested_at=row["ingested_at"],
            source_type=row["source_type"],
            session_id=row["session_id"],
            source_path_hash=row["source_path_hash"],
            speaker=row["speaker"] if has_speaker else None,
        )
        for row in rows
    ]


def _is_known_corroboration(c: _CorroborationRow) -> bool:
    """True when `record_corroboration` accepts this row's values.

    A source written by a newer version, or edited by hand, can hold a
    source type or speaker this version does not know. Checking every row
    before the first write lets migrate skip those rows instead of
    stopping partway with a `ValueError`.
    """
    return c.source_type in CORROBORATION_SOURCE_TYPES and (
        c.speaker is None or c.speaker in CORROBORATION_SPEAKERS
    )


def migrate(
    *,
    legacy_path: Path,
    target_path: Path,
    project_root: Path,
    apply: bool,
    copy_all: bool,
) -> MigrateReport:
    """Copy beliefs, their corroboration rows, and their incident edges
    from `legacy_path` into `target_path`.

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
    # #970: stamp source-'' project rows with the SOURCE repo identity so
    # provenance survives the merge instead of collapsing to ''. '' for a
    # home-dir / non-repo source.
    source_identity = repo_identity_from_db_path(legacy_path)
    try:
        legacy_beliefs, normalized_by_belief = _read_legacy_beliefs(
            legacy_conn, source_identity=source_identity,
        )
        legacy_edges = _read_legacy_edges(legacy_conn)
        legacy_corroborations = _read_legacy_corroborations(legacy_conn)
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
    corroborations_by_belief: dict[str, list[_CorroborationRow]] = {}
    unknown_corroborations: dict[str, int] = {}
    # A source from before v3.8.0 can hold duplicate rows for one source.
    # `record_corroboration` keeps the first row per this key, the key of
    # `uq_belief_corroborations_source`, so the dry run counts it once too.
    # That index exempts consolidation_migration rows: #219 writes one per
    # merged duplicate, so every one of them copies and counts. A user
    # write wins, as in `record_corroboration`: a duplicate from the user
    # makes the kept row the user's.
    kept_at: dict[tuple[str, str, str, str], int] = {}
    for c in legacy_corroborations:
        if _is_known_corroboration(c):
            kept = corroborations_by_belief.setdefault(c.belief_id, [])
            if c.source_type != CORROBORATION_SOURCE_CONSOLIDATION_MIGRATION:
                key = (c.belief_id, c.session_id or "",
                       c.source_path_hash or "", c.source_type)
                if key in kept_at:
                    if c.speaker == CORROBORATION_SPEAKER_USER:
                        i = kept_at[key]
                        kept[i] = dataclasses.replace(kept[i], speaker=c.speaker)
                    continue
                kept_at[key] = len(kept)
            kept.append(c)
        else:
            unknown_corroborations[c.belief_id] = (
                unknown_corroborations.get(c.belief_id, 0) + 1
            )

    inserted_beliefs = 0
    skipped_existing = 0
    inserted_edges = 0
    skipped_orphan_edges = 0
    copied_corroborations = 0
    skipped_unknown_corroborations = 0
    normalized_unknown_values = 0

    if apply:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = MemoryStore(str(target_path))
        try:
            for b in matched:
                # include_retired (#1210): an id-existence check against the target.
                # A tombstone still owns its primary key, so treating it as absent
                # would re-insert and trip a UNIQUE constraint.
                if target.get_belief(b.id, include_retired=True) is not None:
                    skipped_existing += 1
                    continue
                # #1911: one transaction per belief. A run that stops
                # partway leaves each belief either absent from the target
                # or present with its retire and its corroboration rows, so
                # a rerun copies the rest. Without it, a rerun skipped a
                # belief left active or left without its rows.
                with target.transaction():
                    if b.valid_to is None:
                        target.insert_belief(b)
                    else:
                        # A retired belief arrives retired, with the same
                        # valid_to. Retiring through the store also prunes
                        # the FTS row, as it does for the source row.
                        target.insert_belief(
                            dataclasses.replace(b, valid_to=None),
                        )
                        target.soft_delete_belief(b.id, ts=b.valid_to)
                    # A copied belief keeps its corroboration rows. A belief
                    # the target already held is skipped above and keeps
                    # the target's own rows.
                    for c in corroborations_by_belief.get(b.id, ()):
                        target.record_corroboration(
                            c.belief_id, source_type=c.source_type,
                            session_id=c.session_id,
                            source_path_hash=c.source_path_hash,
                            ts=c.ingested_at, speaker=c.speaker,
                        )
                inserted_beliefs += 1
                copied_corroborations += target.count_corroborations(b.id)
                skipped_unknown_corroborations += unknown_corroborations.get(
                    b.id, 0,
                )
                normalized_unknown_values += normalized_by_belief.get(b.id, 0)

            # Copy edges only when both endpoints landed in the target.
            # Edges that reference an unmatched legacy belief are
            # skipped — they'd create orphan edges in the target.
            target_existing_ids = {
                b.id for b in matched if target.get_belief(b.id, include_retired=True) is not None
            }
            valid_endpoints = target_existing_ids | matched_ids
            for e in legacy_edges:
                if e.src in valid_endpoints and e.dst in valid_endpoints:
                    if target.get_edge(e.src, e.dst, e.type) is not None:
                        continue
                    if target.insert_edge(e):
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
                would_insert = [
                    b for b in matched
                    if target.get_belief(b.id, include_retired=True) is None
                ]
            finally:
                target.close()
        else:
            would_insert = matched
        inserted_beliefs = len(would_insert)
        skipped_existing = len(matched) - inserted_beliefs
        for b in would_insert:
            copied_corroborations += len(corroborations_by_belief.get(b.id, ()))
            skipped_unknown_corroborations += unknown_corroborations.get(
                b.id, 0,
            )
            normalized_unknown_values += normalized_by_belief.get(b.id, 0)
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
        copied_corroborations=copied_corroborations,
        skipped_unknown_corroborations=skipped_unknown_corroborations,
        normalized_unknown_values=normalized_unknown_values,
    )
    return MigrateReport(
        legacy_path=legacy_path,
        target_path=target_path,
        counts=counts,
        applied=apply,
    )


def copy_belief(target: MemoryStore, belief: Belief) -> None:
    """Insert `belief` into `target` under its own id.

    For a copy between two stores, such as `aelf store split` (#1796).
    The insert runs here, in a module the write-log gate allows to call
    `insert_belief` directly, because a copy has no ingest row of its own
    to derive the belief from. The caller checks first that neither the
    id nor the content hash is in `target`.
    """
    target.insert_belief(belief)


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
