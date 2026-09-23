#!/usr/bin/env python3
"""Repair onboard-session belief dates and rebuild the temporal spine.

The onboard handshake (`aelfrice.classification.accept_classifications`) can
stamp every belief accepted in one call with a single wall-clock timestamp
instead of each belief's true content date (#1609). When that happens, the
per-session `TEMPORAL_NEXT` chain (`aelfrice.temporal_spine`) encodes scan
order instead of real chronology, even though the project treats that chain
as its most important edge type.

This tool re-dates the affected beliefs from git history and rebuilds the
chain over the corrected dates:

1. For each belief in an onboard session, resolve its `ingest_log.source_path`
   back to a git date. A `git:commit:<sha>` source takes that commit's own
   author date. A `doc:<path>:p<n>` or `ast:<path>:...` source takes the
   path's most recent commit, read from a repo-wide `git log --name-only`
   recency map. A path that map misses because it entered history only
   through a merge (#1612) falls back to a per-path `git log -1 -- <path>`:
   git's own default history simplification then resolves to the merge that
   actually introduced the path, not a later merge that is tree-same to one
   parent for that path.
2. Normalizes every resolved date to UTC, microsecond precision (#1611), so
   plain lexicographic ordering of `created_at` agrees with real time even
   when the source git dates carry mixed local offsets.
3. Deletes the affected session's existing `TEMPORAL_NEXT` edges and
   rebuilds the store's spine with
   `aelfrice.temporal_spine.backfill_temporal_spine`.

Every write goes through `MemoryStore.update_belief`, `MemoryStore.delete_edge`
and `backfill_temporal_spine` -- this tool never writes with raw SQL.

Usage:
    uv run python scripts/repair_onboard_dates.py --db PATH --repo PATH [options]
    uv run python scripts/repair_onboard_dates.py --db PATH --check

Options:
    --db PATH        Path to the aelfrice memory.db to repair. Required.
    --repo PATH      Git work tree to date beliefs against. Required for a
                      repair run. Omit it only when passing --check alone,
                      to verify a store without repairing it.
    --session ID     Restrict the repair to one onboard session id. Default:
                      auto-detect every onboard session in the store -- a
                      session where every belief shares one created_at (the
                      handshake-collapse signature), and whose ingest_log
                      source paths all look like scanner output
                      (`git:commit:<sha>`, `doc:<path>:p<n>`,
                      `ast:<path>:module`, `ast:<path>:func:<name>`, or
                      `ast:<path>:class:<name>`).
    --apply          Write the repair to --db. Without this flag the tool
                      computes and reports what it would change by running
                      the exact same write path against a throwaway scratch
                      copy of --db, and never touches the real file.
    --backup PATH    Copy --db to PATH before making any write. Only takes
                      effect together with --apply.
    --check          Run the read-only spine-order check (does every
                      TEMPORAL_NEXT edge's successor postdate its
                      predecessor in real time?) against the result, and
                      fold it into the exit status. Pass --check alone,
                      with --db and no --repo, to check a store without
                      repairing it.

Exit status: 0 if every requested step succeeded and, when --check ran, no
spine edge was out of real-time order. Non-zero otherwise, including on any
unhandled error.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# Make the repository importable when this script runs from a checkout that
# has not been `pip install -e`'d (mirrors scripts/replay_soak_run.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aelfrice.models import EDGE_TEMPORAL_NEXT  # noqa: E402
from aelfrice.scanner import _build_file_recency_map  # noqa: E402  # pyright: ignore[reportPrivateUsage]
from aelfrice.store import MemoryStore  # noqa: E402
from aelfrice.temporal_spine import backfill_temporal_spine  # noqa: E402

if TYPE_CHECKING:
    from aelfrice.models import Edge

_GIT_LOG_TIMEOUT_SECONDS = 30.0

# Mirrors the source-string shapes `aelfrice.scanner` emits
# (extract_git_log, extract_ast, extract_filesystem).
_DOC_RE = re.compile(r"^doc:(.+):p\d+$")
_AST_MODULE_RE = re.compile(r"^ast:(.+):module$")
_AST_MEMBER_RE = re.compile(r"^ast:(.+):(?:func|class):[^:]+$")
_GIT_RE = re.compile(r"^git:commit:([0-9a-fA-F]+)$")


class RepairError(Exception):
    """Raised to fail the run with a specific, user-facing reason."""


# --- date normalization (#1611) --------------------------------------------


def to_utc_canonical(date_str: str) -> str:
    """Normalize an ISO-8601 timestamp (any offset, or a trailing 'Z') to
    fixed-width UTC, microsecond precision, '+00:00' suffix.

    `backfill_temporal_spine` orders a session's beliefs by the raw
    `created_at` *string*. A git author date (`%aI`) carries a local offset
    (-07:00, +02:00, ...) and second-only precision, so mixed offsets sort
    out of real-time order even when every date is individually correct.
    Converting every date to this one fixed shape makes lexicographic order
    agree with real-time order again.
    """
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_iso(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --- source parsing ----------------------------------------------------------


def parse_source(source: str) -> tuple[str, str] | None:
    """Return ("git", sha) or ("file", rel_path), or None if unrecognized."""
    m = _GIT_RE.match(source)
    if m:
        return ("git", m.group(1))
    for rx in (_DOC_RE, _AST_MODULE_RE, _AST_MEMBER_RE):
        m = rx.match(source)
        if m:
            return ("file", m.group(1))
    return None


def is_scanner_source(source: str) -> bool:
    return parse_source(source) is not None


# --- git date lookups (scanner-equivalent) ----------------------------------


def git_commit_dates(repo: Path) -> dict[str, str]:
    """{7-char short sha: author-date-iso} for every commit in `repo`."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H%x09%aI"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_GIT_LOG_TIMEOUT_SECONDS,
        check=False,
    )
    out: dict[str, str] = {}
    if result.returncode != 0:
        return out
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        full_sha, date = parts
        out.setdefault(full_sha[:7], date)
    return out


def file_recency_map(repo: Path) -> dict[str, str]:
    """Delegates to aelfrice.scanner's own function, so the date-per-file
    logic is byte-identical to what `scan_repo` used at onboard time."""
    return _build_file_recency_map(repo)


def merge_inclusive_file_date(
    repo: Path, rel_path: str, cache: dict[str, str | None]
) -> str | None:
    """Author date of the most recent commit touching `rel_path`, including
    a merge that introduced the path (#1612).

    `_build_file_recency_map` walks a repo-wide `git log --name-only` with
    no `-m`, so a path whose only appearance in history is inside a merge
    commit gets no date there. A plain per-path `git log -1 -- <path>`
    resolves it instead: git's own default history simplification prunes a
    later merge that is tree-same to one parent for that path and keeps the
    merge that actually brought the path in, so this deliberately does not
    pass `-m`/`--diff-merges`, which would surface that later, uninformative
    merge first.
    """
    if rel_path in cache:
        return cache[rel_path]
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=format:%aI", "--", rel_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_GIT_LOG_TIMEOUT_SECONDS,
        check=False,
    )
    date: str | None = None
    if result.returncode == 0:
        stripped = result.stdout.strip()
        date = stripped or None
    cache[rel_path] = date
    return date


def compute_dates(
    repo: Path, mapped: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Returns ({belief_id: utc_iso_date}, {belief_id: reason_undated},
    {belief_id that needed the merge-inclusive fallback})."""
    recency = file_recency_map(repo)
    commit_dates = git_commit_dates(repo)

    dated: dict[str, str] = {}
    undated: dict[str, str] = {}
    pending_fallback: dict[str, str] = {}
    for bid, source in mapped.items():
        parsed = parse_source(source)
        if parsed is None:
            undated[bid] = f"unrecognized source format: {source!r}"
            continue
        kind, key = parsed
        if kind == "git":
            d = commit_dates.get(key[:7])
            if d is None:
                undated[bid] = f"commit {key[:7]} not found in `git log`"
                continue
            dated[bid] = to_utc_canonical(d)
        else:
            d = recency.get(key)
            if d is None:
                pending_fallback[bid] = key
                continue
            dated[bid] = to_utc_canonical(d)

    fallback_cache: dict[str, str | None] = {}
    fallback_used: set[str] = set()
    for bid, rel_path in pending_fallback.items():
        d = merge_inclusive_file_date(repo, rel_path, fallback_cache)
        if d is None:
            undated[bid] = f"file {rel_path!r} has no commit history"
            continue
        dated[bid] = to_utc_canonical(d)
        fallback_used.add(bid)

    return dated, undated, fallback_used


# --- reads against the database (never writes) ------------------------------


def detect_onboard_sessions(db_path: Path) -> list[str]:
    """Read-only. A session qualifies for repair when every belief in it
    shares one `created_at` (the handshake-collapse signature, #1609), and
    its `ingest_log` rows all carry scanner-shaped source paths -- as
    opposed to, say, a session built from `aelf remember` or transcript
    ingest, which this tool has no git-recency logic for."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        candidates = [
            row[0]
            for row in con.execute(
                "SELECT session_id FROM beliefs WHERE session_id IS NOT NULL "
                "GROUP BY session_id HAVING COUNT(DISTINCT created_at) = 1"
            ).fetchall()
        ]
        sessions: list[str] = []
        for session_id in candidates:
            sources = [
                row[0]
                for row in con.execute(
                    "SELECT source_path FROM ingest_log WHERE session_id = ? "
                    "AND source_path IS NOT NULL",
                    (session_id,),
                ).fetchall()
            ]
            if sources and all(is_scanner_source(s) for s in sources):
                sessions.append(session_id)
        return sessions
    finally:
        con.close()


def build_mapping(
    db_path: Path, session_id: str
) -> tuple[list[str], dict[str, str], set[str]]:
    """Returns (session_belief_ids, {belief_id: source_path}, unmapped_ids).

    Read-only. Maps each belief in the session to the `ingest_log` source
    it was derived from. When two rows in the same session resolve to the
    same belief id, the first row in `ingest_log` id order (a ULID, so
    insertion order) is the one that actually created the belief; later
    rows only corroborated it.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT id FROM beliefs WHERE session_id = ?", (session_id,)
        )
        session_belief_ids = [row["id"] for row in cur.fetchall()]
        session_set = set(session_belief_ids)

        cur = con.execute(
            "SELECT id, source_path, derived_belief_ids FROM ingest_log "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        first_source: dict[str, str] = {}
        for row in cur.fetchall():
            raw_ids = row["derived_belief_ids"]
            source_path = row["source_path"]
            if not raw_ids or source_path is None:
                continue
            try:
                ids = json.loads(raw_ids)
            except (TypeError, ValueError):
                continue
            if not isinstance(ids, list):
                continue
            for bid in ids:
                if isinstance(bid, str) and bid not in first_source:
                    first_source[bid] = source_path
    finally:
        con.close()

    mapped = {
        bid: src for bid, src in first_source.items() if bid in session_set
    }
    unmapped = session_set - set(mapped)
    return session_belief_ids, mapped, unmapped


def spine_order_violations(db_path: Path) -> list[tuple[str, str, str, str]]:
    """Read-only. Returns (src, dst, src_created_at, dst_created_at) for
    every TEMPORAL_NEXT edge whose successor (src) predates its predecessor
    (dst) in real time."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        created_at: dict[str, str] = dict(
            con.execute("SELECT id, created_at FROM beliefs").fetchall()
        )
        edges = con.execute(
            "SELECT src, dst FROM edges WHERE type = ?", (EDGE_TEMPORAL_NEXT,)
        ).fetchall()
    finally:
        con.close()

    bad: list[tuple[str, str, str, str]] = []
    for src, dst in edges:
        if src not in created_at or dst not in created_at:
            continue
        if _parse_iso(created_at[src]) < _parse_iso(created_at[dst]):
            bad.append((src, dst, created_at[src], created_at[dst]))
    return bad


# --- writes: MemoryStore.update_belief / delete_edge / backfill only -------


def apply_dates(store: MemoryStore, dated: dict[str, str]) -> tuple[int, int, int]:
    """Writes through `MemoryStore.update_belief`. Returns (changed,
    already_correct, missing_belief_row)."""
    changed = unchanged = missing = 0
    for bid, date in dated.items():
        b = store.get_belief(bid, include_retired=True)
        if b is None:
            missing += 1
            continue
        if b.created_at == date:
            unchanged += 1
            continue
        b.created_at = date
        store.update_belief(b)
        changed += 1
    return changed, unchanged, missing


def rebuild_spine(
    store: MemoryStore, session_belief_ids: list[str]
) -> tuple[int, int]:
    """Deletes the session's own TEMPORAL_NEXT edges through
    `MemoryStore.delete_edge`, then rebuilds the whole store's spine with
    `backfill_temporal_spine`. Returns (edges_deleted, edges_written)."""
    session_set = set(session_belief_ids)
    in_session: list[Edge] = [
        e
        for e in store.edges_for_beliefs(session_belief_ids)
        if e.type == EDGE_TEMPORAL_NEXT
        and e.src in session_set
        and e.dst in session_set
    ]
    for e in in_session:
        store.delete_edge(e.src, e.dst, EDGE_TEMPORAL_NEXT)

    report = backfill_temporal_spine(store, dry_run=False)
    return len(in_session), report.n_edges_written


def copy_db(src_path: Path, dst_path: Path) -> None:
    """WAL-consistent snapshot of `src_path` into a single plain file at
    `dst_path`, via the sqlite backup API -- safe against a live, possibly
    WAL-mode, possibly concurrently written database."""
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(str(dst_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


# --- per-session orchestration ------------------------------------------------


@dataclass
class SessionReport:
    session_id: str
    n_session: int
    n_mapped: int
    n_dated: int
    n_undated_mapped: int
    n_fallback_dated: int
    n_changed: int
    n_already_correct: int
    n_missing_row: int
    edges_deleted: int
    edges_written: int
    ok: bool
    notes: list[str] = field(default_factory=list[str])


def repair_session(
    store: MemoryStore, db_path: Path, repo: Path, session_id: str
) -> SessionReport:
    session_belief_ids, mapped, unmapped = build_mapping(db_path, session_id)
    notes: list[str] = []
    if unmapped:
        notes.append(
            f"{len(unmapped)} belief(s) had no ingest_log mapping; left untouched"
        )

    dated, undated, fallback_used = compute_dates(repo, mapped)
    if undated:
        sample = next(iter(undated.values()))
        notes.append(
            f"{len(undated)} mapped belief(s) could not be dated (e.g. {sample!r})"
        )
    if fallback_used:
        notes.append(
            f"{len(fallback_used)} belief(s) dated via the merge-inclusive fallback"
        )

    changed, unchanged, missing = apply_dates(store, dated)
    if missing:
        notes.append(f"{missing} mapped belief id(s) had no row")

    edges_deleted, edges_written = rebuild_spine(store, session_belief_ids)

    ok = True
    if session_belief_ids and mapped and not dated:
        ok = False
        notes.append("nothing could be dated; spine left as-is")

    return SessionReport(
        session_id=session_id,
        n_session=len(session_belief_ids),
        n_mapped=len(mapped),
        n_dated=len(dated),
        n_undated_mapped=len(undated),
        n_fallback_dated=len(fallback_used),
        n_changed=changed,
        n_already_correct=unchanged,
        n_missing_row=missing,
        edges_deleted=edges_deleted,
        edges_written=edges_written,
        ok=ok,
        notes=notes,
    )


def _print_session_report(report: SessionReport, *, apply: bool) -> None:
    verb = "changed" if apply else "would change"
    print(f"\n=== session {report.session_id} ===")
    print(f"  beliefs in session: {report.n_session}")
    print(f"  mapped via ingest_log: {report.n_mapped}")
    print(
        f"  dated: {report.n_dated} "
        f"(via merge-inclusive fallback: {report.n_fallback_dated}); "
        f"undated-but-mapped: {report.n_undated_mapped}"
    )
    print(f"  {verb}: {report.n_changed}; already correct: {report.n_already_correct}")
    print(
        f"  spine: deleted {report.edges_deleted} in-session edge(s), "
        f"wrote {report.edges_written} edge(s) store-wide"
    )
    for note in report.notes:
        print(f"  note: {note}")


# --- CLI -----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", required=True, type=Path, help="memory.db to repair")
    parser.add_argument(
        "--repo", type=Path, default=None, help="git work tree to date against"
    )
    parser.add_argument(
        "--session", dest="session_id", default=None, help="restrict to one session id"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the repair (default: dry run)"
    )
    parser.add_argument(
        "--backup", type=Path, default=None, help="copy --db here before any write"
    )
    parser.add_argument(
        "--check", action="store_true", help="run the read-only spine-order check"
    )
    return parser


def _run_check(db_path: Path) -> int:
    bad = spine_order_violations(db_path)
    print(
        f"spine-order check: {'PASS' if not bad else 'FAIL'} "
        f"({len(bad)} edge(s) out of real-time order)"
    )
    for src, dst, src_dt, dst_dt in bad[:20]:
        print(f"  {src} -> {dst}: successor {src_dt} predates predecessor {dst_dt}")
    return 1 if bad else 0


def _run_repair(args: argparse.Namespace) -> int:
    repo: Path = args.repo
    if not (repo / ".git").exists():
        raise RepairError(f"--repo is not a git work tree: {repo}")

    if args.session_id:
        session_ids = [args.session_id]
    else:
        session_ids = detect_onboard_sessions(args.db)
        if not session_ids:
            print("no onboard sessions detected; nothing to repair")
            return 0
        print(f"detected {len(session_ids)} onboard session(s)")

    scratch_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.apply:
        if args.backup:
            copy_db(args.db, args.backup)
            print(f"backup written: {args.backup}")
        working_db = args.db
    else:
        scratch_dir = tempfile.TemporaryDirectory(prefix="repair-onboard-dates-")
        working_db = Path(scratch_dir.name) / "scratch.db"
        copy_db(args.db, working_db)
        print("dry run: exercising the real write path against a scratch copy")

    try:
        overall_ok = True
        store = MemoryStore(str(working_db))
        try:
            for session_id in session_ids:
                report = repair_session(store, working_db, repo, session_id)
                _print_session_report(report, apply=args.apply)
                if not report.ok:
                    overall_ok = False
        finally:
            store.close()

        if args.check:
            bad = spine_order_violations(working_db)
            print(
                f"\nspine-order check: {'PASS' if not bad else 'FAIL'} "
                f"({len(bad)} edge(s) out of real-time order)"
            )
            if bad:
                overall_ok = False

        return 0 if overall_ok else 1
    finally:
        if scratch_dir is not None:
            scratch_dir.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.db.exists():
        print(f"error: --db path does not exist: {args.db}", file=sys.stderr)
        return 1

    check_only = args.check and args.repo is None
    if not check_only and args.repo is None:
        print(
            "error: --repo is required for a repair run; pass --check alone "
            "(with no --repo) to only verify a store",
            file=sys.stderr,
        )
        return 1

    try:
        if check_only:
            return _run_check(args.db)
        return _run_repair(args)
    except RepairError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
