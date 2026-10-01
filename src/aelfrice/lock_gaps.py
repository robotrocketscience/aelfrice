"""Typed lock requests that did not take effect and are still unapplied (#1622).

The UserPromptSubmit hook runs a typed `/aelf:lock` itself (#1626) and
reports a failure in that turn. Nothing reported it afterwards, so a
lock the user believed was applied could stay missing with no further
sign. The hook now records each lock's outcome in
`command_outcomes.jsonl`; this module reads those rows back against the
store and returns the requests that are still unapplied. `aelf doctor`
lists them, and SessionStart prints one line while any remains.

What counts as a gap
--------------------
Only the outcomes `exception` and `nonzero_exit`. The input errors
(empty argument, over the length cap, leading `-`) are refused before
anything runs and are reported in the turn; counting them would make
`/aelf:lock --help` nag at every session start.

A gap is resolved from the STORE, not only from a later request:

* a belief with `content_hash == arg_sha256`, `lock_level = 'user'` and
  `valid_to IS NULL` exists, which covers a lock applied later from the
  CLI (which writes no outcome row), or
* a belief carrying the statement has a `feedback_history` row strictly
  after the failed attempt that ends it on purpose. `lock:unlock` always
  counts. `lock:expire` counts only when the belief's `lock_expires_at`
  is also strictly after the failure: the sweep stamps its row when it
  runs, so a lock that lapsed before the failure and was swept after it
  closes nothing, and an expiry with no recorded instant closes nothing.
  `aelf retire`, `aelf delete` and the `/aelf:review` remove verdict
  count only while the belief is still retired or gone: after
  `aelf restore` the statement is back and unlocked, so the gap is open
  again, or
* `ingest_log` maps the statement to a belief that no longer exists, and
  that `ingest_log` row is strictly later than the failed attempt. This
  is a lock applied later from the CLI and then deleted. It survives
  `aelf doctor --gc-orphan-feedback --apply`, which removes the delete's
  own `feedback_history` row. A row at or before the failure may be the
  failed attempt's own ingest, written before the belief was, so it
  closes nothing.

"Strictly after" compares parsed instants at whole-second resolution,
because the outcome row's `ts` is truncated to the second: a removal in
the same second as the failure may have preceded it, and a tie leaves
the gap open. Parsing also makes the formats comparable, since
`lock:expire` and `lock_expires_at` hold an `isoformat()` instant with
microseconds and `+00:00` where every other writer stamps
`%Y-%m-%dT%H:%M:%SZ`.

`aelf delete` removes the `beliefs` row, so `content_hash` alone cannot
find a deleted belief. The detector also reads the `cli_remember` rows
of `ingest_log`, which is append-only and records the statement and the
belief id every `aelf lock` resolved to, and hashes their text.

`arg_sha256` matches `content_hash` because `aelf lock` derives the
belief through `derivation._content_hash`, a plain sha256 of the
statement text; `tests/test_lock_gaps_1622.py` pins that.

Forward-only: rows exist only for requests typed after this shipped.
Earlier failures carry no outcome and are not reported.

Read-only throughout. The store is opened with a `mode=ro` URI, never
through `MemoryStore`, whose constructor writes (DDL, migrations).
Imports only the standard library and `hook_audit`, so the SessionStart
hook pays nothing for it beyond the file read.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, cast

from aelfrice.hook_audit import (
    AUDIT_FILENAME,
    AUDIT_ROTATED_SUFFIX,
    COMMAND_OUTCOME_HOOK,
    command_outcomes_path_for_db,
)

GAP_REASONS: Final[frozenset[str]] = frozenset({"exception", "nonzero_exit"})
"""Outcome reasons that leave an unapplied request.

Spelled as strings, not imported from `aelfrice.hook.CommandReason`,
because importing the hook here would put the retrieval stack on the
doctor and SessionStart path. A test pins the two together.
"""

# `feedback_history.source` values that end a lock on purpose. Literal for
# the same import-cost reason; pinned to `promotion.SOURCE_LOCK_UNLOCK`,
# `models.FEEDBACK_SOURCE_LOCK_EXPIRE`, the literals `aelf retire` and
# `aelf delete` write in `cli.py`, and the one the `/aelf:review` remove
# verdict writes in `review.py`, by tests.
_SOURCE_LOCK_UNLOCK: Final[str] = "lock:unlock"
_SOURCE_LOCK_EXPIRE: Final[str] = "lock:expire"
_LOCK_DROPPED_SOURCES: Final[tuple[str, ...]] = (
    _SOURCE_LOCK_UNLOCK, _SOURCE_LOCK_EXPIRE,
)
"""End the lock and leave the statement in place.

An unlock closes a gap when its row is later than the failure. An expiry
closes one only when the belief's `lock_expires_at` is: the sweep stamps
its row when it runs, which can be long after the lock lapsed.
"""

_BELIEF_REMOVED_SOURCES: Final[tuple[str, ...]] = (
    "user_retired",
    "user_retired_force",
    "user_deleted",
    "user_deleted_force",
    "review:remove",
)
"""Remove the statement. Close a gap only while it stays removed."""

_LOCK_ENDED_SOURCES: Final[tuple[str, ...]] = (
    _LOCK_DROPPED_SOURCES + _BELIEF_REMOVED_SOURCES
)
# `ingest_log.source_kind` of an `aelf lock`; pinned to
# `models.INGEST_SOURCE_CLI_REMEMBER` by a test.
_INGEST_SOURCE_LOCK: Final[str] = "cli_remember"
_LOCK_LEVEL_USER: Final[str] = "user"


@dataclass(frozen=True)
class LockGap:
    """One statement whose typed lock failed and that is still not locked."""

    arg_sha256: str
    statement: str
    arg_len: int
    reason: str
    ts: str
    session_id: str | None
    attempts: int

    @property
    def truncated(self) -> bool:
        """Whether the row holds only a prefix of the statement."""
        return self.arg_len > len(self.statement)

    @property
    def valid_text(self) -> bool:
        """Whether the statement is text a lock could store."""
        try:
            self.statement.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True

    @property
    def display_statement(self) -> str:
        """The statement, printable on a strict UTF-8 stream.

        A lone surrogate is shown as its `\\udXXX` escape. Printed raw,
        it raises from `print` and takes the whole doctor run with it.
        """
        return self.statement.encode(
            "utf-8", errors="backslashreplace"
        ).decode("utf-8")

    @property
    def fix_command(self) -> str | None:
        """The CLI command that applies the lock, or None.

        None when the row holds only a prefix of the statement. Running
        a command built from the prefix would lock text the user never
        typed, at user tier, and the gap would stay open because the
        hash differs. None also when the statement is not valid text,
        because no command can store it.
        """
        if self.truncated or not self.valid_text:
            return None
        return f"aelf lock {shlex.quote(self.statement)}"


@dataclass(frozen=True)
class LockGapReport:
    """The detector's answer. `known` False means it could not tell.

    "Unknown" is reported, never folded into zero: a disabled audit or
    an unreadable store would otherwise read as "nothing failed".
    """

    known: bool
    gaps: tuple[LockGap, ...] = ()
    unknown_reason: str | None = None
    records_seen: int = 0
    first_ts: str | None = None


def read_command_outcomes(path: Path) -> list[dict[str, object]]:
    """Return the command-outcome rows from `path` and its rotated slot.

    Oldest first: the `.1` slot, then the live file. A line that is not
    a JSON object is skipped rather than raised on, because a torn final
    line from a killed hook must not hide every other row. The same holds
    for a line that is not valid UTF-8 and for one nested too deeply to
    parse (`RecursionError`): each line is decoded and parsed on its own,
    so one bad line costs only itself.

    A file that is absent is skipped. Any other failure to read a file
    (a directory at the path, no read permission) raises `OSError`;
    `detect_lock_gaps` reports that as unknown rather than as zero.
    """
    rows: list[dict[str, object]] = []
    rotated = path.with_name(path.name + AUDIT_ROTATED_SUFFIX)
    for p in (rotated, path):
        try:
            data = p.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            continue
        for raw in data.splitlines():
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if not isinstance(parsed, dict):
                continue
            row = cast(dict[str, object], parsed)
            if row.get("hook") == COMMAND_OUTCOME_HOOK:
                rows.append(row)
    return rows


def _failed_lock_candidates(
    rows: list[dict[str, object]],
) -> dict[str, LockGap]:
    """Collapse failed lock rows to one candidate per statement hash.

    Keeps the latest attempt's timestamp: an unlock is "later" only if it
    follows the most recent failure, not merely the first.
    """
    out: dict[str, LockGap] = {}
    for r in rows:
        if r.get("command") != "lock" or r.get("reason") not in GAP_REASONS:
            continue
        digest = r.get("arg_sha256")
        ts = r.get("ts")
        if not isinstance(digest, str) or not isinstance(ts, str):
            continue
        statement = r.get("statement")
        arg_len = r.get("arg_len")
        sid = r.get("session_id")
        prev = out.get(digest)
        out[digest] = LockGap(
            arg_sha256=digest,
            statement=statement if isinstance(statement, str) else "",
            arg_len=arg_len if isinstance(arg_len, int) else 0,
            reason=str(r.get("reason")),
            ts=ts if prev is None or ts >= prev.ts else prev.ts,
            session_id=sid if isinstance(sid, str) else None,
            attempts=1 if prev is None else prev.attempts + 1,
        )
    return out


def _parse_ts(value: object) -> datetime | None:
    """Parse a stored timestamp to a UTC instant truncated to the second.

    Accepts both stored forms: `%Y-%m-%dT%H:%M:%SZ` and `isoformat()`
    with microseconds and `+00:00`. Truncated because the outcome row's
    `ts` is, so an instant inside the failure's second is not "after" it.
    None for anything unparseable, which then closes nothing.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _strictly_after(value: object, failed_at: datetime) -> bool:
    """Whether `value` is a stored instant in a later second than the failure."""
    parsed = _parse_ts(value)
    return parsed is not None and parsed > failed_at


def _statement_belief_ids(
    conn: sqlite3.Connection, digests: set[str],
) -> dict[str, dict[str, list[str]]]:
    """Map each statement hash to every belief id that has carried it.

    The inner value lists the `ingest_log.ts` of each `aelf lock` that
    resolved to that id; it is empty for an id found only in `beliefs`.

    Two sources. `beliefs.content_hash` covers a belief that still exists,
    including a retired one. `ingest_log` covers one `aelf delete`
    removed: it is append-only and records, for every `aelf lock`, the
    statement and the belief id the lock resolved to. Its text is hashed
    here because the log stores no hash. The `source_kind` filter uses
    the log's index and keeps the scan to `aelf lock` and `aelf remember`
    rows.
    """
    out: dict[str, dict[str, list[str]]] = {d: {} for d in digests}
    for digest in digests:
        for (bid,) in conn.execute(
            "SELECT id FROM beliefs WHERE content_hash = ?", (digest,),
        ):
            out[digest].setdefault(str(bid), [])
    for raw_text, derived, ts in conn.execute(
        "SELECT raw_text, derived_belief_ids, ts FROM ingest_log "
        "WHERE source_kind = ?",
        (_INGEST_SOURCE_LOCK,),
    ):
        if not isinstance(raw_text, str) or not isinstance(derived, str):
            continue
        digest = hashlib.sha256(
            raw_text.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        if digest not in out:
            continue
        try:
            ids = json.loads(derived)
        except json.JSONDecodeError:
            continue
        if isinstance(ids, list):
            for i in cast(list[object], ids):
                stamps = out[digest].setdefault(str(i), [])
                if isinstance(ts, str):
                    stamps.append(ts)
    return out


def _is_resolved(
    conn: sqlite3.Connection, gap: LockGap, belief_ids: dict[str, list[str]],
) -> bool:
    """Whether the store shows the request applied or deliberately ended."""
    locked = conn.execute(
        "SELECT 1 FROM beliefs WHERE content_hash = ? AND lock_level = ? "
        "AND valid_to IS NULL LIMIT 1",
        (gap.arg_sha256, _LOCK_LEVEL_USER),
    ).fetchone()
    if locked is not None:
        return True
    failed_at = _parse_ts(gap.ts)
    if failed_at is None:
        return False
    marks = ", ".join("?" for _ in _LOCK_ENDED_SOURCES)
    for bid in sorted(belief_ids):
        row = conn.execute(
            "SELECT valid_to, lock_expires_at FROM beliefs WHERE id = ?",
            (bid,),
        ).fetchone()
        exists = row is not None
        removed = not exists or row[0] is not None
        # The sweep keeps `lock_expires_at` as its audit trace. Unknown
        # for a deleted belief, and then an expiry closes nothing.
        lapsed_after = exists and _strictly_after(row[1], failed_at)
        if not exists and any(
            _strictly_after(t, failed_at) for t in belief_ids[bid]
        ):
            # Locked from the CLI after the failure, then deleted. The
            # delete's feedback row may have been orphan-GC'd since.
            return True
        for source, created_at in conn.execute(
            f"SELECT source, created_at FROM feedback_history "
            f"WHERE belief_id = ? AND source IN ({marks})",
            (bid, *_LOCK_ENDED_SOURCES),
        ):
            if not _strictly_after(created_at, failed_at):
                continue
            if source == _SOURCE_LOCK_EXPIRE:
                if lapsed_after:
                    return True
                continue
            if source == _SOURCE_LOCK_UNLOCK or removed:
                return True
    return False


def detect_lock_gaps(store_path: str, *, audit_enabled: bool) -> LockGapReport:
    """Return the unapplied lock requests for the store at `store_path`.

    `audit_enabled` is the resolved `[hook_audit]` switch. When it is off
    the hook writes no outcome rows, so the answer is unknown. It is also
    unknown when the hook has left no audit trail at all beside the store
    (neither the outcome log nor `hook_audit.jsonl`), and when the store
    exists but cannot be read.

    A store file that does not exist resolves nothing: every failed
    request is a gap, because nothing can have been locked.
    """
    if not audit_enabled:
        return LockGapReport(
            known=False,
            unknown_reason=(
                "the hook audit is disabled (AELFRICE_HOOK_AUDIT=0 or "
                "[hook_audit] enabled = false), so lock outcomes are not "
                "recorded"
            ),
        )
    if store_path == ":memory:":
        return LockGapReport(known=False, unknown_reason="in-memory store")
    db = Path(store_path)
    outcomes = command_outcomes_path_for_db(db)
    main_audit = db.parent / AUDIT_FILENAME
    if not outcomes.exists() and not main_audit.exists():
        return LockGapReport(
            known=False,
            unknown_reason=(
                f"no hook audit found in {db.parent}; the hook has not "
                f"recorded anything for this store"
            ),
        )
    try:
        rows = read_command_outcomes(outcomes)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        # Reported, not raised: an unreadable log must not take the
        # doctor run or the SessionStart notice down with it.
        return LockGapReport(
            known=False,
            unknown_reason=(
                f"the outcome log could not be read: {type(exc).__name__}"
            ),
        )
    first_ts = next(
        (r["ts"] for r in rows if isinstance(r.get("ts"), str)), None,
    )
    candidates = _failed_lock_candidates(rows)
    if not candidates or not db.exists():
        return LockGapReport(
            known=True,
            gaps=tuple(sorted(candidates.values(), key=lambda g: g.ts)),
            records_seen=len(rows),
            first_ts=first_ts if isinstance(first_ts, str) else None,
        )
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ids = _statement_belief_ids(conn, set(candidates))
        open_gaps = [
            g for g in candidates.values()
            if not _is_resolved(conn, g, ids[g.arg_sha256])
        ]
    except sqlite3.Error as exc:
        return LockGapReport(
            known=False,
            unknown_reason=f"the store could not be read: {exc}",
            records_seen=len(rows),
        )
    finally:
        if conn is not None:
            conn.close()
    return LockGapReport(
        known=True,
        gaps=tuple(sorted(open_gaps, key=lambda g: g.ts)),
        records_seen=len(rows),
        first_ts=first_ts if isinstance(first_ts, str) else None,
    )


def session_start_notice(report: LockGapReport) -> str | None:
    """The one SessionStart line while any gap remains, else None."""
    if not report.known or not report.gaps:
        return None
    n = len(report.gaps)
    noun = "request" if n == 1 else "requests"
    return (
        f"aelfrice: {n} /aelf:lock {noun} failed and "
        f"{'is' if n == 1 else 'are'} still not locked"
        f" — `/aelf:doctor` lists the statements and the fix."
    )
