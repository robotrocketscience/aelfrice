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
* that belief has a `lock:unlock` or `lock:expire` row in
  `feedback_history` at or after the failed attempt, which is the user
  deliberately ending the lock.

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

import json
import shlex
import sqlite3
from dataclasses import dataclass
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
# the same import-cost reason; pinned to `promotion.SOURCE_LOCK_UNLOCK`
# and `models.FEEDBACK_SOURCE_LOCK_EXPIRE` by a test.
_LOCK_ENDED_SOURCES: Final[tuple[str, str]] = ("lock:unlock", "lock:expire")
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
    def fix_command(self) -> str:
        """The CLI command that applies the lock."""
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
    line from a killed hook must not hide every other row.
    """
    rows: list[dict[str, object]] = []
    rotated = path.with_name(path.name + AUDIT_ROTATED_SUFFIX)
    for p in (rotated, path):
        try:
            text = p.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
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


def _is_resolved(conn: sqlite3.Connection, gap: LockGap) -> bool:
    """Whether the store shows the request applied or deliberately ended."""
    locked = conn.execute(
        "SELECT 1 FROM beliefs WHERE content_hash = ? AND lock_level = ? "
        "AND valid_to IS NULL LIMIT 1",
        (gap.arg_sha256, _LOCK_LEVEL_USER),
    ).fetchone()
    if locked is not None:
        return True
    ended = conn.execute(
        "SELECT 1 FROM feedback_history f JOIN beliefs b ON b.id = f.belief_id "
        "WHERE b.content_hash = ? AND f.source IN (?, ?) AND f.created_at >= ? "
        "LIMIT 1",
        (gap.arg_sha256, *_LOCK_ENDED_SOURCES, gap.ts),
    ).fetchone()
    return ended is not None


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
    rows = read_command_outcomes(outcomes)
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
        open_gaps = [g for g in candidates.values() if not _is_resolved(conn, g)]
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
