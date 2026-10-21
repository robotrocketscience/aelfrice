"""Retire beliefs that only scheduled prompts produced (#1780).

The host replays a scheduled continuation prompt as a user turn. Before
#1780 the transcript logger wrote those prompts to `turns.jsonl`, and
ingest stored them as the user's beliefs. The host flags the prompt's
record in its own session transcript (`isMeta`, `turnOrigin:
"scheduled"`), so this pass reads those files to find the turns.

The join key is `(session_id, turn_sha)`, computed by
`ingest.user_turn_key` the same way ingest stamps it on each
`ingest_log` row. A key that the same session also holds for an
unflagged record is ambiguous, because a typed prompt with the same text
produced the same key, and it is dropped. A belief is retired only when
every `ingest_log` row that derived or corroborated it carries a flagged
key. Locked beliefs are never retired.

`aelf doctor --retire-scheduled-prompts PATH...` runs the pass. It is a
dry run unless `--apply` is given.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from aelfrice.ingest import user_turn_key
from aelfrice.models import LOCK_USER
from aelfrice.transcript_logger import is_scheduled_host_record

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

TurnKey = tuple[str, str]


@dataclass
class HostTurnScan:
    """Turn keys read from host session files."""

    files_read: int = 0
    unreadable: list[str] = field(default_factory=list[str])
    flagged_records: int = 0
    flagged: set[TurnKey] = field(default_factory=set[TurnKey])
    ambiguous: set[TurnKey] = field(default_factory=set[TurnKey])


@dataclass
class ScheduledPromptReport:
    """Result of `retire_scheduled_prompt_beliefs`."""

    dry_run: bool
    scan: HostTurnScan
    candidates: list[tuple[str, str]] = field(
        default_factory=list[tuple[str, str]],
    )
    locked: list[str] = field(default_factory=list[str])
    retired: int = 0


def host_session_files(paths: Iterable[Path]) -> list[Path]:
    """Each file as given, and each directory's `*.jsonl` files, recursively.

    A directory is walked the way `aelf ingest-transcript --batch` walks
    one. The result is deduplicated and sorted.
    """
    out: set[Path] = set()
    for p in paths:
        if p.is_dir():
            out.update(q for q in p.rglob("*.jsonl") if q.is_file())
        else:
            out.add(p)
    return sorted(out)


def scan_host_sessions(paths: Iterable[Path]) -> HostTurnScan:
    """Collect the flagged turn keys from host session JSONL files.

    A record counts as flagged when `is_scheduled_host_record` says so,
    the same test the transcript logger applies. A line that is not a
    JSON object is skipped. A file that cannot be read is listed in
    `unreadable` and skipped.
    """
    scan = HostTurnScan()
    flagged: set[TurnKey] = set()
    unflagged: set[TurnKey] = set()
    for path in host_session_files(paths):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        obj = json.loads(raw)  # pyright: ignore[reportAny]
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    rec = cast(dict[str, object], obj)
                    if rec.get("type") != "user":
                        continue
                    key = user_turn_key(rec)
                    if key is None:
                        continue
                    if is_scheduled_host_record(rec):
                        scan.flagged_records += 1
                        flagged.add(key)
                    else:
                        unflagged.add(key)
        except OSError:
            scan.unreadable.append(str(path))
            continue
        scan.files_read += 1
    scan.ambiguous = flagged & unflagged
    scan.flagged = flagged - unflagged
    return scan


def _select(
    store: "MemoryStore", flagged: set[TurnKey],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Active beliefs whose every deriving row is flagged, split by lock."""
    every_row_flagged: dict[str, bool] = {}
    for bid, sid, sha in store.list_ingest_turn_derivations():
        hit = sid is not None and sha is not None and (sid, sha) in flagged
        every_row_flagged[bid] = every_row_flagged.get(bid, True) and hit
    candidates: list[tuple[str, str]] = []
    locked: list[str] = []
    for bid in sorted(b for b, ok in every_row_flagged.items() if ok):
        belief = store.get_belief(bid)
        if belief is None:
            continue
        if belief.lock_level == LOCK_USER:
            locked.append(bid)
            continue
        candidates.append((bid, belief.content))
    return candidates, locked


def retire_scheduled_prompt_beliefs(
    store: "MemoryStore", paths: Iterable[Path], *, dry_run: bool = True,
) -> ScheduledPromptReport:
    """List, and with `dry_run=False` retire, beliefs only scheduled turns made.

    Retiring sets `valid_to` (`MemoryStore.soft_delete_belief`), so
    `aelf restore <id>` undoes it. The dry run writes nothing.
    """
    scan = scan_host_sessions(paths)
    report = ScheduledPromptReport(dry_run=dry_run, scan=scan)
    if not scan.flagged:
        return report
    if dry_run:
        report.candidates, report.locked = _select(store, scan.flagged)
        return report
    with store.transaction(immediate=True):
        report.candidates, report.locked = _select(store, scan.flagged)
        for bid, _ in report.candidates:
            store.soft_delete_belief(bid)
    report.retired = len(report.candidates)
    return report


def format_scheduled_prompt_report(report: ScheduledPromptReport) -> str:
    """Human-readable rendering of `retire_scheduled_prompt_beliefs`."""
    scan = report.scan
    lines = [
        f"host session files read: {scan.files_read}",
        f"scheduled or meta user records: {scan.flagged_records} "
        f"({len(scan.flagged)} turn(s); {len(scan.ambiguous)} skipped "
        "because a typed record in the same session has the same text)",
    ]
    for path in scan.unreadable:
        lines.append(f"unreadable, skipped: {path}")
    verb = "would retire" if report.dry_run else "retired"
    lines.append(
        f"beliefs derived only from those turns: {len(report.candidates)}"
    )
    for bid, content in report.candidates:
        lines.append(f"  {verb} {bid}  {content.strip()[:72]}")
    if report.locked:
        lines.append(f"locked, kept: {len(report.locked)}")
        for bid in report.locked:
            lines.append(f"  {bid}")
    if report.dry_run:
        if report.candidates:
            lines.append(
                "dry run: re-run with --apply to retire them "
                "(`aelf restore <id>` undoes one)."
            )
        else:
            lines.append("nothing to do.")
    else:
        lines.append(f"retired {report.retired} belief(s).")
    return "\n".join(lines)
