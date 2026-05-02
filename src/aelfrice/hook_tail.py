"""Reader + pretty-printer for the per-turn hook audit log (#321).

`aelf tail` is the live observability primitive over `hook_audit.jsonl`
(the per-turn audit log shipped in #280 mitigation 3, extended in #321
with `beliefs[]`, `latency_ms`, and `tokens` fields).

Why this layer is separate from `hook.py`: the writer side runs inside
the UserPromptSubmit / SessionStart hook process and must stay
import-cheap (every Claude Code prompt pays its import cost). The
reader side is interactive and only loads when the operator runs
`aelf tail`. Splitting keeps hook startup unburdened by formatting
helpers and reduces the surface area that the hook's non-blocking
contract has to defend.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Final, Iterable

from aelfrice.hook import (
    AUDIT_FILENAME,
    AUDIT_ROTATED_SUFFIX,
    _audit_path_for_db,
)


_FILTER_KEYS: Final[frozenset[str]] = frozenset({"hook", "lane"})
"""Keys accepted by --filter. `hook` matches the record-level value;
`lane` matches at least one belief in the record's beliefs[] array."""

_SINCE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<n>\d+)\s*(?P<unit>[smhd])\s*$"
)
"""Relative `--since` durations: `30s`, `5m`, `2h`, `1d`."""

_TAIL_POLL_INTERVAL: Final[float] = 0.25
"""Seconds between follow-mode polls when no new lines arrive."""


def parse_filter(spec: str) -> tuple[str, str]:
    """Parse a single `--filter key=value` argument.

    Raises ValueError if the spec is malformed or the key is unknown.
    Caller catches and reports; the CLI converts ValueError to an
    error exit.
    """
    if "=" not in spec:
        raise ValueError(
            f"--filter must look like key=value (got {spec!r})"
        )
    key, _, value = spec.partition("=")
    key = key.strip()
    value = value.strip()
    if key not in _FILTER_KEYS:
        raise ValueError(
            f"--filter key {key!r} not recognized "
            f"(supported: {sorted(_FILTER_KEYS)})"
        )
    if not value:
        raise ValueError(f"--filter {key}= has no value")
    return key, value


def parse_since(spec: str) -> timedelta:
    """Parse a `--since` relative duration like `5m` or `2h`.

    Returns a `timedelta`. Raises ValueError on malformed input.
    """
    m = _SINCE_RE.match(spec)
    if m is None:
        raise ValueError(
            f"--since must look like Ns / Nm / Nh / Nd (got {spec!r})"
        )
    n = int(m.group("n"))
    unit = m.group("unit")
    if unit == "s":
        return timedelta(seconds=n)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(days=n)


def _parse_record_ts(record: dict[str, object]) -> datetime | None:
    """Best-effort parse of the `ts` field. Returns None if missing
    or unparseable."""
    raw = record.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        # Audit writes `%Y-%m-%dT%H:%M:%SZ` — Python's fromisoformat
        # in 3.11+ accepts trailing Z directly.
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def record_matches_filters(
    record: dict[str, object], filters: list[tuple[str, str]]
) -> bool:
    """Return True iff the record matches all (key, value) filters.

    `hook=...` matches `record["hook"]` exactly. `lane=...` matches
    iff at least one entry in `record["beliefs"]` has the given lane.
    Records missing the queried field never match — the filter is
    falsifying, not best-effort.
    """
    for key, value in filters:
        if key == "hook":
            if record.get("hook") != value:
                return False
            continue
        if key == "lane":
            beliefs = record.get("beliefs")
            if not isinstance(beliefs, list):
                return False
            hit = False
            for b in beliefs:
                if isinstance(b, dict) and b.get("lane") == value:
                    hit = True
                    break
            if not hit:
                return False
    return True


def format_record(
    record: dict[str, object], *, include_blob: bool = True,
) -> str:
    """Render one audit record as multi-line text suitable for tail output.

    Header line: `<HH:MM:SS>  <hook>  <tokens>tok  <latency>ms  L0×N L1×M`.
    Then one indented line per belief from `beliefs[]` (when present).
    `include_blob=False` suppresses the per-belief snippet bodies and
    leaves just the header line + per-belief identifiers.
    """
    ts = record.get("ts", "?")
    short_ts = _short_ts(ts if isinstance(ts, str) else "?")
    hook = str(record.get("hook", "?"))
    tokens = record.get("tokens")
    latency_ms = record.get("latency_ms")
    beliefs_obj = record.get("beliefs")
    beliefs: list[dict[str, object]] = []
    if isinstance(beliefs_obj, list):
        beliefs = [b for b in beliefs_obj if isinstance(b, dict)]
    n_l0 = sum(1 for b in beliefs if b.get("lane") == "L0")
    n_l1 = sum(1 for b in beliefs if b.get("lane") == "L1")
    parts: list[str] = [short_ts, hook]
    if isinstance(tokens, int):
        parts.append(f"{tokens} tok")
    if isinstance(latency_ms, int):
        parts.append(f"{latency_ms} ms")
    parts.append(f"L0×{n_l0} L1×{n_l1}")
    lines = ["  ".join(parts)]
    for b in beliefs:
        lane = str(b.get("lane", "??"))
        bid = str(b.get("id", "?"))
        bid_short = bid[:8] if len(bid) > 8 else bid
        locked_mark = " locked" if b.get("locked") is True else ""
        if include_blob:
            snippet = str(b.get("snippet", ""))
            lines.append(
                f"  [{lane}{locked_mark:7s}] {bid_short}  {snippet}"
            )
        else:
            lines.append(f"  [{lane}{locked_mark:7s}] {bid_short}")
    return "\n".join(lines)


def _short_ts(ts: str) -> str:
    """Compact `HH:MM:SS` from a full ISO timestamp; passthrough on fail."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    return dt.strftime("%H:%M:%S")


def _read_records(
    path: Path, *, since_cutoff: datetime | None,
) -> list[dict[str, object]]:
    """Read all valid JSON-object records from `path`, optionally
    filtered to records at or after `since_cutoff`. Lines that fail to
    parse as JSON objects are silently skipped — `aelf tail` is a
    diagnostic, not an integrity tool, and a single corrupt line should
    not abort a live tail. (`read_hook_audit` in hook.py raises on
    corruption; that's the contract for batch consumers.)
    """
    if not path.exists():
        return []
    out: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if since_cutoff is not None:
            ts = _parse_record_ts(parsed)
            if ts is None or ts < since_cutoff:
                continue
        out.append(parsed)
    return out


def _emit_records(
    records: Iterable[dict[str, object]],
    filters: list[tuple[str, str]],
    *,
    include_blob: bool,
    out: IO[str],
) -> None:
    """Format and write each record that matches all filters."""
    for rec in records:
        if not record_matches_filters(rec, filters):
            continue
        out.write(format_record(rec, include_blob=include_blob))
        out.write("\n\n")
        out.flush()


def tail_audit(
    *,
    audit_path: Path | None = None,
    filters: list[tuple[str, str]] | None = None,
    since: timedelta | None = None,
    include_blob: bool = True,
    follow: bool = True,
    out: IO[str] | None = None,
    poll_interval: float = _TAIL_POLL_INTERVAL,
    max_iters: int | None = None,
) -> int:
    """Read (and optionally follow) the audit log; emit pretty records.

    Backfill semantics: when `since` is set, replay matching records
    from both the rotated `<path>.1` (oldest) and the live file (newest)
    in order. Without `since`, follow mode starts at end-of-file and
    only emits new records as they land.

    `follow=True` (default) keeps tailing forever, polling for new
    lines and detecting rotation by inode change. `follow=False` reads
    once and returns — useful for tests and one-shot dumps.

    `max_iters` caps the follow loop's poll cycles. Tests pass a small
    integer so the loop returns deterministically; production callers
    leave it None for an infinite tail.
    """
    sink: IO[str] = out if out is not None else sys.stdout
    flt = filters if filters is not None else []
    if audit_path is None:
        from aelfrice.cli import db_path
        audit_path = _audit_path_for_db(db_path())
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)

    cutoff: datetime | None = None
    if since is not None:
        cutoff = datetime.now(timezone.utc) - since

    # Backfill: if --since given, read rotated then live. Otherwise (and
    # in follow mode) skip backfill — start at end-of-file so a fresh
    # `aelf tail` doesn't dump the entire history on every invocation.
    if cutoff is not None:
        backfill: list[dict[str, object]] = []
        backfill.extend(_read_records(rotated, since_cutoff=cutoff))
        backfill.extend(_read_records(audit_path, since_cutoff=cutoff))
        _emit_records(backfill, flt, include_blob=include_blob, out=sink)

    if not follow:
        if cutoff is None:
            # One-shot, no --since: dump everything currently in the
            # live file. Tests rely on this path.
            records = _read_records(audit_path, since_cutoff=None)
            _emit_records(records, flt, include_blob=include_blob, out=sink)
        return 0

    # Follow mode: open the live file at end-of-file and stream new
    # lines. Detect rotation by ino change (st_ino flips when
    # _append_audit renames live → .1 and the next write recreates a
    # fresh inode).
    pos = 0
    last_ino: int | None = None
    if audit_path.exists():
        pos = audit_path.stat().st_size
        last_ino = audit_path.stat().st_ino

    iters = 0
    while True:
        if max_iters is not None and iters >= max_iters:
            return 0
        iters += 1

        if not audit_path.exists():
            time.sleep(poll_interval)
            continue

        st = audit_path.stat()
        if last_ino is not None and st.st_ino != last_ino:
            # Rotation detected: live file was renamed to .1 and a new
            # one was created. Reset position to read from the start.
            pos = 0
        last_ino = st.st_ino

        if st.st_size <= pos:
            time.sleep(poll_interval)
            continue

        with audit_path.open("r", encoding="utf-8") as f:
            f.seek(pos)
            new_text = f.read()
            pos = f.tell()

        new_records: list[dict[str, object]] = []
        for line in new_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            new_records.append(parsed)
        _emit_records(new_records, flt, include_blob=include_blob, out=sink)
