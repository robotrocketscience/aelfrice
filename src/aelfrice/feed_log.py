"""Per-project JSONL event log for belief writes (#931).

Writer module — appended to from every belief-write path so users can
`tail -f` the file and see what aelfrice is doing. Sibling of
`memory.db` under `<git-common-dir>/aelfrice/feed.jsonl`.

Design constraints:

* **Errors are swallowed.** A failed feed write must never break the
  write-side CLI command that triggered it. The feed log is a
  visibility surface; correctness of writes is not.
* **Rotation, not bounded growth.** At 100 MB the active log is
  renamed to `feed.<unix-ts>.jsonl.archive` and a fresh `feed.jsonl`
  starts. Archives are not deleted automatically — users can sweep
  them on their own schedule.
* **No write announcement on stderr.** Stderr-bleed into Claude
  Code's transcript ingestion is a self-feedback risk; the JSONL
  file is the only channel.
* **Default on.** `AELFRICE_FEED_LOG=0` disables. Statusline /
  setup integration can layer richer opt-out in follow-up work.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

ENV_FEED_LOG: Final[str] = "AELFRICE_FEED_LOG"
"""Set to '0' to suppress all feed-log writes."""

FEED_FILENAME: Final[str] = "feed.jsonl"

ROTATE_BYTES: Final[int] = 100 * 1024 * 1024
"""Rotate the active log when it exceeds 100 MB."""


def is_enabled(env: dict[str, str] | None = None) -> bool:
    """True unless `AELFRICE_FEED_LOG=0` is set."""
    src = os.environ if env is None else env
    return src.get(ENV_FEED_LOG) != "0"


def feed_path(db_dir: Path | None = None) -> Path:
    """Return the feed-log path. Sibling of `memory.db`.

    `db_dir=None` resolves via `aelfrice.db_paths.db_path().parent`,
    which honours `AELFRICE_DB` overrides.
    """
    if db_dir is None:
        from aelfrice.db_paths import db_path as _db_path
        db_dir = _db_path().parent
    return db_dir / FEED_FILENAME


def _utc_now_iso() -> str:
    """Return current time as an ISO-Z-suffixed UTC string."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _rotate_if_needed(p: Path) -> None:
    """Rename `p` to `feed.<ts>.jsonl.archive` if it exceeds ROTATE_BYTES.

    Errors are swallowed — if rotation fails the log keeps growing
    rather than the write path failing.
    """
    try:
        size = p.stat().st_size
    except OSError:
        return
    if size < ROTATE_BYTES:
        return
    archive_name = f"feed.{int(time.time())}.jsonl.archive"
    try:
        p.rename(p.with_name(archive_name))
    except OSError:
        return


def append(event: str, **fields: Any) -> None:
    """Append one JSONL row to the feed log.

    `event` is the canonical event type (`belief.locked`,
    `belief.ingested`, `wonder.promoted`, `feedback.applied`).
    Additional structured fields are merged into the row.

    Returns None unconditionally. All errors (env-disabled, missing
    parent dir, json-serialisation, OS write failures) are swallowed
    — the write path must not see them.
    """
    if not is_enabled():
        return
    row: dict[str, Any] = {"ts": _utc_now_iso(), "event": event, **fields}
    try:
        p = feed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(p)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError):
        return


def read_rows(p: Path | None = None) -> list[dict[str, Any]]:
    """Read the active feed log into a list of dicts.

    For the reader CLI. Malformed JSON lines are skipped silently
    (the writer's swallow-errors posture means we may occasionally
    have partial lines from interrupted writes — a malformed row in
    the middle of a file shouldn't prevent the reader from showing
    the rest).
    """
    if p is None:
        p = feed_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return out
    return out
