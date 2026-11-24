"""Per-conversation state for the Antigravity CLI (agy) host.

Two kinds of file, both keyed by agy `conversationId`, under
``~/.aelfrice/antigravity/`` (``$AELFRICE_AGY_STATE_DIR`` overrides):

- ``pending/<id>.txt``: context a PreToolUse hook produced. agy shows the
  model nothing a tool hook prints, so the search-tool context waits here
  and the next PreInvocation (``invocationNum > 0``, which fires after the
  tool and before the model reads its result) injects it.
- ``baseline/<id>``: an empty marker meaning this conversation has had the
  session-start baseline. agy fires no SessionStart on ``agy -c`` resume,
  so the first PreInvocation of a turn injects the baseline when the
  marker is missing.

Every file is bounded, and files older than `STALE_AFTER_S` are removed by
`sweep_stale`. Stdlib only.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Final

STATE_DIR_ENV: Final[str] = "AELFRICE_AGY_STATE_DIR"
PENDING_DIRNAME: Final[str] = "pending"
BASELINE_DIRNAME: Final[str] = "baseline"

# One pending file holds at most this many characters; older context is
# dropped first, because the newest search is the one the model is about
# to read.
MAX_PENDING_CHARS: Final[int] = 32_000
PENDING_SEPARATOR: Final[str] = "\n\n"
STALE_AFTER_S: Final[int] = 14 * 24 * 3600
MAX_SWEEP_FILES: Final[int] = 10_000

_SAFE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def state_dir() -> Path:
    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".aelfrice" / "antigravity"


def _safe_id(conversation_id: str) -> str | None:
    """The id when it is safe as a file name, else None (no path tricks)."""
    if not _SAFE_ID_RE.match(conversation_id) or conversation_id in (".", ".."):
        return None
    return conversation_id


def _pending_path(conversation_id: str) -> Path | None:
    cid = _safe_id(conversation_id)
    return None if cid is None else state_dir() / PENDING_DIRNAME / f"{cid}.txt"


def _baseline_path(conversation_id: str) -> Path | None:
    cid = _safe_id(conversation_id)
    return None if cid is None else state_dir() / BASELINE_DIRNAME / cid


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_pending(conversation_id: str, context: str) -> bool:
    """Queue `context` for the conversation's next model call.

    Keeps the newest `MAX_PENDING_CHARS` characters. Returns whether
    anything was queued.
    """
    path = _pending_path(conversation_id)
    if path is None or not context.strip():
        return False
    try:
        prior = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        prior = ""
    combined = f"{prior}{PENDING_SEPARATOR}{context}" if prior else context
    if len(combined) > MAX_PENDING_CHARS:
        combined = combined[-MAX_PENDING_CHARS:]
    _atomic_write(path, combined)
    return True


def take_pending(conversation_id: str) -> str:
    """Return and delete the conversation's queued context ('' when none)."""
    path = _pending_path(conversation_id)
    if path is None:
        return ""
    claimed = path.with_name(f".{path.name}.{os.getpid()}.taking")
    try:
        os.replace(path, claimed)
    except OSError:
        return ""
    try:
        return claimed.read_text(encoding="utf-8")[:MAX_PENDING_CHARS]
    except OSError:
        return ""
    finally:
        try:
            claimed.unlink()
        except OSError:
            pass


def has_baseline(conversation_id: str) -> bool:
    path = _baseline_path(conversation_id)
    return path is not None and path.exists()


def mark_baseline(conversation_id: str) -> None:
    path = _baseline_path(conversation_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def sweep_stale(now: float | None = None) -> int:
    """Remove state files older than `STALE_AFTER_S`; return how many.

    Looks at no more than `MAX_SWEEP_FILES` files per call.
    """
    cutoff = (time.time() if now is None else now) - STALE_AFTER_S
    removed = 0
    seen = 0
    for sub in (PENDING_DIRNAME, BASELINE_DIRNAME):
        directory = state_dir() / sub
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            seen += 1
            if seen > MAX_SWEEP_FILES:
                return removed
            try:
                if child.is_file() and child.stat().st_mtime < cutoff:
                    child.unlink()
                    removed += 1
            except OSError:
                continue
    return removed
