"""Per-turn transcript logger for Claude Code hook integration.

Wires four hook events into a project-scoped append-only JSONL log
under `<git-common-dir>/aelfrice/transcripts/turns.jsonl`:

- `UserPromptSubmit` -> append `{"role": "user", "text": <prompt>, ...}`.
  Harness-wrapper prompts (rejected by `noise_filter.is_transcript_noise`)
  are dropped before append per #747.
- `Stop` -> append `{"role": "assistant", "text": <last assistant turn>, ...}`.
- `PreCompact` -> write a `compaction_start` marker, rotate
  `turns.jsonl` to `archive/turns-<ts>.jsonl`, spawn
  `aelf ingest-transcript` detached.
- `PostCompact` -> write a `compaction_complete` marker.

All four events share one entry point (`main`). Dispatch is by the
`hook_event_name` field Claude Code includes in every hook JSON
payload.

Non-blocking contract: every failure mode (empty stdin, malformed
JSON, missing fields, filesystem error) writes a stack trace to
stderr and returns exit 0. The conversation must never stall on
logger failure.

Latency budget:
- Per-turn append (user/assistant): sub-10ms p99.
- PreCompact: sub-50ms (rotation + detached spawn; the actual
  ingest runs in the background).

The log lives under `.git/`, which git does not track; transcripts
never cross the git boundary.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess  # noqa: F401 — used by _spawn_background_ingest
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable, Final, cast

# Imported lazily to keep module import fast for the hook hot path.
# `_git_common_dir` is reused from cli.py; if that import is ever
# heavy at import time the hook caller still pays it once per event.

SCHEMA_VERSION: Final[int] = 1
TURNS_FILENAME: Final[str] = "turns.jsonl"
ARCHIVE_DIRNAME: Final[str] = "archive"
TRANSCRIPTS_SUBDIR: Final[str] = "transcripts"
LEGACY_TRANSCRIPTS_DIR: Final[Path] = Path.home() / ".aelfrice" / "transcripts"

EVENT_USER_PROMPT_SUBMIT: Final[str] = "UserPromptSubmit"
EVENT_STOP: Final[str] = "Stop"
EVENT_PRE_COMPACT: Final[str] = "PreCompact"
EVENT_POST_COMPACT: Final[str] = "PostCompact"


def transcripts_dir() -> Path:
    """Resolve the transcripts directory path.

    Resolution order mirrors `cli.db_path()`:
    1. `$AELFRICE_TRANSCRIPTS_DIR` (explicit override).
    2. `<git-common-dir>/aelfrice/transcripts/` when in a git work-tree.
    3. `~/.aelfrice/transcripts/` (non-git fallback).

    The git-tree path lives under `.git/`, which git does not track,
    so transcripts never cross the git boundary.
    """
    override = os.environ.get("AELFRICE_TRANSCRIPTS_DIR")
    if override:
        return Path(override)
    from aelfrice.db_paths import _git_common_dir  # noqa: PLC0415

    git_dir = _git_common_dir()
    if git_dir is not None:
        return git_dir / "aelfrice" / TRANSCRIPTS_SUBDIR
    return LEGACY_TRANSCRIPTS_DIR


def turns_path() -> Path:
    return transcripts_dir() / TURNS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _new_turn_id() -> str:
    return f"{_utc_compact_ts()}-{secrets.token_hex(4)}"


def _read_payload(stdin: IO[str]) -> dict[str, object] | None:
    raw = stdin.read()
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _turn_context() -> dict[str, str | None]:
    """Per-turn context written into every JSONL line.

    Limited to fields cheap enough to collect within the sub-10ms
    per-turn budget. `git rev-parse` / `git symbolic-ref` would each
    fork a subprocess (~5-15ms on macOS), so branch/HEAD are
    deliberately omitted from the hot path. They can be enriched
    later by `ingest_jsonl` when it consumes the archive (which has
    a more generous budget than the live hook).
    """
    return {"cwd": os.getcwd()}


def _append_jsonl(path: Path, line: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(serialized)
        f.write("\n")


def _build_turn_line(
    *, role: str, text: str, session_id: str | None, ctx: dict[str, str | None],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": _now_iso(),
        "role": role,
        "text": text,
        "session_id": session_id,
        "turn_id": _new_turn_id(),
        "context": ctx,
    }


def _last_assistant_text(transcript_path: str | None) -> str | None:
    """Best-effort scan of Claude Code's transcript for the final
    assistant message text. Tolerant of format drift: returns None
    if anything goes wrong rather than raising.

    The transcript is a JSONL file; lines vary in shape across
    Claude Code versions. We scan from the tail for the most
    recent line whose `type`/`role` indicates an assistant message
    and pull its `message.content` text or `text` field.
    """
    if not transcript_path:
        return None
    p = Path(transcript_path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)  # pyright: ignore[reportAny]
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        obj_typed = cast(dict[str, object], obj)
        # Try a few known shapes; degrade silently on miss.
        role = obj_typed.get("role") or obj_typed.get("type")
        if role != "assistant":
            continue
        msg = obj_typed.get("message")
        if isinstance(msg, dict):
            msg_typed = cast(dict[str, object], msg)
            content = msg_typed.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # collected text segments
                parts: list[str] = []
                for seg in cast(list[object], content):
                    if isinstance(seg, dict):
                        seg_typed = cast(dict[str, object], seg)
                        t = seg_typed.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                if parts:
                    return "".join(parts)
        text = obj_typed.get("text")
        if isinstance(text, str):
            return text
    return None


def _handle_user_prompt_submit(payload: dict[str, object]) -> None:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return
    # #747: harness-wrapper prompts (<task-notification>, <summary>Monitor,
    # <tool-result>, etc.) carry no user intent and crowd real turns out of
    # the rebuilder's recent-turns window. Gate the append on the same
    # noise predicate ingest.py uses; fail-soft so a noise_filter import
    # error never breaks the logger.
    try:
        from aelfrice.noise_filter import is_transcript_noise  # noqa: PLC0415

        if is_transcript_noise(prompt):
            return
    except Exception:
        # Fail-soft: any noise_filter regression falls through to the
        # plain-append path. Silent by design — this hook runs on every
        # user prompt and its stderr leaks into the harness output, so
        # logging here would be worse UX than rare unfiltered rows.
        pass
    session_id = payload.get("session_id")
    sid = session_id if isinstance(session_id, str) else None
    line = _build_turn_line(
        role="user", text=prompt, session_id=sid, ctx=_turn_context(),
    )
    _append_jsonl(turns_path(), line)


def _handle_stop(payload: dict[str, object]) -> None:
    transcript_path = payload.get("transcript_path")
    text_path = transcript_path if isinstance(transcript_path, str) else None
    text = _last_assistant_text(text_path)
    if not text:
        # No accessible assistant text — write a stub line so the
        # rebuilder can still see a turn boundary. role='assistant'
        # with empty text degrades gracefully in ingest_jsonl
        # (extract_sentences returns []).
        text = ""
    session_id = payload.get("session_id")
    sid = session_id if isinstance(session_id, str) else None
    line = _build_turn_line(
        role="assistant", text=text, session_id=sid, ctx=_turn_context(),
    )
    _append_jsonl(turns_path(), line)


def _handle_pre_compact(payload: dict[str, object]) -> None:
    """Rotate turns.jsonl into archive/, spawn ingest detached.

    Sub-50ms budget: the rename is O(1); the ingest runs as a
    detached subprocess that does not block this hook's return.
    """
    _ = payload  # currently unused; reserved for compaction metadata
    tdir = transcripts_dir()
    src = tdir / TURNS_FILENAME
    archive_dir = tdir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        # Append the marker BEFORE rotation so the archived file
        # carries it (the rebuilder needs the boundary inside the
        # rotated segment, not after).
        _append_jsonl(src, {
            "schema_version": SCHEMA_VERSION,
            "ts": _now_iso(),
            "event": "compaction_start",
        })
        archived = archive_dir / f"turns-{_utc_compact_ts()}.jsonl"
        os.rename(src, archived)
        _spawn_background_ingest(archived)


def _handle_post_compact(payload: dict[str, object]) -> None:
    _ = payload
    tdir = transcripts_dir()
    tdir.mkdir(parents=True, exist_ok=True)
    target = tdir / TURNS_FILENAME
    _append_jsonl(target, {
        "schema_version": SCHEMA_VERSION,
        "ts": _now_iso(),
        "event": "compaction_complete",
    })


def _spawn_background_ingest(archive_file: Path) -> None:
    """Spawn `aelf ingest-transcript <archive>` detached. Best-effort.

    Detached so the PreCompact hook returns within budget regardless
    of ingest progress. Stdin/out/err -> /dev/null; the ingest
    process owns its own logging via the store's normal pathways.
    """
    try:
        with open(os.devnull, "w") as devnull:
            subprocess.Popen(  # noqa: S603 - args are package-internal
                ["aelf", "ingest-transcript", str(archive_file)],
                stdin=devnull, stdout=devnull, stderr=devnull,
                start_new_session=True, close_fds=True,
            )
    except (FileNotFoundError, OSError):
        # `aelf` not on PATH (highly unusual) or fork failure.
        # Non-blocking: leave the archive in place; a later
        # `aelf ingest-transcript` run picks it up.
        pass


_Handler = Callable[[dict[str, object]], None]
_DISPATCH: Final[dict[str, _Handler]] = {
    EVENT_USER_PROMPT_SUBMIT: _handle_user_prompt_submit,
    EVENT_STOP: _handle_stop,
    EVENT_PRE_COMPACT: _handle_pre_compact,
    EVENT_POST_COMPACT: _handle_post_compact,
}


def main(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract)."""
    sin = stdin if stdin is not None else sys.stdin
    serr = stderr if stderr is not None else sys.stderr
    try:
        payload = _read_payload(sin)
        if payload is None:
            return 0
        event = payload.get("hook_event_name")
        if not isinstance(event, str):
            return 0
        handler = _DISPATCH.get(event)
        if handler is None:
            return 0
        handler(payload)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
