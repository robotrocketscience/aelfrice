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

# #1439: both hosts write an interrupt as a *user*-role record whose only
# content is a harness-generated marker. The Claude-host harness writes
# "[Request interrupted by user]" or "[Request interrupted by user for tool
# use]" (every local occurrence shares the first prefix); the Codex CLI
# writes a `<turn_aborted>` block ("The user interrupted the previous turn
# on purpose...", every local occurrence the record's only segment).
# Neither is a new prompt, so neither may end the tail scan -- the partial
# text the interrupted turn produced is the right answer.
_INTERRUPT_MARKER_PREFIXES: Final[tuple[str, ...]] = (
    "[Request interrupted by user",
    "<turn_aborted>",
)

# #1439: Codex writes three more user-role records that nobody typed. The
# environment block (and the plugin list that rides with it) is re-injected
# after a compaction or a resume, immediately before the real prompt; a
# `<user_shell_command>` record is the transcript of a command the user ran
# in the TUI, written between `exec_command_end` and `task_complete` with no
# model call in between. Same class as the abort marker -- harness-generated
# records, not turn starts -- so the same rule applies to them.
_SYNTHETIC_RECORD_PREFIXES: Final[tuple[str, ...]] = (
    "<environment_context>",
    "<recommended_plugins>",
    "<user_shell_command>",
)

# Matched on either host shape: the tags are Codex's, but the predicate is
# shared, so a Claude-host prompt that opened with one of them would be read
# as harness text too. That is the same trade already taken for
# `<turn_aborted>`, and no local Claude-host user record matches any of them.
_HARNESS_RECORD_PREFIXES: Final[tuple[str, ...]] = (
    _INTERRUPT_MARKER_PREFIXES + _SYNTHETIC_RECORD_PREFIXES
)

# #968: consecutive-duplicate guard. A turn whose (session_id, role, text)
# matches the file's previous turn line within this window is treated as a
# re-fire of one event (duplicated hook registration produces N identical
# lines ~20ms apart) and dropped. Wide enough to absorb a burst, narrow
# enough that a deliberate resend of the same prompt seconds later still logs.
DUP_WINDOW_SECONDS: Final[float] = 2.0
# Bytes scanned from the tail of turns.jsonl to find the previous line.
# Bounds the dedup read at O(1) in file size, holding the per-append cost
# inside the sub-10ms budget even when rotation has not run recently.
_TAIL_READ_BYTES: Final[int] = 65536

# #1011: Stop-cadence ingestion flush. Belief ingestion historically fired
# ONLY on the PreCompact rotation (`_handle_pre_compact`). A session that
# ends without compacting logged its turns to turns.jsonl but never folded
# them into beliefs, so a fresh session could not recall what the user
# stated — the core-capture failure. On Stop, once >= STOP_FLUSH_TURNS new
# turns have accumulated since the last flush, spawn `aelf ingest-transcript`
# over the LIVE turns.jsonl. Ingestion is idempotent per (source_label,
# sentence), so re-ingesting the live file captures only new statements and
# never inflates the store; the file is NOT rotated, so the rebuilder / UPS
# recent-turns window (which reads the live turns.jsonl) is left intact.
DEFAULT_STOP_FLUSH_TURNS: Final[int] = 12
STOP_FLUSH_TURNS_ENV: Final[str] = "AELFRICE_INGEST_STOP_FLUSH_TURNS"
STOP_FLUSH_CURSOR_FILENAME: Final[str] = ".stop_flush_cursor"


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


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_last_turn_line(path: Path) -> dict[str, object] | None:
    """Return the last non-empty JSONL record in `path`, or None.

    Bounded tail read: only the final `_TAIL_READ_BYTES` are scanned, so
    the cost is O(1) in file size. A record whose serialized form is the
    sole line in the window but begins before it (truncated) fails to
    parse and yields None — the guard then fails open to append, never
    dropping a turn it cannot positively confirm as a duplicate.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - _TAIL_READ_BYTES)
            _ = f.seek(start)
            chunk = f.read()
    except OSError:
        return None
    if not chunk:
        return None
    segments = chunk.split(b"\n")
    # When the window starts mid-file the first segment is a partial line;
    # drop it so a truncated record is never mistaken for a complete one.
    if start > 0 and len(segments) > 1:
        segments = segments[1:]
    for raw in reversed(segments):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw.decode("utf-8"))  # pyright: ignore[reportAny]
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(obj, dict):
            return cast(dict[str, object], obj)
        return None
    return None


def _is_consecutive_duplicate(
    last: dict[str, object] | None, candidate: dict[str, object],
) -> bool:
    """True when `candidate` repeats `last` within DUP_WINDOW_SECONDS.

    Match key is (session_id, role, text); turn_id is deliberately not in
    the key — the burst that motivates this guard carries distinct
    turn_ids (#968). Compaction markers (no role/text) and any record
    missing a parseable `ts` compare unequal, so they never trigger a skip.
    """
    if last is None:
        return False
    for field in ("role", "text", "session_id"):
        if last.get(field) != candidate.get(field):
            return False
    last_ts = _parse_iso(last.get("ts"))
    cand_ts = _parse_iso(candidate.get("ts"))
    if last_ts is None or cand_ts is None:
        return False
    return abs((cand_ts - last_ts).total_seconds()) <= DUP_WINDOW_SECONDS


def _record_skipped_duplicate(*, role: str, session_id: str | None) -> None:
    """Note a dropped consecutive-duplicate turn in hook_audit.jsonl.

    Observability only (#968 acceptance): reuses the hook-audit sink so
    `aelf tail` and other readers see the skip alongside belief-injection
    rows, and inherits its rotation. Lazy import keeps this off the normal
    append path — it runs only when a duplicate is actually detected, which
    is rare. Fully fail-soft: audit-disabled, an in-memory DB, or any
    import / I/O error is swallowed per the non-blocking contract.
    """
    try:
        from aelfrice.db_paths import db_path  # noqa: PLC0415
        from aelfrice.hook_audit import (  # noqa: PLC0415
            _append_audit,
            _audit_path_for_db,
            load_hook_audit_config,
        )

        cfg = load_hook_audit_config()
        if not cfg.enabled:
            return
        p = db_path()
        if str(p) == ":memory:":
            return
        record: dict[str, object] = {
            "ts": _now_iso(),
            "hook": "transcript_logger",
            "event": "skipped_duplicate",
            "role": role,
            "session_id": session_id,
        }
        _append_audit(_audit_path_for_db(p), record, cfg.max_bytes)
    except Exception:
        # Fail-soft by design: this audit write is observability-only, so
        # any import or I/O error must be swallowed rather than propagate
        # and break the non-blocking transcript-logger hook path (#968).
        pass


def _append_turn(line: dict[str, object]) -> None:
    """Append a turn line, dropping a consecutive duplicate (#968).

    Guards only turn lines (user/assistant); compaction markers append via
    `_append_jsonl` directly and are never deduped. When the previous
    record in turns.jsonl matches on (session_id, role, text) within
    DUP_WINDOW_SECONDS, the write is skipped and the drop is noted in
    hook_audit.jsonl. The guard targets the sequential burst of re-fires a
    duplicated hook registration produces; it adds no locking, preserving
    the lock-free non-blocking contract.
    """
    path = turns_path()
    last = _read_last_turn_line(path)
    if _is_consecutive_duplicate(last, line):
        role = line.get("role")
        sid = line.get("session_id")
        _record_skipped_duplicate(
            role=role if isinstance(role, str) else "",
            session_id=sid if isinstance(sid, str) else None,
        )
        return
    _append_jsonl(path, line)


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
    """Best-effort scan of the host transcript for the final
    assistant message text. Tolerant of format drift: returns None
    if anything goes wrong rather than raising.

    The transcript is a JSONL file whose per-line shape depends on
    the host (the Claude session transcript or the Codex CLI
    rollout log) and varies across host versions. We scan from the
    tail and try each known shape per line, so schema selection is
    per-record rather than per-file.

    The scan stops at the user prompt that opened the current turn
    (#1439). Without that bound a turn ending in a tool call or an
    interrupt walked past the boundary and returned the *previous*
    turn's answer, which the Stop hook then wrote as the answer to
    this turn. None is the correct result for a turn that produced
    no assistant text; the one caller (`_handle_stop`) already
    degrades it to an empty stub row.
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
        # Try each known shape; degrade silently on miss.
        text = _assistant_text_claude(obj_typed)
        if text is None:
            text = _assistant_text_codex(obj_typed)
        if text is not None:
            return text
        if _is_turn_start_user_record(obj_typed):
            # Turn boundary reached with no assistant text behind it.
            return None
    return None


def _is_turn_start_user_record(obj: dict[str, object]) -> bool:
    """True when this record is the user prompt that opens a turn.

    Host-agnostic (#1439): it recognises the Claude-host shape
    (top-level `role`/`type` == "user") and the Codex rollout shape
    (`payload.role` == "user" on a `message` item), mirroring the
    two assistant extractors, and routes both hosts' content through
    the same exclusions.

    Four kinds of user-role record are *not* turn starts, because
    stopping the scan at one discards assistant text the current turn
    really produced:

    * anything carrying a tool_result segment. The Claude-host shape puts
      mid-turn tool results in user records, and other segments (a
      <system-reminder> as a sibling text segment) can ride along, so
      the test is "carries a tool_result", not "carries only tool
      results". Codex keeps these in `function_call_output` items,
      which are not messages, so this arm can only fire on the
      Claude-host shape.
    * the interrupt marker either harness writes as a plain user record
      — `[Request interrupted by user...]` on the Claude-host shape,
      `<turn_aborted>` on Codex. Both are matched by prefix through
      `_is_harness_record_text`, under both content encodings.
    * the other synthetic records Codex writes —
      `<environment_context>`, `<recommended_plugins>`,
      `<user_shell_command>` — matched by the same prefix rule. A shell
      command run in the TUI is recorded as a user message but calls no
      model, and the environment block is re-injected on resume just
      ahead of the real prompt, which still bounds the scan.
    * anything the host flags `isMeta` — caveat banners, skill
      preambles, Stop-hook feedback: harness text, not a prompt. Codex
      rollout records carry no such flag, so this arm too can only fire
      on the Claude-host shape.

    What is *not* excluded, deliberately: the Claude-host records that
    are harness-shaped but do open a turn the model answers — slash
    commands (`<command-name>`), `<task-notification>` records and
    local command output. Excluding those would suppress real answers.
    """
    if obj.get("type") == "response_item":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return False
        payload_typed = cast(dict[str, object], payload)
        if payload_typed.get("type") != "message":
            return False
        if payload_typed.get("role") != "user":
            return False
        return _user_content_opens_a_turn(payload_typed.get("content"))
    if (obj.get("role") or obj.get("type")) != "user":
        return False
    if obj.get("isMeta"):
        return False
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return True
    msg_typed = cast(dict[str, object], msg)
    return _user_content_opens_a_turn(msg_typed.get("content"))


def _user_content_opens_a_turn(content: object) -> bool:
    """True unless this user-record content is tool plumbing or a marker.

    Shared by both host shapes (#1439), so the tool_result and
    harness-record exclusions apply wherever the content can carry
    them rather than only on the Claude-host arm.
    """
    if isinstance(content, str):
        return not _is_harness_record_text(content)
    if not isinstance(content, list) or not content:
        return True
    texts: list[str] = []
    for seg in cast(list[object], content):
        if not isinstance(seg, dict):
            continue
        seg_typed = cast(dict[str, object], seg)
        if seg_typed.get("type") == "tool_result":
            return False
        t = seg_typed.get("text")
        if isinstance(t, str):
            texts.append(t)
    # Text-only content that says nothing but harness boilerplate is the
    # marker record; anything else with text in it is a real prompt, so a
    # record carrying a marker *and* a typed segment still opens a turn.
    return not (texts and all(_is_harness_record_text(t) for t in texts))


def _is_harness_record_text(text: str) -> bool:
    """True for either host's harness-written record text (#1439).

    Covers both interrupt markers and Codex's other synthetic records;
    see `_INTERRUPT_MARKER_PREFIXES` / `_SYNTHETIC_RECORD_PREFIXES`.
    """
    stripped = text.strip()
    return any(stripped.startswith(p) for p in _HARNESS_RECORD_PREFIXES)


def _assistant_text_claude(obj: dict[str, object]) -> str | None:
    """Extract assistant text from one Claude-host transcript record.

    Shape: top-level `role` (or `type`) == "assistant"; text under
    `message.content` as a string or a list of `{"text": ...}`
    segments, with a top-level `text` field as fallback.
    """
    role = obj.get("role") or obj.get("type")
    if role != "assistant":
        return None
    msg = obj.get("message")
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
    text = obj.get("text")
    if isinstance(text, str):
        return text
    return None


def _assistant_text_codex(obj: dict[str, object]) -> str | None:
    """Extract assistant text from one Codex CLI rollout record (#1051).

    Shape: {"type": "response_item", "payload": {"type": "message",
    "role": "assistant", "content": [{"type": "output_text",
    "text": "..."}]}}. Role and text are nested under `payload`, so
    the Claude parser never matches these records and Stop events
    used to fall through to an empty stub.
    """
    if obj.get("type") != "response_item":
        return None
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_typed = cast(dict[str, object], payload)
    if payload_typed.get("type") != "message":
        return None
    if payload_typed.get("role") != "assistant":
        return None
    content = payload_typed.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for seg in cast(list[object], content):
        if not isinstance(seg, dict):
            continue
        seg_typed = cast(dict[str, object], seg)
        if seg_typed.get("type") not in ("output_text", "text"):
            continue
        t = seg_typed.get("text")
        if isinstance(t, str):
            parts.append(t)
    if parts:
        return "".join(parts)
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
        from aelfrice.noise_filter import (  # noqa: PLC0415
            is_transcript_noise,
            is_transcript_scaffolding,
        )

        # #1371 §1. Two different questions, and conflating them dropped
        # real turns. A *structural* marker — a harness tag, a glyph, a
        # pasted command — describes the whole payload, so the prompt goes
        # as a unit. An ack or a progress emit describes only the sentence
        # it is in, so a prompt is dropped for those reasons only when
        # every sentence in it is noise.
        #
        # Not sentence-granularity across the board, which is what #1371's
        # acceptance text asks for: measured on this repo's 6,274 archived
        # user prompts, that reading would have *kept* 751 multi-sentence
        # `<task-notification>` blocks, because their prose lines are not
        # individually noise. That is precisely the flooding #747 added
        # this gate to stop.
        if is_transcript_scaffolding(prompt):
            return
        if is_transcript_noise(prompt):
            from aelfrice.extraction import extract_sentences  # noqa: PLC0415

            # `all([])` is True, so a prompt whose sentences all fall below
            # `extraction._MIN_LEN` — a bare "Yes." — is still dropped here.
            # An explicit `not parts or` would be dead, and worse: it would
            # short-circuit the quantifier on exactly the prompts a bare-ack
            # test drives, so no test could tell `all` from `any`.
            parts = [s for s in extract_sentences(prompt) if s.strip()]
            if all(is_transcript_noise(s) for s in parts):
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
    _append_turn(line)


def _stop_payload_assistant_text(payload: dict[str, object]) -> str | None:
    """Assistant text carried directly on the Stop payload (#1051).

    Codex CLI 0.146.1 sends `transcript_path: null` and puts the
    answer in `last_assistant_message`, so the rollout parser has
    no file to read and every turn used to degrade to a stub.
    Whitespace-only values count as absent — they carry no answer
    content and must fall through to the transcript instead of
    suppressing it.
    """
    raw = payload.get("last_assistant_message")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw


def _handle_stop(payload: dict[str, object]) -> None:
    # Ordered adapter (#1051): the payload field wins when present,
    # then the host transcript. Exactly one assistant row is written
    # either way, so a payload field plus a readable rollout cannot
    # duplicate the turn.
    text = _stop_payload_assistant_text(payload)
    if text is None:
        transcript_path = payload.get("transcript_path")
        text_path = (
            transcript_path if isinstance(transcript_path, str) else None
        )
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
    _append_turn(line)
    # #1011: fold accumulated turns into beliefs without waiting for a
    # compaction. Fail-soft — a flush error must never break turn logging.
    _maybe_stop_flush(transcripts_dir())


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


def _spawn_background_ingest(archive_file: Path) -> bool:
    """Spawn `aelf ingest-transcript <archive>` detached. Best-effort.

    Detached so the PreCompact hook returns within budget regardless
    of ingest progress. Stdin/out/err -> /dev/null; the ingest
    process owns its own logging via the store's normal pathways.

    Returns True iff the subprocess was launched. The Stop-cadence
    caller (`_maybe_stop_flush`) uses this to advance its cursor only
    on a successful spawn, so a failed launch is retried at the next
    Stop rather than being silently marked flushed (#1012 review).
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
        return False
    return True


def _stop_flush_threshold() -> int:
    """New turns required since the last flush before a Stop triggers an
    ingest. `AELFRICE_INGEST_STOP_FLUSH_TURNS` overrides the default; a
    value <= 0 disables Stop-cadence flushing (PreCompact-only, the
    pre-#1011 behaviour)."""
    raw = os.environ.get(STOP_FLUSH_TURNS_ENV)
    if raw is None:
        return DEFAULT_STOP_FLUSH_TURNS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_STOP_FLUSH_TURNS


def _count_turn_lines(path: Path) -> int:
    """Count role-bearing turn lines in turns.jsonl. The cheap `'"role"'`
    substring test avoids JSON-parsing every line on the hook hot path —
    event markers (compaction_start, etc.) carry no `role` key, so they
    are excluded. Returns 0 on any read error (fail-soft)."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if '"role"' in line)
    except OSError:
        return 0


def _read_flush_cursor(tdir: Path) -> int:
    try:
        text = (tdir / STOP_FLUSH_CURSOR_FILENAME).read_text().strip()
        return int(text) if text else 0
    except (OSError, ValueError):
        return 0


def _write_flush_cursor(tdir: Path, value: int) -> None:
    try:
        (tdir / STOP_FLUSH_CURSOR_FILENAME).write_text(str(value))
    except OSError:
        # Fail-soft: a non-writable transcripts dir must never break the
        # Stop hook. The cursor simply isn't advanced, so the next Stop
        # re-evaluates and re-flushes (ingestion is idempotent).
        pass


def _maybe_stop_flush(tdir: Path) -> bool:
    """#1011: on Stop, ingest the live turns.jsonl once >= threshold new
    turns have accumulated since the last flush. Returns True iff a flush
    fired.

    No rotation: `ingest_jsonl` reads the whole file and dedupes per
    (source_label, sentence), so re-ingesting the live file captures only
    statements not yet in the store — never inflating it — while leaving
    the rebuilder / UPS recent-turns window (which reads the live
    turns.jsonl) intact. The cursor records the turn count at the last
    flush; a count below the cursor means turns.jsonl was rotated
    (PreCompact) or reset, so the cursor is treated as 0.
    """
    threshold = _stop_flush_threshold()
    if threshold <= 0:
        return False
    src = tdir / TURNS_FILENAME
    if not src.exists():
        return False
    now = _count_turn_lines(src)
    last = _read_flush_cursor(tdir)
    if now < last:
        last = 0
    if now - last < threshold:
        return False
    # Advance the cursor only on a successful spawn (#1012 review): if the
    # ingest can't launch, leave the cursor so the next Stop retries rather
    # than silently marking these turns flushed and reopening the recall gap.
    if not _spawn_background_ingest(src):
        return False
    _write_flush_cursor(tdir, now)
    return True


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
