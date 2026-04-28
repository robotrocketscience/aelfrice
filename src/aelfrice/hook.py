"""Claude Code hook entry-points for aelfrice.

This module exposes the script-side half of the v0.7.0 wiring: the
process Claude Code spawns when a `UserPromptSubmit` hook fires. It
reads the JSON event payload from stdin, pulls the user's prompt out
of it, runs aelfrice retrieval against that prompt, and writes the
formatted hits to stdout. Claude Code injects stdout as additional
context above the user's message.

Non-blocking contract: the hook must never fail in a way that
prevents the user's prompt from reaching the model. Every failure
mode (empty payload, malformed JSON, missing prompt field, retrieval
error) returns exit 0 and emits no stdout. Internal exceptions are
written to stderr (Claude Code captures and surfaces these in the
hook log) but do not bubble up.

Output format: a single XML-tag-delimited block. The tag delimiters
are stable; the contents inside are the same per-belief lines the
`aelf search` CLI prints, so a future change to the retrieval format
flows here automatically.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import IO, Final, cast

from aelfrice.cli import db_path
from aelfrice.context_rebuilder import (
    DEFAULT_N_RECENT_TURNS,
    DEFAULT_TOKEN_BUDGET,
    RecentTurn,
    find_aelfrice_log,
    read_recent_turns_aelfrice,
    read_recent_turns_claude_transcript,
    rebuild,
)
from aelfrice.hook_search import search_for_prompt
from aelfrice.models import LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

DEFAULT_HOOK_TOKEN_BUDGET: Final[int] = 1500
"""Conservative default budget for hook-injected context.

Below the CLI default (2000) to leave headroom for the user's
prompt and other concurrent UserPromptSubmit hooks competing for
the same context window.
"""

DEFAULT_SESSION_START_TOKEN_BUDGET: Final[int] = 1500
"""Token budget for the SessionStart context block.

SessionStart fires once at the beginning of a Claude Code session,
before any user prompt. The block surfaces L0 locked beliefs (the
user-asserted ground truth) so the agent enters the session with
durable baseline knowledge already in context. Per-prompt
retrieval continues to fire on every UserPromptSubmit thereafter.
"""

OPEN_TAG: Final[str] = "<aelfrice-memory>"
CLOSE_TAG: Final[str] = "</aelfrice-memory>"
SESSION_START_OPEN_TAG: Final[str] = "<aelfrice-baseline>"
SESSION_START_CLOSE_TAG: Final[str] = "</aelfrice-baseline>"
_PROMPT_KEY: Final[str] = "prompt"
_TRANSCRIPT_PATH_KEY: Final[str] = "transcript_path"
_CWD_KEY: Final[str] = "cwd"


def user_prompt_submit(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    token_budget: int | None = None,
) -> int:
    """Run the UserPromptSubmit hook. Always returns 0.

    Reads a Claude Code UserPromptSubmit JSON payload from `stdin`,
    runs retrieval against the `prompt` field, and writes the
    formatted output to `stdout`. Streams default to the process
    `sys.stdin`/`sys.stdout`/`sys.stderr`.
    """
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        # TTL-gated background update check, completely detached, never
        # blocks the hook. Statusline reads the cache it writes.
        try:
            from aelfrice.lifecycle import maybe_check_for_update_async

            maybe_check_for_update_async()
        except Exception:
            pass
        raw = sin.read()
        prompt = _extract_prompt(raw)
        if prompt is None:
            return 0
        budget = (
            token_budget
            if token_budget is not None
            else DEFAULT_HOOK_TOKEN_BUDGET
        )
        body = _retrieve_and_format(prompt, budget)
        if body:
            sout.write(body)
    except Exception:  # non-blocking: surface but do not fail
        traceback.print_exc(file=serr)
    return 0


def _extract_prompt(raw: str) -> str | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload_typed = cast(dict[str, object], payload)
    prompt = payload_typed.get(_PROMPT_KEY)
    if not isinstance(prompt, str):
        return None
    if not prompt.strip():
        return None
    return prompt


def _retrieve_and_format(prompt: str, token_budget: int) -> str:
    store = _open_store()
    try:
        hits = search_for_prompt(store, prompt, token_budget=token_budget)
    finally:
        store.close()
    if not hits:
        return ""
    return _format_hits(hits)


def _format_hits(hits: list[Belief]) -> str:
    lines: list[str] = [OPEN_TAG]
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        lines.append(f"{prefix} {h.id}: {h.content}")
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(str(p))


def pre_compact(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    n_recent_turns: int = DEFAULT_N_RECENT_TURNS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> int:
    """Run the PreCompact hook. Always returns 0.

    Reads a Claude Code PreCompact JSON payload from `stdin`, locates
    a transcript log (canonical aelfrice turns.jsonl preferred,
    Claude Code internal transcript as fallback), runs the
    context-rebuilder against it, and writes the rebuild block to
    `stdout`. Hook contract: never block, never raise.

    Payload fields used:
      * `cwd` -- working directory; used to find <cwd>/.git/aelfrice/
        transcripts/turns.jsonl (the canonical log).
      * `transcript_path` -- absolute path to Claude Code's per-session
        transcript JSONL. Used as fallback when the canonical log is
        absent (typical pre-transcript_ingest setup).

    With augment-mode coordination (the only mode v1.1.0a0 ships),
    Claude Code will still run its default summarization after this
    hook emits. The rebuild block is therefore additive context, not
    a replacement.
    """
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        raw = sin.read()
        payload = _parse_pre_compact_payload(raw)
        if payload is None:
            return 0
        recent = _read_recent_for_pre_compact(payload, n_recent_turns)
        body = _rebuild_and_format(recent, token_budget)
        if body:
            sout.write(body)
    except Exception:  # non-blocking: surface but do not fail
        traceback.print_exc(file=serr)
    return 0


def _parse_pre_compact_payload(raw: str) -> dict[str, object] | None:
    """Return the parsed payload dict, or None on any malformedness."""
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _read_recent_for_pre_compact(
    payload: dict[str, object], n_recent_turns: int
) -> list[RecentTurn]:
    """Locate a transcript and read its tail.

    Resolution order:
      1. <payload.cwd>/.git/aelfrice/transcripts/turns.jsonl -- the
         canonical aelfrice log written by the per-turn UserPromptSubmit/
         Stop hooks once transcript_ingest ships. Preferred when
         present.
      2. <payload.transcript_path> -- Claude Code's internal per-session
         transcript JSONL. Fallback for the alpha while transcript_ingest
         is unshipped.
      3. Empty list -- both sources missing or unreadable.
    """
    cwd_obj = payload.get(_CWD_KEY)
    if isinstance(cwd_obj, str) and cwd_obj.strip():
        try:
            cwd = Path(cwd_obj)
            log_path = find_aelfrice_log(cwd)
        except OSError:
            log_path = None
        if log_path is not None and log_path.exists():
            return read_recent_turns_aelfrice(log_path, n=n_recent_turns)
    tp_obj = payload.get(_TRANSCRIPT_PATH_KEY)
    if isinstance(tp_obj, str) and tp_obj.strip():
        tp = Path(tp_obj)
        if tp.exists():
            return read_recent_turns_claude_transcript(
                tp, n=n_recent_turns
            )
    return []


def _rebuild_and_format(
    recent: list[RecentTurn], token_budget: int
) -> str:
    store = _open_store()
    try:
        return rebuild(recent, store, token_budget=token_budget)
    finally:
        store.close()


def session_start(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    token_budget: int | None = None,
) -> int:
    """Run the SessionStart hook. Always returns 0.

    Reads the SessionStart JSON payload from stdin (consumed for
    protocol compatibility -- no fields are read from it at MVP) and
    writes a baseline context block of L0 locked beliefs to stdout.
    The block fires once per session, before any user message.

    Empty store / no locked beliefs: emit nothing (return 0). Per the
    non-blocking hook contract, every failure path returns 0; internal
    exceptions write to stderr and are otherwise swallowed.

    Why locked-only: at session start there is no prompt to query
    against, so BM25-driven L1 retrieval would have nothing to score.
    L0 locked beliefs are the user-asserted ground truth that should
    survive across every session regardless of context, which makes
    them the correct content for an unconditional session-start
    injection.
    """
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        # Drain stdin so the hook protocol is honored even though we
        # do not consume any fields.
        try:
            _ = sin.read()
        except Exception:
            pass
        budget = (
            token_budget
            if token_budget is not None
            else DEFAULT_SESSION_START_TOKEN_BUDGET
        )
        body = _retrieve_and_format_baseline(budget)
        if body:
            sout.write(body)
    except Exception:  # non-blocking: surface but do not fail
        traceback.print_exc(file=serr)
    return 0


def _retrieve_and_format_baseline(token_budget: int) -> str:
    """Retrieve L0 locked beliefs and emit them as the baseline block.

    Calls retrieve() with an empty query so only the L0 layer fires.
    Equivalent to MemoryStore.list_locked_beliefs() filtered through
    retrieve()'s budget logic, which leaves L0 untrimmed even when
    the locked set alone exceeds the budget.
    """
    store = _open_store()
    try:
        hits = retrieve(store, "", token_budget=token_budget)
    finally:
        store.close()
    if not hits:
        return ""
    return _format_baseline_hits(hits)


def _format_baseline_hits(hits: list[Belief]) -> str:
    """Format SessionStart block.

    Same per-line shape as `_format_hits` (the UserPromptSubmit
    formatter) but wrapped in distinct <aelfrice-baseline> tags so
    the model can tell which channel a belief arrived through. The
    [locked] prefix renders identically.
    """
    lines: list[str] = [SESSION_START_OPEN_TAG]
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        lines.append(f"{prefix} {h.id}: {h.content}")
    lines.append(SESSION_START_CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Entry point for `python -m aelfrice.hook`."""
    return user_prompt_submit()


def main_pre_compact() -> int:
    """Entry point for the PreCompact hook console script."""
    return pre_compact()


def main_session_start() -> int:
    """Entry point for the SessionStart hook console script."""
    return session_start()


if __name__ == "__main__":
    sys.exit(main())
