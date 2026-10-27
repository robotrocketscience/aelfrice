"""Gemini CLI hook adapter (#1812): the `aelf-gemini-hook` console script.

`aelf setup --host gemini` registers this one script for every Gemini
event aelfrice uses. Gemini names its events differently from the claude
host, sends some fields under other names, and reads context only from
JSON on stdout, so the script translates in both directions and runs the
same handlers the claude host runs:

| Gemini event | Runs as | Handlers |
| --- | --- | --- |
| `SessionStart` | `SessionStart` | session-start baseline |
| `BeforeAgent` | `UserPromptSubmit` | retrieval, then the transcript logger |
| `AfterAgent` | `Stop` | the transcript logger, then the Stop hook |
| `PreCompress` | `PreCompact` | the transcript logger |
| `AfterTool` on `save_memory` | none | the saved-memory mirror below |

Gemini has no event after compression, so the logger's
`compaction_complete` marker is never written on this host.

Payload. The adapter renames `hook_event_name`, copies the `AfterAgent`
reply in `prompt_response` to `last_assistant_message`, where the
transcript logger reads a reply carried on the payload, and blanks
`transcript_path`: Gemini's transcript is one JSON document, which the
claude transcript readers would misread as JSON lines.

Output. Gemini parses stdout as JSON on exit 0 and shows any other stdout
to the user instead of the model, so the handlers' stdout is captured and
the script prints exactly one JSON object: `hookSpecificOutput` with
`additionalContext` on `SessionStart` and `BeforeAgent` when a handler
produced context, and `{}` otherwise. Gemini escapes `<` and `>` in that
text before the model sees it. The exit code is always 0, so no aelfrice
failure blocks a turn.

Host. The script sets `AELFRICE_HOST=gemini` in its own process for the
duration of the call, and the Stop hook reads it to skip the paths that
only the claude host supports.

Saved memories. Gemini's `save_memory` tool appends each fact under a
`## Gemini Added Memories` heading in the global `GEMINI.md`. After a
successful call the adapter reads that section and adds each fact the
store does not already hold as a transcript-sourced belief, so a fact the
store holds is never corroborated again by a later save. The mirror is one
way: aelfrice never writes `GEMINI.md`. Gemini's own `/hooks disable`
turns it off by the hook's name, `aelfrice-memory-mirror`.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import IO, Final, cast

from aelfrice.gemini_paths import (
    AFTER_AGENT_EVENT,
    AFTER_TOOL_EVENT,
    BEFORE_AGENT_EVENT,
    GLOBAL_CONTEXT_FILENAME,
    PRE_COMPRESS_EVENT,
    SAVE_MEMORY_TOOL,
    SESSION_START_EVENT,
    resolve_gemini_dir,
)
from aelfrice.hosts import GEMINI_HOST, HOST_ENV
from aelfrice.stream_encoding import ensure_utf8_streams, read_payload_text

#: The aelfrice event each translated Gemini event runs as.
EVENT_MAP: Final[Mapping[str, str]] = {
    SESSION_START_EVENT: "SessionStart",
    BEFORE_AGENT_EVENT: "UserPromptSubmit",
    AFTER_AGENT_EVENT: "Stop",
    PRE_COMPRESS_EVENT: "PreCompact",
}

#: The events whose `additionalContext` reaches the model, per the Gemini
#: hooks reference. `AfterTool` is one, but the mirror adds no context.
CONTEXT_EVENTS: Final[frozenset[str]] = frozenset(
    {SESSION_START_EVENT, BEFORE_AGENT_EVENT},
)

SAVED_MEMORIES_HEADING: Final[str] = "## Gemini Added Memories"


def to_aelfrice_payload(payload: Mapping[str, object]) -> dict[str, object] | None:
    """The payload the aelfrice handlers read, or None for an unmapped event."""
    event = payload.get("hook_event_name")
    if not isinstance(event, str) or event not in EVENT_MAP:
        return None
    mapped = dict(payload)
    mapped["hook_event_name"] = EVENT_MAP[event]
    mapped["transcript_path"] = ""
    if event == AFTER_AGENT_EVENT:
        reply = payload.get("prompt_response")
        if isinstance(reply, str):
            mapped["last_assistant_message"] = reply
    return mapped


def context_from_output(text: str) -> str:
    """The context a handler's stdout carries, or "" when it carries none.

    The claude-host handlers print context as plain text, except where they
    already print `hookSpecificOutput` JSON. Any other JSON object carries
    no context.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(parsed, dict):
        return stripped
    specific = cast(dict[str, object], parsed).get("hookSpecificOutput")
    if isinstance(specific, dict):
        context = cast(dict[str, object], specific).get("additionalContext")
        if isinstance(context, str):
            return context.strip()
    return ""


def render_output(event: str, context: str) -> str:
    """The one JSON object Gemini reads from this hook's stdout."""
    if not context or event not in CONTEXT_EVENTS:
        return "{}\n"
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": context,
        },
    }) + "\n"


# --- the handlers each event runs ---------------------------------------

_Handler = Callable[[str, IO[str], IO[str]], None]


def _run_session_start(text: str, out: IO[str], err: IO[str]) -> None:
    from aelfrice.hook import session_start  # noqa: PLC0415

    session_start(stdin=io.StringIO(text), stdout=out, stderr=err)


def _run_retrieval(text: str, out: IO[str], err: IO[str]) -> None:
    from aelfrice.hook import user_prompt_submit  # noqa: PLC0415

    user_prompt_submit(stdin=io.StringIO(text), stdout=out, stderr=err)


def _run_stop(text: str, out: IO[str], err: IO[str]) -> None:
    from aelfrice.hook import stop  # noqa: PLC0415

    stop(stdin=io.StringIO(text), stdout=out, stderr=err)


def _run_logger(text: str, out: IO[str], err: IO[str]) -> None:
    _ = out
    from aelfrice.transcript_logger import main as logger_main  # noqa: PLC0415

    logger_main(stdin=io.StringIO(text), stderr=err)


#: Handlers in run order. Retrieval runs before the logger so the prompt
#: being answered is not yet in the turn log it reads; the logger runs
#: before the Stop hook, as on the codex host.
HANDLERS: Final[Mapping[str, tuple[_Handler, ...]]] = {
    SESSION_START_EVENT: (_run_session_start,),
    BEFORE_AGENT_EVENT: (_run_retrieval, _run_logger),
    AFTER_AGENT_EVENT: (_run_logger, _run_stop),
    PRE_COMPRESS_EVENT: (_run_logger,),
}


def _run_handlers(
    event: str, mapped: dict[str, object], serr: IO[str],
) -> list[str]:
    """Run `event`'s handlers on `mapped`; the context each one produced.

    A handler that raises is reported on stderr and the rest still run.
    `sys.stdout` is redirected as well as passed in, so a handler that
    prints past its `stdout` argument cannot add text to the JSON object
    Gemini parses.
    """
    text = json.dumps(mapped)
    contexts: list[str] = []
    for handler in HANDLERS[event]:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                handler(text, buf, serr)
        except Exception:  # non-blocking: one handler never stops the rest
            traceback.print_exc(file=serr)
        context = context_from_output(buf.getvalue())
        if context:
            contexts.append(context)
    return contexts


# --- the saved-memory mirror ----------------------------------------------


def saved_memories(text: str) -> list[str]:
    """The facts under the saved-memories heading of a `GEMINI.md` text.

    Reads from the heading to the next Markdown heading. Each non-blank
    line is one fact, with a leading `-` or `*` list marker removed.
    Gemini documents the heading; the list-item form of each fact is an
    assumption, and a line in another form is kept whole.
    """
    facts: list[str] = []
    inside = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == SAVED_MEMORIES_HEADING:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("#"):
            break
        if line[:2] in ("- ", "* "):
            line = line[2:].strip()
        if line:
            facts.append(line)
    return facts


def _save_succeeded(payload: Mapping[str, object]) -> bool:
    if payload.get("tool_name") != SAVE_MEMORY_TOOL:
        return False
    response = payload.get("tool_response")
    if isinstance(response, dict):
        return not cast(dict[str, object], response).get("error")
    return True


def mirror_saved_memories(
    payload: Mapping[str, object], *, gemini_dir: Path | None = None,
) -> int:
    """Add the saved facts the store lacks; the number added.

    Returns 0 without opening the store unless `payload` is a successful
    `save_memory` call and the global context file holds a fact.
    """
    if not _save_succeeded(payload):
        return 0
    target = gemini_dir if gemini_dir is not None else resolve_gemini_dir()
    try:
        text = (target / GLOBAL_CONTEXT_FILENAME).read_text(
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return 0
    facts = saved_memories(text)
    if not facts:
        return 0

    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.derivation import DerivationInput, derive  # noqa: PLC0415
    from aelfrice.models import (  # noqa: PLC0415
        CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
        INGEST_SOURCE_TRANSCRIPT,
    )
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    path = db_path()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    session = payload.get("session_id")
    store = MemoryStore(str(path))
    added = 0
    try:
        for fact in facts:
            belief = derive(DerivationInput(
                raw_text=fact,
                source_kind=INGEST_SOURCE_TRANSCRIPT,
                session_id=session if isinstance(session, str) else None,
            )).belief
            if belief is None:
                continue
            if store.get_belief_by_content_hash(
                belief.content_hash, include_retired=True,
            ) is not None:
                continue
            store.insert_or_corroborate(
                belief, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            )
            added += 1
    finally:
        store.close()
    return added


# --- entry point ------------------------------------------------------------


@contextlib.contextmanager
def _host_marker() -> Generator[None]:
    """Set `AELFRICE_HOST=gemini` for the call, then restore the old value."""
    previous = os.environ.get(HOST_ENV)
    os.environ[HOST_ENV] = GEMINI_HOST
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(HOST_ENV, None)
        else:
            os.environ[HOST_ENV] = previous


def _read_payload(sin: IO[str], serr: IO[str]) -> dict[str, object] | None:
    raw = read_payload_text(sin, serr)
    if raw is None or not raw.strip():
        return None
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast(dict[str, object], parsed)


def main(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 and prints one JSON object."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    ensure_utf8_streams((sout, serr))
    event = ""
    contexts: list[str] = []
    with _host_marker():
        try:
            payload = _read_payload(sin, serr)
            raw_event = payload.get("hook_event_name") if payload else None
            event = raw_event if isinstance(raw_event, str) else ""
            if payload is not None and event == AFTER_TOOL_EVENT:
                mirror_saved_memories(payload)
            elif payload is not None:
                mapped = to_aelfrice_payload(payload)
                if mapped is not None:
                    contexts = _run_handlers(event, mapped, serr)
        except Exception:  # non-blocking: surface but never raise
            traceback.print_exc(file=serr)
    sout.write(render_output(event, "\n\n".join(contexts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
