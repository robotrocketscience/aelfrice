"""`aelf-agy-hook`: the one hook command the Antigravity CLI (agy) runs.

agy runs this for each of its six hook events (``host_antigravity``
writes the table). It reads the agy payload, builds Claude-shaped
payloads with `aelfrice.agy_payload`, runs the existing hooks in process
on them, and writes the output agy reads. Event mapping, from
``docs/design/antigravity_hook_surface.md``:

==================  ==========================================================
agy event           aelfrice roles
==================  ==========================================================
SessionStart        session_start (baseline); marks the conversation
PreInvocation, 0    session_start if the conversation has no baseline (resume),
                    transcript_ingest (user half), user_prompt_submit
PreInvocation, >0   search_tool_bash: injects context queued by PreToolUse
PreToolUse          pre_issue_guard and search_tool_bash on ``run_command``,
                    agent_context on ``invoke_subagent``
PostToolUse         commit_ingest on ``run_command``
Stop                transcript_ingest (assistant half), stop
==================  ==========================================================

``--skip a,b`` drops roles, so `aelf setup --no-X` flags carry over.

Contract: always exit 0. agy blocks the tool when a PreToolUse hook exits
non-zero, so any failure is caught, reported on stderr, and turned into
the pass output. A PreToolUse pass prints nothing (``{}`` would deny the
tool); every other event prints a JSON object.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import IO, Final, cast

from aelfrice import agy_payload as ap
from aelfrice import agy_state
from aelfrice.agy_payload import (
    ALL_ROLES,
    ROLE_AGENT_CONTEXT,
    ROLE_COMMIT,
    ROLE_PRE_ISSUE,
    ROLE_SEARCH_BASH,
    ROLE_SESSION_START,
    ROLE_STOP,
    ROLE_TRANSCRIPT,
    ROLE_USER_PROMPT,
)


SKIP_FLAG: Final[str] = "--skip"
MAX_PAYLOAD_CHARS: Final[int] = 4 * 1024 * 1024
STOP_ALLOW: Final[dict[str, object]] = {"decision": "stop"}
EMPTY: Final[dict[str, object]] = {}
PRE_ISSUE_BLOCK_CODE: Final[int] = 2
_PRE_ISSUE_DEFAULT_REASON: Final[str] = (
    "aelfrice pre-issue guard: a similar issue or commit already exists"
)

# A hook entrypoint taking (stdin, stdout, stderr) keyword streams.
_StreamHook = Callable[..., int]


def parse_skip(argv: list[str]) -> frozenset[str]:
    """Roles named by `--skip a,b` (repeatable); unknown names are ignored."""
    skipped: set[str] = set()
    for i, arg in enumerate(argv):
        value: str | None = None
        if arg == SKIP_FLAG and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(SKIP_FLAG + "="):
            value = arg.split("=", 1)[1]
        if value:
            skipped.update(v.strip() for v in value.split(","))
    return frozenset(skipped & ALL_ROLES)


def _call(fn: _StreamHook, payload: Mapping[str, object], stderr: IO[str]) -> str:
    """Run a Claude hook entrypoint on `payload`; return what it wrote."""
    sin = io.StringIO(json.dumps(dict(payload)))
    sout = io.StringIO()
    fn(stdin=sin, stdout=sout, stderr=stderr)
    return sout.getvalue()


def _session_start_hook(payload: Mapping[str, object], stderr: IO[str]) -> str:
    from aelfrice.hook import session_start  # noqa: PLC0415

    return ap.hook_output_context(
        _call(session_start, ap.to_claude_session_start(payload), stderr)
    )


def _user_prompt_hook(claude: Mapping[str, object], stderr: IO[str]) -> str:
    from aelfrice.hook import user_prompt_submit  # noqa: PLC0415

    return ap.hook_output_context(_call(user_prompt_submit, claude, stderr))


def _transcript_hook(claude: Mapping[str, object], stderr: IO[str]) -> None:
    from aelfrice.transcript_logger import main as logger_main  # noqa: PLC0415

    logger_main(stdin=io.StringIO(json.dumps(dict(claude))), stderr=stderr)


def on_session_start(
    payload: Mapping[str, object], skip: frozenset[str], stderr: IO[str],
) -> dict[str, object]:
    agy_state.sweep_stale()
    cid = ap.conversation_id(payload)
    if ROLE_SESSION_START in skip:
        return EMPTY
    context = _session_start_hook(payload, stderr)
    if cid is not None:
        agy_state.mark_baseline(cid)
    return ap.inject_output([context]) or EMPTY


def on_pre_invocation(
    payload: Mapping[str, object], skip: frozenset[str], stderr: IO[str],
) -> dict[str, object]:
    cid = ap.conversation_id(payload)
    if ap.invocation_num(payload) > 0:
        if cid is None or ROLE_SEARCH_BASH in skip:
            return EMPTY
        return ap.inject_output([agy_state.take_pending(cid)]) or EMPTY
    texts: list[str] = []
    if cid is not None:
        # Context queued in a turn that ended before its next model call
        # (a cancel) belongs to that turn, not this one.
        agy_state.take_pending(cid)
        if ROLE_SESSION_START not in skip and not agy_state.has_baseline(cid):
            texts.append(_session_start_hook(payload, stderr))
        agy_state.mark_baseline(cid)
    claude = ap.to_claude_prompt(payload)
    if claude is not None:
        if ROLE_TRANSCRIPT not in skip:
            _transcript_hook(claude, stderr)
        if ROLE_USER_PROMPT not in skip:
            texts.append(_user_prompt_hook(claude, stderr))
    return ap.inject_output(texts) or EMPTY


def _pre_issue_reason(claude: dict[str, object], stderr: IO[str]) -> str | None:
    from aelfrice.pre_issue_create_hook import run_guard  # noqa: PLC0415

    buf = io.StringIO()
    code = run_guard(claude, stderr_out=buf)
    if code != PRE_ISSUE_BLOCK_CODE:
        return None
    reason = buf.getvalue().strip()
    stderr.write(reason + "\n")
    return reason or _PRE_ISSUE_DEFAULT_REASON


def on_pre_tool(
    payload: Mapping[str, object], skip: frozenset[str], stderr: IO[str],
) -> dict[str, object] | None:
    claude = ap.to_claude_tool(payload)
    if claude is None:
        return None
    if claude.get("tool_name") == "Agent":
        if ROLE_AGENT_CONTEXT in skip:
            return None
        from aelfrice.hook_agent_context import main as agent_main  # noqa: PLC0415

        updated = ap.hook_output_updated_input(_call(agent_main, claude, stderr))
        tool_input = cast(dict[str, object], claude["tool_input"])
        new_prompt = updated.get("prompt") if updated is not None else None
        if not isinstance(new_prompt, str) or new_prompt == tool_input.get("prompt"):
            return None
        return ap.pre_tool_output(overwrite={ap.SUBAGENT_PROMPT_ARG: new_prompt})
    if ROLE_PRE_ISSUE not in skip:
        reason = _pre_issue_reason(claude, stderr)
        if reason is not None:
            return ap.pre_tool_output(deny_reason=reason)
    cid = ap.conversation_id(payload)
    if ROLE_SEARCH_BASH not in skip and cid is not None:
        from aelfrice.hook_search_tool import main as search_main  # noqa: PLC0415

        context = ap.hook_output_context(_call(search_main, claude, stderr))
        agy_state.append_pending(cid, context)
    return None


def on_post_tool(
    payload: Mapping[str, object], skip: frozenset[str], stderr: IO[str],
) -> dict[str, object]:
    if ROLE_COMMIT in skip:
        return EMPTY
    call = ap.tool_call(payload)
    if call is None or call[0] != ap.SHELL_TOOL:
        return EMPTY
    claude = ap.to_claude_tool(payload, tool_output=ap.read_tool_output(payload))
    if claude is None:
        return EMPTY
    from aelfrice.hook_commit_ingest import main as commit_main  # noqa: PLC0415

    commit_main(stdin=io.StringIO(json.dumps(claude)), stderr=stderr, argv=[])
    return EMPTY


def on_stop(
    payload: Mapping[str, object], skip: frozenset[str], stderr: IO[str],
) -> dict[str, object]:
    transcript = payload.get("transcriptPath")
    text = ap.last_assistant_text(transcript if isinstance(transcript, str) else None)
    claude = ap.to_claude_stop(payload, text)
    if ROLE_TRANSCRIPT not in skip:
        _transcript_hook(claude, stderr)
    if ROLE_STOP in skip:
        return STOP_ALLOW
    from aelfrice.hook import stop  # noqa: PLC0415

    context = ap.hook_output_context(_call(stop, claude, stderr))
    return ap.stop_output(context, payload) or STOP_ALLOW


_Handler = Callable[
    [Mapping[str, object], frozenset[str], IO[str]], "dict[str, object] | None"
]

HANDLERS: Final[dict[str, _Handler]] = {
    ap.EVENT_SESSION_START: on_session_start,
    ap.EVENT_PRE_INVOCATION: on_pre_invocation,
    ap.EVENT_PRE_TOOL: on_pre_tool,
    ap.EVENT_POST_TOOL: on_post_tool,
    ap.EVENT_STOP: on_stop,
}


def _enter_project(payload: Mapping[str, object]) -> None:
    """chdir to the session's project; the hook starts in agy's config dir."""
    cwd = ap.project_dir(payload)
    if cwd is not None and os.path.isdir(cwd):
        os.chdir(cwd)


def dispatch(
    raw: str, skip: frozenset[str], stderr: IO[str],
) -> tuple[str | None, dict[str, object] | None]:
    """(event, output object or None) for one hook fire. May raise."""
    obj: object = json.loads(raw) if raw.strip() else {}
    payload = cast(dict[str, object], obj) if isinstance(obj, dict) else {}
    event = ap.event_name(payload)
    handler = HANDLERS.get(event) if event is not None else None
    if handler is None:
        return event, (None if event == ap.EVENT_PRE_TOOL else EMPTY)
    _enter_project(payload)
    return event, handler(payload, skip, stderr)


def _event_hint(raw: str) -> str | None:
    """The event name when the payload parses; used only on the error path."""
    try:
        obj: object = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return ap.event_name(cast(dict[str, object], obj)) if isinstance(obj, dict) else None


def main(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    skip = parse_skip(sys.argv[1:] if argv is None else argv)
    raw = ""
    try:
        raw = sin.read(MAX_PAYLOAD_CHARS)
        # A stray print() in a wrapped hook must not corrupt the one JSON
        # object agy parses, so process stdout points at stderr meanwhile.
        with contextlib.redirect_stdout(serr):
            event, output = dispatch(raw, skip, serr)
    except Exception:  # never block agy: report and fall back to the pass output
        traceback.print_exc(file=serr)
        event = _event_hint(raw)
        output = None if event == ap.EVENT_PRE_TOOL else (
            STOP_ALLOW if event == ap.EVENT_STOP else EMPTY
        )
    if output is not None:
        sout.write(json.dumps(output))
    sout.flush()
    return 0


def console_main() -> int:
    """`aelf-agy-hook` console script."""
    from aelfrice.stream_encoding import ensure_utf8_streams  # noqa: PLC0415

    ensure_utf8_streams()
    return main()


if __name__ == "__main__":
    sys.exit(console_main())
