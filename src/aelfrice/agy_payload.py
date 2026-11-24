"""Antigravity CLI (agy) hook payload and output translation.

aelfrice's hooks are written against Claude Code's payload and output
shapes. Under agy, one dispatcher (`aelfrice.agy_hook`) receives every
event, builds Claude-shaped payloads with the functions here, runs the
existing hooks on them, and turns their output back into what agy reads.
Every shape here comes from ``docs/design/antigravity_hook_surface.md``
(agy 1.5.7), which records observed behavior where it differs from the
Antigravity hooks reference.

Pure functions plus two bounded file reads. Stdlib only: this runs on
every model call and tool call.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

AGY_CONVERSATION_ENV: Final[str] = "ANTIGRAVITY_CONVERSATION_ID"

EVENT_SESSION_START: Final[str] = "SessionStart"
EVENT_PRE_INVOCATION: Final[str] = "PreInvocation"
EVENT_POST_INVOCATION: Final[str] = "PostInvocation"
EVENT_PRE_TOOL: Final[str] = "PreToolUse"
EVENT_POST_TOOL: Final[str] = "PostToolUse"
EVENT_STOP: Final[str] = "Stop"

# Dispatcher roles, named after the auto-install manifest installers. They
# live here, not in `agy_hook`, so `host_antigravity` (imported by `cli`)
# never imports the dispatcher and through it `aelfrice.hook` (#1631).
ROLE_SESSION_START: Final[str] = "session_start"
ROLE_USER_PROMPT: Final[str] = "user_prompt_submit"
ROLE_TRANSCRIPT: Final[str] = "transcript_ingest"
ROLE_STOP: Final[str] = "stop"
ROLE_SEARCH_BASH: Final[str] = "search_tool_bash"
ROLE_PRE_ISSUE: Final[str] = "pre_issue_guard"
ROLE_AGENT_CONTEXT: Final[str] = "agent_context"
ROLE_COMMIT: Final[str] = "commit_ingest"
ALL_ROLES: Final[frozenset[str]] = frozenset({
    ROLE_SESSION_START, ROLE_USER_PROMPT, ROLE_TRANSCRIPT, ROLE_STOP,
    ROLE_SEARCH_BASH, ROLE_PRE_ISSUE, ROLE_AGENT_CONTEXT, ROLE_COMMIT,
})

SHELL_TOOL: Final[str] = "run_command"
SUBAGENT_TOOL: Final[str] = "invoke_subagent"

# agy tool name -> the Claude Code name the hooks match on.
TOOL_ALIASES: Final[dict[str, str]] = {
    SHELL_TOOL: "Bash",
    SUBAGENT_TOOL: "Agent",
}

# The agy argument that carries the subagent prompt; the agent-context hook
# rewrites it through PreToolUse `overwrite`.
SUBAGENT_PROMPT_ARG: Final[str] = "Prompt"

# Reads of agy-owned files are capped so a huge transcript or tool output
# cannot stall a hook past its timeout.
MAX_TRANSCRIPT_TAIL_BYTES: Final[int] = 2 * 1024 * 1024
MAX_TOOL_OUTPUT_BYTES: Final[int] = 256 * 1024
MAX_TRANSCRIPT_LINES: Final[int] = 20_000

_MODEL_STEP_TYPE: Final[str] = "PLANNER_RESPONSE"
_MODEL_SOURCE: Final[str] = "MODEL"


def is_antigravity_env(environ: Mapping[str, str] | None = None) -> bool:
    """True when this process is an agy hook process.

    agy sets `ANTIGRAVITY_CONVERSATION_ID` in every hook process and no
    other host-identifying variable (surface doc, Hook process environment).
    """
    env = environ if environ is not None else os.environ
    return bool(env.get(AGY_CONVERSATION_ENV))


def _str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def event_name(payload: Mapping[str, object]) -> str | None:
    return _str(payload, "hookEventName")


def conversation_id(payload: Mapping[str, object]) -> str | None:
    return _str(payload, "conversationId")


def invocation_num(payload: Mapping[str, object]) -> int:
    """`invocationNum`; restarts at 0 on each user turn (surface doc)."""
    return _int(payload, "invocationNum")


def execution_num(payload: Mapping[str, object]) -> int:
    """`executionNum` on Stop; 0 on the first stop of a turn.

    This is the loop guard. `continuationActive` stayed false across
    forced continuations on agy 1.5.7, so it is never read.
    """
    return _int(payload, "executionNum")


def project_dir(payload: Mapping[str, object]) -> str | None:
    """The session's project directory.

    The hook process runs in the folder that holds hooks.json, never the
    project, so the payload's `cwd` (then the first workspace path) is the
    only source.
    """
    cwd = _str(payload, "cwd")
    if cwd is not None:
        return cwd
    paths = payload.get("workspacePaths")
    if isinstance(paths, list):
        for item in cast(list[object], paths):
            if isinstance(item, str) and item:
                return item
    return None


def _base(payload: Mapping[str, object], claude_event: str) -> dict[str, object]:
    out: dict[str, object] = {"hook_event_name": claude_event}
    sid = conversation_id(payload)
    if sid is not None:
        out["session_id"] = sid
    cwd = project_dir(payload)
    if cwd is not None:
        out["cwd"] = cwd
    return out


def to_claude_prompt(payload: Mapping[str, object]) -> dict[str, object] | None:
    """A Claude `UserPromptSubmit` payload from a PreInvocation, or None.

    agy's transcript is not Claude JSONL, so `transcript_path` is left out
    rather than pointed at a file the Claude parsers would misread.
    """
    prompt = _str(payload, "lastUserInput")
    if prompt is None or not prompt.strip():
        return None
    out = _base(payload, "UserPromptSubmit")
    out["prompt"] = prompt
    return out


def to_claude_session_start(payload: Mapping[str, object]) -> dict[str, object]:
    out = _base(payload, "SessionStart")
    out["source"] = "startup"
    return out


def tool_call(payload: Mapping[str, object]) -> tuple[str, dict[str, object]] | None:
    """(agy tool name, args) from a tool event payload, or None."""
    call = payload.get("toolCall")
    if not isinstance(call, dict):
        return None
    call_d = cast(dict[str, object], call)
    name = call_d.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = call_d.get("args")
    return name, (cast(dict[str, object], args) if isinstance(args, dict) else {})


def _claude_tool_input(name: str, args: Mapping[str, object]) -> dict[str, object] | None:
    if name == SHELL_TOOL:
        command = args.get("CommandLine")
        return {"command": command} if isinstance(command, str) else None
    if name == SUBAGENT_TOOL:
        prompt = args.get(SUBAGENT_PROMPT_ARG)
        if not isinstance(prompt, str):
            return None
        out: dict[str, object] = {"prompt": prompt}
        role = args.get("Role")
        if isinstance(role, str):
            out["description"] = role
        kind = args.get("TypeName")
        if isinstance(kind, str):
            out["subagent_type"] = kind
        return out
    return None


def to_claude_tool(
    payload: Mapping[str, object],
    *,
    tool_output: str | None = None,
) -> dict[str, object] | None:
    """A Claude PreToolUse/PostToolUse payload, or None for an unmapped tool.

    `tool_output` (PostToolUse only) becomes `tool_response.stdout`; agy's
    payload carries no output, so the caller reads it from disk with
    `read_tool_output`. A non-empty `error` marks the response as failed.
    For the shell tool, its `Cwd` argument wins over the session's `cwd`,
    because that is where the command ran.
    """
    event = event_name(payload)
    if event not in (EVENT_PRE_TOOL, EVENT_POST_TOOL):
        return None
    call = tool_call(payload)
    if call is None:
        return None
    name, args = call
    alias = TOOL_ALIASES.get(name)
    tool_input = _claude_tool_input(name, args)
    if alias is None or tool_input is None:
        return None
    out = _base(payload, event)
    run_cwd = args.get("Cwd")
    if name == SHELL_TOOL and isinstance(run_cwd, str) and run_cwd:
        out["cwd"] = run_cwd
    out["tool_name"] = alias
    out["tool_input"] = tool_input
    if event == EVENT_POST_TOOL:
        error = payload.get("error")
        out["tool_response"] = {
            "stdout": tool_output or "",
            "isError": isinstance(error, str) and bool(error.strip()),
        }
    return out


def to_claude_stop(
    payload: Mapping[str, object], assistant_text: str | None,
) -> dict[str, object]:
    """A Claude Stop payload; `stop_hook_active` follows `executionNum`."""
    out = _base(payload, "Stop")
    out["stop_hook_active"] = execution_num(payload) > 0
    if assistant_text:
        out["last_assistant_message"] = assistant_text
    return out


def _read_tail(path: Path, limit: int) -> str:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - limit))
        data = fh.read(limit)
    return data.decode("utf-8", errors="replace")


def last_assistant_text(transcript_path: str | None) -> str | None:
    """The last model reply in an agy transcript, or None.

    Reads at most the file's last `MAX_TRANSCRIPT_TAIL_BYTES`. A model
    step is `type == "PLANNER_RESPONSE"` with `source == "MODEL"`; its
    reply is `content` (steps that only call tools have none).
    """
    if not transcript_path:
        return None
    path = Path(transcript_path)
    try:
        text = _read_tail(path, MAX_TRANSCRIPT_TAIL_BYTES)
    except OSError:
        return None
    lines = text.splitlines()[-MAX_TRANSCRIPT_LINES:]
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        row_d = cast(dict[str, object], row)
        if row_d.get("type") != _MODEL_STEP_TYPE or row_d.get("source") != _MODEL_SOURCE:
            continue
        content = row_d.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def read_tool_output(payload: Mapping[str, object]) -> str | None:
    """A tool step's output from `<artifactDirectoryPath>/.system_generated/steps/<stepIdx>/output.txt`.

    Capped at `MAX_TOOL_OUTPUT_BYTES` from the start of the file. None
    when the path fields are missing or the file cannot be read.
    """
    base = _str(payload, "artifactDirectoryPath")
    step = payload.get("stepIdx")
    if base is None or isinstance(step, bool) or not isinstance(step, int):
        return None
    path = Path(base) / ".system_generated" / "steps" / str(step) / "output.txt"
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_TOOL_OUTPUT_BYTES)
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def hook_output_context(text: str) -> str:
    """The context a Claude hook wrote to stdout.

    A JSON object with `hookSpecificOutput.additionalContext` yields that
    string; any other non-empty output is plain-text context (the
    UserPromptSubmit and SessionStart hooks write plain text).
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(obj, dict):
            specific = cast(dict[str, object], obj).get("hookSpecificOutput")
            if isinstance(specific, dict):
                ctx = cast(dict[str, object], specific).get("additionalContext")
                return ctx if isinstance(ctx, str) else ""
            return ""
    return stripped


def hook_output_updated_input(text: str) -> dict[str, object] | None:
    """`hookSpecificOutput.updatedInput` from a Claude hook's stdout, or None."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    specific = cast(dict[str, object], obj).get("hookSpecificOutput")
    if not isinstance(specific, dict):
        return None
    updated = cast(dict[str, object], specific).get("updatedInput")
    return cast(dict[str, object], updated) if isinstance(updated, dict) else None


def inject_output(texts: list[str]) -> dict[str, object] | None:
    """agy `injectSteps` output carrying each non-empty text, or None.

    Uses `systemMessage`, the nested-object form: it reaches the model and
    persists for the rest of the turn. `ephemeralMessage` is seen only on
    the call it was injected for (surface doc, Events).
    """
    steps: list[dict[str, object]] = [
        {"systemMessage": {"systemMessage": t}} for t in texts if t.strip()
    ]
    return {"injectSteps": steps} if steps else None


def pre_tool_output(
    *, deny_reason: str | None = None, overwrite: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """agy PreToolUse output, or None to print nothing.

    Printing nothing allows the tool. `{}` (and any object without a valid
    `decision`) denies it, so a pass never prints an object.
    """
    if deny_reason is not None:
        return {"decision": "deny", "reason": deny_reason}
    if overwrite:
        return {"decision": "allow", "overwrite": overwrite}
    return None


def stop_output(continue_reason: str, payload: Mapping[str, object]) -> dict[str, object] | None:
    """agy Stop output that continues the turn once, or None.

    Only the first stop of a turn (`executionNum == 0`) may continue; agy
    would otherwise loop up to its forced-continuation cap.
    """
    if not continue_reason.strip() or execution_num(payload) > 0:
        return None
    return {"decision": "continue", "reason": continue_reason}
