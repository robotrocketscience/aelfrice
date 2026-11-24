"""The agy dispatcher (`aelfrice.agy_hook`): routing, gating, and output.

The wrapped Claude hooks are replaced with fakes that record the payload
they were given and write a fixed output, so these tests pin what the
dispatcher sends and what it prints, not retrieval itself.
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import IO

import pytest

from aelfrice import agy_hook, agy_state

pytestmark = pytest.mark.timeout(30)

Calls = list[tuple[str, dict[str, object]]]


def _fake(name: str, calls: Calls, output: str = "", code: int = 0) -> Callable[..., int]:
    def fn(*, stdin: IO[str], stdout: IO[str] | None = None,
           stderr: IO[str] | None = None, argv: list[str] | None = None) -> int:
        _ = (stderr, argv)
        calls.append((name, json.loads(stdin.read())))
        if stdout is not None:
            stdout.write(output)
        return code
    return fn


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Calls:
    import aelfrice.hook as hook
    import aelfrice.hook_agent_context as agent
    import aelfrice.hook_commit_ingest as commit
    import aelfrice.hook_search_tool as search
    import aelfrice.pre_issue_create_hook as pre_issue
    import aelfrice.transcript_logger as logger

    monkeypatch.setenv(agy_state.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    recorded: Calls = []
    monkeypatch.setattr(hook, "session_start", _fake("session_start", recorded, "BASELINE"))
    monkeypatch.setattr(hook, "user_prompt_submit", _fake("ups", recorded, "<aelfrice-memory>M</aelfrice-memory>"))
    monkeypatch.setattr(hook, "stop", _fake("stop", recorded, ""))
    monkeypatch.setattr(logger, "main", _fake("logger", recorded))
    monkeypatch.setattr(commit, "main", _fake("commit", recorded))
    search_out = json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                    "additionalContext": "SEARCH-CTX"}})
    monkeypatch.setattr(search, "main", _fake("search", recorded, search_out))
    agent_out = json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                   "updatedInput": {"prompt": "BLOCK\n\nfind"}}})
    monkeypatch.setattr(agent, "main", _fake("agent", recorded, agent_out))

    def guard(payload: dict[str, object], *, stderr_out: IO[str]) -> int:
        recorded.append(("guard", payload))
        command = str(payload.get("tool_input", {}))
        if "gh issue create" in command:
            stderr_out.write("BLOCK: duplicate of #12")
            return 2
        return 0
    monkeypatch.setattr(pre_issue, "run_guard", guard)
    return recorded


def _run(payload: dict[str, object], argv: list[str] | None = None) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = agy_hook.main(stdin=io.StringIO(json.dumps(payload)), stdout=out,
                         stderr=err, argv=argv or [])
    assert code == 0
    return out.getvalue(), err.getvalue()


def _base(event: str, cwd: Path, **extra: object) -> dict[str, object]:
    return {"hookEventName": event, "conversationId": "conv-1", "cwd": str(cwd),
            "lastUserInput": "how is the cache keyed?", **extra}


def _shell(cmd: str) -> dict[str, object]:
    return {"id": "c1", "name": "run_command", "args": {"CommandLine": cmd}}


def _injected(out: str) -> list[str]:
    obj = json.loads(out)
    return [s["systemMessage"]["systemMessage"] for s in obj.get("injectSteps", [])]


def test_session_start_injects_baseline_and_marks(calls: Calls, tmp_path: Path) -> None:
    out, _ = _run(_base("SessionStart", tmp_path))
    assert _injected(out) == ["BASELINE"]
    assert calls[0][1]["hook_event_name"] == "SessionStart"
    assert agy_state.has_baseline("conv-1")


def test_first_invocation_runs_prompt_hooks_without_repeating_baseline(
    calls: Calls, tmp_path: Path,
) -> None:
    _run(_base("SessionStart", tmp_path))
    calls.clear()
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=0))
    assert [c[0] for c in calls] == ["logger", "ups"]
    assert calls[1][1]["prompt"] == "how is the cache keyed?"
    assert _injected(out) == ["<aelfrice-memory>M</aelfrice-memory>"]


def test_resumed_conversation_gets_baseline_at_first_invocation(
    calls: Calls, tmp_path: Path,
) -> None:
    # agy -c fires no SessionStart; the first PreInvocation supplies it once.
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=0))
    assert [c[0] for c in calls] == ["session_start", "logger", "ups"]
    assert _injected(out) == ["BASELINE", "<aelfrice-memory>M</aelfrice-memory>"]
    calls.clear()
    _run(_base("PreInvocation", tmp_path, invocationNum=0))
    assert "session_start" not in [c[0] for c in calls]


def test_later_invocations_only_drain_pending(calls: Calls, tmp_path: Path) -> None:
    out, _ = _run(_base("PreToolUse", tmp_path, toolCall=_shell("grep -rn cache src")))
    assert out == ""  # a pass prints nothing; `{}` would deny the tool
    assert [c[0] for c in calls] == ["guard", "search"]
    calls.clear()
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=1))
    assert calls == []
    assert _injected(out) == ["SEARCH-CTX"]
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=2))
    assert json.loads(out) == {}


def test_first_invocation_discards_context_left_from_a_cancelled_turn(
    calls: Calls, tmp_path: Path,
) -> None:
    agy_state.mark_baseline("conv-1")
    agy_state.append_pending("conv-1", "STALE")
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=0))
    assert "STALE" not in out
    assert agy_state.take_pending("conv-1") == ""


def test_pre_issue_block_denies_with_reason(calls: Calls, tmp_path: Path) -> None:
    out, _ = _run(_base("PreToolUse", tmp_path,
                        toolCall=_shell('gh issue create --title "cache bug"')))
    assert json.loads(out) == {"decision": "deny", "reason": "BLOCK: duplicate of #12"}
    assert "search" not in [c[0] for c in calls]


def test_subagent_prompt_is_rewritten_through_overwrite(calls: Calls, tmp_path: Path) -> None:
    call = {"id": "c2", "name": "invoke_subagent", "args": {"Prompt": "find", "Role": "R"}}
    out, _ = _run(_base("PreToolUse", tmp_path, toolCall=call))
    assert json.loads(out) == {"decision": "allow", "overwrite": {"Prompt": "BLOCK\n\nfind"}}
    assert calls[0][1]["tool_name"] == "Agent"


def test_unmapped_tool_passes_silently(calls: Calls, tmp_path: Path) -> None:
    call = {"id": "c3", "name": "read_file", "args": {"Path": "/x"}}
    out, _ = _run(_base("PreToolUse", tmp_path, toolCall=call))
    assert out == ""
    assert calls == []


def test_post_tool_feeds_step_output_to_commit_ingest(calls: Calls, tmp_path: Path) -> None:
    art = tmp_path / "brain"
    (art / ".system_generated" / "steps" / "7").mkdir(parents=True)
    (art / ".system_generated" / "steps" / "7" / "output.txt").write_text("[main abc1234] m\n")
    out, _ = _run(_base("PostToolUse", tmp_path, toolCall=_shell("git commit -m m"),
                        stepIdx=7, error="", artifactDirectoryPath=str(art)))
    assert json.loads(out) == {}
    name, payload = calls[0]
    assert name == "commit"
    assert payload["tool_response"] == {"stdout": "[main abc1234] m\n", "isError": False}


def test_stop_logs_reply_and_allows_stop(calls: Calls, tmp_path: Path) -> None:
    tr = tmp_path / "t.jsonl"
    tr.write_text(json.dumps({"type": "PLANNER_RESPONSE", "source": "MODEL", "content": "done"}) + "\n")
    out, _ = _run(_base("Stop", tmp_path, executionNum=0, transcriptPath=str(tr)))
    assert json.loads(out) == {"decision": "stop"}
    assert [c[0] for c in calls] == ["logger", "stop"]
    assert calls[0][1]["last_assistant_message"] == "done"
    assert calls[1][1]["stop_hook_active"] is False


def test_stop_continuation_only_on_first_stop(
    calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aelfrice.hook as hook
    ctx = json.dumps({"hookSpecificOutput": {"hookEventName": "Stop",
                                             "additionalContext": "LABEL"}})
    monkeypatch.setattr(hook, "stop", _fake("stop", calls, ctx))
    first, _ = _run(_base("Stop", tmp_path, executionNum=0, continuationActive=False))
    again, _ = _run(_base("Stop", tmp_path, executionNum=1, continuationActive=False))
    assert json.loads(first) == {"decision": "continue", "reason": "LABEL"}
    assert json.loads(again) == {"decision": "stop"}


def test_skip_flag_drops_roles(calls: Calls, tmp_path: Path) -> None:
    _run(_base("PreToolUse", tmp_path, toolCall=_shell("grep x")),
         argv=["--skip", "search_tool_bash,pre_issue_guard"])
    _run(_base("PreInvocation", tmp_path, invocationNum=0), argv=["--skip=user_prompt_submit"])
    assert [c[0] for c in calls] == ["session_start", "logger"]


def test_parse_skip_ignores_unknown_roles() -> None:
    assert agy_hook.parse_skip(["--skip", "stop,bogus"]) == frozenset({"stop"})
    assert agy_hook.parse_skip([]) == frozenset()


def test_wrapped_hook_crash_still_passes_the_tool(
    calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aelfrice.pre_issue_create_hook as pre_issue

    def boom(payload: dict[str, object], *, stderr_out: IO[str]) -> int:
        raise RuntimeError("guard exploded")
    monkeypatch.setattr(pre_issue, "run_guard", boom)
    out, err = _run(_base("PreToolUse", tmp_path, toolCall=_shell("ls")))
    assert out == ""
    assert "guard exploded" in err


def test_stray_print_does_not_reach_agy(
    calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aelfrice.hook as hook

    def noisy(*, stdin: IO[str], stdout: IO[str], stderr: IO[str]) -> int:
        print("stray text")
        stdout.write("BASELINE")
        return 0
    monkeypatch.setattr(hook, "session_start", noisy)
    out, err = _run(_base("SessionStart", tmp_path))
    assert _injected(out) == ["BASELINE"]
    assert "stray text" in err


def test_bad_json_and_unknown_events(calls: Calls) -> None:
    out, err = io.StringIO(), io.StringIO()
    assert agy_hook.main(stdin=io.StringIO("{not json"), stdout=out, stderr=err, argv=[]) == 0
    assert json.loads(out.getvalue()) == {}
    out2, _ = _run({"hookEventName": "PostInvocation", "conversationId": "c"})
    assert json.loads(out2) == {}
    assert calls == []


def test_hook_runs_in_the_payload_project(calls: Calls, tmp_path: Path) -> None:
    import os

    proj = tmp_path / "proj"
    proj.mkdir()
    _run(_base("SessionStart", proj))
    assert Path(os.getcwd()) == proj


def test_and_chain_reaches_the_model_through_the_real_search_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An `&&` chain is searched and queued for the next model call.

    Only the dispatcher's state dir is redirected; the search hook is the
    real one. The sandboxed HOME has no store, so the hook emits its
    empty-store block, which still names the lifted query.
    """
    monkeypatch.setenv(agy_state.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    chain = "git diff --name-only HEAD && grep -n -i searchterm state.jsonl && echo ok"
    out, _ = _run(_base("PreToolUse", tmp_path, toolCall=_shell(chain)))
    assert out == ""
    out, _ = _run(_base("PreInvocation", tmp_path, invocationNum=1))
    injected = _injected(out)
    assert len(injected) == 1
    assert 'query="searchterm"' in injected[0]
    assert 'source="bash:grep"' in injected[0]
