"""agy payload translation (`aelfrice.agy_payload`).

Payload shapes are the ones recorded in
docs/design/antigravity_hook_surface.md for agy 1.5.7.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import agy_payload as ap

pytestmark = pytest.mark.timeout(30)

BASE: dict[str, object] = {
    "conversationId": "conv-1",
    "cwd": "/work/proj",
    "workspacePaths": ["/work/proj"],
    "transcriptPath": "/nonexistent/transcript_full.jsonl",
    "artifactDirectoryPath": "/nonexistent/brain/conv-1",
    "modelName": "gemini-3.8-flash-high",
    "lastUserInput": "what did we decide about the cache?",
}


def _p(event: str, **extra: object) -> dict[str, object]:
    return {**BASE, "hookEventName": event, **extra}


def test_env_detection_reads_only_the_conversation_variable() -> None:
    assert ap.is_antigravity_env({"ANTIGRAVITY_CONVERSATION_ID": "x"})
    assert not ap.is_antigravity_env({"ANTIGRAVITY_CONVERSATION_ID": ""})
    assert not ap.is_antigravity_env({"GEMINI_SESSION_ID": "x"})


def test_prompt_payload_maps_last_user_input_and_drops_transcript() -> None:
    out = ap.to_claude_prompt(_p("PreInvocation", invocationNum=0))
    assert out == {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "conv-1",
        "cwd": "/work/proj",
        "prompt": "what did we decide about the cache?",
    }


def test_prompt_payload_is_none_without_text() -> None:
    assert ap.to_claude_prompt(_p("PreInvocation", lastUserInput="   ")) is None


def test_project_dir_falls_back_to_first_workspace() -> None:
    payload = {"workspacePaths": ["", "/w/second"], "hookEventName": "Stop"}
    assert ap.project_dir(payload) == "/w/second"


@pytest.mark.parametrize(
    ("raw", "expected"), [(0, 0), (3, 3), ("2", 2), (True, 0), (None, 0)],
)
def test_invocation_num_parsing(raw: object, expected: int) -> None:
    assert ap.invocation_num({"invocationNum": raw}) == expected


def test_shell_pre_tool_maps_to_bash_and_uses_the_command_cwd() -> None:
    payload = _p(
        "PreToolUse",
        toolCall={"id": "c1", "name": "run_command",
                  "args": {"CommandLine": "grep -rn foo .", "Cwd": "/work/proj/sub"}},
        stepIdx=4,
    )
    out = ap.to_claude_tool(payload)
    assert out is not None
    assert out["tool_name"] == "Bash"
    assert out["tool_input"] == {"command": "grep -rn foo ."}
    assert out["cwd"] == "/work/proj/sub"
    assert "tool_response" not in out


def test_post_tool_carries_output_and_error_flag() -> None:
    call = {"id": "c1", "name": "run_command", "args": {"CommandLine": "git commit -m x"}}
    ok = ap.to_claude_tool(_p("PostToolUse", toolCall=call, error=""), tool_output="[main abc123] x")
    bad = ap.to_claude_tool(_p("PostToolUse", toolCall=call, error="exit status 1"))
    assert ok is not None and bad is not None
    assert ok["tool_response"] == {"stdout": "[main abc123] x", "isError": False}
    assert bad["tool_response"] == {"stdout": "", "isError": True}


def test_subagent_maps_to_agent_prompt() -> None:
    call = {"id": "c2", "name": "invoke_subagent",
            "args": {"Prompt": "find needle", "Role": "Researcher", "TypeName": "research"}}
    out = ap.to_claude_tool(_p("PreToolUse", toolCall=call))
    assert out is not None
    assert out["tool_name"] == "Agent"
    assert out["tool_input"] == {
        "prompt": "find needle", "description": "Researcher", "subagent_type": "research",
    }


def test_unmapped_tool_is_none() -> None:
    call = {"id": "c3", "name": "read_file", "args": {"Path": "/x"}}
    assert ap.to_claude_tool(_p("PreToolUse", toolCall=call)) is None


def test_stop_hook_active_follows_execution_num() -> None:
    first = ap.to_claude_stop(_p("Stop", executionNum=0, continuationActive=False), "hi")
    later = ap.to_claude_stop(_p("Stop", executionNum=1, continuationActive=False), None)
    assert first["stop_hook_active"] is False
    assert first["last_assistant_message"] == "hi"
    assert later["stop_hook_active"] is True
    assert "last_assistant_message" not in later


def test_last_assistant_text_takes_the_last_model_reply(tmp_path: Path) -> None:
    rows = [
        {"type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "q"},
        {"type": "PLANNER_RESPONSE", "source": "MODEL", "content": "first answer"},
        {"type": "PLANNER_RESPONSE", "source": "MODEL", "tool_calls": "[...]"},
        {"type": "PLANNER_RESPONSE", "source": "MODEL", "content": "final answer"},
        {"type": "SYSTEM_MESSAGE", "source": "SYSTEM_SDK", "content": "injected"},
    ]
    path = tmp_path / "transcript_full.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    assert ap.last_assistant_text(str(path)) == "final answer"
    assert ap.last_assistant_text(str(tmp_path / "missing.jsonl")) is None
    assert ap.last_assistant_text(None) is None


def test_read_tool_output_reads_the_step_file(tmp_path: Path) -> None:
    step = tmp_path / ".system_generated" / "steps" / "4"
    step.mkdir(parents=True)
    (step / "output.txt").write_text("[main 1a2b3c] msg\n", encoding="utf-8")
    payload = {"artifactDirectoryPath": str(tmp_path), "stepIdx": 4}
    assert ap.read_tool_output(payload) == "[main 1a2b3c] msg\n"
    assert ap.read_tool_output({"artifactDirectoryPath": str(tmp_path), "stepIdx": 5}) is None
    assert ap.read_tool_output({"artifactDirectoryPath": str(tmp_path), "stepIdx": True}) is None


def test_hook_output_context_reads_both_shapes() -> None:
    js = json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                            "additionalContext": "ctx"}})
    assert ap.hook_output_context(js) == "ctx"
    assert ap.hook_output_context("<aelfrice-memory>x</aelfrice-memory>\n") == (
        "<aelfrice-memory>x</aelfrice-memory>"
    )
    assert ap.hook_output_context("  \n") == ""
    assert ap.hook_output_context(json.dumps({"other": 1})) == ""


def test_inject_output_uses_nested_system_message() -> None:
    assert ap.inject_output(["", "  "]) is None
    assert ap.inject_output(["a", "", "b"]) == {"injectSteps": [
        {"systemMessage": {"systemMessage": "a"}},
        {"systemMessage": {"systemMessage": "b"}},
    ]}


def test_pre_tool_pass_prints_nothing_and_deny_names_the_reason() -> None:
    # `{}` denies the tool on agy, so a pass must be None (print nothing).
    assert ap.pre_tool_output() is None
    assert ap.pre_tool_output(overwrite={}) is None
    assert ap.pre_tool_output(deny_reason="dup") == {"decision": "deny", "reason": "dup"}
    assert ap.pre_tool_output(overwrite={"Prompt": "p"}) == {
        "decision": "allow", "overwrite": {"Prompt": "p"},
    }


def test_stop_continues_only_on_the_first_stop_of_a_turn() -> None:
    # continuationActive stays false on agy 1.5.7; executionNum is the guard.
    first = {"executionNum": 0, "continuationActive": False}
    again = {"executionNum": 1, "continuationActive": False}
    assert ap.stop_output("label these", first) == {
        "decision": "continue", "reason": "label these",
    }
    assert ap.stop_output("label these", again) is None
    assert ap.stop_output("  ", first) is None
