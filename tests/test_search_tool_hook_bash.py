"""Tests for the v1.5.0 #155 Bash matcher in
`aelfrice.hook_search_tool`.

Acceptance-criteria coverage from `docs/search_tool_hook.md
§ Bash extension`:

- AC1 — `test_allowlist_fires`, `test_unknown_command_silent_skip`,
        `test_random_garbage_never_fires`
- AC2 — per-command parser tests below
- AC4 — `test_no_fall_through_to_arbitrary_bash`
- AC5 — `test_per_turn_fire_cap_holds`
- AC6 — `test_bash_block_carries_source_attribute`,
        `test_grep_block_unchanged_no_source_attribute`

ACs 3 (telemetry-gated default-on flip), 7 (`aelf setup
--search-tool-bash`), and 8 (`aelf doctor` surface) are deferred
to a follow-up PR per the v1.5.0 split.
"""
from __future__ import annotations

import json
from io import StringIO
from typing import cast

import pytest

from aelfrice.hook_search_tool import (
    BASH_FIRE_CAP_PER_TURN,
    _extract_bash_query,
    _parse_bash_command,
    _reset_bash_fire_state,
    main,
)


def _payload_bash(command: str, session_id: str | None = "s1") -> str:
    inner: dict[str, object] = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": "/tmp",
    }
    if session_id is not None:
        inner["session_id"] = session_id
    return json.dumps(inner)


def _run_hook(payload: str) -> str:
    sin = StringIO(payload)
    sout = StringIO()
    serr = StringIO()
    rc = main(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    return sout.getvalue()


# --- AC1: allowlist semantics --------------------------------------------


def test_allowlist_fires_grep() -> None:
    parsed = _parse_bash_command("grep -r 'directive-gate' src/")
    assert parsed is not None
    query, cmd = parsed
    assert query == "directive-gate"
    assert cmd == "grep"


def test_allowlist_fires_rg_with_type_flag() -> None:
    parsed = _parse_bash_command("rg --type py foo src/")
    assert parsed is not None
    query, cmd = parsed
    assert query == "foo"
    assert cmd == "rg"


def test_allowlist_fires_ack_with_long_flag_equals_value() -> None:
    parsed = _parse_bash_command("ack --type=python pattern_x")
    assert parsed is not None
    query, cmd = parsed
    assert query == "pattern_x"
    assert cmd == "ack"


def test_allowlist_fires_find_with_name() -> None:
    parsed = _parse_bash_command("find . -name '*.py'")
    assert parsed is not None
    query, cmd = parsed
    assert query == "*.py"
    assert cmd == "find"


def test_find_without_name_silent_skips() -> None:
    """`find . -type f` has no -name; parser returns None."""
    assert _parse_bash_command("find . -type f") is None


def test_allowlist_fires_fd() -> None:
    parsed = _parse_bash_command("fd -e py mymodule")
    assert parsed is not None
    query, cmd = parsed
    assert query == "mymodule"
    assert cmd == "fd"


def test_unknown_command_silent_skip() -> None:
    assert _parse_bash_command("ls -la") is None
    assert _parse_bash_command("cat README.md") is None
    assert _parse_bash_command("cd /tmp") is None
    assert _parse_bash_command("echo hello") is None
    assert _parse_bash_command("python script.py") is None


def test_random_garbage_never_fires() -> None:
    """Property-style: a handful of non-allowlisted commands silent-skip."""
    samples = [
        "",
        " ",
        "true",
        "false",
        "git status",
        "make all",
        "uv run pytest",
        "docker compose up",
        "kubectl apply -f x.yaml",
    ]
    for cmd in samples:
        assert _parse_bash_command(cmd) is None, cmd


# --- AC2: per-command parsing --------------------------------------------


def test_grep_consumes_dash_e_flag_value() -> None:
    parsed = _parse_bash_command("grep -e foo -r src/")
    assert parsed is not None
    query, _ = parsed
    assert query == "src/"


def test_rg_consumes_g_glob_flag_value() -> None:
    parsed = _parse_bash_command("rg -g '*.py' searchterm src")
    assert parsed is not None
    query, _ = parsed
    assert query == "searchterm"


def test_grep_with_a_context_flag_value() -> None:
    parsed = _parse_bash_command("grep -A 3 needle haystack.txt")
    assert parsed is not None
    query, _ = parsed
    assert query == "needle"


def test_grep_pipeline_aborts() -> None:
    """Pipelines must skip; the parser would not know which stage's query."""
    assert _parse_bash_command("grep foo file.txt | head -5") is None


def test_grep_command_substitution_aborts() -> None:
    assert _parse_bash_command("grep $(cat keys.txt) src/") is None


def test_rg_inside_for_loop_aborts() -> None:
    assert _parse_bash_command("for f in *.py; do rg foo $f; done") is None


def test_env_assignment_prefix_skipped() -> None:
    parsed = _parse_bash_command("RUST_LOG=trace rg searchterm")
    assert parsed is not None
    query, _ = parsed
    assert query == "searchterm"


def test_nohup_prefix_skipped() -> None:
    parsed = _parse_bash_command("nohup grep needle file.txt")
    assert parsed is not None
    query, _ = parsed
    assert query == "needle"


def test_absolute_path_command_recognised() -> None:
    parsed = _parse_bash_command("/usr/bin/grep -r foo src/")
    assert parsed is not None
    query, cmd = parsed
    assert query == "foo"
    assert cmd == "grep"


def test_redirection_aborts() -> None:
    assert _parse_bash_command("grep foo file.txt > out.txt") is None


# --- AC4: no fall-through to arbitrary Bash ------------------------------


def test_no_fall_through_to_arbitrary_bash() -> None:
    """Random bash invocation produces no additionalContext block."""
    out = _run_hook(_payload_bash("bash -c 'something opaque'"))
    assert out == ""


def test_unknown_tool_name_silent_skip() -> None:
    """Tool other than Grep / Glob / Bash silent-skips."""
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/x.txt"},
        "cwd": "/tmp",
        "session_id": "s1",
    })
    out = _run_hook(payload)
    assert out == ""


# --- AC5: per-turn fire cap ----------------------------------------------


def test_per_turn_fire_cap_holds() -> None:
    """3 fires within one session emit blocks; the 4th silent-skips."""
    _reset_bash_fire_state()
    fired_count = 0
    for i in range(BASH_FIRE_CAP_PER_TURN + 2):
        out = _run_hook(_payload_bash(
            f"rg query{i} src/", session_id="cap-test",
        ))
        if out:
            # Validate it's a parseable JSON envelope.
            envelope = cast(dict[str, object], json.loads(out))
            assert "hookSpecificOutput" in envelope
            fired_count += 1
    assert fired_count == BASH_FIRE_CAP_PER_TURN


def test_fire_cap_independent_per_session() -> None:
    """Two distinct session_ids each get the full cap."""
    _reset_bash_fire_state()
    for sid in ("session-A", "session-B"):
        for i in range(BASH_FIRE_CAP_PER_TURN):
            out = _run_hook(_payload_bash(
                f"rg query{i} src/", session_id=sid,
            ))
            assert out, f"{sid} fire {i} should have produced output"


# --- AC6: source attribute on Bash blocks --------------------------------


def test_bash_block_carries_source_attribute() -> None:
    _reset_bash_fire_state()
    out = _run_hook(_payload_bash("rg searchterm src/", session_id="src-test"))
    assert out
    envelope = cast(dict[str, object], json.loads(out))
    hso = cast(dict[str, object], envelope["hookSpecificOutput"])
    ctx = cast(str, hso["additionalContext"])
    assert 'source="bash:rg"' in ctx
    assert 'cmd="rg searchterm src/"' in ctx
    assert 'query="searchterm"' in ctx


def test_grep_block_unchanged_no_source_attribute() -> None:
    """Grep tool path emits the v1.2.x output shape (no source attr)."""
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Grep",
        "tool_input": {"pattern": "directive-gate"},
        "cwd": "/tmp",
        "session_id": "grep-test",
    })
    out = _run_hook(payload)
    assert out
    envelope = cast(dict[str, object], json.loads(out))
    hso = cast(dict[str, object], envelope["hookSpecificOutput"])
    ctx = cast(str, hso["additionalContext"])
    assert "source=" not in ctx
    assert "cmd=" not in ctx
    assert 'query="directive-gate"' in ctx


# --- _extract_bash_query integration ------------------------------------


def test_extract_bash_query_returns_truncated_cmd() -> None:
    long_cmd = "rg " + "x" * 200 + " src/"
    payload: dict[str, object] = {
        "tool_name": "Bash",
        "tool_input": {"command": long_cmd},
    }
    extracted = _extract_bash_query(payload)
    assert extracted is not None
    _, _, truncated = extracted
    assert truncated.endswith("...")
    assert len(truncated) <= 80


def test_extract_bash_query_returns_none_for_non_bash() -> None:
    payload: dict[str, object] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
    }
    assert _extract_bash_query(payload) is None


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset the per-process fire-state map between tests."""
    _reset_bash_fire_state()
