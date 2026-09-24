"""A typed /aelf command must run itself (#1626).

aelfrice's promise is that memory does not depend on the model choosing
to act. A slash command is a *skill*: nothing runs it unless the model
decides to, and when it does not, the command silently does not happen.
A user's `/aelf:lock` was lost that way and the same instruction was
restated twelve times over six weeks before anyone noticed (#1620).

The arms here drive the real hook entry point with a real
UserPromptSubmit payload and then read the store back, because the only
claim worth making is that the belief is locked afterwards.

The safety boundary gets as much coverage as the happy path. A gate
that only proves the allowed commands run is half a guard: executing
`/aelf:uninstall` off raw prompt text no human confirmed would be a
worse failure than the one being closed.
"""

from __future__ import annotations

import io
import json

import pytest

from aelfrice.hook import (
    _EXECUTABLE_COMMANDS,
    execute_aelf_command,
    parse_aelf_command,
    user_prompt_submit,
)
from aelfrice.store import MemoryStore

STATEMENT = "Route all questions and decisions through the question tool."


def _run_hook(prompt: str, tmp_path) -> tuple[int, str, str]:
    payload = json.dumps(
        {"prompt": prompt, "session_id": "t1", "cwd": str(tmp_path)}
    )
    out, err = io.StringIO(), io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(payload), stdout=out, stderr=err
    )
    return rc, out.getvalue(), err.getvalue()


def _locked(db) -> list[str]:
    store = MemoryStore(str(db))
    try:
        return [b.content for b in store.list_locked_beliefs()]
    finally:
        store.close()


# --- AC1: the command runs, with no model involvement -------------------


@pytest.mark.timeout(120)
def test_a_typed_lock_produces_a_locked_belief(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The criterion the reported failure would have failed.

    Nothing here invokes a skill or asks a model to do anything. The
    prompt goes in, the belief comes out locked.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    rc, _out, err = _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)

    assert rc == 0
    assert STATEMENT in _locked(db), (
        f"a typed /aelf:lock did not lock the statement; stderr was {err!r}"
    )
    assert "ran /aelf:lock" in err, f"no confirmation on stderr: {err!r}"


@pytest.mark.timeout(120)
def test_the_model_is_told_the_command_already_ran(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook is authoritative, so it must say so.

    Without this the model repeats the command or reports it as undone,
    which is the same outcome the user complained about wearing a
    different hat.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    _rc, out, _err = _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)

    assert "<aelfrice-command-executed>" in out, (
        f"nothing injected to tell the model it ran: {out[:200]!r}"
    )
    assert "Do not run it again" in out


@pytest.mark.timeout(120)
def test_running_the_same_lock_twice_is_idempotent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)
    _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)

    rows = [c for c in _locked(db) if c == STATEMENT]
    assert len(rows) == 1, f"expected one locked row, got {len(rows)}"


# --- AC3: the safety boundary -------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "prompt",
    [
        "/aelf:uninstall",
        "/aelf:delete 7737baed1ab0730b",
        "/aelf:retire 7737baed1ab0730b",
        "/aelf:unlock 7737baed1ab0730b",
        "/aelf:upgrade",
        "/aelf:setup",
        "/aelf:restore 7737baed1ab0730b",
    ],
)
def test_destructive_commands_are_never_executed(
    prompt: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These lose data or reconfigure the machine.

    Running one off prompt text no human confirmed, and which the model
    may merely be quoting, is a worse failure than the one this closes.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    _rc, out, err = _run_hook(prompt, tmp_path)

    assert "ran /aelf:" not in err, f"a destructive command executed: {err!r}"
    assert "<aelfrice-command-executed>" not in out


@pytest.mark.timeout(30)
def test_the_executable_set_excludes_every_destructive_command() -> None:
    """Pin the set itself, or the parametrize above goes vacuous.

    Adding a destructive command to `_EXECUTABLE_COMMANDS` would make
    the arms above fail — but removing one from the test list would
    hide it. This asserts the boundary directly.
    """
    forbidden = {
        "delete", "retire", "unlock", "restore",
        "uninstall", "upgrade", "setup", "unsetup",
    }
    overlap = forbidden & set(_EXECUTABLE_COMMANDS)
    assert not overlap, f"destructive commands are executable: {sorted(overlap)}"
    assert set(_EXECUTABLE_COMMANDS) == {"lock", "confirm", "promote", "scope-out"}


# --- AC4: a mention is not an invocation --------------------------------


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "prompt",
    [
        "The command /aelf:lock did not appear to take effect yesterday.",
        "Did you try /aelf:lock for that?",
        "aelfrice has a /aelf:lock command for ground truth.",
    ],
)
def test_a_mention_of_a_command_does_not_execute_it(prompt: str) -> None:
    assert parse_aelf_command(prompt) is None, (
        f"a sentence mentioning a command parsed as one: {prompt!r}"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("/aelf:lock A statement.", ("lock", "A statement.")),
        ("/aelf:confirm 7737baed1ab0730b", ("confirm", "7737baed1ab0730b")),
        ("/aelf:scope-out benchmarks", ("scope-out", "benchmarks")),
        ("/aelf:uninstall", ("uninstall", "")),
    ],
)
def test_an_invocation_parses(prompt: str, expected: tuple[str, str]) -> None:
    assert parse_aelf_command(prompt) == expected


# --- AC6: failure is loud -----------------------------------------------


@pytest.mark.timeout(60)
def test_a_command_with_no_argument_says_so(tmp_path) -> None:
    """Silence here would reproduce the defect inside its own fix."""
    err = io.StringIO()
    line = execute_aelf_command("/aelf:lock", session_id="t", stderr=err)
    assert line is not None
    assert "needs an argument" in line
    assert "needs an argument" in err.getvalue()


@pytest.mark.timeout(60)
def test_a_failing_command_reports_loudly(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure must reach the user, not be swallowed."""
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("synthetic failure")

    from aelfrice import cli

    monkeypatch.setattr(cli, "main", _boom)
    err = io.StringIO()
    line = execute_aelf_command(
        "/aelf:lock A statement.", session_id="t", stderr=err
    )
    assert line is not None and "FAILED" in line
    assert "FAILED" in err.getvalue()


@pytest.mark.timeout(120)
def test_a_failing_command_does_not_cost_the_prompt_its_injection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook must still return 0 and still inject memory.

    A command that explodes must not take the turn's memory with it.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    from aelfrice import cli

    def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "main", _boom)
    rc, _out, err = _run_hook("/aelf:lock A statement.", tmp_path)
    assert rc == 0, "a failing command took down the hook"
    assert "FAILED" in err


# --- AC7: aelf search precedes every search tool ------------------------


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Grep", {"pattern": "lock_level user"}),
        ("Glob", {"pattern": "**/*.py"}),
        ("WebSearch", {"query": "sqlite fts5 ranking"}),
        ("WebFetch", {"url": "https://x.test/a", "prompt": "how bm25 scoring works"}),
    ],
)
def test_every_search_tool_triggers_the_memory_lane(
    tool: str, tool_input: dict[str, str]
) -> None:
    """The value of this hook is ORDERING.

    aelfrice runs first, so the model already holds the relevant
    brain-graph context before it chooses grep, the web, or anything
    else. Covering only Grep and Glob left the web tools reaching out
    with no brain-graph context at all.
    """
    from aelfrice.hook_search_tool import _is_search_tool_call

    assert _is_search_tool_call({"tool_name": tool, "tool_input": tool_input}), (
        f"{tool} does not trigger an aelf search first"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("tool", "tool_input", "must_contain"),
    [
        ("Grep", {"pattern": "lock_level user"}, "lock_level"),
        ("WebSearch", {"query": "sqlite fts5 ranking"}, "sqlite"),
        # The prompt, not the URL: a bare URL tokenises into host
        # fragments that match nothing useful.
        (
            "WebFetch",
            {"url": "https://example.test/a", "prompt": "how bm25 scoring works"},
            "bm25",
        ),
    ],
)
def test_the_query_comes_from_each_tools_own_field(
    tool: str, tool_input: dict[str, str], must_contain: str
) -> None:
    from aelfrice.hook_search_tool import _extract_query

    q = _extract_query({"tool_name": tool, "tool_input": tool_input})
    assert q is not None and must_contain in q, (
        f"{tool}: expected {must_contain!r} in the query, got {q!r}"
    )


@pytest.mark.timeout(30)
def test_the_installed_matcher_covers_the_web_tools() -> None:
    """The hook only fires for tools the matcher names."""
    from aelfrice.setup import SEARCH_TOOL_MATCHER

    for tool in ("Grep", "Glob", "WebSearch", "WebFetch"):
        assert tool in SEARCH_TOOL_MATCHER, (
            f"{tool} is not in the installed matcher {SEARCH_TOOL_MATCHER!r}"
        )


@pytest.mark.timeout(30)
def test_a_non_search_tool_does_not_trigger_the_lane() -> None:
    """Widening must not make every tool call pay for retrieval."""
    from aelfrice.hook_search_tool import _is_search_tool_call

    for tool in ("Read", "Write", "Edit", "TodoWrite"):
        assert not _is_search_tool_call(
            {"tool_name": tool, "tool_input": {"file_path": "/x"}}
        ), f"{tool} should not trigger the memory lane"
