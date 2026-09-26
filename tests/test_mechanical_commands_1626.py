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
import os

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


def _exclusions_for(session_id: str) -> list[str]:
    """The exclusions `session_id` would actually get at retrieval time.

    Resolved the way the hook resolves them, not by reading the file
    directly: the whole defect was that the file said one thing and
    `load_exclusions` returned [] because the stored session did not
    match. A test that read the JSON would have passed throughout.
    """
    from aelfrice.hook import _session_state_path
    from aelfrice.session_exclusions import exclusions_path, load_exclusions

    state_path = _session_state_path()
    assert state_path is not None
    return load_exclusions(exclusions_path(state_path.parent), session_id)


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
        # Every one carries an argument. Three of these used to be bare
        # (`/aelf:uninstall`, `/aelf:upgrade`, `/aelf:setup`), so the
        # empty-argument guard refused them BEFORE the allowlist was
        # consulted — they passed on the wrong rule, and kept passing
        # under a mutation that disabled the allowlist entirely.
        "/aelf:uninstall --yes",
        "/aelf:delete 7737baed1ab0730b",
        "/aelf:retire 7737baed1ab0730b",
        "/aelf:unlock 7737baed1ab0730b",
        "/aelf:upgrade now",
        "/aelf:setup claude",
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
    out = execute_aelf_command("/aelf:lock", session_id="t", stderr=err)
    assert out is not None
    assert "needs an argument" in out.line
    assert "needs an argument" in err.getvalue()
    # The flag, not the prose, is what the caller branches on.
    assert out.took_effect is False


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
    out = execute_aelf_command(
        "/aelf:lock A statement.", session_id="t", stderr=err
    )
    assert out is not None and "FAILED" in out.line
    assert "FAILED" in err.getvalue()
    assert out.took_effect is False


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


# --- the argument is the first line only --------------------------------


@pytest.mark.timeout(120)
def test_the_argument_stops_at_the_first_newline(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command followed by more prose must not lock the whole prompt.

    `re.DOTALL` with `(.*)` swallowed the remainder, so
    `/aelf:lock Always use uv.\n\nAlso draft the release note` locked the
    entire message as ONE user-locked belief — the highest-trust tier in
    the product, re-injected as standing ground truth on every later
    turn. Typing a command and then continuing the message is the most
    natural thing a user does.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    statement = "Always use uv for Python environments."

    _run_hook(
        f"/aelf:lock {statement}\n\nAlso, unrelated: draft the "
        "release note for me.",
        tmp_path,
    )

    locked = _locked(db)
    assert statement in locked, f"the statement was not locked: {locked}"
    assert not any("release note" in c for c in locked), (
        f"the rest of the prompt was locked as ground truth: {locked}"
    )


@pytest.mark.timeout(30)
def test_the_parser_keeps_only_the_first_line() -> None:
    assert parse_aelf_command("/aelf:lock One line.\nSecond line.") == (
        "lock",
        "One line.",
    )


# --- the argument is text, never a flag ---------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "prompt",
    [
        "/aelf:lock --help",
        "/aelf:lock -h",
        "/aelf:lock --advanced",
        "/aelf:scope-out --clear",
        "/aelf:confirm --db /tmp/elsewhere",
    ],
)
def test_an_argument_may_not_be_a_flag(prompt: str) -> None:
    """`argv` is `[command, argument]`, so a leading `-` is an option.

    `/aelf:lock --help` dumped 2,265 characters of argparse usage onto
    this hook's stdout protocol channel, and `/aelf:lock --advanced`
    exited 0 and reported a lock that never happened. The executable set
    is meant to be additive and idempotent, not arbitrary CLI.
    """
    err = io.StringIO()
    out = execute_aelf_command(prompt, session_id="t", stderr=err)
    assert out is not None
    assert "may not start with" in out.line, out
    assert "ran /aelf:" not in out.line
    assert out.took_effect is False


@pytest.mark.timeout(120)
def test_nothing_the_command_prints_escapes_onto_the_protocol_channel(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """argparse writes to the real sys.stdout, which IS the protocol.

    Anything printed there is injected verbatim into the model's
    context, outside COMMAND_NOTE_CAP and outside the writer
    enumeration that exists to make that impossible. So the CLI call
    runs under a stdout redirect, not merely with `out=`.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    capsys.readouterr()

    err = io.StringIO()
    execute_aelf_command(
        "/aelf:lock A statement that should lock cleanly.",
        session_id="t",
        stderr=err,
    )
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"the command leaked onto the real stdout: {captured.out[:200]!r}"
    )


# --- the bounded note ----------------------------------------------------


@pytest.mark.timeout(60)
def test_the_note_is_capped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The note carries the command's own stdout, which can be long.

    The cap is on the command's OUTPUT, so the output is what has to be
    long here. An earlier version of this arm locked a very long
    statement, which proved nothing: `aelf lock` prints
    `locked: <id>` whatever the statement length, so the note stayed
    short and removing the cap passed.
    """
    from aelfrice import cli
    from aelfrice.hook import COMMAND_NOTE_CAP

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _chatty(_argv: object, out: object = None) -> int:
        print("y" * (COMMAND_NOTE_CAP + 5000), file=out)  # type: ignore[arg-type]
        return 0

    monkeypatch.setattr(cli, "main", _chatty)
    _rc, out, _err = _run_hook("/aelf:lock A statement.", tmp_path)

    start = out.index("<aelfrice-command-executed>")
    end = out.index("</aelfrice-command-executed>")
    body = out[start:end]
    assert len(body) < COMMAND_NOTE_CAP + 500, (
        f"the note is unbounded: {len(body)} characters"
    )


@pytest.mark.timeout(60)
def test_a_clean_systemexit_is_not_reported_as_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SystemExit(0)` is success, and must not be reported as failure.

    `int(exc.code or 1)` turned a clean 0 into 1, because `0 or 1` is 1
    — so a command that succeeded through an exit path was announced to
    the user and the model as having failed. Announcing a false failure
    is the mirror of the false success this issue is about.
    """
    from aelfrice import cli

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _clean_exit(_argv: object, out: object = None) -> int:
        raise SystemExit(0)

    monkeypatch.setattr(cli, "main", _clean_exit)
    err = io.StringIO()
    out = execute_aelf_command(
        "/aelf:lock A statement.", session_id="t", stderr=err
    )
    assert out is not None
    assert "FAILED" not in out.line, f"a clean exit reported as failure: {out!r}"
    assert "ran /aelf:lock" in out.line
    assert out.took_effect is True


@pytest.mark.timeout(60)
def test_a_nonzero_systemexit_is_reported_as_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction, so the fix is not "never report failure"."""
    from aelfrice import cli

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _bad_exit(_argv: object, out: object = None) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(cli, "main", _bad_exit)
    err = io.StringIO()
    out = execute_aelf_command(
        "/aelf:lock A statement.", session_id="t", stderr=err
    )
    assert out is not None and "FAILED" in out.line, out
    assert "exit 2" in out.line
    assert out.took_effect is False


# --- session attribution is restored ------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("preset", [None, "PREVIOUS"])
def test_the_session_env_var_is_restored(
    preset: str | None, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor mutates os.environ; it must put it back.

    Pinned because making the restore a no-op passed every other arm.
    Both directions matter: a previously-unset variable must not be left
    set, and a previously-set one must not be clobbered.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    if preset is None:
        monkeypatch.delenv("AELF_SESSION_ID", raising=False)
    else:
        monkeypatch.setenv("AELF_SESSION_ID", preset)

    err = io.StringIO()
    execute_aelf_command(
        "/aelf:lock A statement for the restore check.",
        session_id="during-the-call",
        stderr=err,
    )

    assert os.environ.get("AELF_SESSION_ID") == preset, (
        f"AELF_SESSION_ID left as {os.environ.get('AELF_SESSION_ID')!r}, "
        f"expected {preset!r}"
    )


@pytest.mark.timeout(60)
def test_the_session_env_var_is_restored_after_a_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("AELF_SESSION_ID", "PREVIOUS")

    from aelfrice import cli

    def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "main", _boom)
    execute_aelf_command("/aelf:lock A statement.", session_id="x", stderr=io.StringIO())

    assert os.environ.get("AELF_SESSION_ID") == "PREVIOUS"


# --- upgrading an existing install must not duplicate the hook ----------


@pytest.mark.timeout(60)
def test_upgrading_retires_the_superseded_search_matcher(tmp_path) -> None:
    """Widening the matcher must not leave two entries behind.

    `_install_or_replace_entry` keys on (command, matcher), so a wider
    matcher appends rather than replaces. Both then match Grep and Glob,
    so every Grep call would run the hook twice — two store opens, two
    retrievals, and the locked block injected twice. Every existing
    install is the affected case, and `prune_broken_aelf_hooks` does not
    clean it because the entry is superseded, not broken.
    """
    from aelfrice.setup import (
        SEARCH_TOOL_MATCHER,
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        install_search_tool_hook,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/opt/bin/aelf-search-tool-hook",
                                }
                            ],
                        },
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/opt/bin/aelf-search-tool-hook",
                                }
                            ],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_search_tool_hook(
        settings, command="/usr/local/bin/aelf-search-tool-hook"
    )

    entries = json.loads(settings.read_text())["hooks"]["PreToolUse"]
    matchers = [e["matcher"] for e in entries]
    assert SEARCH_TOOL_MATCHER in matchers
    for superseded in SUPERSEDED_SEARCH_TOOL_MATCHERS:
        assert superseded not in matchers, (
            f"a superseded entry survived the upgrade: {matchers}"
        )
    # The unrelated Bash entry is not this hook's to remove.
    assert "Bash" in matchers


@pytest.mark.timeout(30)
def test_a_foreign_entry_on_the_old_matcher_is_left_alone(tmp_path) -> None:
    """Only THIS hook's entry is retired, not a user's own.

    Scoping by command is what keeps the migration from deleting
    somebody else's PreToolUse entry that happens to sit on the same
    matcher.
    """
    from aelfrice.setup import (
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        install_search_tool_hook,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
                            "hooks": [
                                {"type": "command", "command": "/usr/bin/somebody-else"}
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_search_tool_hook(
        settings, command="/usr/local/bin/aelf-search-tool-hook"
    )

    entries = json.loads(settings.read_text())["hooks"]["PreToolUse"]
    foreign = [
        e
        for e in entries
        if any(h.get("command") == "/usr/bin/somebody-else" for h in e["hooks"])
    ]
    assert foreign, f"a foreign entry was removed: {entries}"


# --- review round 4: the defects a blind adversarial pass found --------


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "terminator,name",
    [
        ("\n", "LF"),
        ("\r", "CR"),
        ("\r\n", "CRLF"),
        ("\v", "VT"),
        ("\f", "FF"),
        ("\x1c", "FS"),
        ("\x1d", "GS"),
        ("\x1e", "RS"),
        ("\x85", "NEL"),
        (" ", "LS"),
        (" ", "PS"),
    ],
)
def test_every_line_terminator_ends_the_argument(
    terminator: str, name: str
) -> None:
    """"First line" must mean what `str.splitlines` means.

    The first fix for the swallowing defect excluded `\\n` alone, so the
    whole defect stayed reachable from a CR-only client and from every
    other terminator Python recognises: the rest of the message was
    still locked as one user-locked belief. Ten of these eleven arms
    failed before the character class was widened.
    """
    rest = "Also draft the release note for me."
    parsed = parse_aelf_command(
        f"/aelf:lock {STATEMENT}{terminator}{terminator}{rest}"
    )
    assert parsed is not None
    command, argument = parsed
    assert command == "lock"
    assert argument == STATEMENT, f"{name} leaked: {argument!r}"
    assert rest not in argument, f"{name} swallowed the rest of the prompt"


@pytest.mark.timeout(60)
def test_a_refused_command_is_not_announced_as_having_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #1620 class, reproduced inside its own fix.

    A refusal used to get the identical frame as a success: "aelfrice
    ran this command itself ... Do not run it again". Nothing had run,
    nothing was written, and the one fallback that could still have
    saved it -- the model running the command -- was explicitly
    disabled. That is the silent-loss shape #1626 exists to close.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    _rc, out, _err = _run_hook("/aelf:lock --advanced", tmp_path)

    assert "<aelfrice-command-failed>" in out, out
    assert "<aelfrice-command-executed>" not in out, out
    assert "ran this command itself" not in out, out
    assert "did NOT take effect" in out, out


@pytest.mark.timeout(120)
def test_a_command_that_took_effect_still_says_so(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction, so the fix is not "never claim success".

    Without this arm, making `took_effect` always False would pass the
    refusal arm above and silently disable the authoritative report
    that stops the model running the command a second time.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _rc, out, _err = _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)

    assert "<aelfrice-command-executed>" in out, out
    assert "<aelfrice-command-failed>" not in out, out
    assert "ran this command itself" in out, out
    assert STATEMENT in _locked(db)


@pytest.mark.timeout(120)
def test_scope_out_attaches_to_the_session_that_typed_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor must run AFTER the session-state file is stamped.

    `is_session_first_prompt` is the only writer of the top-level
    `session_id` key, and `aelf scope-out` resolves its target session
    from that key. Executing ahead of it wrote the exclusion under
    whichever session most recently *started*, so `load_exclusions`
    returned [] on the mismatch -- a dead exclusion, reported as
    applied. A model-invoked `aelf scope-out` necessarily ran after the
    hook, so this was a regression introduced by executing early.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _run(prompt: str, sid: str) -> str:
        payload = json.dumps(
            {"prompt": prompt, "session_id": sid, "cwd": str(tmp_path)}
        )
        out, err = io.StringIO(), io.StringIO()
        user_prompt_submit(stdin=io.StringIO(payload), stdout=out, stderr=err)
        return out.getvalue()

    # Session A starts first and stamps the state file.
    _run("hello", "SESSION-A")
    # Session B then types the command. The exclusion must be B's.
    out_b = _run("/aelf:scope-out benchmarks", "SESSION-B")

    assert "<aelfrice-command-failed>" not in out_b, out_b
    assert _exclusions_for("SESSION-B") == ["benchmarks"], (
        "the exclusion did not attach to the session that typed it"
    )


@pytest.mark.timeout(120)
def test_scope_out_works_on_a_sessions_very_first_prompt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The degenerate case of the same ordering bug.

    With no state file yet, the command failed outright -- and the same
    turn then created the file, so it worked on the second try and
    looked like a flake.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    payload = json.dumps(
        {
            "prompt": "/aelf:scope-out vendor",
            "session_id": "ONLY-SESSION",
            "cwd": str(tmp_path),
        }
    )
    out, err = io.StringIO(), io.StringIO()
    user_prompt_submit(stdin=io.StringIO(payload), stdout=out, stderr=err)

    assert "no active session" not in out.getvalue() + err.getvalue()
    assert _exclusions_for("ONLY-SESSION") == ["vendor"]


@pytest.mark.timeout(60)
def test_the_stdout_redirect_holds_without_the_leading_dash_guard(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Bind the redirect on its own, not through the dash guard.

    The first version of this arm was vacuous: removing the redirect
    left the whole file green, because the only input that reached
    argparse was a flag, and the dash guard refused those first. Two
    guards, one test, and the mutation of either was survived by the
    other. This drives a command whose own body writes to the real
    `sys.stdout` directly, so the redirect is the only thing between it
    and the protocol channel.
    """
    from aelfrice import cli

    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))

    def _chatty(_argv: object, out: object = None) -> int:
        import sys

        sys.stdout.write("LEAK" * 1000)
        return 0

    monkeypatch.setattr(cli, "main", _chatty)
    capsys.readouterr()
    execute_aelf_command(
        f"/aelf:lock {STATEMENT}", session_id="t", stderr=io.StringIO()
    )
    captured = capsys.readouterr()
    assert "LEAK" not in captured.out, (
        f"{len(captured.out)} characters reached the real stdout, "
        "which IS this hook's protocol channel"
    )


@pytest.mark.timeout(60)
def test_an_overlong_argument_is_refused_rather_than_truncated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paste must not become a user-locked belief.

    The CLI has always accepted an arbitrarily long statement. What is
    new is that no human confirms a Bash call: the hook writes off raw
    prompt text, so a pasted blob beginning with `/aelf:lock` reached
    the highest-trust tier and could crowd the whole lock budget.
    Refused rather than truncated, because a truncated lock stores
    something the user did not type.
    """
    from aelfrice.hook import COMMAND_ARGUMENT_CAP

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    blob = "x" * (COMMAND_ARGUMENT_CAP + 1)
    err = io.StringIO()
    outcome = execute_aelf_command(
        f"/aelf:lock {blob}", session_id="t", stderr=err
    )

    assert outcome is not None
    assert outcome.took_effect is False
    assert "over the" in outcome.line, outcome.line
    assert _locked(db) == [], "an overlong paste was locked"


@pytest.mark.timeout(60)
def test_an_argument_at_the_cap_is_still_accepted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary, so the cap is not "refuse everything long"."""
    from aelfrice.hook import COMMAND_ARGUMENT_CAP

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    statement = "y" * COMMAND_ARGUMENT_CAP
    outcome = execute_aelf_command(
        f"/aelf:lock {statement}", session_id="t", stderr=io.StringIO()
    )

    assert outcome is not None and outcome.took_effect is True, outcome
    assert _locked(db) == [statement]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "stored,shape",
    [
        ("/opt/bin/aelf-search-tool-hook --debug", "trailing argument"),
        ('"/opt/my bin/aelf-search-tool-hook"', "quoted spaced path"),
        ("/opt/my bin/aelf-search-tool-hook", "unquoted spaced path"),
    ],
)
def test_the_superseded_entry_is_retired_whatever_shape_it_was_stored_in(
    stored: str, shape: str, tmp_path
) -> None:
    """Ownership is decided by the project's key rule, not `Path().name`.

    The first version rolled its own basename with `Path(command).name`,
    which keeps a trailing argument, keeps a closing quote, and on
    Windows keeps an `.EXE` suffix. In each of those shapes the stored
    command did not compare equal to the installed one, the old entry
    survived, and the duplicate this exists to prevent came back --
    silently, because the install still reported success.

    `_command_basename` / `launcher.command_program_keys` is the rule
    the rest of the module already uses, hardened for exactly these
    shapes by #1412 and #1482.
    """
    from aelfrice.setup import (
        SEARCH_TOOL_MATCHER,
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        install_search_tool_hook,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
                            "hooks": [{"type": "command", "command": stored}],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_search_tool_hook(
        settings, command="/usr/local/bin/aelf-search-tool-hook"
    )

    matchers = [
        e.get("matcher")
        for e in json.loads(settings.read_text())["hooks"]["PreToolUse"]
    ]
    assert SUPERSEDED_SEARCH_TOOL_MATCHERS[0] not in matchers, (
        f"a {shape} left the superseded entry behind: {matchers}"
    )
    assert SEARCH_TOOL_MATCHER in matchers


@pytest.mark.timeout(60)
def test_a_grouped_entry_keeps_the_foreign_hook_and_loses_ours(tmp_path) -> None:
    """Granular to the inner hook, not the whole entry.

    A settings file may group our hook and a user's onto one matcher.
    Retiring the whole entry deletes theirs; keeping the whole entry
    leaves the duplicate. Neither is acceptable, so only our hook is
    dropped out of the inner list.
    """
    from aelfrice.setup import (
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        install_search_tool_hook,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/old/bin/aelf-search-tool-hook",
                                },
                                {
                                    "type": "command",
                                    "command": "/usr/bin/somebody-else",
                                },
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_search_tool_hook(
        settings, command="/usr/local/bin/aelf-search-tool-hook"
    )

    entries = json.loads(settings.read_text())["hooks"]["PreToolUse"]
    superseded = [
        e
        for e in entries
        if e.get("matcher") == SUPERSEDED_SEARCH_TOOL_MATCHERS[0]
    ]
    assert len(superseded) == 1, entries
    commands = [h.get("command") for h in superseded[0]["hooks"]]
    assert commands == ["/usr/bin/somebody-else"], (
        f"the foreign hook must survive and ours must not: {commands}"
    )


@pytest.mark.timeout(60)
def test_install_converges_when_both_the_old_and_new_entry_exist(
    tmp_path,
) -> None:
    """The reachable half-upgraded state must converge, not stall.

    A settings merge, or an upgrade that was interrupted between
    retiring and installing, leaves both entries present. Without the
    `and not removed_superseded` term the install sees the new matcher
    already there, reports `already_present`, and returns before
    writing -- so the retirement is computed and then thrown away, and
    the duplicate persists across every subsequent upgrade.
    """
    from aelfrice.setup import (
        SEARCH_TOOL_MATCHER,
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        install_search_tool_hook,
    )

    command = "/usr/local/bin/aelf-search-tool-hook"
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
                            "hooks": [{"type": "command", "command": command}],
                        },
                        {
                            "matcher": SEARCH_TOOL_MATCHER,
                            "hooks": [{"type": "command", "command": command}],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    install_search_tool_hook(settings, command=command)

    matchers = [
        e.get("matcher")
        for e in json.loads(settings.read_text())["hooks"]["PreToolUse"]
    ]
    assert matchers == [SEARCH_TOOL_MATCHER], (
        f"install did not converge from the half-upgraded state: {matchers}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "installed,shape",
    [
        ('"/opt/my bin/aelf-search-tool-hook"', "quoted install path"),
        ("/opt/bin/aelf-search-tool-hook --quiet", "argument-bearing command"),
    ],
)
def test_ownership_is_keyed_the_same_way_on_the_side_we_resolved(
    installed: str, shape: str
) -> None:
    """The *target* side needs the project's key rule too, not just the
    stored side.

    The arms above vary how the OLD entry was written and all pass under
    a plain `Path(command).name`, because the command being installed in
    those is a clean path where the two rules agree. They disagree the
    moment the installed command is quoted, carries an argument, or (on
    Windows) ends in `.EXE` -- and there `Path().name` yields
    `aelf-search-tool-hook"` or `aelf-search-tool-hook --quiet`, which
    matches no stored entry, so the superseded entry silently survives.

    Driven against the function directly rather than through
    `install_search_tool_hook`, on purpose: `_resolve_script` does not
    produce these shapes on POSIX today, so a test routed through the
    current caller cannot reach them and a mutation back to
    `Path(command).name` survives the whole suite. The contract being
    pinned is the function's own -- "retire the entries belonging to
    THIS command" -- which is what a future caller, and the Windows
    install path, both depend on.
    """
    from aelfrice.setup import (
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        _drop_superseded_search_entries,
    )

    entries: list[dict[str, object]] = [
        {
            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
            "hooks": [
                {
                    "type": "command",
                    "command": "/old/prefix/aelf-search-tool-hook",
                }
            ],
        }
    ]

    removed = _drop_superseded_search_entries(entries, command=installed)

    assert removed is True, f"a {shape} left the superseded entry behind"
    assert entries == [], entries


# --- review round 5 ----------------------------------------------------


@pytest.mark.timeout(60)
def test_what_was_stored_is_reported_even_when_it_is_not_what_was_typed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated lock must never be silent.

    `/aelf:lock Always use uv<U+2028>for python projects.` stores
    "Always use uv". The separator ends the first line, as it must --
    but to the person who typed it that was one sentence, and
    `aelf lock` reports only `locked: <id>`, so nothing told the user
    or the model that half of it was dropped.

    The length cap refuses rather than truncates because "too long" is
    decidable. "They meant this to be one line" is not, so the answer
    here is to show what was stored rather than to guess.
    """
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    typed = "Always use uv for python projects."
    outcome = execute_aelf_command(
        f"/aelf:lock {typed}", session_id="t", stderr=io.StringIO()
    )

    assert outcome is not None and outcome.took_effect is True
    stored = _locked(db)
    assert stored == ["Always use uv"], stored
    # The report has to carry the stored text, not just the id.
    assert "Always use uv" in outcome.line, outcome.line
    assert "for python projects" not in outcome.line, (
        "the report claims text that was not stored"
    )


@pytest.mark.timeout(120)
def test_the_injected_block_shows_what_was_stored(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same, end to end: the model sees the stored text."""
    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _rc, out, _err = _run_hook(f"/aelf:lock {STATEMENT}", tmp_path)
    assert STATEMENT in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "typo",
    [
        "/aelf:lockFOO bar baz",
        "/aelf:lock_it or not",
        "/aelf:scope-out.stuff",
        "/aelf:lock!now",
    ],
)
def test_a_mistyped_command_name_is_not_a_command(typo: str) -> None:
    """A typo must fall through to the model, not write a belief.

    `[a-z0-9-]*` stops at the first character outside the class, so
    without a boundary `/aelf:lockFOO bar` parsed as command `lock`
    with argument `FOO bar` and wrote it at the highest trust tier.
    No destructive command was reachable this way -- none has an
    allowlisted prefix -- but a user-locked belief from a typo is its
    own defect.
    """
    parsed = parse_aelf_command(typo)
    assert parsed is None or parsed[0] not in _EXECUTABLE_COMMANDS, (
        f"{typo!r} parsed as {parsed!r} and would have executed"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("/aelf:lock A real statement.", ("lock", "A real statement.")),
        ("/aelf:lock\tTab separated.", ("lock", "Tab separated.")),
        ("/aelf:lock", ("lock", "")),
        ("/aelf:scope-out benchmarks", ("scope-out", "benchmarks")),
    ],
)
def test_the_boundary_does_not_break_a_real_invocation(
    prompt: str, expected: tuple[str, str]
) -> None:
    """The other direction, so the boundary is not "refuse everything"."""
    assert parse_aelf_command(prompt) == expected


@pytest.mark.timeout(60)
def test_a_bare_string_hook_on_the_old_matcher_is_retired(tmp_path) -> None:
    """A hand-edited settings file stores the command as a plain string.

    Skipping non-dict inner hooks left the superseded entry in place,
    so the duplicate survived for exactly the users most likely to have
    edited the file by hand -- and the install still reported success.
    """
    from aelfrice.setup import (
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        _drop_superseded_search_entries,
    )

    entries: list[dict[str, object]] = [
        {
            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
            "hooks": ["/old/prefix/aelf-search-tool-hook"],
        }
    ]
    removed = _drop_superseded_search_entries(
        entries, command="/usr/local/bin/aelf-search-tool-hook"
    )
    assert removed is True
    assert entries == []


@pytest.mark.timeout(60)
def test_a_bare_string_hook_that_is_not_ours_is_left_alone(tmp_path) -> None:
    """The safe direction still holds for the string shape."""
    from aelfrice.setup import (
        SUPERSEDED_SEARCH_TOOL_MATCHERS,
        _drop_superseded_search_entries,
    )

    entries: list[dict[str, object]] = [
        {
            "matcher": SUPERSEDED_SEARCH_TOOL_MATCHERS[0],
            "hooks": ["/usr/bin/somebody-else"],
        }
    ]
    removed = _drop_superseded_search_entries(
        entries, command="/usr/local/bin/aelf-search-tool-hook"
    )
    assert removed is False
    assert entries[0]["hooks"] == ["/usr/bin/somebody-else"]


# --- the import cycle #1626 created, and closed -------------------------


def _aelfrice_src() -> "pathlib.Path":
    import pathlib

    import aelfrice

    return pathlib.Path(aelfrice.__file__).parent


def _import_edges(module: str, target: str) -> list[tuple[int, str]]:
    """Every import of `aelfrice.<target>` in `aelfrice.<module>`.

    Parsed, not grepped: the claim is about the module graph, and a
    substring scan counts the word in a comment or a docstring -- of
    which the two modules involved have several, precisely because this
    boundary has been reasoned about before.
    """
    import ast

    tree = ast.parse((_aelfrice_src() / f"{module}.py").read_text())
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == f"aelfrice.{target}" or (
                mod == "aelfrice"
                and any(a.name == target for a in node.names)
            ):
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == f"aelfrice.{target}":
                    found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


@pytest.mark.timeout(60)
def test_cli_does_not_import_hook_so_there_is_no_cycle() -> None:
    """#1626 created a cli<->hook cycle; this keeps it closed.

    `hook` has to import `cli` -- running a typed command is the point.
    `cli` imported exactly one thing from `hook`,
    `ENV_SESSIONSTART_RECAP`, and only to interpolate the name into a
    help string. Together those two edges closed a cycle that CodeQL
    flagged.

    Both edges were function-local, so nothing failed at interpreter
    start. That is not a defence: a cycle survivable only because every
    edge is deferred fails the first time someone needs one of them at
    module scope, and the failure lands at import of the whole package.
    The constant moved to `env_names`, which imports nothing from
    `aelfrice` by construction, so the `cli -> hook` edge is gone rather
    than documented.

    Asserted in the cheap direction: `cli` must not import `hook` at
    all. Permitting "only lazily" is what allowed this to accumulate.
    """
    edges = _import_edges("cli", "hook")
    assert edges == [], (
        "aelfrice.cli imports aelfrice.hook, which re-closes the cycle "
        f"with hook's own import of cli: {edges}. Put anything both "
        "modules need in aelfrice.env_names, or another module that "
        "imports nothing from aelfrice."
    )
    # The other direction must still exist, or this test passes for the
    # wrong reason -- a cycle is also absent when the feature is gone.
    assert _import_edges("hook", "cli"), (
        "hook no longer imports cli, so this test would pass even with "
        "the cycle reintroduced later"
    )


@pytest.mark.timeout(60)
def test_env_names_imports_nothing_from_aelfrice() -> None:
    """The property that makes `env_names` a safe place to put a shared
    constant.

    A module with no outgoing edges cannot be part of a cycle. The
    moment it imports from `aelfrice`, it stops being a solution and
    becomes another way to build one.
    """
    import ast

    tree = ast.parse((_aelfrice_src() / "env_names.py").read_text())
    offenders = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "aelfrice" in ast.unparse(node)
    ]
    assert offenders == [], (
        f"env_names must import nothing from aelfrice: {offenders}"
    )


@pytest.mark.timeout(60)
def test_the_constant_has_exactly_one_home() -> None:
    """`env_names` owns the name, and `hook` does not shadow it.

    A re-export would have kept `hook.ENV_SESSIONSTART_RECAP` working,
    but it costs a module-scope import and so a place on the pinned
    import-cost budget -- for a string constant with one reader. Asserted
    as an absence, because a re-export added back later is exactly the
    regression: it is invisible at the call site and only shows up as
    `test_hook_import_cost_1351` going from 18 to 19.
    """
    import aelfrice.env_names
    import aelfrice.hook

    assert (
        aelfrice.env_names.ENV_SESSIONSTART_RECAP
        == "AELFRICE_SESSIONSTART_RECAP"
    )
    assert not hasattr(aelfrice.hook, "ENV_SESSIONSTART_RECAP"), (
        "hook re-exports ENV_SESSIONSTART_RECAP again, which puts "
        "env_names back on hook's import-time graph"
    )
