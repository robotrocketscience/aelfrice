"""Codex handler fields and the opt-in summary line (#1812 host parity).

Codex adds hook stdout to the model's context and never renders it, so a
Codex user can't see that aelfrice ran. This build closes three gaps:

* `aelf setup --host codex --status-message TEXT` writes the Codex
  `statusMessage` handler field instead of refusing the option (#1429).
* `additionalContextLimit` on the SessionStart, UserPromptSubmit, and
  PreToolUse search-tool handlers, so Codex doesn't cut aelfrice's block
  to a preview.
* `[hook] summary_line` / `AELFRICE_HOOK_SUMMARY_LINE`: with it on, the
  Codex SessionStart hook prints JSON carrying the same context plus a
  `systemMessage` for the user. With it off, stdout doesn't change.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_doctor_codex, main
from aelfrice.hook import _host_from_argv, _SessionStartTally, session_start
from aelfrice.hook_payload import HOOK_PAYLOAD_CHAR_LIMIT
from aelfrice.hook_summary import ENV_HOOK_SUMMARY_LINE
from aelfrice.host_codex import (
    CODEX_ADDITIONAL_CONTEXT_LIMIT,
    _rust_regex_rejects,
    _wrapped_owned_keys,
    install_codex_hooks,
    install_codex_skills,
)
from aelfrice.launcher import command_launcher_key
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

_SUMMARY_ENVS = (ENV_HOOK_SUMMARY_LINE, "AELFRICE_SESSIONSTART_RECAP")


# --- fixtures ----------------------------------------------------------------


@pytest.fixture()
def codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An existing `$CODEX_HOME` under a sandboxed `HOME`."""
    import aelfrice.auto_install as auto_install
    import aelfrice.setup as setup_mod

    home = tmp_path / "home"
    home.mkdir()
    codex = tmp_path / "codex"
    codex.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(auto_install, "AELFRICE_DOTDIR", home / ".aelfrice")
    monkeypatch.setattr(
        auto_install, "OPT_OUT_PATH", home / ".aelfrice" / "opt-out-hooks.json",
    )
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json",
    )
    return codex


def _handlers(hooks_path: Path) -> list[tuple[str, dict[str, object]]]:
    """Every (event, handler) pair in a hooks.json, in file order."""
    doc = json.loads(hooks_path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, dict[str, object]]] = []
    for event, groups in doc["hooks"].items():
        for group in groups:
            for handler in group["hooks"]:
                pairs.append((event, handler))
    return pairs


def _key(handler: dict[str, object]) -> str:
    return command_launcher_key(str(handler["command"]))


# --- statusMessage ------------------------------------------------------------


@pytest.mark.timeout(60)
def test_status_message_reaches_the_three_handlers_it_covers(
    codex_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        ["setup", "--host", "codex", "--status-message", "aelfrice memory"],
        io.StringIO(),
    )
    assert rc == 0, capsys.readouterr().err
    with_status = sorted(
        (event, _key(h)) for event, h in _handlers(codex_home / "hooks.json")
        if "statusMessage" in h
    )
    assert with_status == [
        ("SessionStart", "aelf-session-start-hook"),
        ("Stop", "aelf-stop-hook"),
        ("UserPromptSubmit", "aelf-hook"),
    ]
    for _event, handler in _handlers(codex_home / "hooks.json"):
        if "statusMessage" in handler:
            assert handler["statusMessage"] == "aelfrice memory"


@pytest.mark.timeout(60)
def test_a_rerun_without_the_option_drops_the_field(
    codex_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Setup replaces its handlers whole, as on the claude host."""
    hooks = codex_home / "hooks.json"
    assert main(
        ["setup", "--host", "codex", "--status-message", "x"], io.StringIO(),
    ) == 0
    assert any("statusMessage" in h for _e, h in _handlers(hooks))
    assert main(["setup", "--host", "codex"], io.StringIO()) == 0
    assert not any("statusMessage" in h for _e, h in _handlers(hooks))
    capsys.readouterr()


# --- additionalContextLimit ---------------------------------------------------


@pytest.mark.timeout(60)
def test_context_limit_is_on_the_three_context_handlers_only(
    codex_home: Path,
) -> None:
    """Only these handlers return `additionalContext`, and Codex reads the
    field on their events, PreToolUse among them."""
    install_codex_hooks(codex_home / "hooks.json")
    limited = sorted(
        (event, _key(h), h["additionalContextLimit"])
        for event, h in _handlers(codex_home / "hooks.json")
        if "additionalContextLimit" in h
    )
    assert limited == [
        ("PreToolUse", "aelf-search-tool-hook", HOOK_PAYLOAD_CHAR_LIMIT),
        ("SessionStart", "aelf-session-start-hook", HOOK_PAYLOAD_CHAR_LIMIT),
        ("UserPromptSubmit", "aelf-hook", HOOK_PAYLOAD_CHAR_LIMIT),
    ]


def test_context_limit_covers_the_whole_payload_bound() -> None:
    """One token per character is the most any 4-bytes-per-token estimate
    charges, so the largest fire aelfrice writes fits under the limit."""
    assert CODEX_ADDITIONAL_CONTEXT_LIMIT >= HOOK_PAYLOAD_CHAR_LIMIT
    assert CODEX_ADDITIONAL_CONTEXT_LIMIT > 2_500  # Codex's default


# --- the SessionStart host argument ------------------------------------------


@pytest.mark.timeout(60)
def test_setup_names_the_codex_host_on_the_session_start_command(
    codex_home: Path,
) -> None:
    install_codex_hooks(codex_home / "hooks.json")
    commands = [
        str(h["command"]) for event, h in _handlers(codex_home / "hooks.json")
        if event == "SessionStart"
    ]
    assert len(commands) == 1
    assert commands[0].endswith(" --host codex")
    assert _host_from_argv(commands[0].split()[1:]) == "codex"


@pytest.mark.parametrize(
    ("argv", "host"),
    [
        ([], "claude"),
        (["--host", "codex"], "codex"),
        (["--host=codex"], "codex"),
        (["--host", "nonesuch"], "claude"),
        (["--host"], "claude"),
        (["--other", "codex"], "claude"),
    ],
)
def test_host_from_argv(argv: list[str], host: str) -> None:
    assert _host_from_argv(argv) == host


# --- doctor (#1430) -------------------------------------------------------------


def _doctor(codex_dir: Path, skills: Path) -> tuple[int, str]:
    out = io.StringIO()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out,
        codex_dir=codex_dir, skills_dest=skills,
    )
    return rc, out.getvalue()


@pytest.mark.timeout(60)
def test_doctor_accepts_the_new_fields(tmp_path: Path) -> None:
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json", status_message="aelfrice")
    install_codex_skills(tmp_path / "skills")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "aelfrice_handlers=10" in out


@pytest.mark.timeout(60)
def test_doctor_still_faults_a_tampered_session_start_handler(
    tmp_path: Path,
) -> None:
    """The new fields don't make a foreign command look like ours."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks, status_message="aelfrice")
    install_codex_skills(tmp_path / "skills")
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    handler = doc["hooks"]["SessionStart"][0]["hooks"][0]
    handler["command"] = "/usr/bin/true --host codex"
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart" in out


def _session_start_command(hooks: Path) -> str:
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    return str(doc["hooks"]["SessionStart"][0]["hooks"][0]["command"])


def _set_session_start_commands(hooks: Path, *commands: str) -> None:
    """Make setup's SessionStart group hold exactly these handlers."""
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"] = [
        {"type": "command", "command": c, "timeout": 30} for c in commands
    ]
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _installed(tmp_path: Path) -> tuple[Path, Path, str]:
    """A Codex dir after setup, its hooks.json, and the session-start
    program path without the host argument."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks)
    install_codex_skills(tmp_path / "skills")
    program = _session_start_command(hooks).removesuffix(" --host codex")
    return codex_dir, hooks, program


# Each wrapper runs the session-start program as its first word. `{p}` is
# the program path.
_DIRECT = {
    "env": "/usr/bin/env {p}",
    # The flag after a prefix word is skipped, so the program is the first
    # word.
    "env -i": "/usr/bin/env -i {p} --host codex",
    "assignment": "AELFRICE_HOOK_SUMMARY_LINE=1 {p} --host codex",
    # `-p` sets a default PATH and still runs the program.
    "command -p": "command -p {p} --host codex",
    # The option's operand is read with it, so the program comes next.
    "env -u VAR": "/usr/bin/env -u FOO {p} --host codex",
    "env -iu VAR": "env -iu FOO {p} --host codex",
    "env --unset VAR": "env --unset FOO {p} --host codex",
    "exec -a NAME": "exec -a aelf {p} --host codex",
    # The operand is in the option's own word.
    "env -uVAR": "env -uFOO {p} --host codex",
    "env --unset=VAR": "env --unset=FOO {p} --host codex",
    "env -C DIR": "env -C /tmp {p} --host codex",
    "env -iC DIR": "env -iC /tmp {p} --host codex",
    "env --chdir DIR": "env --chdir /tmp {p} --host codex",
    "env -P PATH": "/usr/bin/env -P /bin {p} --host codex",
    # Only the `time` program takes `-o FILE`: a path to it, or a `time`
    # that a prefix word finds on PATH.
    "/usr/bin/time -o FILE": "/usr/bin/time -o /dev/null {p} --host codex",
    "relative time -o FILE": "bin/time -o /dev/null {p} --host codex",
    "time --output FILE": "/usr/bin/time --output /dev/null {p} --host codex",
    "command time -o FILE": "command time -o /dev/null {p} --host codex",
    "env time -o FILE": "env time -o /dev/null {p} --host codex",
    "exec time -o FILE": "exec time -o /dev/null {p} --host codex",
    "nohup time -o FILE": "nohup time -o /dev/null {p} --host codex",
    # A shell reads no quoted or escaped word as its reserved `time`, and
    # none after an assignment or after its own `time`, so these are the
    # `time` program.
    "escaped time -o FILE": "\\time -o /dev/null {p} --host codex",
    "single-quoted time -o FILE": "'time' -o /dev/null {p} --host codex",
    "double-quoted time -o FILE": '"time" -o /dev/null {p} --host codex',
    "partly quoted time -o FILE": "ti'me' -o /dev/null {p} --host codex",
    "assignment time -o FILE": "X=1 time -o /dev/null {p} --host codex",
    "second time -o FILE": "time time -o /dev/null {p} --host codex",
    # The shell's own `time` takes `-p` in bash and sh.
    "time -p": "time -p {p} --host codex",
    # env splits the string into its own arguments, so the program can be
    # in it.
    "env -S": "/usr/bin/env -S '{p} --host codex'",
    "env -S after a flag": "env -i -S '{p} --host codex'",
    "env -S holding options": "env -S '-i FOO=1 {p} --host codex'",
    "env -S attached": "env '-S{p} --host codex'",
    "env --split-string": "env --split-string '{p} --host codex'",
    "env --split-string=": "env '--split-string={p} --host codex'",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_DIRECT))
def test_doctor_faults_setup_run_over_a_direct_session_start_wrapper(
    tmp_path: Path, shape: str,
) -> None:
    """The state the move-off-a-wrapper steps start from.

    A user replaces setup's SessionStart handler with a wrapper, then runs
    setup again. Setup doesn't own the wrapper, so it keeps the wrapper's
    group and adds its own handler in a new group, and Codex injects the
    baseline twice.
    """
    codex_dir, hooks, program = _installed(tmp_path)
    wrapper = _DIRECT[shape].format(p=program)
    _set_session_start_commands(hooks, wrapper)
    install_codex_hooks(hooks)
    groups = json.loads(hooks.read_text(encoding="utf-8"))["hooks"][
        "SessionStart"
    ]
    assert [[h["command"] for h in g["hooks"]] for g in groups] == [
        [wrapper], [f"{program} --host codex"],
    ]
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out
    assert "may also run" not in out


def _warning_for(out: str, command: str) -> str | None:
    """The doctor warning line that names `command`, if there is one."""
    return next(
        (
            ln for ln in out.splitlines()
            if ln.startswith("[warn] SessionStart handler may also run "
                             "aelf-session-start-hook")
            and ln.endswith(f"delete this one: {command}")
        ),
        None,
    )


# Each wrapper names the session-start program in a shell script or an
# argument, which doctor can't prove the wrapper runs. `{p}` is the program
# path.
_WRAPPERS = {
    "python": "/usr/bin/python3 /x/aelf-visible.py SessionStart {p}",
    "sh -c": "sh -c '{p} --host codex'",
    "bash -lc pipe": "bash -lc '{p} | cat'",
    "sh -c sequence": "sh -c 'echo start; {p}'",
    "bash -o -c": "/bin/bash -o pipefail -c '{p} | cat'",
    "env assignment bash -c": (
        "/usr/bin/env AELFRICE_HOOK_SUMMARY_LINE=1 "
        "bash -c '{p} --host codex'"
    ),
    "sh -c runs $0": "sh -c 'exec \"$0\" --host codex' {p}",
    "bash -c runs $@": "bash -c '\"$@\"' _ {p} --host codex",
    "sh -c printf $(...)": "sh -c 'printf \"%s\" \"$({p} --host codex)\"'",
    "echo $(...)": "echo \"$({p} --host codex)\"",
    "sh -c backquotes": "sh -c 'echo `{p} --host codex`'",
    "printf into sh": "printf '%s' {p} | sh",
    "echo into xargs": "echo {p} | xargs -I{{}} sh -c '{{}} --host codex'",
    "timeout sh -c": "timeout 30 sh -c '{p} --host codex | cat'",
    "nice bash -c": "nice -n 5 bash -c '{p} --host codex | cat'",
    "env -i sh -c": "env -i sh -c '{p} --host codex | cat'",
    "python one argument": "/usr/bin/python3 /x/w.py '{p} --host codex'",
    # The script opens with another command, so reading the script is the
    # only way to find the program.
    "bash -lc after cd": "bash -lc 'cd /tmp && {p} | cat'",
    "bash -o -c after cd": "/bin/bash -o pipefail -c 'cd /tmp && {p} | cat'",
    "timeout sh -c after cd": "timeout 30 sh -c 'cd /tmp && {p} | cat'",
    "sh -c -- runs $0": "sh -c -- 'exec \"$0\" --host codex' {p}",
    # Only the first command's first word is a direct run.
    "after cd": "cd /tmp && {p} --host codex",
    # The shell's own `time` takes no `-o`, so bash, zsh, and sh run `-o`
    # as the program and never run the hook.
    "bare time -o FILE": "time -o /dev/null {p} --host codex",
    "bare time -p -o FILE": "time -p -o /dev/null {p} --host codex",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_WRAPPERS))
def test_doctor_warns_about_setup_run_over_a_session_start_wrapper(
    tmp_path: Path, shape: str,
) -> None:
    """Setup keeps a wrapper it doesn't own and adds its handler beside it.

    Doctor reads the program from the wrapper's script or arguments, so it
    names the wrapper in a warning and doesn't fail.
    """
    codex_dir, hooks, program = _installed(tmp_path)
    wrapper = _WRAPPERS[shape].format(p=program)
    _set_session_start_commands(hooks, wrapper)
    install_codex_hooks(hooks)
    groups = json.loads(hooks.read_text(encoding="utf-8"))["hooks"][
        "SessionStart"
    ]
    assert [[h["command"] for h in g["hooks"]] for g in groups] == [
        [wrapper], [f"{program} --host codex"],
    ]
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, wrapper) is not None, out


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "C:\\bin\\aelf-session-start-hook.exe --host codex"',
        'cmd /c "C:\\bin\\aelf-session-start-hook.exe --host codex"',
    ],
)
def test_a_windows_wrapper_given_the_command_as_one_argument_counts(
    command: str,
) -> None:
    """Windows doesn't read the shell script, but the argument's program
    still counts."""
    handler = {"type": "command", "command": command}
    assert _wrapped_owned_keys(handler, windows=True) == {
        "aelf-session-start-hook",
    }


@pytest.mark.timeout(60)
def test_doctor_faults_a_copied_session_start_handler(tmp_path: Path) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _set_session_start_commands(hooks, ours, ours)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out


def _add_group(
    hooks: Path, event: str, command: str, matcher: object = None,
) -> None:
    """Append a matcher group with one handler; None leaves out `matcher`."""
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    group: dict[str, object] = {
        "hooks": [{"type": "command", "command": command}],
    }
    if matcher is not None:
        group["matcher"] = matcher
    doc["hooks"][event].append(group)
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _narrow_setup_matcher(hooks: Path, matcher: str) -> None:
    """Give setup's SessionStart group, the first one, this matcher."""
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["matcher"] = matcher
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# The SessionStart matcher that releases before this one wrote.
_OLD_SESSION_START_MATCHER = "startup|resume|clear|compact"


def _session_start_groups(hooks: Path) -> list[dict[str, object]]:
    return json.loads(hooks.read_text(encoding="utf-8"))["hooks"][
        "SessionStart"
    ]


@pytest.mark.timeout(60)
def test_setup_session_start_matcher_lists_every_source(
    tmp_path: Path,
) -> None:
    """A forked thread starts with source `fork` and gets the baseline."""
    _codex_dir, hooks, _program = _installed(tmp_path)
    [group] = _session_start_groups(hooks)
    assert group["matcher"] == "startup|resume|clear|compact|fork"


@pytest.mark.timeout(60)
def test_setup_rerun_moves_an_earlier_session_start_group_to_fork(
    tmp_path: Path,
) -> None:
    _codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _narrow_setup_matcher(hooks, _OLD_SESSION_START_MATCHER)
    result = install_codex_hooks(hooks)
    assert result.error is None
    assert result.changed
    groups = _session_start_groups(hooks)
    assert [g["matcher"] for g in groups] == [
        "startup|resume|clear|compact|fork",
    ]
    assert [[h["command"] for h in g["hooks"]] for g in groups] == [[ours]]


_STALE_MATCHER_FAULT = (
    "SessionStart runs aelfrice under the matcher "
    "'startup|resume|clear|compact' that an earlier release wrote"
)


@pytest.mark.timeout(60)
def test_doctor_faults_the_earlier_session_start_matcher_until_setup_reruns(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    _narrow_setup_matcher(hooks, _OLD_SESSION_START_MATCHER)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert _STALE_MATCHER_FAULT in out, out
    assert "run `aelf setup --host codex` again" in out, out
    install_codex_hooks(hooks)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "earlier release" not in out, out


@pytest.mark.timeout(60)
def test_doctor_passes_a_foreign_group_under_the_earlier_matcher(
    tmp_path: Path,
) -> None:
    """Only a group that holds an aelfrice handler is a stale install."""
    codex_dir, hooks, _program = _installed(tmp_path)
    _add_group(hooks, "SessionStart", _UNRELATED, _OLD_SESSION_START_MATCHER)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "earlier release" not in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_the_earlier_matcher_as_stale_in_session_start_only(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["PreCompact"][0]["matcher"] = _OLD_SESSION_START_MATCHER
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "earlier release" not in out, out


def _first_command(hooks: Path, event: str) -> str:
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    return str(doc["hooks"][event][0]["hooks"][0]["command"])


# A second SessionStart group beside setup's. `{p}` is the program path and
# `{ours}` setup's own command. Codex reads a matcher of letters and `|` as
# exact names and any other matcher as a regex that can match part of a
# source. Setup's matcher lists every source, so the disjoint cases narrow
# it to `startup|resume` first, as a user who edits it might.
_DISJOINT_SESSION_START = {
    "env under fork": ("/usr/bin/env {p}", "fork"),
    "copy under fork": ("{ours}", "fork"),
    "copy under start": ("{ours}", "start"),
    "copy under ^fork$": ("{ours}", "^fork$"),
}
_OVERLAPPING_SESSION_START = {
    "env under resume": ("/usr/bin/env {p}", "resume"),
    # Setup's matcher lists `fork`, so a forked thread runs both.
    "env under fork": ("/usr/bin/env {p}", "fork"),
    "copy under fork": ("{ours}", "fork"),
    "copy under ^fork$": ("{ours}", "^fork$"),
    "copy under ^res": ("{ours}", "^res"),
    "copy under *": ("{ours}", "*"),
    "copy under empty": ("{ours}", ""),
    "copy under a number": ("{ours}", 5),
    # Rust's regex reads these, so Codex runs the group.
    "copy under (?i)": ("{ours}", "(?i)RESUME"),
    "copy under a class holding (?=": ("{ours}", "[(?=]?resume"),
}
# Matchers Codex can't compile. It logs "invalid matcher" and skips the
# group, so a copy under one never runs. Each would match `resume` if
# Codex could compile it.
_SKIPPED_MATCHERS = {
    "unclosed group": "(",
    "lookahead": "(?=resume)",
    "negative lookahead": "(?!fork)resume",
    "lookbehind": "(?<=r)esume",
    "negative lookbehind": "(?<!x)resume",
    "backreference": r"(r)\1?esume",
    "named backreference": "(?P<a>r)(?P=a)?esume",
    "atomic group": "(?>resume)",
    # Python's `re` reads the `{` and the inner `[` as literals and matches
    # `resume`, but Rust's `regex` rejects both patterns.
    "unclosed counted repetition": "resume|x{",
    "nested class opener": "resume|[[]",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_DISJOINT_SESSION_START))
def test_doctor_passes_a_second_run_under_a_disjoint_matcher(
    tmp_path: Path, shape: str,
) -> None:
    """No SessionStart source runs both handlers, so neither runs twice."""
    codex_dir, hooks, program = _installed(tmp_path)
    _narrow_setup_matcher(hooks, "startup|resume")
    command, matcher = _DISJOINT_SESSION_START[shape]
    _add_group(
        hooks, "SessionStart",
        command.format(p=program, ours=_session_start_command(hooks)),
        matcher,
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_OVERLAPPING_SESSION_START))
def test_doctor_faults_a_second_run_under_an_overlapping_matcher(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    command, matcher = _OVERLAPPING_SESSION_START[shape]
    _add_group(
        hooks, "SessionStart",
        command.format(p=program, ours=_session_start_command(hooks)),
        matcher,
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_SKIPPED_MATCHERS))
def test_doctor_skips_a_group_whose_matcher_codex_cannot_compile(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    matcher = _SKIPPED_MATCHERS[shape]
    _add_group(hooks, "SessionStart", _session_start_command(hooks), matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert (
        f"[warn] SessionStart group with matcher {matcher!r} never runs"
        in out
    ), out


@pytest.mark.timeout(60)
def test_doctor_skips_a_wrapper_under_a_matcher_codex_cannot_compile(
    tmp_path: Path,
) -> None:
    """A skipped group's wrapper gets no `may also run` warning either."""
    codex_dir, hooks, program = _installed(tmp_path)
    _add_group(hooks, "SessionStart", f"sh -c '{program}'", "(?=resume)")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "may also run" not in out
    assert "never runs" in out, out


# A foreign handler that runs no aelfrice program and that doctor doesn't
# warn about, so a note about its group's matcher would be noise.
_UNRELATED = "/usr/bin/python3 /x/other-hook.py SessionStart"


@pytest.mark.timeout(60)
@pytest.mark.parametrize("matcher", ["(?=resume)", "(?<s>resume)"])
def test_doctor_says_nothing_about_an_unrelated_group_matcher(
    tmp_path: Path, matcher: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    _add_group(hooks, "SessionStart", _UNRELATED, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert repr(matcher) not in out, out


# Matchers that Rust's `regex` compiles and Python's `re` rejects. Each
# matches `resume` in Rust, so Codex runs the copy beside setup's handler.
_PYTHON_ONLY_MATCHERS = {
    "named group": "(?<s>resume)",
    "flag after the start": "^(?i)resume",
    "swap greed": "(?U)resume",
    "braced hex escape": r"\x{72}esume",
    "one-letter class": r"\pL+",
    "Unicode class": r"\p{L}+",
    "repeated repetition": "resume|a**",
    "repeated counted repetition": "resume|x{1,}{2}",
    # Python reads `\z` from 3.14 on.
    "end of text": r"^resume\z",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_PYTHON_ONLY_MATCHERS))
def test_doctor_counts_a_matcher_it_cannot_read_as_every_fire(
    tmp_path: Path, shape: str,
) -> None:
    """Doctor counts the copy as running on every source and says it
    can't read the matcher."""
    codex_dir, hooks, _program = _installed(tmp_path)
    matcher = _PYTHON_ONLY_MATCHERS[shape]
    _add_group(hooks, "SessionStart", _session_start_command(hooks), matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out
    assert "never runs" not in out
    unread = (
        f"[warn] SessionStart group with matcher {matcher!r}: doctor can't "
        "read this matcher and counted it as matching every fire"
    )
    python_reads = shape == "end of text" and sys.version_info >= (3, 14)
    assert (unread in out) is not python_reads, out


@pytest.mark.timeout(60)
def test_doctor_skips_a_tool_group_whose_matcher_codex_cannot_compile(
    tmp_path: Path,
) -> None:
    """Codex compiles a tool event's matcher too, and skips the group."""
    codex_dir, hooks, _program = _installed(tmp_path)
    _add_group(
        hooks, "PreToolUse", _first_command(hooks, "PreToolUse"), "(?<=B)ash",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out


@pytest.mark.parametrize(
    ("pattern", "rejected"),
    [
        ("(?=x)", True),
        ("(?<!x)", True),
        (r"\1", True),
        (r"[\1]", True),
        ("(?P=a)", True),
        ("(?#c)", True),
        ("(?a)x", True),
        (r"x\Z", True),
        (r"[\b]", True),
        ("x{,3}", True),
        # A `{` always starts a counted repetition in Rust.
        ("x{", True),
        ("{", True),
        ("^{", True),
        (".{", True),
        ("x{a}", True),
        ("x{2,1}", True),
        ("x{2, }", True),
        # A `[` in a class opens a nested class.
        ("[[]", True),
        ("[[]]", True),
        ("[a[b]", True),
        # Errors that Python's `re` shares.
        ("(", True),
        ("x)", True),
        ("[", True),
        ("[]", True),
        ("*x", True),
        ("x|*", True),
        ("(?i)*", True),
        ("(?)", True),
        ("[z-a]", True),
        (r"[\d-z]", True),
        (r"\x4", True),
        (r"\x{}", True),
        (r"\e", True),
        ("x{2}", False),
        ("x{ 2 }", False),
        ("x{1,}{2}", False),
        ("x**", False),
        (r"\b{2}", False),
        ("[a[b]c]", False),
        ("[[:alpha:]]", False),
        ("[[:foo:]]", False),
        ("[]a]", False),
        ("[a&&[^b]]", False),
        ("(?i:x)", False),
        ("(?-s)x", False),
        ("(?P<a>x)", False),
        ("(?<a>x)", False),
        ("[(?=]", False),
        ("[](?=]", False),
        (r"\(?=x", False),
        (r"\bx\b", False),
        ("x{2,}", False),
    ],
)
def test_rust_regex_rejects_what_regex_1_12_rejects(
    pattern: str, rejected: bool,
) -> None:
    """Each verdict is what regex 1.12.3's `Regex::new` returned."""
    assert _rust_regex_rejects(pattern) is rejected


@pytest.mark.timeout(60)
def test_doctor_counts_runs_on_one_source_not_across_sources(
    tmp_path: Path,
) -> None:
    """Three handlers, but no source runs more than two of them."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _add_group(hooks, "SessionStart", ours, "startup")
    _add_group(hooks, "SessionStart", ours, "resume")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out


@pytest.mark.timeout(60)
def test_doctor_faults_two_runs_on_a_forked_session(tmp_path: Path) -> None:
    """Codex also starts a session with source `fork`, which setup's
    matcher lists, so a fork fire runs setup's handler and both copies."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _add_group(hooks, "SessionStart", ours, "fork")
    _add_group(hooks, "SessionStart", ours, "fork")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 3 times" in out


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("ours_matcher", "copy_matcher", "rc"),
    [("auto", "manual", 0), (None, "manual", 1), ("auto", "auto", 1)],
)
def test_doctor_reads_the_compaction_trigger_matcher(
    tmp_path: Path, ours_matcher: str | None, copy_matcher: str, rc: int,
) -> None:
    """PreCompact matches its trigger, `manual` or `auto`."""
    codex_dir, hooks, _program = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    if ours_matcher is not None:
        doc["hooks"]["PreCompact"][0]["matcher"] = ours_matcher
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _add_group(
        hooks, "PreCompact", _first_command(hooks, "PreCompact"), copy_matcher,
    )
    got, out = _doctor(codex_dir, tmp_path / "skills")
    assert got == rc, out
    assert ("PreCompact runs aelf-transcript-logger 2 times" in out) == bool(rc)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("event", "matcher"),
    [
        # Doctor can't list the tool names a regex matches, so it counts
        # every handler.
        ("PreToolUse", "^Web"),
        ("PostToolUse", "Web.*"),
        # Codex ignores a matcher on Stop.
        ("Stop", "never"),
    ],
)
def test_doctor_counts_every_handler_where_it_reads_no_matcher(
    tmp_path: Path, event: str, matcher: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    command = _first_command(hooks, event)
    _add_group(hooks, event, command, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert f"{event} runs {command_launcher_key(command)} 2 times" in out


# Copies of the event's first handler, each in its own group under one of
# these matchers, beside setup's `Bash` group. Codex matches an exact tool
# matcher against the tool's name and its aliases: `apply_patch` is also
# `Write` and `Edit`, and `spawn_agent` is also `Agent`.
_TOOL_MATCHERS = {
    "WebSearch": ("PreToolUse", ["WebSearch"], 0),
    "Write": ("PostToolUse", ["Write"], 0),
    "Write and WebSearch": ("PreToolUse", ["Write", "WebSearch"], 0),
    "Bash|WebSearch": ("PreToolUse", ["Bash|WebSearch"], 2),
    "Bash": ("PostToolUse", ["Bash"], 2),
    "Write and Edit": ("PreToolUse", ["Write", "Edit"], 2),
    "Write and apply_patch": ("PostToolUse", ["Write", "apply_patch"], 2),
    "Agent and spawn_agent": ("PreToolUse", ["Agent", "spawn_agent"], 2),
    "Edit|Agent and spawn_agent": (
        "PreToolUse", ["Edit|Agent", "spawn_agent"], 2,
    ),
    # A group under no matcher or `*` counts as running on every fire, so
    # the copy under `WebSearch` counts as well.
    "no matcher and WebSearch": ("PreToolUse", [None, "WebSearch"], 3),
    "* and WebSearch": ("PreToolUse", ["*", "WebSearch"], 3),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_TOOL_MATCHERS))
def test_doctor_reads_exact_tool_matchers(tmp_path: Path, shape: str) -> None:
    """The last item is the runs doctor counts, and 0 means it passes."""
    codex_dir, hooks, _program = _installed(tmp_path)
    event, matchers, runs = _TOOL_MATCHERS[shape]
    command = _first_command(hooks, event)
    for matcher in matchers:
        _add_group(hooks, event, command, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == (1 if runs else 0), out
    if runs:
        key = command_launcher_key(command)
        assert f"{event} runs {key} {runs} times" in out, out
    else:
        assert "times" not in out, out


@pytest.mark.timeout(60)
def test_setup_and_doctor_pass_an_env_search_handler_under_websearch(
    tmp_path: Path,
) -> None:
    """The review's repro: setup keeps the foreign group, and no tool call
    runs both it and setup's `Bash` group."""
    codex_dir, hooks, _program = _installed(tmp_path)
    _add_group(
        hooks, "PreToolUse",
        f"/usr/bin/env {_first_command(hooks, 'PreToolUse')}", "WebSearch",
    )
    install_codex_hooks(hooks)
    groups = json.loads(hooks.read_text(encoding="utf-8"))["hooks"][
        "PreToolUse"
    ]
    assert [g.get("matcher") for g in groups] == ["WebSearch", "Bash"]
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out


# Each foreign handler names the program without running it, or runs none.
_NOT_RUNNING = {
    "other hook": "/usr/bin/python3 /x/other-hook.py SessionStart --verbose",
    "echo": "echo aelf-session-start-hook",
    "sh -c printf": "sh -c 'printf \"%s\\n\" {p}'",
    "echo or true": "echo {p} || true",
    "printf into a file": "printf '%s\\n' {p} > /dev/null",
    # The flag after a prefix word is skipped, so echo is the program.
    "env -i echo": "env -i echo {p}",
    # The assignment is skipped, so echo is the program.
    "assignment echo": "AELFRICE_HOOK_SUMMARY_LINE=1 echo {p}",
    # The option's operand is skipped, so echo is the program.
    "exec -a echo": "exec -a name echo {p}",
    "env -C echo": "env -C {p} echo hello",
    "time -o echo": "/usr/bin/time -o {p} echo hello",
    # The split string's first word is the program.
    "env -S echo": "env -S 'echo {p}'",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_NOT_RUNNING))
def test_doctor_ignores_a_foreign_handler_that_runs_no_aelfrice_program(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    _set_session_start_commands(
        hooks,
        _NOT_RUNNING[shape].format(p=program),
        _session_start_command(hooks),
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert "may also run" not in out


# Each foreign handler only checks that the program is there, but the
# script scan reads the program as one the handler runs.
_CHECKS = {
    "command -v": "sh -c 'command -v {p} >/dev/null || exit 1'",
    # `command` is a prefix word, but `-v` only looks the program up.
    "top-level command -v": "command -v {p} >/dev/null",
    "top-level command -V": "command -V {p}",
    # The lookup option can follow another option or share a word with one.
    "command -p -v": "command -p -v {p}",
    "command -p -V": "command -p -V {p}",
    "command -pv": "command -pv {p}",
    "test -x": "sh -c 'test -x {p} || echo missing'",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_CHECKS))
def test_doctor_warns_and_passes_a_handler_that_checks_for_the_program(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    check = _CHECKS[shape].format(p=program)
    _set_session_start_commands(hooks, check, _session_start_command(hooks))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, check) is not None, out


# A wrapper in its own group beside setup's handlers. `{p}` is the
# session-start program path and `{search}` setup's PreToolUse search-tool
# command. The wrapper gets a warning only when one fire runs it and
# another handler of the program.
_WRAPPER_MATCHERS = {
    # Setup's matcher lists `fork`.
    "sh -c under fork": ("SessionStart", "fork", "sh -c '{p}'", True),
    "python under fork": (
        "SessionStart", "fork", "/usr/bin/python3 w.py {p}", True,
    ),
    "sh -c under ^fork$": ("SessionStart", "^fork$", "sh -c '{p}'", True),
    # `start` lists one exact name, and no source is `start`.
    "sh -c under start": ("SessionStart", "start", "sh -c '{p}'", False),
    "sh -c under resume": ("SessionStart", "resume", "sh -c '{p}'", True),
    "sh -c under ^res": ("SessionStart", "^res", "sh -c '{p}'", True),
    "sh -c under WebSearch": (
        "PreToolUse", "WebSearch", "sh -c '{search}'", False,
    ),
    "sh -c under Bash": ("PreToolUse", "Bash", "sh -c '{search}'", True),
    "sh -c under ^Web": ("PreToolUse", "^Web", "sh -c '{search}'", True),
    # No handler of the program runs in PreCompact.
    "lone sh -c in PreCompact": ("PreCompact", None, "sh -c '{p}'", False),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_WRAPPER_MATCHERS))
def test_doctor_warns_about_a_wrapper_only_where_a_fire_runs_it_twice(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    event, matcher, template, warns = _WRAPPER_MATCHERS[shape]
    wrapper = template.format(
        p=program, search=_first_command(hooks, "PreToolUse"),
    )
    _add_group(hooks, event, wrapper, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    line = next(
        (
            ln for ln in out.splitlines()
            if ln.startswith(f"[warn] {event} handler may also run ")
            and ln.endswith(f"delete this one: {wrapper}")
        ),
        None,
    )
    assert (line is not None) == warns, out


@pytest.mark.timeout(60)
def test_doctor_warns_about_two_wrappers_under_one_disjoint_matcher(
    tmp_path: Path,
) -> None:
    """A `fork` fire runs both wrappers, though setup's handler, narrowed
    to `startup|resume`, doesn't run on it."""
    codex_dir, hooks, program = _installed(tmp_path)
    _narrow_setup_matcher(hooks, "startup|resume")
    first, second = f"sh -c '{program}'", f"bash -c '{program}'"
    _add_group(hooks, "SessionStart", first, "fork")
    _add_group(hooks, "SessionStart", second, "fork")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert _warning_for(out, first) is not None, out
    assert _warning_for(out, second) is not None, out


@pytest.mark.timeout(60)
def test_doctor_warns_about_the_overlapping_wrapper_and_not_the_disjoint_one(
    tmp_path: Path,
) -> None:
    """A `resume` fire runs setup's handler and one wrapper. With setup's
    matcher narrowed to `startup|resume`, no fire runs the `fork` wrapper
    beside another handler."""
    codex_dir, hooks, program = _installed(tmp_path)
    _narrow_setup_matcher(hooks, "startup|resume")
    overlapping, disjoint = f"sh -c '{program}'", f"bash -c '{program}'"
    _add_group(hooks, "SessionStart", overlapping, "resume")
    _add_group(hooks, "SessionStart", disjoint, "fork")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert _warning_for(out, overlapping) is not None, out
    assert _warning_for(out, disjoint) is None, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("folder", ["gone", "first last"])
def test_doctor_names_a_missing_program_without_its_arguments(
    tmp_path: Path, folder: str,
) -> None:
    """The fault names the program path, as it did before the SessionStart
    command gained `--host codex`. A path with a space stays whole."""
    codex_dir, hooks, _program = _installed(tmp_path)
    missing = f"{tmp_path / folder / 'aelf-session-start-hook'}"
    _set_session_start_commands(hooks, f"{missing} --host codex")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    line = next(
        ln for ln in out.splitlines() if "not present on disk" in ln
    )
    assert line.endswith(f"not present on disk: {missing}"), line


@pytest.mark.timeout(60)
def test_doctor_passes_a_hooks_json_written_before_this_build(
    tmp_path: Path,
) -> None:
    """The upgrade path: handlers with no new field and no host argument.

    This is the shape `desired_codex_hooks` wrote on v5.4.0: `type` and
    `command` only.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks, status_message="aelfrice")
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    for groups in doc["hooks"].values():
        for group in groups:
            for handler in group["hooks"]:
                handler.pop("statusMessage", None)
                handler.pop("additionalContextLimit", None)
                handler["command"] = str(handler["command"]).removesuffix(
                    " --host codex",
                )
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    assert "--host" not in hooks.read_text(encoding="utf-8")
    install_codex_skills(tmp_path / "skills")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out


# --- the summary line -------------------------------------------------------------


def _belief(bid: str, content: str, lock_level: str) -> Belief:
    locked = lock_level == LOCK_USER
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0 if locked else 1.0,
        beta=0.5 if locked else 1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-04-26T00:00:00Z" if locked else None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


_BELIEFS = (
    _belief("L1", "primary db is sqlite", LOCK_USER),
    _belief("L2", "tests run under uv", LOCK_USER),
    _belief("F1", "an unlocked belief", LOCK_NONE),
)


def _fire(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    host: str,
    beliefs: tuple[Belief, ...] = _BELIEFS,
    env: dict[str, str] | None = None,
    toml: str | None = None,
) -> str:
    """One SessionStart fire against a fresh store and home under `root`.

    Each fire gets its own store and home, so the recap timestamp and the
    injection ledger one fire writes can't change what the next prints.
    """
    root.mkdir(parents=True)
    home = root / "home"
    home.mkdir()
    project = root / "project"
    project.mkdir()
    db = root / "memory.db"
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()
    if toml is not None:
        (project / ".aelfrice.toml").write_text(toml, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_USER_DB", str(root / "user" / "memory.db"))
    for name in _SUMMARY_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    payload = json.dumps({
        "session_id": "s-1812", "cwd": str(project),
        "hook_event_name": "SessionStart", "source": "startup",
    })
    sout = io.StringIO()
    rc = session_start(
        stdin=io.StringIO(payload), stdout=sout, stderr=io.StringIO(),
        host=host,
    )
    assert rc == 0
    return sout.getvalue()


@pytest.mark.timeout(60)
def test_opt_in_off_codex_stdout_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default host runs the unchanged path, so it is the reference."""
    reference = _fire(tmp_path / "claude", monkeypatch, host="claude")
    codex = _fire(tmp_path / "codex", monkeypatch, host="codex")
    assert "<belief" in reference
    assert codex == reference


@pytest.mark.timeout(60)
def test_opt_in_env_prints_json_with_the_same_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = _fire(tmp_path / "plain", monkeypatch, host="codex")
    raw = _fire(
        tmp_path / "json", monkeypatch, host="codex",
        env={ENV_HOOK_SUMMARY_LINE: "1"},
    )
    doc = json.loads(raw)
    assert raw.endswith("\n") and raw.count("\n") == 1
    assert doc["hookSpecificOutput"] == {
        "hookEventName": "SessionStart",
        "additionalContext": plain,
    }
    assert doc["systemMessage"] == "aelfrice: 2 beliefs injected (2 locked)"


@pytest.mark.timeout(60)
def test_opt_in_toml_in_the_payload_cwd_turns_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fire(
        tmp_path / "t", monkeypatch, host="codex",
        toml="[hook]\nsummary_line = true\n",
    )
    assert json.loads(raw)["systemMessage"].startswith("aelfrice: 2 beliefs")


@pytest.mark.timeout(60)
def test_env_off_overrides_toml_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fire(
        tmp_path / "t", monkeypatch, host="codex",
        env={ENV_HOOK_SUMMARY_LINE: "0"},
        toml="[hook]\nsummary_line = true\n",
    )
    assert raw.startswith("<")


@pytest.mark.timeout(60)
def test_opt_in_does_nothing_on_the_claude_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fire(
        tmp_path / "t", monkeypatch, host="claude",
        env={ENV_HOOK_SUMMARY_LINE: "1"},
    )
    assert raw.startswith("<")
    assert "systemMessage" not in raw


@pytest.mark.timeout(60)
def test_opt_in_with_nothing_to_inject_still_says_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fire(
        tmp_path / "t", monkeypatch, host="codex", beliefs=(),
        env={ENV_HOOK_SUMMARY_LINE: "1", "AELFRICE_SESSIONSTART_RECAP": "0"},
    )
    assert json.loads(raw) == {"systemMessage": "aelfrice: ran, nothing injected"}


_ELEMENT_RE = re.compile(r'<belief id="([^"]+)"([^>]*)>')


def _context_counts(context: str) -> tuple[int, int]:
    """Distinct belief ids in a context, and how many of them are locks.

    Both renderers mark a lock: the baseline writes `lock="user"` and the
    rebuild block writes `locked="true"`.
    """
    locked: dict[str, bool] = {}
    for bid, attrs in _ELEMENT_RE.findall(context):
        is_lock = 'lock="user"' in attrs or 'locked="true"' in attrs
        locked[bid] = locked.get(bid, False) or is_lock
    return len(locked), sum(locked.values())


def _fire_compact(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, env: dict[str, str],
) -> str:
    """A post-compaction SessionStart fire on the #1031 AC1 fixture.

    One lock, one session-scoped belief, and one general belief, with a
    turn log the rebuilder reads, so the baseline block and the rebuild
    block both reach stdout.
    """
    repo = root / "repo"
    transcripts = repo / ".git" / "aelfrice" / "transcripts"
    transcripts.mkdir(parents=True)
    home = root / "home"
    home.mkdir()
    db = root / "memory.db"
    session = _belief("S1sess", "session note about session_id propagation",
                      LOCK_NONE)
    session.session_id = "s1"
    store = MemoryStore(str(db))
    try:
        for b in (
            _belief("L1lock", "user prefers uv over pip", LOCK_USER),
            session,
            _belief("F1gen", "BM25 ranks results by frequency and rarity",
                    LOCK_NONE),
        ):
            store.insert_belief(b)
    finally:
        store.close()
    (repo / ".aelfrice.toml").write_text(
        '[rebuilder]\ntrigger_mode = "threshold"\n', encoding="utf-8",
    )
    (transcripts / "turns.jsonl").write_text(
        "\n".join(json.dumps({"session_id": "s1", **turn}) for turn in (
            {"role": "user",
             "text": "How does session_id propagate through ingest?"},
            {"role": "assistant",
             "text": "session_id is plumbed end-to-end via ingest_turn."},
        )) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_USER_DB", str(root / "user" / "memory.db"))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))
    for name in _SUMMARY_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    payload = json.dumps({
        "session_id": "s1", "cwd": str(repo), "source": "compact",
        "hook_event_name": "SessionStart",
    })
    sout = io.StringIO()
    rc = session_start(
        stdin=io.StringIO(payload), stdout=sout, stderr=io.StringIO(),
        host="codex",
    )
    assert rc == 0
    return sout.getvalue()


@pytest.mark.timeout(60)
def test_a_compact_fire_counts_the_rebuild_block_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rebuild block shares the fire's stdout with the baseline, so the
    line counts every distinct belief either block shows the model."""
    doc = json.loads(_fire_compact(
        tmp_path, monkeypatch, env={ENV_HOOK_SUMMARY_LINE: "1"},
    ))
    context = doc["hookSpecificOutput"]["additionalContext"]
    assert "<aelfrice-rebuild>" in context
    assert _context_counts(context) == (2, 1)
    assert doc["systemMessage"] == "aelfrice: 2 beliefs injected (1 locked)"


@pytest.mark.timeout(60)
def test_locks_the_payload_bound_cuts_are_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1639 names the locks that don't fit instead of showing them, so the
    line counts the locks the model got, not the locks in the store."""
    locks = tuple(
        _belief(
            f"L{i:03d}",
            f"lock {i:03d} " + "keep this standing rule in mind always " * 3,
            LOCK_USER,
        )
        for i in range(150)
    )
    doc = json.loads(_fire(
        tmp_path / "t", monkeypatch, host="codex", beliefs=locks,
        env={ENV_HOOK_SUMMARY_LINE: "1", "AELFRICE_SESSIONSTART_RECAP": "0"},
    ))
    shown, shown_locked = _context_counts(
        doc["hookSpecificOutput"]["additionalContext"],
    )
    assert 0 < shown < len(locks)
    assert shown_locked == shown
    assert doc["systemMessage"] == (
        f"aelfrice: {shown} beliefs injected ({shown} locked)"
    )


def test_the_tally_counts_an_id_once_and_keeps_its_lock() -> None:
    """A lock stays a lock when the other block's element doesn't say so."""
    tally = _SessionStartTally()
    tally.note("a", True)
    tally.note("a", False)
    tally.note_block(
        '<belief id="b" locked="false">x</belief>\n'
        '<belief id="a" locked="false">y</belief>\n',
    )
    assert (tally.n_beliefs, tally.n_locked) == (2, 1)


@pytest.mark.timeout(60)
def test_one_belief_reads_singular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _fire(
        tmp_path / "t", monkeypatch, host="codex", beliefs=_BELIEFS[:1],
        env={ENV_HOOK_SUMMARY_LINE: "1", "AELFRICE_SESSIONSTART_RECAP": "0"},
    )
    assert json.loads(raw)["systemMessage"] == (
        "aelfrice: 1 belief injected (1 locked)"
    )


@pytest.mark.timeout(60)
def test_a_store_the_hook_cannot_read_says_the_hook_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fire must not print the empty-store line, which tells the
    user the store has nothing to inject."""
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "memory.db"
    db.write_bytes(b"not a sqlite database " * 20)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user" / "memory.db"))
    monkeypatch.setenv(ENV_HOOK_SUMMARY_LINE, "1")
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")
    payload = json.dumps({
        "session_id": "s", "cwd": str(tmp_path),
        "hook_event_name": "SessionStart", "source": "startup",
    })
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr, host="codex",
    )
    assert rc == 0
    assert "DatabaseError" in serr.getvalue()
    assert json.loads(sout.getvalue()) == {
        "systemMessage": "aelfrice: hook failed (DatabaseError)",
    }


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["eager-guard", "lazy-import"])
def test_an_incomplete_install_says_the_hook_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str,
) -> None:
    """Both #1527 install-incomplete paths count as a failed fire."""
    import aelfrice.hook as hook_mod

    missing = ImportError("No module named 'numpy'", name="numpy")
    if where == "eager-guard":
        monkeypatch.setattr(hook_mod, "_IMPORTS_OK", False)
        monkeypatch.setattr(hook_mod, "_IMPORT_ERR", missing)
    else:
        def raise_missing() -> object:
            raise missing

        monkeypatch.setattr(
            hook_mod, "_retrieve_baseline_with_block", raise_missing,
        )
    raw = _fire(
        tmp_path / "t", monkeypatch, host="codex",
        env={ENV_HOOK_SUMMARY_LINE: "1", "AELFRICE_SESSIONSTART_RECAP": "0"},
    )
    assert json.loads(raw) == {
        "systemMessage": "aelfrice: hook failed (ImportError)",
    }


@pytest.mark.timeout(60)
def test_a_fire_that_fails_after_injecting_says_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline reached the model, so the line keeps its count, and the
    failed rebuild still shows."""
    import aelfrice.hook as hook_mod

    def boom(_payload: dict[str, object]) -> str:
        raise RuntimeError("rebuild broke")

    monkeypatch.setattr(hook_mod, "_build_rebuild_block_from_payload", boom)
    doc = json.loads(_fire_compact(
        tmp_path, monkeypatch, env={ENV_HOOK_SUMMARY_LINE: "1"},
    ))
    assert _context_counts(doc["hookSpecificOutput"]["additionalContext"]) == (
        1, 1,
    )
    assert doc["systemMessage"] == (
        "aelfrice: 1 belief injected (1 locked); hook failed (RuntimeError)"
    )


@pytest.mark.timeout(120)
def test_console_entry_reads_the_host_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through `main_session_start`, the script setup writes."""
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        for b in _BELIEFS:
            store.insert_belief(b)
    finally:
        store.close()
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "AELFRICE_DB": str(db),
        "AELFRICE_USER_DB": str(tmp_path / "user" / "memory.db"),
        "AELFRICE_NO_AUTO_INSTALL": "1",
        ENV_HOOK_SUMMARY_LINE: "1",
    }
    payload = json.dumps({"session_id": "s", "cwd": str(tmp_path)})
    probe = (
        "import sys; from aelfrice.hook import main_session_start; "
        "sys.exit(main_session_start())"
    )

    def run(*args: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", probe, *args],
            input=payload, capture_output=True, text=True, env=env,
            timeout=90, check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    codex = json.loads(run("--host", "codex"))
    assert codex["systemMessage"] == "aelfrice: 2 beliefs injected (2 locked)"
    assert run().startswith("<")
