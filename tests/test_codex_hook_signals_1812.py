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
import random
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import _cmd_doctor_codex, main
from aelfrice.hook import _host_from_argv, _SessionStartTally, session_start
from aelfrice.hook_payload import HOOK_PAYLOAD_CHAR_LIMIT
from aelfrice.hook_summary import ENV_HOOK_SUMMARY_LINE
from aelfrice.host_codex import (
    CODEX_ADDITIONAL_CONTEXT_LIMIT,
    _OWNED_BASENAMES,
    _json_object,
    _matched_values,
    _named_owned_keys,
    _rejected_config_hooks,
    _rejected_hooks_file,
    doctor_codex,
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


def _lone_warning_for(
    out: str, event: str, key: str, command: str, where: str = "",
) -> str | None:
    """The doctor warning line for a handler that names `key` and that no
    fire runs beside another handler of it, if there is one."""
    return next(
        (
            ln for ln in out.splitlines()
            if ln.startswith(
                f"[warn] {event} handler{where} names {key}, and doctor "
                f"can't tell whether it runs {key}. Doctor counts a run only "
                "for a command byte-identical to one `aelf setup --host "
                "codex` writes (",
            )
            and "and setup's own handler is the supported form." in ln
            and ln.endswith(f" Handler: {command}")
        ),
        None,
    )


# Wrappers that name the session-start program. Setup owns none of them,
# so it keeps each one and adds its own handler beside it. None is
# byte-identical to the command setup writes, so doctor warns about each
# one and exits 0, whether or not a shell runs the program. `{p}` is the
# program path.
_WRAPPERS = {
    '!': '! {p} --host codex',
    '/usr/bin/time -o FILE': '/usr/bin/time -o f {p} --host codex',
    'after cd': 'cd /tmp && {p} --host codex',
    'assignment': 'AELFRICE_HOOK_SUMMARY_LINE=1 {p} --host codex',
    'assignment !': 'X=1 ! {p} --host codex',
    'assignment exec': 'X=1 exec {p} --host codex',
    'assignment nice': 'X=1 nice {p} --host codex',
    'assignment time': 'X=1 time {p} --host codex',
    'assignment time -p': 'X=1 time -p {p} --host codex',
    'bare time -o FILE': 'time -o f {p} --host codex',
    'bare time -p': 'time -p {p} --host codex',
    'bash -c runs $@': 'bash -c \'"$@"\' _ {p} --host codex',
    'bash -lc after cd': "bash -lc 'cd /tmp && {p} | cat'",
    'bash -lc pipe': "bash -lc '{p} | cat'",
    'bash -o -c': "/bin/bash -o pipefail -c '{p} | cat'",
    'bash -o -c after cd': "/bin/bash -o pipefail -c 'cd /tmp && {p} | cat'",
    'command': 'command {p} --host codex',
    'command -p': 'command -p {p} --host codex',
    'command assignment': 'command X=1 {p} --host codex',
    'double-quoted assignment': '"X=1" {p} --host codex',
    'echo $(...)': 'echo "$({p} --host codex)"',
    'echo into xargs': "echo {p} | xargs -I{{}} sh -c '{{}} --host codex'",
    'env': '/usr/bin/env {p}',
    'env --split-string': "env --split-string '{p} --host codex'",
    'env --unset VAR': 'env --unset FOO {p} --host codex',
    'env -C DIR': 'env -C /tmp {p} --host codex',
    'env -P PATH': '/usr/bin/env -P /bin {p} --host codex',
    'env -S': "/usr/bin/env -S '{p} --host codex'",
    'env -S after a flag': "env -i -S '{p} --host codex'",
    'env -S assignment': "env -S 'X=1 {p} --host codex'",
    'env -S attached': "env '-S{p} --host codex'",
    'env -S nice assignment': "env -S 'nice X=1 {p} --host codex'",
    'env -i': '/usr/bin/env -i {p} --host codex',
    'env -i sh -c': "env -i sh -c '{p} --host codex | cat'",
    'env -iu VAR': 'env -iu FOO {p} --host codex',
    'env -u VAR': '/usr/bin/env -u FOO {p} --host codex',
    'env -u VAR assignment': 'env -u FOO X=1 {p} --host codex',
    'env -uVAR': 'env -uFOO {p} --host codex',
    'env assignment': 'env X=1 {p} --host codex',
    'env assignment bash -c': (
        "/usr/bin/env AELFRICE_HOOK_SUMMARY_LINE=1 bash -c '{p} --host codex'"
    ),
    'env nice assignment': 'env nice X=1 {p} --host codex',
    'env time -o FILE': 'env time -o f {p} --host codex',
    'env time program -p': 'env time -p {p} --host codex',
    'escaped time -o FILE': '\\time -o f {p} --host codex',
    'escaped time program -p': '\\time -p {p} --host codex',
    'exec': 'exec {p} --host codex',
    'exec --': 'exec -- {p} --host codex',
    'exec -a NAME': 'exec -a aelf {p} --host codex',
    'exec assignment': 'exec X=1 {p} --host codex',
    'exec nice': 'exec nice -n 5 {p} --host codex',
    'flock sh -c after cd': "flock /tmp/aelf.lock sh -c 'cd /tmp && {p} | cat'",
    'group': '{{ {p}; }}',
    'group with arguments': '{{ {p} --host codex; }}',
    'if': 'if {p}; then :; fi',
    'if time assignment': 'if time X=1 {p} --host codex; then :; fi',
    'if with arguments': 'if {p} --host codex; then :; fi',
    'nice': 'nice {p} --host codex',
    'nice !': 'nice ! {p} --host codex',
    'nice --adjustment ADJUSTMENT': 'nice --adjustment 5 {p} --host codex',
    'nice -n ADJUSTMENT': 'nice -n 5 {p} --host codex',
    'nice -n assignment': 'nice -n 5 X=1 {p} --host codex',
    'nice assignment': 'nice X=1 {p} --host codex',
    'nice bash -c': "nice -n 5 bash -c '{p} --host codex | cat'",
    'nice env assignment': 'nice env X=1 {p} --host codex',
    'nohup': 'nohup {p} --host codex',
    'nohup assignment': 'nohup X=1 {p} --host codex',
    'partly quoted assignment': "X'=1' {p} --host codex",
    'printf into sh': "printf '%s' {p} | sh",
    'python': '/usr/bin/python3 /x/aelf-visible.py SessionStart {p}',
    'python one argument': "/usr/bin/python3 /x/w.py '{p} --host codex'",
    'quoted time program -p': "'time' -p {p} --host codex",
    'second time': 'time time {p} --host codex',
    'second time --': 'time time -- {p} --host codex',
    'second time -o FILE': 'time time -o f {p} --host codex',
    'second time -p': 'time time -p {p} --host codex',
    'sh -c': "sh -c '{p} --host codex'",
    'sh -c -- runs $0': 'sh -c -- \'exec "$0" --host codex\' {p}',
    'sh -c backquotes': "sh -c 'echo `{p} --host codex`'",
    'sh -c printf $(...)': 'sh -c \'printf "%s" "$({p} --host codex)"\'',
    'sh -c runs $0': 'sh -c \'exec "$0" --host codex\' {p}',
    'sh -c sequence': "sh -c 'echo start; {p}'",
    'single-quoted assignment': "'X=1' {p} --host codex",
    'time': 'time {p} --host codex',
    'time assignment': 'time X=1 {p} --host codex',
    'time assignment time': 'time X=1 time {p} --host codex',
    'time program -p': '/usr/bin/time -p {p} --host codex',
    'time program assignment': '\\time X=1 {p} --host codex',
    'timeout -k DURATION': 'timeout -k 5 30 {p} --host codex',
    'timeout -s SIGNAL': 'timeout -s KILL 30 {p} --host codex',
    'timeout DURATION': 'timeout 30 {p} --host codex',
    'timeout assignment': 'timeout 30 X=1 {p} --host codex',
    'timeout sh -c': "timeout 30 sh -c '{p} --host codex | cat'",
    'timeout sh -c after cd': "timeout 30 sh -c 'cd /tmp && {p} | cat'",
    'trailing backslash': '{p} --host codex\\',
    'two assignments': 'X=1 Y=2 {p} --host codex',
    'while time assignment': 'while time X=1 {p} --host codex; do break; done',
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_WRAPPERS))
def test_doctor_warns_about_setup_run_over_a_session_start_wrapper(
    tmp_path: Path, shape: str,
) -> None:
    """The state the move-off-a-wrapper steps start from.

    A user replaces setup's SessionStart handler with a wrapper, then runs
    setup again. Setup doesn't own the wrapper, so it keeps the wrapper's
    group and adds its own handler in a new group. Doctor names the
    wrapper in a warning and doesn't fail.
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


# The commands the round-10 review gave for the exit-1 check. A shell
# runs the program for some of them, and doctor still only warns, because
# none is byte-identical to the command setup writes. `{p}` is the program
# path.
_NOT_SETUP_COMMANDS = {
    "PATH nohup": "PATH=/opt/tools/bin nohup {p}",
    "env assignment redirect": (
        "env AELFRICE_HOOK_SUMMARY_LINE=1 {p} --host codex 2>/dev/null"
    ),
    "nice": "nice {p}",
    "UID assignment": "UID=0 {p}",
    "group": "{{ {p}; }}",
    "absolute nice that isn't there": "/nonexistent/nice {p}",
    "setup's command with one extra space": "{p}  --host codex",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_NOT_SETUP_COMMANDS))
def test_doctor_warns_about_a_command_setup_does_not_write(
    tmp_path: Path, shape: str,
) -> None:
    """The handler beside setup's in one group gets the warning, which
    names the command setup writes, and doctor exits 0."""
    codex_dir, hooks, program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    other = _NOT_SETUP_COMMANDS[shape].format(p=program)
    _set_session_start_commands(hooks, ours, other)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    line = _warning_for(out, other)
    assert line is not None, out
    assert f"(`{ours}`)" in line, line


# Copies of setup's handler that aren't byte-identical to it. Setup reads
# each one as its own, because the program is the first word.
_OWNED_NOT_SETUP = {
    "piped": "{p} --host codex | cat",
    "redirected": "{p} --host codex > /dev/null",
    "syntax error after": "{p} --host codex; if",
    "trailing space": "{p} --host codex ",
    "tab": "{p}\t--host codex",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_OWNED_NOT_SETUP))
def test_doctor_warns_about_an_owned_handler_setup_does_not_write(
    tmp_path: Path, shape: str,
) -> None:
    """A copy of setup's handler counts as a run only when it is
    byte-identical to it, the rule a foreign handler follows."""
    codex_dir, hooks, program = _installed(tmp_path)
    copy = _OWNED_NOT_SETUP[shape].format(p=program)
    _set_session_start_commands(hooks, _session_start_command(hooks), copy)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, copy) is not None, out


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "C:\\bin\\aelf-session-start-hook.exe --host codex"',
        'cmd /c "C:\\bin\\aelf-session-start-hook.exe --host codex"',
        'cmd /c "C:\\bin\\AELF-SESSION-START-HOOK.CMD"',
        "C:\\bin\\aelf-session-start-hook",
    ],
)
def test_a_windows_command_names_the_program_with_a_launcher_suffix(
    command: str,
) -> None:
    """On Windows the name can end in a launcher suffix, and its case
    doesn't matter."""
    handler = {"type": "command", "command": command}
    assert set(_named_owned_keys(handler, windows=True)) == {
        "aelf-session-start-hook",
    }


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize(
    "command",
    [
        "/x/aelf-session-start-hook.exe",
        "/x/AELF-SESSION-START-HOOK",
        "/x/Aelf-Session-Start-Hook",
        "/usr/bin/env -Saelf-session-start-hook",
        "cmd /caelf-session-start-hook",
        "/x/xaelf-session-start-hook_2",
    ],
)
def test_a_command_names_the_program_in_any_case_on_every_platform(
    command: str, windows: bool,
) -> None:
    """The name is a plain substring in any case, wherever it is."""
    handler = {"type": "command", "command": command}
    assert set(_named_owned_keys(handler, windows=windows)) == {
        "aelf-session-start-hook",
    }


_HOOK = "aelf-session-start-hook"

# Install directories whose names hold characters a shell reads in some
# positions. Setup writes the program path unquoted.
_INSTALL_DIRS = {
    "plain": "plain",
    "non-ASCII letter": "D\u00e9veloppement",
    "#": "C#",
    "~": "proj~old",
    "space": "first last",
}


def _installed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str,
) -> tuple[Path, Path]:
    """A Codex dir after setup resolves every program in `folder`/bin on
    `PATH`, and its hooks.json. Doctor also looks for `aelf` on `PATH`."""
    bindir = tmp_path / folder / "bin"
    bindir.mkdir(parents=True)
    for name in (*_OWNED_BASENAMES, "aelf"):
        stub = bindir / name
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks)
    install_codex_skills(tmp_path / "skills")
    return codex_dir, hooks


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform == "win32", reason="the program stubs are POSIX scripts",
)
@pytest.mark.parametrize("folder", sorted(_INSTALL_DIRS))
def test_doctor_faults_a_copy_of_setup_handler_at_any_install_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str,
) -> None:
    """Setup's own handler counts as a run wherever setup installs it."""
    codex_dir, hooks = _installed_at(
        tmp_path, monkeypatch, _INSTALL_DIRS[folder],
    )
    ours = _session_start_command(hooks)
    program = tmp_path / _INSTALL_DIRS[folder] / "bin" / _HOOK
    assert ours == f"{program} --host codex"
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    _set_session_start_commands(hooks, ours, ours)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out
    assert "may also run" not in out


@pytest.mark.timeout(60)
def test_doctor_faults_a_copied_session_start_handler(tmp_path: Path) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _set_session_start_commands(hooks, ours, ours)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out


# Copies of setup's SessionStart handler under a `type` Codex runs no
# command for: it skips `prompt` and `agent` handlers, and an `mcp_tool`
# handler calls a tool. A handler with no `type` or an unknown one makes
# Codex reject the whole file (`_REJECTED_HANDLERS`).
_NOT_COMMAND_TYPES = {
    "prompt": {"type": "prompt"},
    "agent": {"type": "agent"},
    "mcp_tool": {"type": "mcp_tool", "server": "s", "tool": "t"},
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_NOT_COMMAND_TYPES))
def test_doctor_counts_no_run_for_a_handler_codex_runs_no_command_for(
    tmp_path: Path, shape: str,
) -> None:
    """The copy names the program, so it gets the warning."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"].append(
        {**_NOT_COMMAND_TYPES[shape], "command": ours},
    )
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, ours) is not None, out


# --- config.toml [hooks], event names, and files Codex rejects ---------------


def _write_config_hooks(codex_dir: Path, event: str, *commands: str) -> None:
    """Write a config.toml whose `[hooks]` table has one `event` group
    under setup's SessionStart matcher with these command handlers."""
    lines = [
        f"[[hooks.{event}]]",
        'matcher = "startup|resume|clear|compact|fork"',
    ]
    for command in commands:
        lines += [
            f"[[hooks.{event}.hooks]]",
            'type = "command"',
            f"command = {json.dumps(command)}",
        ]
    (codex_dir / "config.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


@pytest.mark.timeout(60)
def test_doctor_counts_a_copy_of_setup_handler_in_config_toml(
    tmp_path: Path,
) -> None:
    """Codex loads config.toml `[hooks]` beside hooks.json and runs both, so
    setup's handler in hooks.json and its copy in config.toml are two runs."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    _write_config_hooks(codex_dir, "SessionStart", ours)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out
    assert (
        f"[warn] SessionStart: the [hooks] table in {codex_dir / 'config.toml'}"
        " holds a copy of the aelf-session-start-hook command that `aelf setup "
        "--host codex` writes, and Codex runs the handlers in that table as "
        f"well as those in {hooks}\n" in out
    ), out


# hooks.json beside a config.toml whose `[hooks]` table holds two copies of
# setup's SessionStart handler: none, one with no event, and one with only
# a foreign handler.
_NO_OWNED_HOOKS_JSON: dict[str, object] = {
    "absent": None,
    "no events": {"hooks": {}},
    "foreign only": {
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "/bin/true"}]}],
        },
    },
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_NO_OWNED_HOOKS_JSON))
def test_doctor_faults_copies_of_setup_handler_only_in_config_toml(
    tmp_path: Path, shape: str,
) -> None:
    """A duplicate run fails doctor whichever file holds it, and the
    warning names hooks.json only when hooks.json holds a copy."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    doc = _NO_OWNED_HOOKS_JSON[shape]
    if doc is None:
        hooks.unlink()
    else:
        hooks.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    _write_config_hooks(codex_dir, "SessionStart", ours, ours)
    config = codex_dir / "config.toml"
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times" in out
    ), out
    assert (
        f"[warn] SessionStart: the [hooks] table in {config} holds a copy of "
        "the aelf-session-start-hook command that `aelf setup --host codex` "
        "writes, and Codex runs the handlers in that table\n" in out
    ), out
    if doc is None:
        assert (
            f"[warn] {hooks} does not exist; only the [hooks] table in "
            f"{config} runs aelfrice hooks for the Codex host" in out
        ), out
        assert "no aelfrice hooks are installed" not in out


@pytest.mark.timeout(60)
def test_doctor_warns_about_a_config_toml_handler_that_names_the_program(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    wrapper = f"/usr/bin/env {program}"
    _write_config_hooks(codex_dir, "SessionStart", wrapper)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    where = f" in the [hooks] table of {codex_dir / 'config.toml'}"
    assert any(
        ln.startswith(f"[warn] SessionStart handler{where} may also run {_HOOK}.")
        and ln.endswith(f"delete this one: {wrapper}")
        for ln in out.splitlines()
    ), out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("event", ["Notification", "sessionstart"])
@pytest.mark.parametrize("where", ["hooks.json", "config.toml"])
def test_doctor_counts_no_run_under_a_key_codex_reads_no_event_for(
    tmp_path: Path, event: str, where: str,
) -> None:
    """Codex skips a key that isn't one of its event names, so two copies of
    setup's handler there never run. Doctor warns about each one."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    if where == "hooks.json":
        doc = json.loads(hooks.read_text(encoding="utf-8"))
        doc["hooks"][event] = [
            {"hooks": [{"type": "command", "command": ours}] * 2},
        ]
        hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    else:
        _write_config_hooks(codex_dir, event, ours, ours)
    report = doctor_codex(codex_dir)
    assert report.duplicate_runs == []
    assert any(
        w.startswith(f"{event!r} handler")
        and f"names {_HOOK}, but Codex reads no hook event by that name" in w
        and w.endswith(f"never runs this handler: {ours}")
        for w in report.warnings
    ), report.warnings
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out


# Changes to setup's hooks.json that make Codex reject the whole file
# (`HooksFile` and the types it holds). Each one returns the document with
# a copy of setup's SessionStart handler beside it, so a doctor that kept
# reading the file would report two runs.
def _reject_untyped(ours: str) -> dict[str, object]:
    return {"command": ours}


def _reject_unknown_type(ours: str) -> dict[str, object]:
    return {"type": "Command", "command": ours}


def _reject_no_command(ours: str) -> dict[str, object]:
    return {"type": "command", "commandWindows": ours}


def _reject_both_windows_fields(ours: str) -> dict[str, object]:
    return {
        "type": "command", "command": ours,
        "commandWindows": ours, "command_windows": ours,
    }


_REJECTED_HANDLERS = {
    "no type": _reject_untyped,
    "unknown type": _reject_unknown_type,
    "type command with no command": _reject_no_command,
    "both Windows fields": _reject_both_windows_fields,
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("shape", sorted(_REJECTED_HANDLERS))
def test_doctor_counts_nothing_in_a_hooks_json_codex_rejects(
    tmp_path: Path, shape: str, windows: bool,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"] += [
        {"type": "command", "command": ours},
        _REJECTED_HANDLERS[shape](ours),
    ]
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    report = doctor_codex(codex_dir, windows=windows)
    assert report.duplicate_runs == []
    assert report.hooks_file_rejected is not None
    assert any(
        w.startswith(f"Codex loads no hook from {hooks}, because it rejects")
        for w in report.warnings
    ), report.warnings


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "text",
    [
        # A matcher that isn't a string.
        '{"hooks": {"SessionStart": [{"matcher": 5, "hooks": []}]}}',
        # A top-level key other than `description` and `hooks`.
        '{"hooks": {}, "version": 1}',
        # A field written twice.
        '{"hooks": {"Stop": [{"hooks": [{"type": "command",'
        ' "command": "a", "command": "b"}]}]}}',
    ],
)
def test_doctor_exits_1_on_a_hooks_json_codex_rejects(
    tmp_path: Path, text: str,
) -> None:
    """The exit an unparsable hooks.json gets."""
    codex_dir, hooks, _program = _installed(tmp_path)
    hooks.write_text(text, encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert f"[warn] Codex loads no hook from {hooks}" in out, out


@pytest.mark.timeout(60)
def test_doctor_exits_1_on_a_config_toml_hooks_table_codex_rejects(
    tmp_path: Path,
) -> None:
    """A copy of setup's handler with no `type` in config.toml makes Codex
    reject the table, so it adds no run to setup's handler in hooks.json."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    (codex_dir / "config.toml").write_text(
        "[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\n"
        f"command = {json.dumps(ours)}\n",
        encoding="utf-8",
    )
    report = doctor_codex(codex_dir)
    assert report.duplicate_runs == []
    assert report.config_hooks_rejected is not None
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[warn] Codex loads no hook from the [hooks] table in "
        f"{codex_dir / 'config.toml'}" in out
    ), out


# Each case and whether serde, built from the Codex types at
# openai/codex@69231e119bfca0b14453515e97d99688ce1c47c9, rejects it. Every
# outcome here was checked against that build. JSON cases are hooks.json
# text; TOML cases are config.toml text.
_SERDE_CASES = {
    "json: valid": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x"}]}]}}', False),
    "json: no type": ('{"hooks": {"Stop": [{"hooks": [{"command": "x"}]}]}}', True),
    "json: type null": ('{"hooks": {"Stop": [{"hooks": [{"type": null}]}]}}', True),
    "json: unknown type": ('{"hooks": {"Stop": [{"hooks": [{"type": "shell"}]}]}}', True),
    "json: command null": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": null}]}]}}', True),
    "json: commandWindows only": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "commandWindows": "x"}]}]}}', True),
    "json: both Windows fields null": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "commandWindows": null, "command_windows": null}]}]}}', True),
    "json: commandWindows null": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "commandWindows": null}]}]}}', False),
    "json: async null": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "async": null}]}]}}', True),
    "json: timeout -1": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "timeout": -1}]}]}}', True),
    "json: timeout 1.0": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "timeout": 1.0}]}]}}', True),
    "json: timeout null": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "timeout": null}]}]}}', False),
    "json: additionalContextLimit string": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "additionalContextLimit": "1"}]}]}}', True),
    "json: statusMessage number": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "statusMessage": 1}]}]}}', True),
    "json: prompt with a number command": ('{"hooks": {"Stop": [{"hooks": [{"type": "prompt", "command": 1}]}]}}', False),
    "json: mcp_tool with no server": ('{"hooks": {"Stop": [{"hooks": [{"type": "mcp_tool", "tool": "t"}]}]}}', True),
    "json: mcp_tool input with null": ('{"hooks": {"Stop": [{"hooks": [{"type": "mcp_tool", "server": "s", "tool": "t", "input": {"a": null}}]}]}}', True),
    "json: mcp_tool input 2**63": ('{"hooks": {"Stop": [{"hooks": [{"type": "mcp_tool", "server": "s", "tool": "t", "input": {"a": 9223372036854775808}}]}]}}', True),
    "json: mcp_tool input 2**64": ('{"hooks": {"Stop": [{"hooks": [{"type": "mcp_tool", "server": "s", "tool": "t", "input": {"a": 18446744073709551616}}]}]}}', False),
    "json: handler null": ('{"hooks": {"Stop": [{"hooks": [null]}]}}', True),
    "json: group hooks null": ('{"hooks": {"Stop": [{"hooks": null}]}}', True),
    "json: group with an unknown key": ('{"hooks": {"Stop": [{"x": 1, "hooks": []}]}}', False),
    "json: matcher number": ('{"hooks": {"Stop": [{"matcher": 1}]}}', True),
    "json: matcher null": ('{"hooks": {"Stop": [{"matcher": null}]}}', False),
    "json: event null": ('{"hooks": {"Stop": null}}', True),
    "json: unknown event null": ('{"hooks": {"Notification": null}}', False),
    "json: unknown event with no type": ('{"hooks": {"Notification": [{"hooks": [{"command": "x"}]}]}}', False),
    "json: event twice": ('{"hooks": {"Stop": [], "Stop": []}}', True),
    "json: unknown key twice": ('{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "z": 1, "z": 2}]}]}}', False),
    "json: top-level extra key": ('{"hooks": {}, "x": 1}', True),
    "json: description number": ('{"description": 1}', True),
    "json: hooks null": ('{"hooks": null}', True),
    "toml: valid": ('[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = "x"\n', False),
    "toml: no type": ('[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ncommand = "x"\n', True),
    "toml: timeout 1.5": ('[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = "x"\ntimeout = 1.5\n', True),
    "toml: command a date": ('[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = 1979-05-27\n', False),
    "toml: event a table": ('[hooks.Stop]\nmatcher = "x"\n', True),
    "toml: state only": ('[hooks.state."k"]\ntrusted_hash = "h"\n', False),
    "toml: hooks a number": ('hooks = 5\n', True),
    # Codex reads config.toml's `hooks` as `HooksToml`, whose events are
    # flattened beside a `state` table of `HookStateToml` entries.
    "toml: hooks an array": (
        'hooks = [[], [], [], [], [], [], [], '
        '[{hooks = [{type = "command", command = "x"}]}]]\n',
        True,
    ),
    "toml: state entry a number": ('[hooks.state]\nfoo = 1\n', True),
    "toml: state a number": ('[hooks]\nstate = 1\n', True),
    "toml: state enabled a string": (
        '[hooks.state."k"]\nenabled = "yes"\n', True,
    ),
    "toml: state trusted_hash a number": (
        '[hooks.state."k"]\ntrusted_hash = 1\n', True,
    ),
    "toml: state trusted_content_hash a number": (
        '[hooks.state."k"]\ntrusted_content_hash = 1\n', True,
    ),
    "toml: state with an unknown key": (
        '[hooks.state."k"]\nenabled = true\nother = 1\n', False,
    ),
}


@pytest.mark.parametrize("case", sorted(_SERDE_CASES))
def test_doctor_rejects_what_codex_serde_rejects(case: str) -> None:
    text, rejects = _SERDE_CASES[case]
    if case.startswith("json: "):
        reason = _rejected_hooks_file(
            json.loads(text, object_pairs_hook=_json_object),
        )
    else:
        reason = _rejected_config_hooks(tomllib.loads(text)["hooks"])
    assert (reason is not None) == rejects, reason


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "text",
    [
        'hooks = [[], [], [], [], [], [], [], [{hooks = [{type = "command", '
        'command = {ss}}]}]]\n',
        '[hooks.state]\nfoo = 1\n',
        '[hooks.state."k"]\nenabled = "yes"\n',
    ],
)
def test_doctor_exits_1_on_a_config_toml_hooks_value_codex_rejects(
    tmp_path: Path, text: str,
) -> None:
    """Codex's config load rejects a `hooks` value written as an array and
    a malformed `[hooks.state]`, so doctor exits 1 as it does for any other
    `[hooks]` table Codex rejects."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = json.dumps(_session_start_command(hooks))
    (codex_dir / "config.toml").write_text(
        text.replace("{ss}", ours), encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[warn] Codex loads no hook from the [hooks] table in "
        f"{codex_dir / 'config.toml'}, because it rejects the whole table" in out
    ), out


@pytest.mark.timeout(60)
def test_doctor_passes_a_config_toml_with_approved_hooks_state(
    tmp_path: Path,
) -> None:
    codex_dir, _hooks, _program = _installed(tmp_path)
    (codex_dir / "config.toml").write_text(
        '[hooks.state."k"]\nenabled = true\ntrusted_hash = "h"\n',
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "rejects the whole table" not in out, out


def _windows_report(
    codex_dir: Path, hooks: Path, extra: dict[str, object],
) -> tuple[list[tuple[str, str, int]], list[str]]:
    """Doctor's duplicate runs and warnings on Windows after `extra` is
    added beside setup's SessionStart handler."""
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"].append(
        {"type": "command", **extra},
    )
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    report = doctor_codex(codex_dir, windows=True)
    return report.duplicate_runs, report.warnings


@pytest.mark.timeout(60)
@pytest.mark.parametrize("field", ["commandWindows", "command_windows"])
def test_doctor_on_windows_counts_the_windows_command(
    tmp_path: Path, field: str,
) -> None:
    """Codex runs `commandWindows`, or its alias, on Windows in place of
    `command`."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    runs, _warnings = _windows_report(
        codex_dir, hooks, {"command": "true", field: ours},
    )
    assert runs == [("SessionStart", _HOOK, 2)]


@pytest.mark.timeout(60)
def test_doctor_on_windows_warns_about_setup_command_that_codex_skips(
    tmp_path: Path,
) -> None:
    """A handler with setup's `command` and another `commandWindows` runs
    the other command on Windows, so doctor counts no run for it and warns
    that it names the program."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    runs, warnings = _windows_report(
        codex_dir, hooks, {"command": ours, "commandWindows": "true"},
    )
    assert runs == []
    assert any(
        w.startswith(f"SessionStart handler may also run {_HOOK}.")
        and w.endswith(f"delete this one: {ours}")
        for w in warnings
    ), warnings


@pytest.mark.timeout(60)
def test_doctor_warns_about_a_windows_command_that_names_the_program(
    tmp_path: Path,
) -> None:
    """On POSIX Codex runs `command`, so doctor counts no run for the
    Windows field and warns that it names the program."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"].append(
        {"type": "command", "command": "true", "commandWindows": ours},
    )
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    report = doctor_codex(codex_dir, windows=False)
    assert report.duplicate_runs == []
    assert any(
        w.startswith(f"SessionStart handler may also run {_HOOK}.")
        and w.endswith(f"delete this one: {ours}")
        for w in report.warnings
    ), report.warnings


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
    doc["hooks"].setdefault(event, []).append(group)
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
    # Setup's matcher lists `fork`, so a forked thread runs both.
    "copy under fork": ("{ours}", "fork"),
    "copy under ^fork$": ("{ours}", "^fork$"),
    "copy under ^res": ("{ours}", "^res"),
    "copy under up$": ("{ours}", "up$"),
    "copy under *": ("{ours}", "*"),
    "copy under empty": ("{ours}", ""),
    "copy under .*": ("{ours}", ".*"),
    "copy under ^startup$|ork": ("{ours}", "^startup$|ork"),
}
# Matchers doctor doesn't read, so it counts no run in the group and warns
# about each handler there that names a program. Codex can't compile some
# of them, and skips the group. Rust's `regex` reads the others, and some
# of those mean something else to Python's `re`, such as `[[:alpha:]]`.
_UNREAD_MATCHERS = {
    "unclosed group": "(",
    "lookahead": "(?=resume)",
    "backreference": r"(r)\1?esume",
    "unclosed counted repetition": "resume|x{",
    "nested class opener": "resume|[[]",
    "named group": "(?<s>resume)",
    "flag": "(?i)RESUME",
    "class holding (?=": "[(?=]?resume",
    "braced hex escape": r"\x{72}esume",
    "Unicode class": r"\p{L}+",
    "POSIX class": "^[[:alpha:]]+$",
    "word class": "[[:word:]]+",
    "class intersection": "[a-z&&[^q]]+",
    "word start": r"\b{start}fork",
    "group": "^(startup|fork)$",
    "empty branch": "^resume||x",
    "anchor alone": "^",
}


def _unread_warning_for(
    out: str, event: str, key: str, command: str, why: str,
) -> str | None:
    """The doctor warning line for a handler in a group doctor counts no
    run in, if there is one."""
    return next(
        (
            ln for ln in out.splitlines()
            if ln.startswith(f"[warn] {event} handler at `hooks.{event}[")
            and f"names {key}, and doctor counts no run in its group, because "
            f"{why}." in ln
            and ln.endswith(f"on the same fire. Handler: {command}")
        ),
        None,
    )


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
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_UNREAD_MATCHERS))
def test_doctor_counts_no_run_under_a_matcher_it_does_not_read(
    tmp_path: Path, shape: str,
) -> None:
    """A copy of setup's command under the matcher gets a warning, and
    doctor exits 0 whether or not Codex runs the group."""
    codex_dir, hooks, _program = _installed(tmp_path)
    matcher = _UNREAD_MATCHERS[shape]
    ours = _session_start_command(hooks)
    _add_group(hooks, "SessionStart", ours, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    why = f"doctor doesn't read the matcher {matcher!r}"
    assert _unread_warning_for(out, "SessionStart", _HOOK, ours, why), out


@pytest.mark.timeout(60)
def test_doctor_gives_a_wrapper_under_an_unread_matcher_one_warning(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    wrapper = f"sh -c '{program}'"
    _add_group(hooks, "SessionStart", wrapper, "(?=resume)")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "may also run" not in out
    why = "doctor doesn't read the matcher '(?=resume)'"
    assert _unread_warning_for(out, "SessionStart", _HOOK, wrapper, why), out


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


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("event", "matcher"),
    [
        ("PreToolUse", "(?<=B)ash"),
        # A tool event's values are an open set of tool names, so doctor
        # reads a regex there only when each branch is `^name$`.
        ("PreToolUse", "^Ba"),
        ("PreToolUse", "^Web"),
        ("PostToolUse", "Web.*"),
        ("PostToolUse", "^Bash$|Read"),
    ],
)
def test_doctor_counts_no_run_under_a_tool_matcher_it_does_not_read(
    tmp_path: Path, event: str, matcher: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    command = _first_command(hooks, event)
    _add_group(hooks, event, command, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    why = f"doctor doesn't read the matcher {matcher!r}"
    assert _unread_warning_for(
        out, event, command_launcher_key(command), command, why,
    ), out


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("event", "first", "second"),
    [
        # Codex fires SessionEnd with one reason and matches
        # PermissionRequest against tool names, and doctor lists neither.
        ("SessionEnd", "clear", "startup"),
        ("SessionEnd", None, None),
        ("PermissionRequest", "Bash", "Read"),
        ("PermissionRequest", "Bash", "Bash"),
    ],
)
def test_doctor_counts_no_run_in_an_event_whose_values_it_does_not_list(
    tmp_path: Path, event: str, first: str | None, second: str | None,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = next(
        str(h["command"]) for h in json.loads(
            hooks.read_text(encoding="utf-8"),
        )["hooks"]["Stop"][0]["hooks"]
        if "aelf-stop-hook" in str(h["command"])
    )
    _add_group(hooks, event, ours, first)
    _add_group(hooks, event, ours, second)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    why = f"doctor doesn't list the values Codex tests a {event} matcher against"
    lines = [
        ln for ln in out.splitlines()
        if _unread_warning_for(ln, event, "aelf-stop-hook", ours, why)
    ]
    assert len(lines) == 2, out


# Matchers and the SessionStart sources Rust's `regex` 1.12.3 matches with
# each, from the rv15 oracle, which compiles a matcher the way
# codex-rs/hooks/src/engine/matcher.rs does. `None` means Rust rejects the
# pattern. Doctor reads the first group, and must agree with Rust on each.
# It reads none of the second group, whose meaning differs between Rust's
# `regex` and Python's `re` or that only a wider reader could list.
_READ_BY_DOCTOR: dict[str, frozenset[str]] = {
    "": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "*": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    ".*": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "startup|resume": frozenset({"startup", "resume"}),
    "start": frozenset(),
    "startup|": frozenset({"startup"}),
    "|": frozenset(),
    "^res": frozenset({"resume"}),
    "up$": frozenset({"startup"}),
    # Only letters and `|`, so a list of one exact name to Codex.
    "c": frozenset(),
    "a||b": frozenset(),
    "^c": frozenset({"clear", "compact"}),
    "^fork$": frozenset({"fork"}),
    "^startup$|ork": frozenset({"startup", "fork"}),
    "^s|e$|ar": frozenset({"startup", "resume", "clear"}),
    "STARTUP$": frozenset(),
    "^r$": frozenset(),
}
_NOT_READ_BY_DOCTOR: dict[str, frozenset[str] | None] = {
    "^[[:alpha:]]+$": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "[[:word:]]+": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "[a-z&&[^q]]+": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    r"\b{start}fork": frozenset({"fork"}),
    r"\>": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "(?i)FORK": frozenset({"fork"}),
    "x{": None,
    "(?=a)": None,
    "^(startup|fork)$": frozenset({"startup", "fork"}),
    "^a||b": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "^": frozenset({"startup", "resume", "clear", "compact", "fork"}),
    "st.rtup": frozenset({"startup"}),
    "^ fork$": frozenset(),
}


@pytest.mark.parametrize("matcher", sorted(_READ_BY_DOCTOR))
def test_doctor_reads_its_matcher_subset_the_way_rust_regex_does(
    matcher: str,
) -> None:
    got = _matched_values("SessionStart", {"matcher": matcher})
    assert got == _READ_BY_DOCTOR[matcher]


@pytest.mark.parametrize("matcher", sorted(_NOT_READ_BY_DOCTOR))
def test_doctor_reads_no_matcher_outside_its_subset(matcher: str) -> None:
    assert _matched_values("SessionStart", {"matcher": matcher}) is None
    assert _matched_values("PreToolUse", {"matcher": matcher}) is None


@pytest.mark.parametrize(
    ("matcher", "names"),
    [
        ("Bash", frozenset({"Bash"})),
        ("Bash|Read", frozenset({"Bash", "Read"})),
        ("^Bash$", frozenset({"Bash"})),
        ("^Bash$|^Read$", frozenset({"Bash", "Read"})),
        ("^Ba", None),
        ("sh$", None),
        ("Ba", frozenset({"Ba"})),
        ("^Bash$|Read", None),
    ],
)
def test_doctor_reads_a_tool_matcher_only_as_exact_names(
    matcher: str, names: frozenset[str] | None,
) -> None:
    """`Ba` is a list of one exact name to Codex, and `^Ba` a regex."""
    assert _matched_values("PreToolUse", {"matcher": matcher}) == names


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
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out


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
    assert "SessionStart starts the aelf-session-start-hook command 3 times" in out


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
    assert ("PreCompact starts the aelf-transcript-logger command 2 times" in out) == bool(rc)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("event", "matcher"),
    [
        # `.*` matches every tool name, so the copy runs beside setup's.
        ("PreToolUse", ".*"),
        ("PostToolUse", ".*"),
        # Codex ignores a matcher on Stop.
        ("Stop", "never"),
        ("Stop", "(?=x)"),
    ],
)
def test_doctor_counts_every_handler_where_every_fire_runs_it(
    tmp_path: Path, event: str, matcher: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    command = _first_command(hooks, event)
    _add_group(hooks, event, command, matcher)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert f"{event} starts the {command_launcher_key(command)} command 2 times" in out


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
    # A group under no matcher or `*` runs on every fire, so a `Bash` call
    # runs it and setup's handler, and a `WebSearch` call runs it and the
    # copy under `WebSearch`. No call runs all three.
    "no matcher and WebSearch": ("PreToolUse", [None, "WebSearch"], 2),
    "* and WebSearch": ("PreToolUse", ["*", "WebSearch"], 2),
    "no matcher and Bash": ("PreToolUse", [None, "Bash"], 3),
    # A regex whose branches are each `^name$` lists exact names.
    "^Bash$|^Read$": ("PreToolUse", ["^Bash$|^Read$"], 2),
    "^WebSearch$": ("PreToolUse", ["^WebSearch$"], 0),
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
        assert f"{event} starts the {key} command {runs} times" in out, out
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


# Each foreign handler holds no owned program name in any case. A name
# built at run time or a link with another name escapes the check.
_NAMES_NO_PROGRAM = {
    "other hook": "/usr/bin/python3 /x/other-hook.py SessionStart --verbose",
    "built from parts": "X=hook; exec aelf-session-start-$X --host codex",
    "one letter short": "/x/aelf-session-start-hoo --host codex",
    "a link with another name": "/x/session-start-link --host codex",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_NAMES_NO_PROGRAM))
def test_doctor_ignores_a_foreign_handler_that_names_no_aelfrice_program(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    _set_session_start_commands(
        hooks,
        _NAMES_NO_PROGRAM[shape].format(p=program),
        _session_start_command(hooks),
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert "may also run" not in out


# Each foreign handler holds the program name inside other text: right
# after a letter or `-`, after an escape such as `\n` or `\t`, in another
# case, or inside a longer word. The round-12 review found doctor silent
# on the first seven, and a shell runs the program for each of them.
# `{p}` is the program path and `{n}` its name.
_INSIDE_TEXT = {
    "env -S attached": "/usr/bin/env -S{n}",
    "default expansion": "exec \"${{CMD:-{n}}}\"",
    "default expansion without :": "exec ${{CMD-{n}}}",
    "python3 -c after \\n": (
        "python3 -c 'import os; os.system(\"true\\n{n}\")'"
    ),
    "printf \\n into sh": "printf 'true\\n{n}\\n' | sh",
    "bash -c $'...\\n'": "bash -c $'true\\n{n}'",
    "printf \\t into sh": "printf '\\t{n}' | sh",
    "other case": "Aelf-Session-Start-Hook",
    "upper case path": "{P} --host codex",
    "suffix after a dot": "/usr/bin/python3 {p}.py --host codex",
    "suffix after a dash": "{p}-old --host codex",
    "suffix after an underscore": "{p}_2 --host codex",
    "suffix after a digit": "{p}2 --host codex",
    "prefix letter": "/x/xaelf-session-start-hook",
    "prefix dot": "/x/.aelf-session-start-hook",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_INSIDE_TEXT))
def test_doctor_warns_about_a_name_inside_other_text(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    other = _INSIDE_TEXT[shape].format(p=program, n=_HOOK, P=program.upper())
    _set_session_start_commands(hooks, _session_start_command(hooks), other)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, other) is not None, out


# Each foreign handler only prints the program's name. Doctor reads no
# shell syntax, so it warns about each one.
_PRINTS = {
    "echo": "echo aelf-session-start-hook",
    "sh -c printf": "sh -c 'printf \"%s\\n\" {p}'",
    "echo or true": "echo {p} || true",
    "printf into a file": "printf '%s\\n' {p} > /dev/null",
    "env -i echo": "env -i echo {p}",
    "assignment echo": "AELFRICE_HOOK_SUMMARY_LINE=1 echo {p}",
    "exec -a echo": "exec -a name echo {p}",
    "timeout -s echo": "timeout -s KILL 30 echo {p}",
    "nice -n echo": "nice -n 5 echo {p}",
    "env -S echo": "env -S 'echo {p}'",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_PRINTS))
def test_doctor_warns_about_a_handler_that_only_prints_the_program(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    other = _PRINTS[shape].format(p=program)
    _set_session_start_commands(hooks, other, _session_start_command(hooks))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, other) is not None, out


# Handlers that run the program from a script string, from the output of
# a command substitution, or from a variable. The round-11 review found
# doctor silent on each one. `{p}` is the program path and `{n}` its name.
_STRING_RUNS = {
    "python3 -c subprocess": (
        "python3 -c \"import subprocess; subprocess.run(['{p}'])\""
    ),
    "python3 -c os.system": (
        "python3 -c 'import os; os.system(\"{p} --host codex\")'"
    ),
    "perl -e": "perl -e 'system(\"{p}\")'",
    "node -e": "node -e \"require('child_process').execSync('{p}')\"",
    "awk": "awk 'BEGIN{{system(\"{p}\")}}'",
    "osascript": "osascript -e \"do shell script \\\"{p}\\\"\"",
    "$(echo)": "$(echo {p})",
    "backquotes": "`echo {p}`",
    "eval $(echo)": "eval \"$(echo {p})\"",
    "exec $(printf)": "exec $(printf %s {p})",
    "default expansion": "exec \"${{CMD:-{p}}}\"",
    "variable": "CMD={n}; exec $CMD",
    "variable with a path": "CMD={p}; $CMD",
    "--cmd=": "wrapper --cmd={n}",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_STRING_RUNS))
def test_doctor_warns_about_a_program_run_from_a_string(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    other = _STRING_RUNS[shape].format(p=program, n=_HOOK)
    _set_session_start_commands(hooks, _session_start_command(hooks), other)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, other) is not None, out


# Each foreign handler names the program where a prefix word takes an
# operand, so it runs no aelfrice program, but it names one, and doctor
# warns about every handler that names one.
_OPERANDS = {
    "exec -a": "exec -a {p}",
    "env -u": "env -u {p} --host codex",
    "timeout": "timeout {p} --host codex",
    "env -C echo": "env -C {p} echo hello",
    "time -o echo": "/usr/bin/time -o {p} echo hello",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_OPERANDS))
def test_doctor_warns_about_a_handler_that_names_the_program_as_an_operand(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    other = _OPERANDS[shape].format(p=program)
    _set_session_start_commands(hooks, other, _session_start_command(hooks))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert _warning_for(out, other) is not None, out


# Each foreign handler only checks that the program is there. Doctor reads
# no shell syntax, so it warns about each one.
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
# command. The wrapper gets the "may also run" warning when one fire runs
# it and another handler of the program, and the lone-handler warning
# otherwise.
_WRAPPER_MATCHERS = {
    "env under resume": ("SessionStart", "resume", "/usr/bin/env {p}", True),
    # Setup's matcher lists `fork`.
    "env under fork": ("SessionStart", "fork", "/usr/bin/env {p}", True),
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
    "sh -c under ^Bash$": ("PreToolUse", "^Bash$", "sh -c '{search}'", True),
    # No handler of the program runs in PreCompact.
    "lone sh -c in PreCompact": ("PreCompact", None, "sh -c '{p}'", False),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_WRAPPER_MATCHERS))
def test_doctor_words_a_wrapper_warning_by_whether_a_fire_runs_it_twice(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, program = _installed(tmp_path)
    event, matcher, template, cofires = _WRAPPER_MATCHERS[shape]
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
    assert (line is not None) == cofires, out
    key = "aelf-search-tool-hook" if "{search}" in template else _HOOK
    lone = _lone_warning_for(out, event, key, wrapper)
    assert (lone is not None) != cofires, out
    assert lone is None or "A rerun" not in lone, lone


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
def test_doctor_gives_the_disjoint_wrapper_the_lone_handler_warning(
    tmp_path: Path,
) -> None:
    """A `resume` fire runs setup's handler and one wrapper. With setup's
    matcher narrowed to `startup|resume`, no fire runs the `fork` wrapper
    beside another handler, so it gets the lone-handler warning."""
    codex_dir, hooks, program = _installed(tmp_path)
    _narrow_setup_matcher(hooks, "startup|resume")
    overlapping, disjoint = f"sh -c '{program}'", f"bash -c '{program}'"
    _add_group(hooks, "SessionStart", overlapping, "resume")
    _add_group(hooks, "SessionStart", disjoint, "fork")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert _warning_for(out, overlapping) is not None, out
    assert _warning_for(out, disjoint) is None, out
    assert _lone_warning_for(out, "SessionStart", _HOOK, overlapping) is None
    assert _lone_warning_for(out, "SessionStart", _HOOK, disjoint) is not None


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["hooks.json", "config.toml"])
@pytest.mark.parametrize("event", ["SessionStart", "PreCompact"])
def test_doctor_warns_about_a_copy_of_setup_command_in_another_event(
    tmp_path: Path, event: str, where: str,
) -> None:
    """Setup writes the aelf-hook handler in UserPromptSubmit only. One copy
    of its command in another event is no duplicate run, and it isn't
    setup's handler there, so it gets a warning."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _first_command(hooks, "UserPromptSubmit")
    if where == "hooks.json":
        _add_group(hooks, event, ours)
        label, group = "", 1
    else:
        _write_config_hooks(codex_dir, event, ours)
        label = f" in the [hooks] table of {codex_dir / 'config.toml'}"
        group = 0
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert out.count("runs aelf-hook with a command") == 1, out
    assert (
        f"[warn] {event} handler{label} runs aelf-hook with a command `aelf "
        "setup --host codex` writes, but setup writes no aelf-hook handler "
        f"in {event}; delete this one, at `hooks.{event}[{group}].hooks[0]`: "
        f"{ours}\n" in out
    ), out


def _old_install_program(tmp_path: Path, name: str) -> str:
    """An aelfrice program an older install left at another path."""
    old = tmp_path / "old" / "bin" / name
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    old.chmod(0o755)
    return str(old)


# A handler that names aelf-stop-hook where no fire runs another handler
# of it: (event, file, whether the handler is the stop program itself at
# another path, whether setup's next run replaces it). Setup writes Stop
# in hooks.json, writes no handler in Codex's agent stop event, and never
# touches config.toml. Python joins the event name split across literals.
_AGENT_STOP = "Sub" "agentStop"
_LONE_HANDLERS = {
    "older install in Stop": ("Stop", "hooks.json", True, True),
    "older install in the agent stop event": (
        _AGENT_STOP, "hooks.json", True, False,
    ),
    "older install in config.toml Stop": ("Stop", "config.toml", True, False),
    "wrapper in the agent stop event": (
        _AGENT_STOP, "hooks.json", False, False,
    ),
    "wrapper in config.toml in the agent stop event": (
        _AGENT_STOP, "config.toml", False, False,
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform == "win32", reason="the program stub is a POSIX script",
)
@pytest.mark.parametrize("shape", sorted(_LONE_HANDLERS))
def test_doctor_warns_about_a_lone_handler_that_names_the_program(
    tmp_path: Path, shape: str,
) -> None:
    """Every handler that names a program and isn't a byte-identical copy
    of setup's command gets a warning, though no fire runs it twice."""
    codex_dir, hooks, _program = _installed(tmp_path)
    event, where, owned, replaced = _LONE_HANDLERS[shape]
    old = _old_install_program(tmp_path, "aelf-stop-hook")
    command = old if owned else f"bash -c {old}"
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    stop = doc["hooks"]["Stop"][0]["hooks"]
    ours = next(h for h in stop if "aelf-stop-hook" in h["command"])
    if where == "hooks.json" and event == "Stop":
        # The older install's handler stands where setup's was.
        ours["command"] = command
    else:
        stop.remove(ours)
        if where == "hooks.json":
            doc["hooks"].setdefault(event, []).append(
                {"hooks": [{"type": "command", "command": command}]},
            )
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if where == "config.toml":
        _write_config_hooks(codex_dir, event, command)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert "times" not in out
    assert "may also run" not in out
    label = (
        f" in the [hooks] table of {codex_dir / 'config.toml'}"
        if where == "config.toml" else ""
    )
    line = _lone_warning_for(out, event, "aelf-stop-hook", command, label)
    assert line is not None, out
    assert (
        " A rerun of `aelf setup --host codex` replaces this aelfrice "
        "handler with setup's, as it does one an older install wrote at "
        "another path." in line
    ) == replaced, line
    if replaced:
        assert rc == 0, out
        install_codex_hooks(hooks)
        rc, out = _doctor(codex_dir, tmp_path / "skills")
        assert rc == 0, out
        assert old not in out, out


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


# --- Codex's agent start and stop events --------------------------------------

# Two groups that each hold a copy of setup's UserPromptSubmit handler under
# one of Codex's agent events: (first matcher, second matcher, exit code).
# Codex reads the matcher there, tested against a name doctor doesn't list,
# so groups that list exact names run together only when they share one,
# and a group under no matcher runs with every group. Doctor counts no run
# under any other regex, and warns instead.
_AGENT_EVENT_MATCHERS: dict[str, tuple[str | None, str, int]] = {
    "disjoint exact names": ("a", "b", 0),
    "a shared exact name": ("a|b", "b", 1),
    "a matcher Rust rejects": ("(", "(", 0),
    "an anchored name beside the same exact name": ("^b$", "b", 1),
    # Each branch of a regex in these events must be `^name$`.
    "a partly anchored regex beside an exact name": ("^x", "b", 0),
    "no matcher beside an exact name": (None, "b", 1),
}
_AGENT_EVENTS = ("Sub" "agentStart", "Sub" "agentStop")


@pytest.mark.timeout(60)
@pytest.mark.parametrize("event", _AGENT_EVENTS)
@pytest.mark.parametrize("shape", sorted(_AGENT_EVENT_MATCHERS))
def test_doctor_reads_the_matcher_of_codex_agent_events(
    tmp_path: Path, shape: str, event: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _first_command(hooks, "UserPromptSubmit")
    first, second, code = _AGENT_EVENT_MATCHERS[shape]
    _add_group(hooks, event, ours, first)
    _add_group(hooks, event, ours, second)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == code, out
    assert (f"{event} starts the aelf-hook command 2 times" in out) == bool(
        code,
    ), out


# --- strings anywhere in a handler, and handlers written as arrays --------------

# A handler or group in its own Stop group beside setup's, built from setup's
# stop command `{p}`. Codex reads an array by position, and doctor counts no
# run in one but warns about any string in it that names a program, as it
# does for a string in any field of a handler.
_ANYWHERE: dict[str, Callable[[str], object]] = {
    "positional handler": lambda p: {"hooks": [["command", p]]},
    "positional group": lambda p: [
        "", [{"type": "command", "command": p}],
    ],
    "positional group of a positional handler": lambda p: ["", [["command", p]]],
    "statusMessage": lambda p: {
        "hooks": [
            {"type": "command", "command": "/bin/true", "statusMessage": p},
        ],
    },
    "mcp_tool input key": lambda p: {
        "hooks": [
            {"type": "mcp_tool", "server": "s", "tool": "t", "input": {p: 1}},
        ],
    },
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_ANYWHERE))
def test_doctor_warns_about_a_name_in_any_string_of_a_handler_or_group(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    stop = next(
        str(h["command"]) for h in doc["hooks"]["Stop"][0]["hooks"]
        if "aelf-stop-hook" in str(h["command"])
    )
    doc["hooks"]["Stop"].append(_ANYWHERE[shape](stop))
    hooks.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out
    assert any(
        ln.startswith("[warn] Stop handler may also run aelf-stop-hook.")
        and ln.endswith(f"delete this one: {stop}")
        for ln in out.splitlines()
    ), out


# --- the last check: every string that names a program --------------------------

# A path written the way doctor writes one, such as `hooks.Stop[1].hooks[0]`.
_PATH_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _render(path: tuple[object, ...]) -> str:
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif _PATH_NAME.fullmatch(str(part)):
            out += f".{part}" if out else str(part)
        else:
            out += f"[{json.dumps(str(part))}]"
    return out


def _named_strings(
    value: object, path: tuple[object, ...],
) -> list[tuple[tuple[object, ...], str]]:
    """Every string in `value` that names an owned program, keys included,
    each with its path; a key's path is its member's."""
    found: list[tuple[tuple[object, ...], str]] = []
    if isinstance(value, str):
        if any(k in value.casefold() for k in _OWNED_BASENAMES):
            found.append((path, value))
    elif isinstance(value, dict):
        for key, child in cast(dict[str, object], value).items():
            found += _named_strings(key, (*path, key))
            found += _named_strings(child, (*path, key))
    elif isinstance(value, list):
        for i, child in enumerate(cast(list[object], value)):
            found += _named_strings(child, (*path, i))
    return found


def _unreported(
    warnings: list[str], config: Path, tag: str,
    strings: list[tuple[tuple[object, ...], str]],
) -> list[str]:
    """The paths of the strings that no warning names, with the program,
    by the string's path or a path above it in the same file."""
    missed: list[str] = []
    for path, text in strings:
        keys = [k for k in _OWNED_BASENAMES if k in text.casefold()]
        above = [f"`{_render(path[:n])}`" for n in range(1, len(path) + 1)]
        if not any(
            (str(config) in w) == (tag == "toml")
            and all(k in w for k in keys) and any(a in w for a in above)
            for w in warnings
        ):
            missed.append(f"{tag} {_render(path)}: {text}")
    return missed


def _toml_value(value: object) -> str:
    """`value` as a TOML inline value, keys always quoted."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return json.dumps(value)
    if isinstance(value, list):
        items = cast(list[object], value)
        return "[" + ", ".join(_toml_value(v) for v in items) + "]"
    assert isinstance(value, dict)
    return "{" + ", ".join(
        f"{json.dumps(k)} = {_toml_value(v)}"
        for k, v in cast(dict[str, object], value).items()
    ) + "}"


# Text that names a program but is no command setup writes.
_NAMING = (
    "aelf-hook", "AELF-STOP-HOOK", "/opt/aelf-search-tool-hook --x",
    "x aelf-commit-ingest", "aelf-pre-issue-hook.py", "aelf-transcript-logger;",
    "env aelf-session-start-hook",
)
_PLAIN = ("x", "echo hi", "Bash", "")


def _junk(rng: random.Random, depth: int) -> object:
    """A random JSON and TOML value whose strings may name a program."""
    roll = rng.random()
    if depth <= 0 or roll < 0.35:
        return rng.choice([
            rng.choice(_NAMING), rng.choice(_PLAIN), 1, True,
        ])
    if roll < 0.65:
        return [_junk(rng, depth - 1) for _ in range(rng.randint(0, 3))]
    return {
        (rng.choice(_NAMING) if rng.random() < 0.4 else rng.choice("abc")):
        _junk(rng, depth - 1)
        for _ in range(rng.randint(0, 3))
    }


def _junk_handler(rng: random.Random) -> object:
    roll = rng.random()
    if roll < 0.15:
        return ["command", _junk(rng, 2)]
    if roll < 0.3:
        return {
            "type": "mcp_tool", "server": "s", "tool": "t",
            "input": {"a": _junk(rng, 2)},
        }
    handler: dict[str, object] = {
        "type": rng.choice(["command", "prompt"]),
        "command": rng.choice(_NAMING + _PLAIN),
    }
    if rng.random() < 0.3:
        handler["commandWindows"] = rng.choice(_NAMING)
    if rng.random() < 0.3:
        handler["statusMessage"] = rng.choice(_NAMING)
    if rng.random() < 0.4:
        handler["extra"] = _junk(rng, 3)
    return handler


# Matchers for junk groups: some doctor reads, some it doesn't, one Codex
# rejects.
_JUNK_MATCHERS = (None, "", "*", "startup", "^res", "(?i)x", "(", "Bash")


def _junk_events(rng: random.Random) -> dict[str, object]:
    """Junk groups under Codex events and junk values under other keys."""
    events: dict[str, object] = {}
    for event in rng.sample(
        ["Stop", "SessionStart", "SessionEnd", "PreToolUse", "PreCompact"],
        rng.randint(0, 2),
    ):
        groups: list[object] = []
        for _ in range(rng.randint(1, 2)):
            if rng.random() < 0.15:
                groups.append(["", [_junk_handler(rng)]])
                continue
            group: dict[str, object] = {
                "hooks": [_junk_handler(rng) for _ in range(rng.randint(1, 2))],
            }
            matcher = rng.choice(_JUNK_MATCHERS)
            if matcher is not None:
                group["matcher"] = matcher
            if rng.random() < 0.2:
                group["extra"] = _junk(rng, 2)
            groups.append(group)
        events[event] = groups
    for key in rng.sample(["Notification", "aelf-hook", "x y"], rng.randint(0, 2)):
        events[key] = _junk(rng, 3)
    return events


@pytest.mark.timeout(180)
@pytest.mark.parametrize("seed", range(4))
def test_doctor_reports_every_program_name_in_random_nestings(
    tmp_path: Path, seed: int,
) -> None:
    """Strings that name a program anywhere in hooks.json or config.toml,
    at any depth, as keys or values, under events, other keys, extra
    fields, and arrays, each get a warning that names their path. Setup's
    own commands are the only strings left out."""
    rng = random.Random(1812 + seed)
    codex_dir, hooks, _program = _installed(tmp_path)
    setup_doc = json.loads(hooks.read_text(encoding="utf-8"))
    config = codex_dir / "config.toml"
    for _case in range(10):
        doc = json.loads(json.dumps(setup_doc))
        for event, groups in _junk_events(rng).items():
            if isinstance(groups, list) and event in doc["hooks"]:
                doc["hooks"][event].extend(groups)
            else:
                doc["hooks"][event] = groups
        if rng.random() < 0.3:
            doc["description"] = rng.choice(_NAMING)
        stop = doc["hooks"]["Stop"][0]["hooks"][1]
        if rng.random() < 0.3:
            stop["extra"] = _junk(rng, 2)
        hooks.write_text(json.dumps(doc), encoding="utf-8")
        toml_hooks = _junk_events(rng)
        config.write_text(f"hooks = {_toml_value(toml_hooks)}\n", encoding="utf-8")
        own = {
            ("hooks", event, 0, "hooks", h, "command")
            for event, groups in setup_doc["hooks"].items()
            for h in range(len(groups[0]["hooks"]))
        }
        json_strings = [
            (p, t) for p, t in _named_strings(doc, ()) if p not in own
        ]
        toml_strings = _named_strings(toml_hooks, ("hooks",))
        for windows in (False, True):
            report = doctor_codex(codex_dir, windows=windows)
            assert report.duplicate_runs == []
            assert report.hooks_file_rejected is None
            assert report.config_hooks_rejected is None
            missed = _unreported(
                report.warnings, config, "json", json_strings,
            ) + _unreported(report.warnings, config, "toml", toml_strings)
            assert missed == [], (missed, report.warnings)


def _repeated_key(doc: dict[str, object], ours: str) -> str:
    """hooks.json text with a handler field written twice; json.loads keeps
    the second value."""
    text = json.dumps(doc)
    return text.replace(
        '"Stop": [{"hooks": [',
        '"Stop": [{"hooks": [{"type": "command", "command": "true", '
        f'"z": {json.dumps(ours)}, "z": 1}}, ',
        1,
    )


# A form of hooks.json that names setup's stop command `{s}` where no other
# check reads it, and the path doctor names. Codex reads a `hooks` value
# written as an array by position.
_UNREAD_FORMS: dict[
    str, tuple[Callable[[dict[str, object], str], str], Callable[[str], str]],
] = {
    "description": (
        lambda d, s: json.dumps({**d, "description": s}),
        lambda _s: "description",
    ),
    "hooks written as an array": (
        lambda _d, s: json.dumps({"hooks": [
            [], [], [], [], [], [], [],
            [{"hooks": [{"type": "command", "command": s}]}],
        ]}),
        lambda _s: "hooks[7][0].hooks[0].command",
    ),
    "object under a key that names no event": (
        lambda d, s: json.dumps({"hooks": {
            **cast_dict(d["hooks"]),
            "Bogus": {"hooks": [{"type": "command", "command": s}]},
        }}),
        lambda _s: "hooks.Bogus.hooks[0].command",
    ),
    "a key that names the program": (
        lambda d, s: json.dumps({"hooks": {**cast_dict(d["hooks"]), s: []}}),
        lambda s: f"hooks[{json.dumps(s)}]",
    ),
    "a field of setup's handler other than its command": (
        lambda d, s: json.dumps(d).replace(
            f'"command": {json.dumps(s)}',
            f'"command": {json.dumps(s)}, "note": {json.dumps(s)}',
        ),
        lambda _s: "hooks.Stop[0].hooks[1].note",
    ),
    "the first value of a repeated field": (
        _repeated_key, lambda _s: "hooks.Stop[0].hooks[0].z",
    ),
}


def cast_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_UNREAD_FORMS))
def test_doctor_names_the_path_of_a_program_name_no_other_check_reads(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, _program = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    stop = next(
        str(h["command"]) for h in doc["hooks"]["Stop"][0]["hooks"]
        if "aelf-stop-hook" in str(h["command"])
    )
    build, path_of = _UNREAD_FORMS[shape]
    hooks.write_text(build(doc, stop), encoding="utf-8")
    path = path_of(stop)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"in {hooks} names aelf-stop-hook, outside the handlers `aelf setup "
        "--host codex` writes, and no other doctor check covers it." in out
    ), out
    assert f"at `{path}` in {hooks}" in out, out


@pytest.mark.timeout(60)
def test_doctor_names_a_config_toml_handler_outside_setup_handlers(
    tmp_path: Path,
) -> None:
    """One copy of setup's command in config.toml `[hooks]` is no duplicate
    run when hooks.json holds none, and setup never writes config.toml."""
    codex_dir, hooks, _program = _installed(tmp_path)
    ours = _session_start_command(hooks)
    hooks.unlink()
    _write_config_hooks(codex_dir, "SessionStart", ours)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        "[warn] The string at `hooks.SessionStart[0].hooks[0].command` in the "
        f"[hooks] table of {codex_dir / 'config.toml'} names "
        f"{_HOOK}, outside the handlers" in out
    ), out


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
