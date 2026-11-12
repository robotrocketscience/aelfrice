"""Codex handler fields and the opt-in summary line (#1812 host parity).

Codex adds hook stdout to the model's context and never renders it, so a
Codex user can't see that aelfrice ran. This build closes three gaps:

* `aelf setup --host codex --status-message TEXT` writes the Codex
  `statusMessage` handler field instead of refusing the option (#1429).
* `additionalContextLimit` on the SessionStart and UserPromptSubmit
  handlers, so Codex doesn't cut aelfrice's block to a preview.
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
def test_context_limit_is_on_the_two_context_handlers_only(
    codex_home: Path,
) -> None:
    """Codex reads the field on SessionStart and UserPromptSubmit and warns
    on any other event, and only these two handlers write context."""
    install_codex_hooks(codex_home / "hooks.json")
    limited = sorted(
        (event, _key(h), h["additionalContextLimit"])
        for event, h in _handlers(codex_home / "hooks.json")
        if "additionalContextLimit" in h
    )
    assert limited == [
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


# Each wrapper runs the session-start program. `{p}` is the program path.
_WRAPPERS = {
    "python": "/usr/bin/python3 /x/aelf-visible.py SessionStart {p}",
    "env": "/usr/bin/env {p}",
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
    "env -S": "/usr/bin/env -S '{p} --host codex'",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_WRAPPERS))
def test_doctor_faults_setup_run_over_a_session_start_wrapper(
    tmp_path: Path, shape: str,
) -> None:
    """The state the move-off-a-wrapper steps start from.

    A user replaces setup's SessionStart handler with a wrapper, then runs
    setup again. Setup doesn't own the wrapper, so it keeps the wrapper's
    group and adds its own handler in a new group, and Codex injects the
    baseline twice.
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
    assert rc == 1, out
    assert "SessionStart runs aelf-session-start-hook 2 times" in out


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


# Each foreign handler names the program without running it, or runs none.
_NOT_RUNNING = {
    "other hook": "/usr/bin/python3 /x/other-hook.py SessionStart --verbose",
    "echo": "echo aelf-session-start-hook",
    "sh -c printf": "sh -c 'printf \"%s\\n\" {p}'",
    "echo or true": "echo {p} || true",
    "printf into a file": "printf '%s\\n' {p} > /dev/null",
    # The flag after a prefix word is skipped, so echo is the program.
    "env -i echo": "env -i echo {p}",
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
