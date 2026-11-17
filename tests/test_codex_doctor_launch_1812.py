"""`aelf doctor --host codex` scores what a plain `codex` launch starts
(#1812): the files it can read, the approvals it honors, and the profile
settings it ignores.

Each case writes the files setup writes plus one change, and checks the
exit code and the lines doctor prints.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Final, cast

import pytest

from aelfrice.cli import (
    _cmd_doctor_codex,
    _cmd_setup_codex,
    _cmd_unsetup_codex,
    main,
)
from aelfrice.host_codex import (
    _PROGRAM_NAMES,
    _names_by_line,
    _names_in,
    _run_command,
    _trust_hash,
    install_codex_hooks,
    install_codex_skills,
)
from tests.codex_approvals import approve_codex_hooks


def _doctor(
    codex_dir: Path, skills: Path, *, approve: bool = True,
) -> tuple[int, str]:
    """Doctor's exit code and output, after approving every handler as
    `/hooks` does, unless `approve` is False."""
    if approve:
        approve_codex_hooks(codex_dir)
    out = io.StringIO()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out,
        codex_dir=codex_dir, skills_dest=skills,
    )
    return rc, out.getvalue()


def _installed(tmp_path: Path) -> tuple[Path, Path, str]:
    """A Codex dir after setup, its hooks.json, and setup's SessionStart
    command."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks)
    install_codex_skills(tmp_path / "skills")
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    return codex_dir, hooks, str(doc["hooks"]["SessionStart"][0]["hooks"][0]["command"])


# --- files Codex can't read --------------------------------------------------


@pytest.mark.timeout(60)
def test_doctor_reads_a_utf16_hooks_json_as_codex_does(tmp_path: Path) -> None:
    """Codex reads hooks.json with `fs::read_to_string`, which fails on
    bytes that aren't UTF-8, and loads no hook from it. Doctor says so and
    exits 1, as it does for a hooks.json that isn't JSON."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    hooks.write_bytes(hooks.read_text(encoding="utf-8").encode("utf-16"))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        f"[warn] {hooks} is unreadable as hooks config (the file isn't "
        "UTF-8 text: byte 0 is 0xff); Codex will ignore or reject it"
    ) in out, out


#: hooks.json contents that aren't UTF-8, and the first bad byte: setup's
#: file as Windows PowerShell 5.1 saves it (UTF-16 with a byte order
#: mark), and a JSON file with one Latin-1 byte in a string.
_NOT_UTF8: Final[dict[str, tuple[bytes, int, int]]] = {
    "utf-16": ('{"hooks": {}}'.encode("utf-16"), 0, 0xFF),
    "latin-1": (b'{"hooks": {}, "description": "x\xe9"}', 31, 0xE9),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("case", sorted(_NOT_UTF8))
def test_setup_and_unsetup_treat_a_hooks_json_that_is_not_utf8_as_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """Setup and unsetup treat a hooks.json that isn't UTF-8 as one that
    isn't JSON: setup and unsetup exit 1, name the file and the first bad
    byte, and leave it as it is, and `setup --force` replaces it."""
    data, offset, byte = _NOT_UTF8[case]
    hooks = tmp_path / "codex" / "hooks.json"
    hooks.parent.mkdir()
    hooks.write_bytes(data)
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=False,
    )
    bad = f"{hooks} isn't UTF-8 text: byte {offset} is {byte:#04x}"
    assert _cmd_setup_codex(args, io.StringIO(), hooks_path=hooks) == 1
    assert _cmd_unsetup_codex(
        args, io.StringIO(), hooks_path=hooks, skills_dest=tmp_path / "s",
    ) == 1
    assert capsys.readouterr().err == (
        f"setup --host codex: existing {bad}; re-run with --force to "
        "replace it\n"
        f"unsetup --host codex: {bad}; not modified\n"
    )
    assert hooks.read_bytes() == data
    args.force = True
    assert _cmd_setup_codex(args, io.StringIO(), hooks_path=hooks) == 0
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    assert sorted(doc) == ["hooks"], doc
    assert "SessionStart" in doc["hooks"], doc


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", ["hooks.json", "config.toml"])
def test_doctor_reads_a_directory_in_place_of_a_file_as_unreadable(
    tmp_path: Path, name: str,
) -> None:
    """A hooks.json that is a directory loads no hook, and codex-cli
    0.167.1 refuses to start on a config.toml that is one. Doctor names
    the file as one Codex can't read, doesn't say it doesn't exist, and
    exits 1."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    if name == "hooks.json":
        hooks.unlink()
    (codex_dir / name).mkdir()
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "does not exist" not in out, out
    why = "Codex can't read the file: it is a directory, not a file"
    if name == "hooks.json":
        assert "[-] codex hooks.json: present=True valid=False " in out, out
        assert (
            f"[warn] {hooks} is unreadable as hooks config ({why}); Codex "
            "will ignore or reject it\n"
        ) in out, out
    else:
        assert (
            f"[warn] Codex refuses to start with {codex_dir / name}, because "
            f"{why}\n"
        ) in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_latin1_config_toml_as_codex_does(tmp_path: Path) -> None:
    """Codex reads config.toml with `String::from_utf8` and fails on bytes
    that aren't UTF-8. Doctor names the byte and exits 1, as it does for a
    config.toml Codex's TOML parser rejects."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    config = codex_dir / "config.toml"
    config.write_bytes(b'notify = ["abc\xe9"]\n')
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        f"{config}, because the file isn't UTF-8 text: byte 14 is 0xe9"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_latin1_profile_file_without_raising(
    tmp_path: Path,
) -> None:
    """A profile file that isn't UTF-8 gets a warning that names it."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    profile = codex_dir / "work.config.toml"
    profile.write_bytes(b'notify = ["abc\xe9"]\n')
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert f"the profile file {profile}" in out, out


# --- the launch doctor scores ------------------------------------------------


def _toml_copy(command: str, matcher: str = "startup") -> str:
    return (
        f'[[hooks.SessionStart]]\nmatcher = "{matcher}"\n'
        "[[hooks.SessionStart.hooks]]\n"
        f'type = "command"\ncommand = {json.dumps(command)}\n'
    )


# Profile settings that `codex --profile work` reads and a plain `codex`
# launch doesn't: (config.toml text, profile files, the warning doctor
# gives, where `{config}` and `{dir}` are filled in).
_PROFILE_ONLY: dict[str, tuple[str, dict[str, str], str]] = {
    "a profile table": (
        "[profiles.work.features]\nhooks = false\n", {},
        "[warn] [profiles.work.features] in {config} sets `hooks`, which no "
        "launch reads: a plain `codex` launch reads the `hooks` feature from "
        "[features] only, and `codex --profile work` refuses to start while "
        "config.toml has [profiles.work]\n",
    ),
    "a profile file's [features]": (
        "", {"work.config.toml": "[features]\nhooks = false\n"},
        "[warn] `codex --profile work` also reads the profile file "
        "{dir}/work.config.toml, and its [features] turns Codex's `hooks` "
        "feature off. Doctor scores a plain `codex` launch, which doesn't "
        "read that file, so this doesn't change what doctor counts\n",
    ),
    "a profile file's [hooks]": (
        "", {"work.config.toml": (
            '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\n'
            'command = "/bin/true"\n'
        )},
        "[warn] `codex --profile work` also reads the profile file "
        "{dir}/work.config.toml, and its [hooks] table adds handlers.",
    ),
    "a profile file's structured [features] table": (
        "", {"work.config.toml": "[features.code_mode]\nenabled = true\n"},
        "[warn] `codex --profile work` also reads the profile file "
        "{dir}/work.config.toml, and doctor doesn't check the value of "
        "`code_mode` in its [features], so it can't tell whether `codex "
        "--profile work` starts.",
    ),
    "a profile file's `profile` key": (
        "", {"work.config.toml": 'profile = "x"\n'},
        "[warn] `codex --profile work` also reads the profile file "
        "{dir}/work.config.toml, and its top-level `profile` key makes "
        "`codex --profile work` refuse to start.",
    ),
    "an unreadable profile file": (
        "", {"work.config.toml": "[features\n"},
        "[warn] doctor can't read the profile file {dir}/work.config.toml, "
        "which `codex --profile work` reads: could not parse",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_PROFILE_ONLY))
def test_doctor_scores_a_plain_launch_and_names_the_profile(
    tmp_path: Path, shape: str,
) -> None:
    """A plain `codex` launch reads no profile file and ignores
    `[profiles.NAME]`, so it starts both copies of setup's handler. Doctor
    fails on that launch and warns about the profile."""
    codex_dir, hooks, ours = _installed(tmp_path)
    hooks.unlink()
    settings, files, warning = _PROFILE_ONLY[shape]
    config = codex_dir / "config.toml"
    config.write_text(
        settings + _toml_copy(ours) + _toml_copy(ours, "startup|resume"),
        encoding="utf-8",
    )
    for name, text in files.items():
        (codex_dir / name).write_text(text, encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times"
    ) in out, out
    assert warning.format(config=config, dir=codex_dir) in out, out
    assert "doctor counts no duplicate run" not in out, out


def _two_in_hooks_json(hooks: Path) -> None:
    """Put a second copy of setup's SessionStart handler in its group."""
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    group = doc["hooks"]["SessionStart"][0]
    group["hooks"].append(dict(group["hooks"][0]))
    hooks.write_text(json.dumps(doc), encoding="utf-8")


# config.toml text that makes Codex refuse to start.
_REFUSED = {
    "nesting": "x = " + "[" * 81 + "]" * 81 + "\n",
    "a [hooks] table Codex rejects": '[hooks.state."k"]\nenabled = "yes"\n',
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_REFUSED))
def test_doctor_counts_no_run_in_hooks_json_when_codex_refuses_to_start(
    tmp_path: Path, shape: str,
) -> None:
    """Codex starts no hook when it refuses to start with config.toml, so
    two copies in hooks.json are no duplicate run."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _two_in_hooks_json(hooks)
    config = codex_dir / "config.toml"
    config.write_text(_REFUSED[shape], encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "times" not in out, out
    assert (
        f"[warn] doctor counts no duplicate run, because Codex refuses to "
        f"start with {config}\n"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_says_nothing_about_a_profile_that_changes_no_hook(
    tmp_path: Path,
) -> None:
    """A profile file that leaves the feature as config.toml sets it, and
    adds no hook setting, gets no line."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    (codex_dir / "config.toml").write_text(
        "[features]\nhooks = true\n", encoding="utf-8",
    )
    (codex_dir / "work.config.toml").write_text(
        'model = "x"\n[features]\ncodex_hooks = true\n', encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "work.config.toml" not in out, out


# --- approvals ---------------------------------------------------------------

_SS = "/opt/aelf/bin/aelf-session-start-hook --host codex"
# Handlers whose hash Codex's own code computed, with `hook_hash` and
# `version_for_toml` at openai/codex@69231e119bfca0b14453515e97d99688ce1c47c9
# copied into a Rust build with serde 1.0.228, serde_json 1.0.149, toml
# 0.9.11, and sha2 0.10.9: (event, group, handler, hash with `command`,
# hash with `commandWindows`).
_GOLDEN: dict[str, tuple[str, dict[str, object], dict[str, object], str, str]] = {
    "setup's SessionStart handler": (
        "SessionStart", {"matcher": "startup|resume|clear|compact|fork"},
        {"type": "command", "command": _SS, "timeout": 30,
         "statusMessage": "aelfrice", "additionalContextLimit": 9500},
        "sha256:18fc543fef9367ab5a4b414a33b6e7cb06419326010da69c200c27c84be90ebd",
        "sha256:18fc543fef9367ab5a4b414a33b6e7cb06419326010da69c200c27c84be90ebd",
    ),
    "the default timeout and context limit": (
        "SessionStart", {"matcher": "startup"},
        {"type": "command", "command": _SS, "additionalContextLimit": 2500},
        "sha256:60f4f235542c39019826f694c2bf4ad0c38f3257cad08a87cecdb0243bb69326",
        "sha256:60f4f235542c39019826f694c2bf4ad0c38f3257cad08a87cecdb0243bb69326",
    ),
    "an event whose matcher Codex ignores": (
        "Stop", {"matcher": "ignored"},
        {"type": "command", "command": "/opt/aelf/bin/aelf-stop-hook",
         "timeout": 0},
        "sha256:8d4ec4dca7fc2571332d652e5d36f21be315b87704028360827b38485367bb5b",
        "sha256:8d4ec4dca7fc2571332d652e5d36f21be315b87704028360827b38485367bb5b",
    ),
    "a SessionEnd timeout past its cap": (
        "SessionEnd", {},
        {"type": "command", "command": "/opt/aelf/bin/aelf-stop-hook",
         "timeout": 9},
        "sha256:3e0c9cf5f84ec3a036358abc7620f11d570f3cff58e9dbdc98865b40b735efc8",
        "sha256:3e0c9cf5f84ec3a036358abc7620f11d570f3cff58e9dbdc98865b40b735efc8",
    ),
    "a context limit the event drops": (
        "PostCompact", {"matcher": "auto"},
        {"type": "command", "command": "/opt/aelf/bin/aelf-transcript-logger",
         "additionalContextLimit": 7, "async": True},
        "sha256:f7b80ca55162d2df39635d72dd777568b95d04dc03c04acf4617aee250aa31b8",
        "sha256:f7b80ca55162d2df39635d72dd777568b95d04dc03c04acf4617aee250aa31b8",
    ),
    "a Windows command and a non-ASCII status": (
        "UserPromptSubmit", {},
        {"type": "command", "command": "/opt/aelf/bin/aelf-hook",
         "commandWindows": "C:\\aelf\\aelf-hook.exe", "statusMessage": "\u00e9\n"},
        "sha256:814ef22baaeaeee943c7b58692077840e039ecdbd637726167b970dfe8bbb3ba",
        "sha256:eb61f3843d384a445ea1e2bdc1338db8a23c46bb726a17cf55801082a00d11b8",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("case", sorted(_GOLDEN))
def test_doctor_hashes_a_handler_as_codex_does(case: str) -> None:
    event, group, handler, posix, windows = _GOLDEN[case]
    for win, want in ((False, posix), (True, windows)):
        command = _run_command(handler, windows=win)
        assert command is not None
        assert _trust_hash(event, group, handler, command) == (want, ""), win


def _state_entry(key: str, digest: str) -> str:
    return f"[hooks.state.{json.dumps(key)}]\ntrusted_hash = {json.dumps(digest)}\n"


def _hash_of(config: Path, g: int) -> str:
    """The hash Codex approves group `g` of config.toml's SessionStart by."""
    doc = tomllib.loads(config.read_text(encoding="utf-8"))
    group = doc["hooks"]["SessionStart"][g]
    handler = group["hooks"][0]
    digest, _why = _trust_hash("SessionStart", group, handler, handler["command"])
    assert digest is not None
    return digest


def _two_toml_copies(tmp_path: Path, extra: str = "") -> tuple[Path, Path, str]:
    """Setup's files, with hooks.json removed and two copies of setup's
    SessionStart handler in config.toml whose matchers share `startup`."""
    codex_dir, hooks, ours = _installed(tmp_path)
    hooks.unlink()
    config = codex_dir / "config.toml"
    config.write_text(
        _toml_copy(ours) + _toml_copy(ours, "startup|resume") + extra,
        encoding="utf-8",
    )
    return codex_dir, config, ours


def _approve(config: Path, *groups: int, digest: str | None = None) -> None:
    config.write_text(
        config.read_text(encoding="utf-8") + "".join(
            _state_entry(
                f"{config}:session_start:{g}:0", digest or _hash_of(config, g),
            )
            for g in groups
        ),
        encoding="utf-8",
    )


@pytest.mark.timeout(60)
def test_doctor_fails_two_approved_copies(tmp_path: Path) -> None:
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    _approve(config, 0, 1)
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times"
    ) in out, out


# Approval states that leave a copy unapproved: (groups approved, the hash
# they hold, why doctor counts no run for group 1).
_UNAPPROVED: dict[str, tuple[tuple[int, ...], str | None, str]] = {
    "no entry": (
        (0,), None,
        "no [hooks.state] entry in {config} approves it under the key Codex "
        "reads, `{config}:session_start:1:0`",
    ),
    "neither has an entry": (
        (), None,
        "no [hooks.state] entry in {config} approves it under the key Codex "
        "reads, `{config}:session_start:1:0`",
    ),
    "a stale hash": (
        (0, 1), "sha256:" + "0" * 64,
        "its [hooks.state] `trusted_hash` in {config} isn't the hash of the "
        "handler as it is now, so Codex skips it as modified",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("state", sorted(_UNAPPROVED))
def test_doctor_counts_no_run_for_a_copy_codex_has_not_approved(
    tmp_path: Path, state: str,
) -> None:
    """Codex skips a user handler whose [hooks.state] entry doesn't hold
    the hash it computes for it now, so the copy is no second run. Doctor
    says that approving both copies starts the command twice."""
    codex_dir, config, ours = _two_toml_copies(tmp_path)
    groups, digest, reason = _UNAPPROVED[state]
    for g in groups:
        _approve(config, g, digest=digest)
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert "times" not in out.replace("twice", ""), out
    assert (
        "[warn] SessionStart handler at `hooks.SessionStart[1].hooks[0]` in "
        f"the [hooks] table of {config} is a copy of the "
        "aelf-session-start-hook command `aelf setup --host codex` writes, "
        f"and doctor counts no run for it, because {reason.format(config=config)}. "
        "Codex skips a handler you haven't approved, but once both copies are "
        "approved, Codex starts the command twice on one fire; keep one copy "
        f"and delete the others: {ours}\n"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_counts_no_run_for_a_copy_it_can_not_hash(tmp_path: Path) -> None:
    """Codex reads a TOML date `statusMessage` as its text, which doctor
    doesn't render, so doctor can't compute the hash and counts no run."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    text = config.read_text(encoding="utf-8")
    config.write_text(
        text.rstrip("\n") + "\nstatusMessage = 1979-05-27\n", encoding="utf-8",
    )
    _approve(config, 0, digest=_hash_of(config, 0))
    config.write_text(
        config.read_text(encoding="utf-8")
        + _state_entry(f"{config}:session_start:1:0", "sha256:" + "1" * 64),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert (
        "and doctor counts no run for it, because doctor can't compute the "
        "hash Codex approves it by, because its `statusMessage` isn't a string."
    ) in out, out


def _resolved_doctor(skills: Path) -> tuple[int, str]:
    """Doctor's exit code and output for the Codex home it resolves."""
    out = io.StringIO()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out, skills_dest=skills,
    )
    return rc, out.getvalue()


def _two_copies_in(codex_dir: Path, skills: Path) -> Path:
    """Setup's skills, and two copies of setup's SessionStart handler in
    the config.toml of `codex_dir`, whose matchers share `startup`."""
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks)
    install_codex_skills(skills)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    ours = str(doc["hooks"]["SessionStart"][0]["hooks"][0]["command"])
    hooks.unlink()
    config = codex_dir / "config.toml"
    config.write_text(
        _toml_copy(ours) + _toml_copy(ours, "startup|resume"), encoding="utf-8",
    )
    return config


def _approve_under(config: Path, home: Path) -> None:
    """Approve both copies under keys that spell the Codex home `home`."""
    config.write_text(
        config.read_text(encoding="utf-8") + "".join(
            _state_entry(
                f"{home / 'config.toml'}:session_start:{g}:0",
                _hash_of(config, g),
            )
            for g in (0, 1)
        ),
        encoding="utf-8",
    )


def _check_approvals(rc: int, out: str, counted: bool, home: Path) -> None:
    """Both copies count when `counted`, and otherwise neither does, and
    doctor names the key under `home` that Codex reads."""
    if counted:
        assert rc == 1, out
        assert (
            "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
            "times"
        ) in out, out
        assert "haven't approved" not in out, out
    else:
        assert rc == 0, out
        assert "times" not in out.replace("twice", ""), out
        assert (
            "approves it under the key Codex reads, "
            f"`{home / 'config.toml'}:session_start:0:0`"
        ) in out, out


def _symlinked_dot_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """`~/.codex` as a symlink, as a dotfile manager writes it, with
    `CODEX_HOME` unset: the home as written, and canonical."""
    real = tmp_path / "dotfiles" / "codex"
    real.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    written = home / ".codex"
    written.symlink_to(real, target_is_directory=True)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    return written, real


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("spelling", "counted"), [("as written", True), ("canonical", False)],
)
def test_doctor_reads_approvals_through_a_symlinked_home_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str,
    counted: bool,
) -> None:
    """With `CODEX_HOME` unset, Codex keys approvals by `~/.codex` as
    written, so through a symlinked `~/.codex` an entry under the
    canonical path approves nothing."""
    written, real = _symlinked_dot_codex(tmp_path, monkeypatch)
    config = _two_copies_in(written, tmp_path / "skills")
    _approve_under(config, written if spelling == "as written" else real)
    rc, out = _resolved_doctor(tmp_path / "skills")
    _check_approvals(rc, out, counted, written)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("spelling", "counted"), [("as written", True), ("canonical", False)],
)
def test_doctor_reads_an_empty_codex_home_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str,
    counted: bool,
) -> None:
    """Codex ignores an empty `CODEX_HOME`, so through a symlinked
    `~/.codex` it keys approvals by `~/.codex` as written."""
    written, real = _symlinked_dot_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("CODEX_HOME", "")
    config = _two_copies_in(written, tmp_path / "skills")
    _approve_under(config, written if spelling == "as written" else real)
    rc, out = _resolved_doctor(tmp_path / "skills")
    _check_approvals(rc, out, counted, written)


#: `CODEX_HOME` values that start with `~`. `Path.expanduser` raises on the
#: second, which names no user.
_TILDE_HOMES: Final[list[str]] = ["~/c", "~nosuchuser1812/c"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("raw", _TILDE_HOMES)
def test_doctor_reads_a_codex_home_that_starts_with_a_tilde_as_codex_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    """Codex doesn't expand `~` in `CODEX_HOME`: it reads the value as a
    path under the folder you start it in. So doctor reads that folder,
    and counts the two approved copies there, not the ones under `$HOME`."""
    home = tmp_path / "home"
    (home / "c").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _two_copies_in(home / "c", tmp_path / "skills")
    work = tmp_path / "work"
    literal = work / raw
    literal.mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("CODEX_HOME", raw)
    config = _two_copies_in(literal, tmp_path / "skills")
    _approve_under(config, Path(os.path.realpath(literal)))
    rc, out = _resolved_doctor(tmp_path / "skills")
    assert f"[i] codex home: {literal}\n" in out, out
    _check_approvals(rc, out, True, literal)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("raw", _TILDE_HOMES)
def test_doctor_reports_a_missing_codex_home_that_starts_with_a_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], raw: str,
) -> None:
    """Where no folder by that name is under the folder you start Codex
    in, Codex refuses to start, so doctor names that path and exits 1,
    even when `$HOME` holds the folder `~` would expand to."""
    home = tmp_path / "home"
    (home / "c").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", raw)
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), io.StringIO(),
        skills_dest=tmp_path / "skills",
    )
    assert rc == 1
    assert (
        f"doctor --host codex: $CODEX_HOME points to {tmp_path / raw}, but "
        "that path does not exist"
    ) in capsys.readouterr().err


@pytest.mark.timeout(60)
@pytest.mark.parametrize("raw", _TILDE_HOMES)
@pytest.mark.parametrize("exists", [True, False], ids=["folder", "no folder"])
def test_setup_refuses_a_codex_home_that_starts_with_a_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], raw: str, exists: bool,
) -> None:
    """Setup exits 1 and writes nothing for a `CODEX_HOME` that starts with
    `~`, which Codex doesn't expand, and asks for the expanded path."""
    home = tmp_path / "home"
    (home / "c").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    if exists:
        (tmp_path / raw).mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", raw)
    before = sorted(p for p in tmp_path.rglob("*"))
    rc = main(["setup", "--host", "codex", "--no-codex-skills"])
    assert rc == 1
    assert (
        f"setup --host codex: $CODEX_HOME is {raw!r}, which starts with `~`. "
        "Codex doesn't expand `~` in it: it reads the value as a path under "
        "the folder you start Codex in. Set CODEX_HOME to the expanded path, "
        "and run setup again\n"
    ) in capsys.readouterr().err
    assert sorted(p for p in tmp_path.rglob("*")) == before


@pytest.mark.timeout(60)
@pytest.mark.parametrize("raw", _TILDE_HOMES)
def test_unsetup_reads_a_codex_home_that_starts_with_a_tilde_as_codex_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], raw: str,
) -> None:
    """Unsetup reads a `CODEX_HOME` that starts with `~` as Codex and
    doctor do, as a path under the folder you run it in. It removes setup's
    handlers there and leaves the folder that `~` expands to alone."""
    home = tmp_path / "home"
    expanded = home / "c"
    expanded.mkdir(parents=True)
    install_codex_hooks(expanded / "hooks.json")
    untouched = (expanded / "hooks.json").read_bytes()
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    literal = work / raw
    literal.mkdir(parents=True)
    install_codex_hooks(literal / "hooks.json")
    monkeypatch.chdir(work)
    monkeypatch.setenv("CODEX_HOME", raw)
    rc = main(["unsetup", "--host", "codex"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (
        f"removed aelfrice entries from {literal / 'hooks.json'} "
    ) in captured.out, captured.out
    assert (expanded / "hooks.json").read_bytes() == untouched


# --- the home folder Codex starts from ---------------------------------------


class _Passwd:
    """The one field of a password database entry that Codex reads."""

    def __init__(self, pw_dir: str) -> None:
        self.pw_dir = pw_dir


def _pin_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: str | None,
) -> Path:
    """Set `HOME` to `home`, or unset it for None, with every way Python
    finds a home folder pinned under `tmp_path`, so no case can reach the
    real home: the password database entry for your user names
    `tmp_path / "passwd"`, it knows no user by name, and `Path.home` and
    `os.path.expanduser` read only those. Returns that folder."""
    import pwd

    passwd = tmp_path / "passwd"
    passwd.mkdir(exist_ok=True)
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _Passwd(str(passwd)))

    def no_user(name: str) -> object:
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", no_user)
    expanduser = os.path.expanduser

    def pinned_expanduser(path: str) -> str:
        expanded = expanduser(path)
        assert not os.path.isabs(expanded) or _inside(Path(expanded), tmp_path) or (
            expanded == "/"
        ), expanded
        return expanded

    def pinned_home(cls: type[Path]) -> Path:
        # `Path.home` as Python 3.12 and 3.13 write it.
        expanded = os.path.expanduser("~")
        if expanded[:1] == "~":
            raise RuntimeError("Could not determine home directory.")
        return cls(expanded)

    monkeypatch.setattr(os.path, "expanduser", pinned_expanduser)
    monkeypatch.setattr(Path, "home", classmethod(pinned_home))
    if home is None:
        monkeypatch.delenv("HOME", raising=False)
    else:
        monkeypatch.setenv("HOME", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return passwd


def _inside(path: Path, root: Path) -> bool:
    """Whether `path` is under `root`, with both made canonical."""
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(root))


#: `HOME` values, None for unset, and where Codex reads its home for each,
#: under the folder you start it in or the password database's home folder.
#: `~/h` reads `~/h/h/.codex`: Codex expands the `~` of `~/h/.codex` to
#: `$HOME` itself (`AbsolutePathBuf::maybe_expand_home_directory`).
_HOME_CASES: Final[dict[str, tuple[str | None, str]]] = {
    "unset": (None, "passwd/.codex"),
    "empty": ("", "passwd/.codex"),
    "tilde": ("~", "work/~/.codex"),
    "tilde slash": ("~/h", "work/~/h/h/.codex"),
    "tilde user": ("~nosuchuser1812", "work/~nosuchuser1812/.codex"),
    "relative": ("rel", "work/rel/.codex"),
    "dot dot": ("rel/../h", "work/h/.codex"),
}


def _home_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> tuple[Path, Path]:
    """Pin `HOME` for `case`, run from `tmp_path / "work"`, and return the
    Codex home Codex reads, and the password database's home folder."""
    home, codex_rel = _HOME_CASES[case]
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    passwd = _pin_home(tmp_path, monkeypatch, home)
    codex_dir = tmp_path / codex_rel
    from aelfrice.host_codex import resolve_codex_home

    resolved = resolve_codex_home()
    assert resolved == codex_dir and _inside(resolved, tmp_path), resolved
    return codex_dir, passwd


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
@pytest.mark.parametrize("case", sorted(_HOME_CASES))
def test_doctor_reads_the_codex_home_codex_reads_for_any_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """With `CODEX_HOME` unset, doctor reads the Codex home Codex reads for
    any `HOME`: the password database's home folder for an empty or unset
    `HOME`, and a folder under the one you run it in for a relative one or
    one that starts with `~`. Doctor doesn't raise, and reads no skills
    folder where Codex reads none."""
    codex_dir, passwd = _home_case(tmp_path, monkeypatch, case)
    codex_dir.mkdir(parents=True)
    install_codex_hooks(codex_dir / "hooks.json")
    out = io.StringIO()
    rc = _cmd_doctor_codex(argparse.Namespace(host="codex"), out)
    text = out.getvalue()
    assert rc in (0, 1), text
    assert f"[i] codex home: {codex_dir}\n" in text, text
    assert "present=True valid=True" in text, text
    if case in ("unset", "empty"):
        skills = passwd / ".agents" / "skills"
        assert (
            "[i] codex skills ($aelf-*): 0 installed (0 orphaned) in "
            f"{skills}\n"
        ) in text, text
    else:
        assert (
            "[i] codex skills ($aelf-*): none read: $HOME is "
            f"{_HOME_CASES[case][0]!r}, which isn't an absolute path, so "
            "Codex reads no user skills folder under it\n"
        ) in text, text


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
@pytest.mark.parametrize(
    "case", sorted(set(_HOME_CASES) - {"unset"}),
)
def test_setup_refuses_a_home_that_is_empty_or_not_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """With `CODEX_HOME` unset, setup exits 1 for an empty `HOME`, a
    relative one, or one that starts with `~`, asks for an absolute `HOME`
    or a `CODEX_HOME`, and writes nothing."""
    codex_dir, _passwd = _home_case(tmp_path, monkeypatch, case)
    home = _HOME_CASES[case][0]
    before = sorted(tmp_path.rglob("*"))
    rc = main(["setup", "--host", "codex"])
    err = capsys.readouterr().err
    assert rc == 1, err
    what = (
        "$HOME is empty, which Codex reads as unset, so Codex reads its home "
        f"under {tmp_path / 'passwd'}, the home folder the password database "
        "names for your user"
        if home == ""
        else f"$HOME is {home!r}, which isn't an absolute path, so Codex "
        "reads its home as a path under the folder you start Codex in"
    )
    assert err == (
        f"setup --host codex: {what}, and $CODEX_HOME is unset. Set HOME to "
        "an absolute path, or set CODEX_HOME to the folder Codex reads, and "
        "run setup again\n"
    ), err
    assert sorted(tmp_path.rglob("*")) == before
    assert not codex_dir.exists()


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
def test_setup_installs_under_the_password_database_home_for_an_unset_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With `HOME` and `CODEX_HOME` unset, setup writes hooks.json and the
    skills under the home folder the password database names, as Codex
    reads them."""
    codex_dir, passwd = _home_case(tmp_path, monkeypatch, "unset")
    codex_dir.mkdir(parents=True)
    rc = main(["setup", "--host", "codex"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (codex_dir / "hooks.json").is_file()
    assert (passwd / ".agents" / "skills" / "aelf-onboard" / "SKILL.md").is_file()


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
@pytest.mark.parametrize("skills", [True, False], ids=["skills", "no skills"])
def test_setup_refuses_skills_under_a_home_codex_reads_none_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], skills: bool,
) -> None:
    """With `CODEX_HOME` set and `HOME` relative, Codex reads no user
    skills folder. Setup exits 1 and writes nothing unless you pass
    `--no-codex-skills`, and then writes hooks.json only."""
    _codex_dir, _passwd = _home_case(tmp_path, monkeypatch, "tilde user")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    before = sorted(tmp_path.rglob("*"))
    rc = main(
        ["setup", "--host", "codex"] + ([] if skills else ["--no-codex-skills"]),
    )
    captured = capsys.readouterr()
    if skills:
        assert rc == 1, captured.err
        assert captured.err == (
            "setup --host codex: $HOME is '~nosuchuser1812', which isn't an "
            "absolute path, so Codex reads no user skills folder under it; "
            "nothing was written. Set HOME to an absolute path, or pass "
            "--no-codex-skills\n"
        ), captured.err
        assert sorted(tmp_path.rglob("*")) == before
    else:
        assert rc == 0, captured.err
        assert sorted(p.name for p in codex_dir.iterdir()) == [
            "hooks.json", "hooks.json.lock",
        ]


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
@pytest.mark.parametrize("case", sorted(_HOME_CASES))
def test_unsetup_reads_the_codex_home_doctor_reads_for_any_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """Unsetup removes setup's handlers from the Codex home doctor reads,
    removes skills only from a skills folder Codex reads, and changes no
    other file under `tmp_path`."""
    codex_dir, passwd = _home_case(tmp_path, monkeypatch, case)
    codex_dir.mkdir(parents=True)
    install_codex_hooks(codex_dir / "hooks.json")
    skills = passwd / ".agents" / "skills"
    install_codex_skills(skills)
    # A copy where Python's `Path.home` reads `~` for a relative `HOME`.
    decoy = tmp_path / "work" / "~" / ".agents" / "skills"
    install_codex_skills(decoy)

    def files() -> dict[str, bytes]:
        return {
            str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*"))
            if p.is_file() and not _inside(p, codex_dir)
            and (case not in ("unset", "empty") or not _inside(p, skills))
        }

    before = files()
    rc = main(["unsetup", "--host", "codex"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert (
        f"removed aelfrice entries from {codex_dir / 'hooks.json'} "
    ) in captured.out, captured.out
    if case in ("unset", "empty"):
        assert f"Codex skill(s) from {skills}: " in captured.out, captured.out
        assert not (skills / "aelf-onboard").exists()
    else:
        assert captured.out.endswith(
            "no aelfrice Codex skills to remove: $HOME is "
            f"{_HOME_CASES[case][0]!r}, which isn't an absolute path, so "
            "Codex reads no user skills folder under it\n",
        ), captured.out
    assert files() == before


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="Codex reads HOME on POSIX")
def test_doctor_setup_and_unsetup_exit_1_with_no_home_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With `HOME` and `CODEX_HOME` unset and no password database entry
    for your user, Codex can't find its home and refuses to start, so
    doctor, setup and unsetup exit 1 and say so, and write nothing."""
    import pwd

    _codex_dir, _passwd = _home_case(tmp_path, monkeypatch, "unset")

    def no_entry(uid: int) -> object:
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", no_entry)
    before = sorted(tmp_path.rglob("*"))
    for verb in ("doctor", "setup", "unsetup"):
        assert main([verb, "--host", "codex"]) == 1, verb
        assert capsys.readouterr().err.startswith(
            f"{verb} --host codex: $HOME is unset or empty, and the password "
            "database names no home folder for your user, so Codex can't "
            "find its home and refuses to start.",
        ), verb
    assert sorted(tmp_path.rglob("*")) == before


def _strict_utf8() -> io.TextIOWrapper:
    """A text stream that raises on any character UTF-8 can't encode, as
    a Linux stdout under a UTF-8 locale does."""
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")


def _written(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return cast(io.BytesIO, stream.buffer).getvalue().decode("utf-8")


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    os.name == "nt", reason="Windows paths are text, not bytes",
)
@pytest.mark.parametrize("given", ["argument", "missing", "existing"])
def test_doctor_prints_a_codex_home_that_is_not_utf8_on_a_strict_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: str,
) -> None:
    """A path can hold a byte that isn't UTF-8, which Python reads as a
    lone surrogate (`os.fsdecode`), and a strict UTF-8 stream can't encode
    that. Doctor writes the byte as an escape instead of raising."""
    raw = os.fsdecode(os.fsencode(str(tmp_path)) + b"/c\xff")
    shown = f"{tmp_path}/c\\udcff"
    out, err = _strict_utf8(), _strict_utf8()
    monkeypatch.setattr(sys, "stderr", err)
    codex_dir: Path | None = Path(raw)
    if given != "argument":
        if given == "existing":
            try:
                os.mkdir(os.fsencode(raw))
            except OSError:
                pytest.skip("this file system stores only UTF-8 names")
        monkeypatch.setenv("CODEX_HOME", raw)
        codex_dir = None
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out,
        codex_dir=codex_dir, skills_dest=tmp_path / "skills",
    )
    if given == "missing":
        assert rc == 1
        assert (
            f"doctor --host codex: $CODEX_HOME points to {shown}, but that "
            "path does not exist"
        ) in _written(err), _written(err)
    else:
        assert f"[i] codex home: {shown}\n" in _written(out), _written(out)


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    os.name == "nt", reason="Windows paths are text, not bytes",
)
def test_setup_and_unsetup_print_paths_that_are_not_utf8_on_strict_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup and unsetup print each path through the escape doctor uses,
    on stdout and stderr. The skills folder and the second hooks.json need
    not exist, so this runs where a name must be UTF-8, such as macOS APFS,
    where setup's skills install fails and prints the path to stderr."""
    raw = os.fsencode(str(tmp_path))
    skills = Path(os.fsdecode(raw + b"/s\xff"))
    missing = Path(os.fsdecode(raw + b"/c\xff/hooks.json"))
    hooks = tmp_path / "codex" / "hooks.json"
    hooks.parent.mkdir()
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=True,
    )
    streams: dict[str, tuple[io.TextIOWrapper, io.TextIOWrapper]] = {}
    for verb, call, path in (
        ("setup", _cmd_setup_codex, hooks),
        ("unsetup", _cmd_unsetup_codex, missing),
    ):
        out, err = _strict_utf8(), _strict_utf8()
        monkeypatch.setattr(sys, "stderr", err)
        call(args, out, hooks_path=path, skills_dest=skills)
        streams[verb] = (out, err)
    setup = _written(streams["setup"][0]) + _written(streams["setup"][1])
    assert f"{tmp_path}/s\\udcff" in setup, setup
    unsetup = _written(streams["unsetup"][0])
    assert unsetup == (
        f"no aelfrice entries in {tmp_path}/c\\udcff/hooks.json\n"
        f"no aelfrice Codex skills in {tmp_path}/s\\udcff\n"
    ), unsetup + _written(streams["unsetup"][1])


@pytest.mark.timeout(60)
def test_setup_and_unsetup_print_errors_that_are_not_utf8_on_a_strict_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error message that names a path with a byte that isn't UTF-8
    prints as an escape on stderr. The error is stubbed, so the case runs
    on every system."""
    import aelfrice.host_codex as host_codex
    from aelfrice.host_codex import CodexInstallResult

    bad = Path(f"{tmp_path}/c\udcff/hooks.json")
    failed = CodexInstallResult(path=bad, changed=False, error=f"x {bad}")
    monkeypatch.setattr(host_codex, "install_codex_hooks", lambda *a, **k: failed)
    monkeypatch.setattr(host_codex, "remove_codex_hooks", lambda *a, **k: failed)
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=True,
    )
    shown = f"x {tmp_path}/c\\udcff/hooks.json\n"
    for verb, call in (
        ("setup", _cmd_setup_codex), ("unsetup", _cmd_unsetup_codex),
    ):
        err = _strict_utf8()
        monkeypatch.setattr(sys, "stderr", err)
        rc = call(args, io.StringIO(), hooks_path=bad, skills_dest=tmp_path)
        assert rc == 1
        assert _written(err) == f"{verb} --host codex: {shown}", _written(err)


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    os.name == "nt", reason="Windows paths are text, not bytes",
)
def test_setup_and_unsetup_print_a_codex_home_that_is_not_utf8_on_a_strict_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `CODEX_HOME` that names a missing folder with a byte that isn't
    UTF-8 prints as an escape in setup's and unsetup's refusal. The folder
    need not exist, so this runs on every POSIX file system."""
    raw = os.fsdecode(os.fsencode(str(tmp_path)) + b"/c\xff")
    monkeypatch.setenv("CODEX_HOME", raw)
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=False,
    )
    for verb, call in (
        ("setup", _cmd_setup_codex), ("unsetup", _cmd_unsetup_codex),
    ):
        err = _strict_utf8()
        monkeypatch.setattr(sys, "stderr", err)
        rc = call(args, io.StringIO(), skills_dest=tmp_path / "s")
        assert rc == 1
        assert _written(err).startswith(
            f"{verb} --host codex: $CODEX_HOME points to {tmp_path}/c\\udcff, "
            "but that path does not exist",
        ), _written(err)


@pytest.mark.timeout(60)
def test_setup_and_unsetup_print_skills_lines_that_are_not_utf8_on_strict_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup's up-to-date line and unsetup's failed-removal line name a
    skills folder that can hold a byte that isn't UTF-8, as on Linux ext4.
    The skills calls are stubbed to report such a folder, so the case runs
    on every system."""
    import aelfrice.host_codex as host_codex
    from aelfrice.host_codex import CodexSkillsResult

    bad = Path(f"{tmp_path}/s\udcff")
    monkeypatch.setattr(
        host_codex, "install_codex_skills",
        lambda *_a, **_k: CodexSkillsResult(dest_dir=bad),
    )

    def fail(*_a: object, **_k: object) -> CodexSkillsResult:
        raise OSError(f"can't remove {bad}")

    monkeypatch.setattr(host_codex, "remove_codex_skills", fail)
    hooks = tmp_path / "codex" / "hooks.json"
    hooks.parent.mkdir()
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=True,
    )
    out, err = _strict_utf8(), _strict_utf8()
    monkeypatch.setattr(sys, "stderr", err)
    assert _cmd_setup_codex(args, out, hooks_path=hooks, skills_dest=bad) == 0
    assert (
        f"Codex skills already up to date in {tmp_path}/s\\udcff\n"
    ) in _written(out), _written(out)
    err = _strict_utf8()
    monkeypatch.setattr(sys, "stderr", err)
    rc = _cmd_unsetup_codex(
        args, io.StringIO(), hooks_path=hooks, skills_dest=bad,
    )
    assert rc == 1
    assert _written(err) == (
        "unsetup --host codex: skills removal failed: can't remove "
        f"{tmp_path}/s\\udcff\n"
    ), _written(err)


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    os.name == "nt", reason="Windows paths are text, not bytes",
)
def test_doctor_prints_a_skills_folder_that_is_not_utf8_on_a_strict_stream(
    tmp_path: Path,
) -> None:
    """The skills line names the skills folder, which can hold a byte that
    isn't UTF-8. The folder need not exist, so this runs on a file system
    that stores only UTF-8 names, such as macOS APFS."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    skills = Path(os.fsdecode(os.fsencode(str(tmp_path)) + b"/s\xff"))
    out = _strict_utf8()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out,
        codex_dir=codex_dir, skills_dest=skills,
    )
    assert rc in (0, 1), _written(out)
    assert (
        "[i] codex skills ($aelf-*): 0 installed (0 orphaned) in "
        f"{tmp_path}/s\\udcff\n"
    ) in _written(out), _written(out)


@pytest.mark.timeout(60)
def test_doctor_prints_a_fail_line_that_is_not_utf8_on_a_strict_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `[FAIL]` line can name a skill folder that holds a byte that isn't
    UTF-8, as on Linux ext4. The scan is stubbed to report such a name, so
    the case runs on every system."""
    import aelfrice.host_codex as host_codex

    codex_dir, _hooks, _ours = _installed(tmp_path)
    approve_codex_hooks(codex_dir)
    monkeypatch.setattr(
        host_codex, "modified_codex_skills", lambda _dir: ["aelf-x\udcff"],
    )
    out = _strict_utf8()
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), out,
        codex_dir=codex_dir, skills_dest=tmp_path / "skills",
    )
    assert rc == 1, _written(out)
    assert (
        "[FAIL] installed $aelf-* skill(s) do not match what this build "
        "generates (edited, or left stale by an upgrade): aelf-x\\udcff "
    ) in _written(out), _written(out)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("spelling", "counted"), [("made absolute", True), ("as written", False)],
)
def test_doctor_reads_approvals_under_home_with_dot_dot_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str,
    counted: bool,
) -> None:
    """With `CODEX_HOME` unset, Codex removes `.` and `..` from
    `$HOME/.codex` before it keys approvals by it."""
    (tmp_path / "home" / ".codex").mkdir(parents=True)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home" / ".." / "home"))
    written = tmp_path / "home" / ".." / "home" / ".codex"
    config = _two_copies_in(written, tmp_path / "skills")
    absolute = tmp_path / "home" / ".codex"
    _approve_under(config, absolute if spelling == "made absolute" else written)
    rc, out = _resolved_doctor(tmp_path / "skills")
    _check_approvals(rc, out, counted, absolute)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("where", "spelling", "counted"),
    [
        ("a symlink", "canonical", True),
        ("a symlink", "as written", False),
        ("/tmp", "canonical", True),
        ("/tmp", "as written", False),
    ],
)
def test_doctor_reads_approvals_under_codex_home_as_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str,
    spelling: str, counted: bool,
) -> None:
    """With `CODEX_HOME` set, Codex canonicalizes it, so for a
    `CODEX_HOME` through a symlink, or under `/tmp`, which macOS links to
    `/private/tmp`, it keys approvals by the canonical path."""
    if where == "/tmp" and sys.platform != "darwin":
        pytest.skip("only macOS links /tmp to /private/tmp")
    if where == "/tmp":
        written = Path(tempfile.mkdtemp(dir="/tmp"))
    else:
        (tmp_path / "real").mkdir()
        written = tmp_path / "link"
        written.symlink_to(tmp_path / "real", target_is_directory=True)
    try:
        real = Path(os.path.realpath(written))
        assert real != written
        monkeypatch.setenv("CODEX_HOME", str(written))
        config = _two_copies_in(written, tmp_path / "skills")
        _approve_under(config, real if spelling == "canonical" else written)
        rc, out = _resolved_doctor(tmp_path / "skills")
        _check_approvals(rc, out, counted, real)
    finally:
        if where == "/tmp":
            shutil.rmtree(written)


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS realpath(3)")
@pytest.mark.parametrize(
    ("spelling", "counted"), [("canonical", True), ("as written", False)],
)
def test_doctor_reads_approvals_under_codex_home_in_the_case_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str,
    counted: bool,
) -> None:
    """Codex canonicalizes `CODEX_HOME` with realpath(3), which on macOS
    corrects its letter case to the folder's, so for a `CODEX_HOME` in
    another case it keys approvals by the case on disk."""
    (tmp_path / "real").mkdir()
    written = tmp_path / "REAL"
    if not written.is_dir():
        pytest.skip("this volume tells letter case apart")
    real = Path(os.path.realpath(tmp_path / "real"))
    monkeypatch.setenv("CODEX_HOME", str(written))
    config = _two_copies_in(written, tmp_path / "skills")
    _approve_under(
        config,
        real if spelling == "canonical"
        else Path(os.path.realpath(tmp_path)) / "REAL",
    )
    rc, out = _resolved_doctor(tmp_path / "skills")
    _check_approvals(rc, out, counted, real)
    assert "can't canonicalize" not in out, out


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS realpath(3)")
def test_doctor_says_when_it_can_not_call_realpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the realpath(3) call fails, doctor keys approvals as
    `os.path.realpath` spells the home and says the letter case can
    differ from Codex's key."""
    import aelfrice.host_codex as host_codex

    def fail(path: str) -> str:
        raise OSError(5, "Input/output error", path)

    monkeypatch.setattr(host_codex, "_darwin_realpath", fail)
    (tmp_path / "real").mkdir()
    real = Path(os.path.realpath(tmp_path / "real"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "real"))
    config = _two_copies_in(tmp_path / "real", tmp_path / "skills")
    _approve_under(config, real)
    rc, out = _resolved_doctor(tmp_path / "skills")
    _check_approvals(rc, out, True, real)
    assert (
        f"[warn] doctor can't canonicalize {tmp_path / 'real'} with "
        "realpath(3), as Codex does ([Errno 5] Input/output error: "
        f"'{tmp_path / 'real'}'), so it reads [hooks.state] keys under {real}, "
        "which can differ in letter case from the key Codex reads\n"
    ) in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("spelling", "disabled"), [("as written", True), ("canonical", False)],
)
def test_doctor_reads_enabled_through_a_symlinked_home_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str,
    disabled: bool,
) -> None:
    """`enabled = false` names a handler by the key an approval does, so
    through a symlinked `~/.codex` with `CODEX_HOME` unset, an entry under
    the canonical path disables nothing."""
    written, real = _symlinked_dot_codex(tmp_path, monkeypatch)
    config = _two_copies_in(written, tmp_path / "skills")
    _approve_under(config, written)
    home = written if spelling == "as written" else real
    # A leading space, which Codex trims, keeps the approval's key apart.
    key = f" {home / 'config.toml'}:session_start:1:0"
    config.write_text(
        config.read_text(encoding="utf-8")
        + f"[hooks.state.{json.dumps(key)}]\nenabled = false\n",
        encoding="utf-8",
    )
    rc, out = _resolved_doctor(tmp_path / "skills")
    assert rc == (0 if disabled else 1), out
    assert ("sets `enabled = false` for it" in out) is disabled, out


@pytest.mark.timeout(60)
def test_doctor_reads_the_last_enabled_of_a_key(tmp_path: Path) -> None:
    """Codex lets a later entry's `enabled` win, so `enabled = true` under
    one spaced spelling of the key, after `enabled = false` under another,
    leaves the copy enabled."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    _approve(config, 0, 1)
    key = f"{config}:session_start:1:0"
    config.write_text(
        config.read_text(encoding="utf-8")
        + f"[hooks.state.{json.dumps(chr(9) + key)}]\nenabled = false\n"
        + f"[hooks.state.{json.dumps(' ' + key)}]\nenabled = true\n",
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 1, out
    assert "starts the aelf-session-start-hook command 2 times" in out, out


@pytest.mark.timeout(60)
def test_doctor_keeps_a_trusted_hash_key_codex_keeps_whole(
    tmp_path: Path,
) -> None:
    """Rust's `str::trim` keeps U+001C to U+001F, which Python's
    `str.strip()` removes, so an approval under a key that ends in U+001F
    approves no handler, and neither copy counts a run."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8") + "".join(
            f'[hooks.state."{config}:session_start:{g}:0\\u001f"]\n'
            f"trusted_hash = {json.dumps(_hash_of(config, g))}\n"
            for g in (0, 1)
        ),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert "times" not in out.replace("twice", ""), out
    assert (
        f"approves it under the key Codex reads, `{config}:session_start:1:0`"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_the_last_trusted_hash_of_a_key(tmp_path: Path) -> None:
    """Codex trims each key and lets a later entry's `trusted_hash` win, so
    a stale hash under a spaced spelling of the key, after the right one,
    leaves the copy unapproved."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    _approve(config, 0, 1)
    config.write_text(
        config.read_text(encoding="utf-8")
        + _state_entry(f" {config}:session_start:1:0\t", "sha256:" + "2" * 64),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert "isn't the hash of the handler as it is now" in out, out


@pytest.mark.timeout(60)
def test_doctor_keeps_a_state_key_codex_keeps_whole(tmp_path: Path) -> None:
    """Rust's `str::trim` removes Unicode White_Space, which leaves out
    U+001C to U+001F that Python's `str.strip()` removes. A `[hooks.state]`
    key that ends in U+001F matches no handler, so its `enabled = false`
    disables nothing and both approved copies run."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    _approve(config, 0, 1)
    config.write_text(
        config.read_text(encoding="utf-8")
        + f'[hooks.state."{config}:session_start:1:0\\u001f"]\nenabled = false\n',
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 1, out
    assert "starts the aelf-session-start-hook command 2 times" in out, out


# --- a config.toml tomllib can't parse ---------------------------------------


@pytest.mark.timeout(60)
def test_doctor_names_programs_in_a_config_toml_tomllib_rejects(
    tmp_path: Path,
) -> None:
    """Codex's toml reads TOML 1.1, such as a newline inside an inline
    table, which `tomllib` rejects. Doctor can't read the file's approvals,
    so it counts no run anywhere, and it names each line that names a
    program, the way it reads a string."""
    codex_dir, hooks, ours = _installed(tmp_path)
    hooks.unlink()
    config = codex_dir / "config.toml"
    config.write_text(
        '[mcp_servers.mem]\ncommand = "aelf-hook"\nenv = {\n  A = "1"\n}\n'
        'notify = ["Aelf-Stop-Ho\\\nok"]\n'
        + _toml_copy(ours) + _toml_copy(ours, "startup|resume"),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert "times" not in out, out
    for number, key, line in (
        (2, "aelf-hook", 'command = "aelf-hook"'),
        (6, "aelf-stop-hook", 'notify = ["Aelf-Stop-Ho\\'),
        (12, "aelf-session-start-hook", f"command = {json.dumps(ours)}"),
        (17, "aelf-session-start-hook", f"command = {json.dumps(ours)}"),
    ):
        assert (
            f"[warn] Line {number} of {config} names {key}, and doctor counts "
            "no run from that file, because it can't parse it. Check what "
            f"Codex runs from this line: {line}\n"
        ) in out, out
    assert (
        f"[warn] doctor counts no duplicate run, because doctor can't parse "
        f"{config}, which holds the [hooks] table and the [hooks.state] "
        "approvals Codex reads\n"
    ) in out, out
    assert "no aelfrice hooks are installed" not in out, out
    assert (
        f"{hooks} does not exist; doctor can't parse {config}, so it can't "
        "tell whether aelfrice hooks are installed"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_counts_no_run_in_hooks_json_beside_a_config_toml_it_can_not_parse(
    tmp_path: Path,
) -> None:
    """The approvals of hooks.json handlers live in config.toml, so doctor
    can't count them either."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _two_in_hooks_json(hooks)
    approve_codex_hooks(codex_dir)
    config = codex_dir / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + "x = { a = 1, }\n", encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 0, out
    assert "times" not in out, out
    assert (
        "[warn] doctor counts no duplicate run, because doctor can't parse "
        f"{config}, which holds the [hooks] table and the [hooks.state] "
        "approvals Codex reads\n"
    ) in out, out
    # Codex may still load both copies, so each one gets a line.
    for h in (0, 1):
        assert (
            f"[warn] SessionStart handler at `hooks.SessionStart[0].hooks[{h}]` "
            "is a copy of the aelf-session-start-hook command `aelf setup "
            "--host codex` writes, and doctor counts no run for it, because "
            f"doctor can't parse {config}, which holds the [hooks] table and "
            "the [hooks.state] approvals Codex reads. If Codex starts its "
            "hooks, a fire that runs all 2 copies starts the command once for "
            "each; keep one copy and delete the others: "
        ) in out, out


@pytest.mark.timeout(60)
def test_doctor_names_programs_in_a_profile_file_tomllib_rejects(
    tmp_path: Path,
) -> None:
    codex_dir, _hooks, _ours = _installed(tmp_path)
    profile = codex_dir / "ci.config.toml"
    profile.write_text('x = { a = 1, }\nnotify = ["aelf-hook"]\n', encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] Line 2 of the profile file {profile} names aelf-hook, and "
        "doctor can't parse that file, which `codex --profile ci` reads. "
        'Check what Codex runs from this line: notify = ["aelf-hook"]\n'
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_does_not_claim_no_hooks_when_config_toml_names_one(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, _ours = _installed(tmp_path)
    hooks.unlink()
    config = codex_dir / "config.toml"
    config.write_text('notify = ["aelf-transcript-logger"]\n', encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "no aelfrice hooks are installed" not in out, out
    assert (
        f"{hooks} does not exist; {config} names an aelfrice program, as the "
        "lines above say, in no handler `aelf setup --host codex` writes"
    ) in out, out


# --- a byte order mark ---------------------------------------------------------

#: The UTF-8 byte order mark, as bytes.
_BOM_BYTES: Final[bytes] = b"\xef\xbb\xbf"


@pytest.mark.timeout(60)
@pytest.mark.parametrize("marks", [1, 2])
def test_doctor_reads_a_config_toml_that_starts_with_a_byte_order_mark(
    tmp_path: Path, marks: int,
) -> None:
    """Codex's TOML parser skips one UTF-8 byte order mark, so doctor reads
    the approvals in a config.toml that starts with one and counts both
    copies there. With a second mark, Codex refuses to start and `tomllib`
    can't parse the file, so doctor counts no run."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    config = _two_copies_in(codex_dir, tmp_path / "skills")
    approve_codex_hooks(codex_dir)
    config.write_bytes(_BOM_BYTES * marks + config.read_bytes())
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    if marks == 1:
        assert rc == 1, out
        assert (
            "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
            "times"
        ) in out, out
        assert "parse" not in out, out
    else:
        assert rc == 0, out
        assert "times" not in out, out
        assert f"doctor can't parse {config}, which holds" in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_profile_file_that_starts_with_a_byte_order_mark(
    tmp_path: Path,
) -> None:
    """A profile file goes through the same reader as config.toml."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    profile = codex_dir / "work.config.toml"
    profile.write_bytes(_BOM_BYTES + b"[features]\nhooks = false\n")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] `codex --profile work` also reads the profile file {profile}, "
        "and its [features] turns Codex's `hooks` feature off."
    ) in out, out
    assert "can't" not in out, out


@pytest.mark.timeout(60)
def test_doctor_rejects_a_hooks_json_that_starts_with_a_byte_order_mark(
    tmp_path: Path,
) -> None:
    """serde_json reads no byte order mark, so Codex loads no hook from a
    hooks.json that starts with one, and doctor exits 1 and counts neither
    copy in it."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _two_in_hooks_json(hooks)
    approve_codex_hooks(codex_dir)
    hooks.write_bytes(_BOM_BYTES + hooks.read_bytes())
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 1, out
    assert f"[warn] {hooks} is unreadable as hooks config (" in out, out
    assert "times" not in out, out


# --- numbers and keys Codex rejects ------------------------------------------


#: A decimal integer longer than Python's `int` reads.
_LONG_INT = "1" * (sys.get_int_max_str_digits() + 1)


#: Places in hooks.json for a value: the path of a field that two copies of
#: setup's SessionStart handler hold, and the path doctor names, which for
#: a value nested past serde_json's limit is the path of the value it cuts.
_PLACES: dict[str, tuple[tuple[object, ...], str]] = {
    "a key Codex skips": (("Foo",), "`hooks.Foo`"),
    "a field Codex doesn't read": (
        ("SessionStart", 0, "hooks", 1, "x"),
        "`hooks.SessionStart[0].hooks[1].x`",
    ),
    "a timeout": (
        ("SessionStart", 0, "hooks", 1, "timeout"),
        "`hooks.SessionStart[0].hooks[1].timeout`",
    ),
    "an mcp_tool input": (
        ("SessionStart", 0, "hooks", 2, "input", "x"),
        "`hooks.SessionStart[0].hooks[2].input.x`",
    ),
    "a value nested deep under a key Codex skips": (
        ("Foo",),
        "`hooks.Foo[0][0][0][0]...`, in a value nested 128 levels deep",
    ),
}


def _hooks_with(hooks: Path, place: str, value: str) -> None:
    """Two copies of setup's SessionStart handler, and `value`, a JSON
    token, at `place` (`_PLACES`)."""
    _two_in_hooks_json(hooks)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"].append(
        {"type": "mcp_tool", "server": "s", "tool": "t", "input": {}},
    )
    path, _shown = _PLACES[place]
    parent: Any = doc["hooks"]
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = "@@VALUE@@"
    if place.startswith("a value nested deep"):
        value = "[" * 140 + value + "]" * 140
    hooks.write_text(
        json.dumps(doc).replace('"@@VALUE@@"', value), encoding="utf-8",
    )


#: Numbers a 64-bit float can't hold. The codex binary builds serde_json
#: with `arbitrary_precision`, so none of them fails the parse.
_WIDE_NUMBERS: Final[list[str]] = ["1e400", "-1e400", "1" + "0" * 400, _LONG_INT]
_WIDE_IDS: Final[list[str]] = ["1e400", "-1e400", "401 digits", "past the digit limit"]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "place", sorted(p for p in _PLACES if p != "a timeout"),
)
@pytest.mark.parametrize("value", _WIDE_NUMBERS, ids=_WIDE_IDS)
def test_doctor_counts_runs_beside_a_number_a_float_can_not_hold(
    tmp_path: Path, place: str, value: str,
) -> None:
    """Codex loads hooks.json with a number a 64-bit float can't hold in a
    value it skips or doesn't read as a typed field, so doctor counts the
    two copies there, and doesn't raise on an integer longer than Python's
    `int` reads."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _hooks_with(hooks, place, value)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times"
    ) in out, out
    assert "Codex loads no hook" not in out, out
    assert "float" not in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("timeout", "has a `timeout` that isn't a count of seconds"),
        (
            "additionalContextLimit",
            "has an `additionalContextLimit` that isn't a count",
        ),
    ],
)
@pytest.mark.parametrize("value", _WIDE_NUMBERS, ids=_WIDE_IDS)
def test_doctor_rejects_a_hooks_json_with_such_a_number_in_a_typed_field(
    tmp_path: Path, field: str, reason: str, value: str,
) -> None:
    """Such a number reaches a typed field of a handler as a map, which the
    field rejects, so Codex loads no hook from the file. Doctor exits 1
    and names the handler, and counts neither copy."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _two_in_hooks_json(hooks)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["SessionStart"][0]["hooks"][1][field] = "@@VALUE@@"
    hooks.write_text(
        json.dumps(doc).replace('"@@VALUE@@"', value), encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        f"[warn] Codex loads no hook from {hooks}, because it rejects the "
        f"whole file: SessionStart group 1, handler 2 {reason}\n"
    ) in out, out
    assert "times" not in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("value", ["1e308", "-1e308", "1e-400", "1" * 308])
def test_doctor_counts_runs_beside_a_number_a_float_holds(
    tmp_path: Path, value: str,
) -> None:
    """A number a 64-bit float holds, or one that underflows to 0, which
    serde_json reads, leaves the count as it is."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _hooks_with(hooks, "a field Codex doesn't read", value)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "starts the aelf-session-start-hook command 2 times" in out, out
    assert "64-bit float" not in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("place", sorted(_PLACES))
@pytest.mark.parametrize("name", ["NaN", "Infinity", "-Infinity"])
def test_doctor_rejects_a_hooks_json_with_a_json_constant(
    tmp_path: Path, place: str, name: str,
) -> None:
    """serde_json rejects `NaN`, `Infinity`, and `-Infinity` at any place in
    a file, in a value Codex skips or nested past its depth limit too, so
    Codex loads no hook from it and the two copies in it never run."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _hooks_with(hooks, place, name)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "times" not in out, out
    assert (
        f"[warn] Codex loads no hook from {hooks}, because it rejects the "
        f"whole file: it holds `{name}` at {_PLACES[place][1]}, which "
        "serde_json, which Codex reads the file with, rejects at any place "
        "in a file\n"
    ) in out, out


# A top-level value in config.toml, and why Codex's TOML parser rejects
# the file, or None when it reads it.
_TOML_NUMBERS: dict[str, tuple[str, str | None]] = {
    "the largest i64": ("9223372036854775807", None),
    "the smallest i64": ("-9223372036854775808", None),
    "past the largest i64": (
        "9223372036854775808",
        "the integer at `x` is outside the range of a 64-bit signed integer",
    ),
    "past the smallest i64": (
        "-9223372036854775809",
        "the integer at `x` is outside the range of a 64-bit signed integer",
    ),
    "a hex u64": (
        "0xffffffffffffffff",
        "the integer at `x` is outside the range of a 64-bit signed integer",
    ),
    "in a nested array": (
        "[[1, 9223372036854775808]]",
        "the integer at `x[0][1]` is outside the range of a 64-bit signed "
        "integer",
    ),
    "in an inline table": (
        "{ a = { b = -9223372036854775809 } }",
        "the integer at `x.a.b` is outside the range of a 64-bit signed "
        "integer",
    ),
    "a float that overflows": (
        "1e400", "the float `1e400` overflows a 64-bit float",
    ),
    "a negative float that overflows": (
        "-1_0e400", "the float `-1_0e400` overflows a 64-bit float",
    ),
    "inf": ("inf", None),
    "-inf": ("-inf", None),
    "a float that underflows": ("1e-400", None),
    "a long integer": (
        _LONG_INT,
        "it holds an integer of more than "
        f"{sys.get_int_max_str_digits()} digits, outside the range of a "
        "64-bit signed integer",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_TOML_NUMBERS))
def test_doctor_reads_toml_numbers_as_codex_does(
    tmp_path: Path, shape: str,
) -> None:
    """toml 0.9 rejects the whole config.toml over an integer an `i64`
    can't hold or a float literal that overflows, so Codex refuses to
    start and doctor counts no run. `inf` and an underflow are fine."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    value, why = _TOML_NUMBERS[shape]
    _approve(config, 0, 1)
    config.write_text(
        f"x = {value}\n" + config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    assert rc == 1, out
    if why is None:
        assert "starts the aelf-session-start-hook command 2 times" in out, out
        assert "refuses to start" not in out, out
    else:
        assert "times" not in out, out
        assert (
            f"[warn] Codex refuses to start with {config}, because its TOML "
            f"parser rejects the whole file: {why}\n"
        ) in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_profile_file_with_a_long_integer(
    tmp_path: Path,
) -> None:
    """A profile file `tomllib` can't read an integer of doesn't stop
    doctor: the plain launch doesn't read it, and doctor names it."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    profile = codex_dir / "ci.config.toml"
    profile.write_text(f"x = {_LONG_INT}\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] doctor can't read the profile file {profile}, which `codex "
        "--profile ci` reads: its TOML parser rejects the whole file: it "
        f"holds an integer of more than {sys.get_int_max_str_digits()} digits"
    ) in out, out


# `[features]` tables, and what doctor makes of each: Codex reads it,
# Codex refuses to start because of the key named, or doctor doesn't check
# the value of the key named, in the table named.
_FEATURES: dict[str, tuple[str, str, str, str]] = {
    "a text value": (
        '[features]\nweb_search = "live"\n', "refuses", "web_search",
        "[features]",
    ),
    "a number": ("[features]\nshell_tool = 1\n", "refuses", "shell_tool", "[features]"),
    "a table under a plain key": (
        "[features.memories]\nx = 1\n", "refuses", "memories", "[features]",
    ),
    "a profile table's text value": (
        '[profiles.ci.features]\nweb_search = "live"\n', "refuses",
        "web_search", "[profiles.ci.features]",
    ),
    "true and false": (
        "[features]\nweb_search = true\nx = false\n", "reads", "", "",
    ),
    "true under a structured key": (
        "[features]\ncode_mode = true\nnetwork_proxy = false\n", "reads", "",
        "",
    ),
    "a table under a structured key": (
        "[features.code_mode]\nenabled = true\n", "unchecked", "code_mode",
        "[features]",
    ),
    "a number under a structured key": (
        "[features]\ntoken_budget = 100000\n", "unchecked", "token_budget",
        "[features]",
    ),
    "a network_proxy table": (
        "[features.network_proxy]\nenabled = true\n", "unchecked",
        "network_proxy", "[features]",
    ),
    "a profile table's structured table": (
        "[profiles.ci.features.code_mode]\nenabled = true\n", "unchecked",
        "code_mode", "[profiles.ci.features]",
    ),
    "tool_registry set to true": (
        "[features]\ntool_registry = true\n", "refuses", "tool_registry",
        "[features]",
    ),
    "a tool_registry table": (
        "[features.tool_registry]\nerror_on_tool_collisions = true\n",
        "unchecked", "tool_registry", "[features]",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_FEATURES))
def test_doctor_reads_every_features_key_as_codex_does(
    tmp_path: Path, shape: str,
) -> None:
    """`FeaturesToml` reads each `[features]` key outside its own fields as
    true or false, and `tool_registry` as a table only, so another value
    makes Codex refuse to start. It reads each other field of its own as
    true, false, or a table, which doctor doesn't check. Doctor counts no
    run when Codex refuses to start or when it doesn't check a value."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    table, kind, key, where = _FEATURES[shape]
    _approve(config, 0, 1)
    config.write_text(
        config.read_text(encoding="utf-8") + table, encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    if kind == "reads":
        assert rc == 1, out
        assert "starts the aelf-session-start-hook command 2 times" in out, out
        assert "refuses to start" not in out, out
        assert "doesn't check the value" not in out, out
        return
    assert rc == 0, out
    assert "times" not in out, out
    # Codex may start both copies only when doctor can't tell.
    copy_line = (
        "is a copy of the aelf-session-start-hook command `aelf setup --host "
        "codex` writes, and doctor counts no run for it, because "
    )
    assert (out.count(copy_line) == 2) is (kind == "unchecked"), out
    if kind == "refuses":
        assert (
            f"[warn] doctor counts no duplicate run, because `{key}` in "
            f"{where} in {config} "
        ) in out, out
        assert ", so Codex refuses to start\n" in out, out
    else:
        assert (
            "[warn] doctor counts no duplicate run, because doctor doesn't "
            f"check the value of `{key}` in {where} in {config} against the "
            "type Codex reads it as, so it can't tell whether Codex starts\n"
        ) in out, out


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize(
    ("mode", "warns"),
    [(0o311, True), (0o644, False)],
    ids=["listed no", "searched no"],
)
def test_doctor_reads_a_codex_home_it_can_not_list_or_search(
    tmp_path: Path, mode: int, warns: bool,
) -> None:
    """Doctor doesn't raise on a Codex home it can't list, which only
    hides profile files, or can't search, where Python 3.12's
    `Path.is_file` raises."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    codex_dir, _hooks, _ours = _installed(tmp_path)
    codex_dir.chmod(mode)
    try:
        rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    finally:
        codex_dir.chmod(0o755)
    assert rc in (0, 1), out
    listed = (
        f"[warn] doctor can't list the files in {codex_dir}, so it reads no "
        "profile file that `codex --profile NAME` reads: "
    )
    assert (listed in out) is warns, out


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize(
    ("locked", "mode"), [("agents", 0), ("agents/skills", 0o444)],
)
def test_doctor_reads_a_skills_folder_it_can_not_search(
    tmp_path: Path, locked: str, mode: int,
) -> None:
    """Doctor doesn't raise on a `~/.agents`, or a `~/.agents/skills`, it
    can't search, where Python 3.12's `Path.is_dir` raises, and counts no
    skill there."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    codex_dir, _hooks, _ours = _installed(tmp_path)
    agents = tmp_path / "agents"
    install_codex_skills(agents / "skills")
    (tmp_path / locked).chmod(mode)
    try:
        rc, out = _doctor(codex_dir, agents / "skills")
    finally:
        (tmp_path / locked).chmod(0o755)
    assert rc == 0, out
    assert (
        "[i] codex skills ($aelf-*): 0 installed (0 orphaned) in "
        f"{agents / 'skills'}\n"
    ) in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_skill_file_that_is_not_utf8(tmp_path: Path) -> None:
    """A SKILL.md that isn't UTF-8 reads as a skill that isn't aelfrice's,
    and doctor doesn't raise."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    skill = tmp_path / "skills" / "aelf-local"
    skill.mkdir()
    (skill / "SKILL.md").write_bytes(b"AELFRICE-CODEX-SKILL \xff\n")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "(0 orphaned)" in out, out


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_doctor_reports_a_codex_home_under_a_folder_it_can_not_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `CODEX_HOME` under a folder doctor can't search is one Codex can't
    read either, so doctor reports it and exits 1, where Python 3.12's
    `Path.exists` raises."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    locked = tmp_path / "locked"
    (locked / "codex").mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(locked / "codex"))
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    locked.chmod(0)
    try:
        rc = _cmd_doctor_codex(
            argparse.Namespace(host="codex"), io.StringIO(),
            skills_dest=tmp_path / "skills",
        )
    finally:
        locked.chmod(0o755)
    assert rc == 1, err.getvalue()
    assert (
        f"doctor --host codex: $CODEX_HOME points to {locked / 'codex'}, "
        "which aelfrice can't read ("
    ) in err.getvalue(), err.getvalue()


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize("mode", [0o644, 0], ids=["644", "000"])
def test_setup_and_unsetup_refuse_a_codex_home_they_can_not_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], mode: int,
) -> None:
    """In a Codex home you can't search, setup and unsetup exit 1 with a
    message that names hooks.json and the system error, where Python 3.12
    and 3.13 raise `PermissionError`, and write nothing."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks)
    before = (sorted(p.name for p in codex_dir.iterdir()), hooks.read_bytes())
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    args = argparse.Namespace(
        host="codex", force=False, status_message=None, codex_skills=True,
    )
    codex_dir.chmod(mode)
    try:
        rcs = {
            "setup": _cmd_setup_codex(
                args, io.StringIO(), skills_dest=tmp_path / "skills",
            ),
            "unsetup": _cmd_unsetup_codex(
                args, io.StringIO(), skills_dest=tmp_path / "skills",
            ),
        }
    finally:
        codex_dir.chmod(0o755)
    err = capsys.readouterr().err
    assert rcs == {"setup": 1, "unsetup": 1}, err
    assert err == (
        f"setup --host codex: can't read {hooks} ([Errno 13] Permission "
        f"denied: '{hooks}'); nothing was written\n"
        f"unsetup --host codex: can't read {hooks} ([Errno 13] Permission "
        f"denied: '{hooks}'); nothing was changed\n"
    ), err
    assert (sorted(p.name for p in codex_dir.iterdir()), hooks.read_bytes()) == before
    assert not (tmp_path / "skills").exists()


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
@pytest.mark.parametrize("mode", [0o644, 0], ids=["644", "000"])
def test_doctor_names_the_files_in_a_codex_home_it_can_not_search(
    tmp_path: Path, mode: int,
) -> None:
    """Doctor can't tell whether hooks.json exists in a Codex home it can't
    search, so it says so, names the file, and exits 1, and it reads
    config.toml as a file Codex can't read and refuses to start with."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    codex_dir, hooks, _ours = _installed(tmp_path)
    approve_codex_hooks(codex_dir)
    codex_dir.chmod(mode)
    try:
        rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    finally:
        codex_dir.chmod(0o755)
    assert rc == 1, out
    assert "[-] codex hooks.json: present=unknown valid=False " in out, out
    assert (
        f"[warn] {hooks} is unreadable as hooks config (doctor can't tell "
        "whether the file exists, because a stat of it fails: [Errno 13] "
        f"Permission denied: '{hooks}'); Codex will ignore or reject it\n"
    ) in out, out
    config = codex_dir / "config.toml"
    assert (
        f"[warn] Codex refuses to start with {config}, because Codex can't "
        f"read the file: [Errno 13] Permission denied: '{config}'\n"
    ) in out, out
    assert "does not exist" not in out, out


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_doctor_reports_a_default_codex_home_under_a_folder_it_can_not_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `CODEX_HOME` unset, a `~/.codex` under a folder doctor can't
    search is one Codex can't read either, so doctor names it and exits 1
    instead of saying that Codex isn't installed."""
    if os.geteuid() == 0:
        pytest.skip("root reads any directory")
    locked = tmp_path / "locked"
    (locked / "home" / ".codex").mkdir(parents=True)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(locked / "home"))
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    locked.chmod(0)
    try:
        rc = _cmd_doctor_codex(
            argparse.Namespace(host="codex"), out,
            skills_dest=tmp_path / "skills",
        )
    finally:
        locked.chmod(0o755)
    assert rc == 1, out.getvalue() + err.getvalue()
    assert err.getvalue().startswith(
        "doctor --host codex: doctor can't tell whether "
        f"{locked / 'home' / '.codex'} exists ([Errno 13] Permission denied",
    ), err.getvalue()
    assert "not found" not in out.getvalue(), out.getvalue()


# --- the programs doctor names ----------------------------------------------


def _package_scripts() -> list[str]:
    """The console scripts in pyproject.toml's `[project.scripts]`."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return sorted(data["project"]["scripts"])


def test_doctor_names_every_program_the_package_installs() -> None:
    """A script pyproject.toml adds is a program doctor must name."""
    scripts = _package_scripts()
    assert "aelf" in scripts and len(scripts) >= 11, scripts
    assert sorted(set(scripts) - _PROGRAM_NAMES) == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", _package_scripts())
def test_doctor_warns_about_a_handler_that_runs_any_package_program(
    tmp_path: Path, name: str,
) -> None:
    """A handler of your own that runs any program the package installs
    gets a `[warn]` line that names the program, and doctor exits 0."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["Stop"][0]["hooks"].append(
        {"type": "command", "command": f"/usr/bin/env {name} --flag"},
    )
    hooks.write_text(json.dumps(doc), encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    lines = [
        line for line in out.splitlines()
        if line.startswith("[warn] Stop handler ")
        and "`hooks.Stop[0].hooks[2]`" in line
        and f" {name}. " in line
    ]
    assert len(lines) == 1, out


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("/opt/aelf/bin/aelf-hook", ["aelf-hook"]),
        ("/home/me/aelfrice/.venv/bin/aelf-stop-hook", ["aelf-stop-hook"]),
        ("aelf search x", ["aelf"]),
        ("AELF.exe status", ["aelf"]),
        ("uv run a'el'f session-delta", ["aelf"]),
        ("python -m aelfrice.cli", []),
        ("myaelf x_aelf aelf_x aelfx", []),
    ],
)
def test_doctor_reads_the_bare_cli_as_a_word(
    text: str, names: list[str],
) -> None:
    """The bare `aelf` counts only as a word, so a folder named `aelf`, the
    package name, and the other programs don't name it."""
    assert _names_in(text) == names


@pytest.mark.timeout(60)
@pytest.mark.parametrize("text", ["aelf memory", "aelf-hook memory"])
def test_doctor_reads_no_name_in_the_status_message_of_setup_handlers(
    tmp_path: Path, text: str,
) -> None:
    """`aelf setup --host codex --status-message TEXT` writes TEXT on its
    own handlers. Doctor reads a handler whose command is byte-identical to
    setup's as setup's, so a program name in TEXT gets no line."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    install_codex_hooks(codex_dir / "hooks.json", status_message=text)
    install_codex_skills(tmp_path / "skills")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "[warn]" not in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_the_status_message_of_a_handler_of_your_own(
    tmp_path: Path,
) -> None:
    """A handler whose command setup doesn't write keeps the line for a
    program its `statusMessage` names."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["Stop"][0]["hooks"].append({
        "type": "command", "command": "/usr/bin/true",
        "statusMessage": "aelf memory",
    })
    hooks.write_text(json.dumps(doc), encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    lines = [
        line for line in out.splitlines()
        if line.startswith("[warn] Stop handler ")
        and "`hooks.Stop[0].hooks[2]`" in line and " names aelf, " in line
    ]
    assert len(lines) == 1, out


@pytest.mark.timeout(60)
def test_doctor_keeps_the_command_of_a_setup_handler_with_a_status_message(
    tmp_path: Path,
) -> None:
    """The exemption drops only the names in the `statusMessage` of setup's
    handler. A copy of that handler under a key that isn't a Codex event
    still gets the line that names the program its command runs."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    hooks = codex_dir / "hooks.json"
    install_codex_hooks(hooks, status_message="aelf memory")
    install_codex_skills(tmp_path / "skills")
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["NotAnEvent"] = [doc["hooks"]["Stop"][0]]
    hooks.write_text(json.dumps(doc), encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    lines = [
        line for line in out.splitlines()
        if line.startswith(
            "[warn] 'NotAnEvent' handler at `hooks.NotAnEvent[0].hooks[",
        )
        and " but Codex reads no hook event by that name" in line
    ]
    named = sorted(
        line.split(" names ", 1)[1].split(",", 1)[0] for line in lines
    )
    assert named == ["aelf-stop-hook", "aelf-transcript-logger"], out


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("C:\\Users\\me\\.local\\bin\\aelf.exe search", ["aelf"]),
        ("C:\\tools\\aelf search", ["aelf"]),
        ("aelf\\ x", ["aelf"]),
        ("C:\\aelf\\bin\\tool.exe", []),
        ("C:\\bin\\myaelf.exe", []),
    ],
)
def test_doctor_reads_a_backslash_as_a_path_separator_for_the_bare_cli(
    text: str, names: list[str],
) -> None:
    """A bare `aelf` counts as a word in the string with `\\` deleted or in
    the string with `\\` read as `/`, so a Windows path to the CLI names it,
    and a Windows folder named `aelf` doesn't."""
    assert _names_in(text) == names


def test_doctor_names_the_line_of_a_windows_path_to_the_bare_cli() -> None:
    """In a file `tomllib` can't parse, the bare CLI in a Windows path
    counts on the line it is on, after a line of backslashes that the
    first reading deletes."""
    text = (
        'x = "' + "\\" * 12 + '"\n'
        "y = = 1\n"
        "notify = ['C:\\bin\\aelf.exe']\n"
        "z = 2\n"
    )
    assert _names_by_line(text) == [
        (3, "aelf", "notify = ['C:\\bin\\aelf.exe']"),
    ]


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("true\\naelf search", ["aelf"]),
        ("x\\taelf", ["aelf"]),
        ("bash -c $'\\raelf search'", ["aelf"]),
        ("env -Saelf search", []),
        ("C:\\aelf\\bin\\tool.exe", []),
    ],
)
def test_doctor_reads_an_escape_before_the_bare_cli_as_a_space(
    text: str, names: list[str],
) -> None:
    """A `\\` and the one character after it, such as the escape `\\n`,
    leave a bare `aelf` after them a word, and doctor reads no other shell
    syntax, so an option glued to the name hides it."""
    assert _names_in(text) == names


def test_doctor_names_the_line_of_an_escape_before_the_bare_cli() -> None:
    """In a file `tomllib` can't parse, the bare CLI after an escape counts
    on the line it is on, after a line that a backslash-newline joins and a
    line of backslashes, which each reading shortens by its own count."""
    text = (
        'x = "a\\\nb"\n'
        'w = "' + "\\" * 40 + '"\n'
        "y = = 1\n"
        "notify = ['true\\naelf']\n"
        "z = 2\n"
        "v = 3\n"
    )
    assert _names_by_line(text) == [
        (5, "aelf", "notify = ['true\\naelf']"),
    ]


@pytest.mark.timeout(60)
def test_doctor_warns_about_the_bare_cli_after_an_escape(
    tmp_path: Path,
) -> None:
    """A handler whose command runs the CLI after an escape that a shell
    reads as a newline gets the warning that names it."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    doc["hooks"]["Stop"][0]["hooks"].append({
        "type": "command", "command": "bash -c $'true\\naelf search'",
    })
    hooks.write_text(json.dumps(doc), encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    lines = [
        line for line in out.splitlines()
        if line.startswith("[warn] Stop handler ")
        and "`hooks.Stop[0].hooks[2]`" in line and " names aelf, " in line
    ]
    assert len(lines) == 1, out
