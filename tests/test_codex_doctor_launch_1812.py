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

import pytest

from aelfrice.cli import _cmd_doctor_codex
from aelfrice.host_codex import (
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


# --- numbers and keys Codex rejects ------------------------------------------


#: A decimal integer longer than Python's `int` reads.
_LONG_INT = "1" * (sys.get_int_max_str_digits() + 1)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["an event Codex skips", "a timeout"])
def test_doctor_reads_a_long_integer_in_hooks_json_as_unreadable(
    tmp_path: Path, where: str,
) -> None:
    """Python's `json` raises a plain ValueError for an integer longer
    than `int` reads. Doctor reads such a hooks.json as one it can't read,
    exits 1, and doesn't raise."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    text = hooks.read_text(encoding="utf-8")
    if where == "a timeout":
        text = text.replace(
            '"type": "command"', f'"type": "command", "timeout": {_LONG_INT}',
            1,
        )
    else:
        text = text.replace('"hooks": {', f'"hooks": {{"Foo": {_LONG_INT}, ', 1)
    hooks.write_text(text, encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        f"[warn] {hooks} is unreadable as hooks config (it holds an integer "
        f"of more than {sys.get_int_max_str_digits()} digits, which doctor "
        "doesn't read); Codex will ignore or reject it\n"
    ) in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", ["NaN", "Infinity", "-Infinity"])
def test_doctor_rejects_a_hooks_json_with_a_json_constant(
    tmp_path: Path, name: str,
) -> None:
    """serde_json rejects `NaN`, `Infinity`, and `-Infinity` anywhere in a
    file, even under a key Codex skips, so Codex loads no hook from it and
    the two copies in it never run."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    _two_in_hooks_json(hooks)
    text = hooks.read_text(encoding="utf-8")
    hooks.write_text(
        text.replace('"hooks": {', f'"hooks": {{"Foo": {name}, ', 1),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "times" not in out, out
    assert (
        f"[warn] {hooks} is unreadable as hooks config (it holds `{name}`, "
        "which serde_json, which Codex reads the file with, rejects anywhere "
        "in a file); Codex will ignore or reject it\n"
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


# `[features]` tables, and the key whose value makes Codex refuse to
# start, or None when Codex reads the table.
_FEATURES: dict[str, tuple[str, str | None]] = {
    "a text value": ('[features]\nweb_search = "live"\n', "web_search"),
    "a number": ("[features]\nshell_tool = 1\n", "shell_tool"),
    "a table under a plain key": ("[features.memories]\nx = 1\n", "memories"),
    "a profile table's text value": (
        '[profiles.ci.features]\nweb_search = "live"\n', "web_search",
    ),
    "true and false": ("[features]\nweb_search = true\nx = false\n", None),
    "a table under a structured key": (
        "[features.code_mode]\nenabled = true\n", None,
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_FEATURES))
def test_doctor_reads_every_features_key_as_codex_does(
    tmp_path: Path, shape: str,
) -> None:
    """`FeaturesToml` reads each `[features]` key outside its own fields as
    true or false, so any other value makes Codex refuse to start, and
    doctor counts no run."""
    codex_dir, config, _ours = _two_toml_copies(tmp_path)
    table, bad = _FEATURES[shape]
    _approve(config, 0, 1)
    config.write_text(
        config.read_text(encoding="utf-8") + table, encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills", approve=False)
    if bad is None:
        assert rc == 1, out
        assert "starts the aelf-session-start-hook command 2 times" in out, out
    else:
        assert rc == 0, out
        assert "times" not in out, out
        assert (
            f"[warn] doctor counts no duplicate run, because `{bad}` in "
        ) in out, out
        assert "isn't true or false, so Codex refuses to start\n" in out, out


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
