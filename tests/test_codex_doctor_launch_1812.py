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
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_doctor_codex
from aelfrice.host_codex import install_codex_hooks, install_codex_skills


def _doctor(codex_dir: Path, skills: Path) -> tuple[int, str]:
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
    config.write_bytes(b'notify = ["caf\xe9"]\n')
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
    profile.write_bytes(b'notify = ["caf\xe9"]\n')
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
