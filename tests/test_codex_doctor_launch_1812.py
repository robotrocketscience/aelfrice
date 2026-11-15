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
