"""`aelf doctor --host codex` on files Codex reads in ways the duplicate
check must match (#1812): nesting depth, value types, disabled handlers,
the `hooks` feature, and program names that a shell's quote removal
spells.

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
from aelfrice.host_codex import (
    _toml_representable,
    install_codex_hooks,
    install_codex_skills,
)


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


def _nest(depth: int, leaf: object) -> object:
    """`leaf` inside `depth` lists, built without recursion."""
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


# --- depth -------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_the_mcp_tool_input_check_walks_any_depth() -> None:
    """The check keeps its own stack, so a value deeper than Python's
    recursion limit gets an answer instead of a RecursionError."""
    assert _toml_representable(_nest(100_000, 1)) is True
    assert _toml_representable(_nest(100_000, None)) is False
    assert _toml_representable({"a": _nest(100_000, 2**63)}) is False


@pytest.mark.timeout(60)
def test_doctor_reads_a_config_toml_mcp_tool_input_of_any_depth(
    tmp_path: Path,
) -> None:
    """TOML builds a deep `input` from inline tables of dotted keys, each
    within the limits of Codex's TOML parser."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    value = "1"
    for _ in range(40):
        value = "{" + ".".join(["k"] * 40) + " = " + value + "}"
    (codex_dir / "config.toml").write_text(
        '[[hooks.PreToolUse]]\nmatcher = "Bash"\n'
        "[[hooks.PreToolUse.hooks]]\n"
        'type = "mcp_tool"\nserver = "s"\ntool = "t"\n'
        f"input = {value}\n",
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
