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
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def _splice(hooks: Path, marker: str, depth: int) -> None:
    """Replace the JSON string `marker` in hooks.json with `depth` nested
    arrays, written as text so no Python recursion builds them."""
    text = hooks.read_text(encoding="utf-8")
    assert text.count(json.dumps(marker)) == 1
    hooks.write_text(
        text.replace(json.dumps(marker), "[" * depth + "]" * depth),
        encoding="utf-8",
    )


def _edit_hooks(hooks: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    doc = json.loads(hooks.read_text(encoding="utf-8"))
    edit(doc)
    hooks.write_text(json.dumps(doc), encoding="utf-8")


def _mcp_input(doc: dict[str, Any]) -> None:
    doc["hooks"].setdefault("PreToolUse", []).append({
        "matcher": "Bash", "hooks": [{
            "type": "mcp_tool", "server": "s", "tool": "t",
            "input": {"a": "DEEP"},
        }],
    })


# Places serde reads, where a deep nest rejects the file, with the depth
# of the arrays that replace the marker, counting the top-level object as
# depth 1.
_READ_PLACES: dict[str, tuple[Callable[[dict[str, Any]], object], int]] = {
    # {} hooks{} PreToolUse[] group{} hooks[] handler{} input{} a=[
    "mcp_tool input": (_mcp_input, 8),
    # {} hooks{} Stop[] group{} hooks[] handler{} zz=[
    "unknown handler field": (
        lambda d: d["hooks"]["Stop"][0]["hooks"][0].__setitem__("zz", "DEEP"),
        7,
    ),
}
# Places serde skips with `ignore_value`, where any depth loads.
_SKIPPED_PLACES: dict[str, Callable[[dict[str, Any]], object]] = {
    "unknown event": lambda d: d["hooks"].__setitem__("Bogus", "DEEP"),
    "unknown group key": (
        lambda d: d["hooks"]["Stop"][0].__setitem__("x", "DEEP")
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("place", sorted(_READ_PLACES))
@pytest.mark.parametrize("total", [127, 128, 129, 100_000])
def test_doctor_rejects_a_hooks_json_past_serde_json_depth_limit(
    tmp_path: Path, place: str, total: int,
) -> None:
    """serde_json rejects an array or object 128 levels deep, so Codex
    loads no hook from the file, and doctor exits 1 as it does on any file
    Codex rejects. At 127 levels Codex loads the file."""
    codex_dir, hooks, _ours = _installed(tmp_path)
    edit, marker_depth = _READ_PLACES[place]
    _edit_hooks(hooks, edit)
    _splice(hooks, "DEEP", total - marker_depth + 1)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    rejected = (
        f"[warn] Codex loads no hook from {hooks}, because it rejects the "
        "whole file: an array or object at `"
    )
    if total < 128:
        assert rc == 0, out
        assert rejected not in out, out
    else:
        assert rc == 1, out
        assert rejected in out, out
        assert (
            "nests 128 levels deep, where serde_json, which Codex reads the "
            "file with, stops" in out
        ), out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("place", sorted(_SKIPPED_PLACES))
@pytest.mark.parametrize("total", [129, 100_000])
def test_doctor_reads_a_hooks_json_nested_deep_where_serde_skips(
    tmp_path: Path, place: str, total: int,
) -> None:
    """serde skips a value under a key it doesn't read without counting
    depth, so Codex loads the file and runs a copy of setup's handler in
    it. Doctor counts the copy and says which value it didn't read."""
    codex_dir, hooks, ours = _installed(tmp_path)

    def edit(doc: dict[str, Any]) -> None:
        _SKIPPED_PLACES[place](doc)
        doc["hooks"]["SessionStart"].append(
            {"hooks": [{"type": "command", "command": ours}]},
        )

    _edit_hooks(hooks, edit)
    _splice(hooks, "DEEP", total)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times" in out
    ), out
    assert "Codex loads no hook from" not in out, out
    assert any(
        ln.startswith("[warn] doctor didn't read the value at `hooks.")
        and ln.endswith(
            f"in {hooks}, which nests 128 levels deep under a key Codex "
            "skips, and checked no string in it for an aelfrice program name",
        )
        for ln in out.splitlines()
    ), out
