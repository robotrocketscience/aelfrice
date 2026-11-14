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


def _inline_tables(depth: int) -> str:
    return "{a = " * depth + "1" + "}" * depth


_NESTING = "it nests arrays or inline tables more than 80 levels deep"
_KEY_PARTS = "it has a key of more than 80 dotted parts"
# config.toml lines `n` levels or parts deep, past the limits of Codex's
# TOML parser when `n` is over 80: 80 levels of arrays and inline tables,
# and 80 parts in a key. Two follow a string whose end the scan must find
# for the brackets after it to count.
_TOML_TOO_DEEP: dict[str, tuple[Callable[[int], str], str]] = {
    "inline arrays": (lambda n: "x = " + "[" * n + "]" * n, _NESTING),
    "inline tables": (lambda n: "x = " + _inline_tables(n), _NESTING),
    "after a string closed by extra quotes": (
        lambda n: 'x = ["""a"""", ' + "[" * (n - 1) + "]" * n, _NESTING,
    ),
    "after an escaped quote": (
        lambda n: 'x = ["""\\"""a""", ' + "[" * (n - 1) + "]" * n, _NESTING,
    ),
    "dotted key": (lambda n: ".".join(["a"] * n) + " = 1", _KEY_PARTS),
    "table header": (lambda n: "[" + ".".join(["b"] * n) + "]", _KEY_PARTS),
    "array of tables header": (
        lambda n: "[[" + ".".join(["c"] * n) + "]]", _KEY_PARTS,
    ),
    "inline table key": (
        lambda n: "y = {" + ".".join(["d"] * n) + " = 1}", _KEY_PARTS,
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_TOML_TOO_DEEP))
@pytest.mark.parametrize("depth", [80, 81, 129, 100_000])
def test_doctor_rejects_a_config_toml_past_codex_toml_limits(
    tmp_path: Path, shape: str, depth: int,
) -> None:
    """Codex's TOML parser rejects the whole file past its limits, so Codex
    loads no hook from it, and doctor exits 1. `tomllib` never reads such
    a file, so no depth can exhaust Python's recursion limit."""
    codex_dir, _hooks, ours = _installed(tmp_path)
    config = codex_dir / "config.toml"
    config.write_text(
        '[[hooks.SessionStart]]\nmatcher = "startup"\n'
        "[[hooks.SessionStart.hooks]]\n"
        f'type = "command"\ncommand = {json.dumps(ours)}\n'
        + _TOML_TOO_DEEP[shape][0](depth) + "\n",
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    why = _TOML_TOO_DEEP[shape][1]
    rejected = (
        f"[warn] Codex loads no hook from {config}, because its TOML parser "
        f"rejects the whole file: {why}, past the limit of 80"
    )
    if depth <= 80:
        # Codex loads the file, and its copy of setup's handler is a
        # second run.
        assert rc == 1, out
        assert rejected not in out, out
        assert "starts the aelf-session-start-hook command 2 times" in out
    else:
        assert rc == 1, out
        assert rejected in out, out
        assert "times" not in out, out


# Brackets, dots, and quotes that Codex's TOML parser reads as text.
_TOML_TEXT = {
    "basic string": 'x = "' + "[" * 90 + '"',
    "literal string": "x = '" + "[" * 90 + "'",
    "multiline basic string": 'x = """\n' + "{" * 90 + '\n\\""" """',
    "multiline literal string": "x = '''\n" + "[" * 90 + "\n'' '''",
    "string closed by extra quotes": 'x = """' + "[" * 90 + '""""',
    "comment": "# x = " + "[" * 90,
    "quoted key": '"' + ".".join(["a"] * 90) + '" = 1',
    "quoted header part": '["' + ".".join(["b"] * 90) + '"]',
    "float": "x = [" + ", ".join(["1.5"] * 90) + "]",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_TOML_TEXT))
def test_doctor_counts_no_nesting_inside_toml_text(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, _hooks, _ours = _installed(tmp_path)
    config = codex_dir / "config.toml"
    config.write_text(_TOML_TEXT[shape] + "\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "rejects the whole file" not in out, out
    assert "could not parse" not in out, out


@pytest.mark.timeout(60)
def test_doctor_says_where_it_stopped_reading_a_deep_config_toml(
    tmp_path: Path,
) -> None:
    """Inline tables of dotted keys nest past the scan's depth bound within
    the parser's limits, and doctor says it didn't read below it."""
    codex_dir, _hooks, _ours = _installed(tmp_path)
    value = '"aelf-hook"'
    for _ in range(5):
        value = "{" + ".".join(["k"] * 60) + " = " + value + "}"
    config = codex_dir / "config.toml"
    config.write_text(f"[hooks]\nBogus = {value}\n", encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] doctor read the [hooks] table of {config} only to a depth "
        "of 256 and checked no string below it for an aelfrice program name"
    ) in out, out

