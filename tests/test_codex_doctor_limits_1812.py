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
        f"[warn] doctor read {config} only to a depth of 256 and checked no "
        "string below it for an aelfrice program name"
    ) in out, out


# --- value types --------------------------------------------------------------

# TOML dates and times, which Codex's toml reads as their text. A matcher
# such as "1979-05-27" matches no session source or tool name.
_TOML_DATES = {
    "date": "1979-05-27",
    "time": "07:32:00",
    "offset date-time": "1979-05-27T07:32:00Z",
    "local date-time": "1979-05-27T07:32:00",
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("event", ["SessionStart", "PreToolUse"])
@pytest.mark.parametrize("kind", sorted(_TOML_DATES))
def test_doctor_counts_no_run_under_a_toml_date_matcher(
    tmp_path: Path, kind: str, event: str,
) -> None:
    codex_dir, hooks, ours = _installed(tmp_path)
    if event == "PreToolUse":
        doc = json.loads(hooks.read_text(encoding="utf-8"))
        ours = str(doc["hooks"]["PreToolUse"][0]["hooks"][0]["command"])
    config = codex_dir / "config.toml"
    config.write_text(
        f"[[hooks.{event}]]\nmatcher = {_TOML_DATES[kind]}\n"
        f"[[hooks.{event}.hooks]]\n"
        f'type = "command"\ncommand = {json.dumps(ours)}\n',
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out, out
    assert any(
        ln.startswith(f"[warn] {event} handler at `hooks.{event}[0].hooks[0]`")
        and "is a TOML date or time, which Codex reads as text and doctor "
        "doesn't read." in ln
        and ln.endswith(f"Handler: {ours}")
        for ln in out.splitlines()
    ), out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("field", ["commandWindows", "command_windows"])
@pytest.mark.parametrize("windows", [False, True])
def test_doctor_reads_no_toml_date_windows_command_on_windows(
    tmp_path: Path, field: str, windows: bool,
) -> None:
    """On Windows, Codex runs the date's text, not `command`, so the copy of
    setup's command in `command` is no run there. Elsewhere it is one."""
    from aelfrice.host_codex import doctor_codex

    codex_dir, _hooks, ours = _installed(tmp_path)
    (codex_dir / "config.toml").write_text(
        '[[hooks.SessionStart]]\nmatcher = "startup"\n'
        "[[hooks.SessionStart.hooks]]\n"
        f'type = "command"\ncommand = {json.dumps(ours)}\n'
        f"{field} = 1979-05-27\n",
        encoding="utf-8",
    )
    report = doctor_codex(codex_dir, windows=windows)
    if windows:
        assert report.duplicate_runs == [], report.duplicate_runs
        assert any(
            w.startswith("SessionStart handler in the [hooks] table of ")
            and "may also run aelf-session-start-hook." in w
            for w in report.warnings
        ), report.warnings
    else:
        assert report.duplicate_runs == [
            ("SessionStart", "aelf-session-start-hook", 2),
        ], report.duplicate_runs


# --- lone surrogates ----------------------------------------------------------

_LONE = "\ud800"


def _session_start_copy(ours: str, **fields: object) -> dict[str, Any]:
    return {
        "matcher": "startup",
        "hooks": [{"type": "command", "command": ours, **fields}],
    }


# Places serde_json reads as text, where a lone surrogate escape rejects
# the file. Each adds a copy of setup's SessionStart handler, which Codex
# never runs, since it loads no hook from the file.
_SURROGATE_READ: dict[str, Callable[[dict[str, Any], str], object]] = {
    "statusMessage": lambda d, c: d["hooks"]["SessionStart"].append(
        _session_start_copy(c, statusMessage=_LONE),
    ),
    "description": lambda d, c: (
        d.__setitem__("description", _LONE),
        d["hooks"]["SessionStart"].append(_session_start_copy(c)),
    ),
    "unknown event name": lambda d, c: (
        d["hooks"].__setitem__("Bo" + _LONE, 1),
        d["hooks"]["SessionStart"].append(_session_start_copy(c)),
    ),
    "unknown group key name": lambda d, c: d["hooks"]["SessionStart"].append(
        {"x" + "\udc00": 1, **_session_start_copy(c)},
    ),
    "unknown handler field": lambda d, c: d["hooks"]["SessionStart"].append(
        _session_start_copy(c, zz=_LONE),
    ),
    "mcp_tool input": lambda d, c: d["hooks"]["SessionStart"].append({
        "matcher": "startup", "hooks": [
            {"type": "command", "command": c},
            {"type": "mcp_tool", "server": "s", "tool": "t",
             "input": {"k": [_LONE]}},
        ],
    }),
    "prompt handler field": lambda d, c: d["hooks"]["SessionStart"].append({
        "matcher": "startup", "hooks": [
            {"type": "command", "command": c}, {"type": "prompt", "zz": _LONE},
        ],
    }),
    "high then high": lambda d, c: d["hooks"]["SessionStart"].append(
        _session_start_copy(c, statusMessage="\ud800𐀀"),
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("place", sorted(_SURROGATE_READ))
def test_doctor_rejects_a_hooks_json_with_a_lone_surrogate_codex_reads(
    tmp_path: Path, place: str,
) -> None:
    codex_dir, hooks, ours = _installed(tmp_path)
    _edit_hooks(hooks, lambda d: _SURROGATE_READ[place](d, ours))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        f"[warn] Codex loads no hook from {hooks}, because it rejects the "
        "whole file: the string at `"
    ) in out, out
    assert (
        "holds a lone UTF-16 surrogate escape, which serde_json rejects "
        "where Codex reads the value as text"
    ) in out, out
    assert "times" not in out, out


@pytest.mark.timeout(60)
def test_doctor_reads_a_repeated_member_that_json_loads_drops(
    tmp_path: Path,
) -> None:
    """serde reads both members of a repeated `input` key, so the dropped
    one's lone surrogate rejects the file."""
    codex_dir, hooks, ours = _installed(tmp_path)
    _edit_hooks(hooks, lambda d: d["hooks"]["SessionStart"].append({
        "matcher": "startup", "hooks": [
            {"type": "command", "command": ours},
            {"type": "mcp_tool", "server": "s", "tool": "t",
             "input": {"k": "DROPPED"}},
        ],
    }))
    text = hooks.read_text(encoding="utf-8")
    hooks.write_text(
        text.replace('{"k": "DROPPED"}', '{"k": "\\ud800", "k": "ok"}'),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "holds a lone UTF-16 surrogate escape" in out, out


# Places serde skips, where a lone surrogate loads.
_SURROGATE_SKIPPED: dict[str, Callable[[dict[str, Any], str], object]] = {
    "under an unknown event": lambda d, c: d["hooks"].__setitem__(
        "Bogus", [{"hooks": [{"type": "command", "command": c + _LONE}]}],
    ),
    "under an unknown group key": lambda d, c: d["hooks"]["Stop"][0]
    .__setitem__("x", {"run": c + _LONE}),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("place", sorted(_SURROGATE_SKIPPED))
def test_doctor_reads_a_lone_surrogate_where_serde_skips_it(
    tmp_path: Path, place: str,
) -> None:
    """Codex loads such a file, so a copy of setup's handler in it counts,
    and the warning that quotes the string prints on a UTF-8 stream."""
    codex_dir, hooks, ours = _installed(tmp_path)

    def edit(doc: dict[str, Any]) -> None:
        _SURROGATE_SKIPPED[place](doc, ours)
        doc["hooks"]["SessionStart"].append(_session_start_copy(ours))

    _edit_hooks(hooks, edit)
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="utf-8")
    rc = _cmd_doctor_codex(
        argparse.Namespace(host="codex"), stream,
        codex_dir=codex_dir, skills_dest=tmp_path / "skills",
    )
    stream.flush()
    out = buffer.getvalue().decode("utf-8")
    assert rc == 1, out
    assert "SessionStart starts the aelf-session-start-hook command 2 times" in out
    assert any(
        ln.startswith("[warn]") and ln.endswith(ours + "\\ud800")
        for ln in out.splitlines()
    ), out


# --- handlers Codex doesn't start ----------------------------------------------


def _toml_copy(command: str, matcher: str = "startup") -> str:
    """A config.toml SessionStart group that holds `command`."""
    return (
        f'[[hooks.SessionStart]]\nmatcher = "{matcher}"\n'
        "[[hooks.SessionStart.hooks]]\n"
        f'type = "command"\ncommand = {json.dumps(command)}\n'
    )


# config.toml settings and profile files that turn the `hooks` feature off
# for some launch, or leave doctor unable to tell: (config.toml text, profile
# files, the reason doctor gives).
_HOOKS_OFF: dict[str, tuple[str, dict[str, str], str]] = {
    "hooks false": (
        "[features]\nhooks = false\n", {},
        "[features] in {config} sets `hooks` to false",
    ),
    "codex_hooks false": (
        "[features]\ncodex_hooks = false\n", {},
        "[features] in {config} sets `codex_hooks` to false",
    ),
    "hooks false over codex_hooks true": (
        "[features]\ncodex_hooks = true\nhooks = false\n", {},
        "[features] in {config} sets `hooks` to false",
    ),
    "not true or false": (
        '[features]\nhooks = "off"\n', {},
        "`hooks` in [features] in {config} isn't true or false",
    ),
    "features not a table": (
        "features = 1\n", {},
        "[features] in {config} isn't a table",
    ),
    "a profile table": (
        "[profiles.work.features]\nhooks = false\n", {},
        "[profiles.work.features] in {config} sets `hooks` to false",
    ),
    "a profile file": (
        "", {"work.config.toml": "[features]\nhooks = false\n"},
        "[features] in the profile file {dir}/work.config.toml sets `hooks` "
        "to false",
    ),
    "a profile file's profile table": (
        "", {"work.config.toml": "[profiles.x.features]\ncodex_hooks = false\n"},
        "[profiles.x.features] in the profile file {dir}/work.config.toml "
        "sets `codex_hooks` to false",
    ),
    "an unreadable profile file": (
        "", {"work.config.toml": "[features\n"},
        "doctor can't read the profile file {dir}/work.config.toml, which "
        "Codex reads when you pass `--profile work`",
    ),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_HOOKS_OFF))
def test_doctor_counts_no_run_while_hooks_may_be_off(
    tmp_path: Path, shape: str,
) -> None:
    """Two copies of setup's command that only config.toml holds fail
    doctor, unless a setting can turn the `hooks` feature off. Then Codex
    may start neither, so doctor counts no run and warns about each."""
    codex_dir, hooks, ours = _installed(tmp_path)
    hooks.unlink()
    settings, files, reason = _HOOKS_OFF[shape]
    config = codex_dir / "config.toml"
    config.write_text(
        settings + _toml_copy(ours) + _toml_copy(ours, "startup|resume"),
        encoding="utf-8",
    )
    for name, text in files.items():
        (codex_dir / name).write_text(text, encoding="utf-8")
    why = reason.format(config=config, dir=codex_dir)
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out, out
    assert (
        f"[warn] doctor counts no duplicate run, because {why}, and Codex "
        "starts no hook of yours while its `hooks` feature is off\n"
    ) in out, out
    for g in (0, 1):
        assert (
            f"[warn] SessionStart handler at `hooks.SessionStart[{g}].hooks[0]`"
            f" in the [hooks] table of {config} names aelf-session-start-hook, "
            f"and doctor counts no run for it, because {why}. Handler: {ours}\n"
        ) in out, out


# Settings that leave the feature on for every launch, and a file that
# isn't a profile file.
_HOOKS_ON: dict[str, tuple[str, dict[str, str]]] = {
    "codex_hooks false under hooks true": (
        "[features]\ncodex_hooks = false\nhooks = true\n", {},
    ),
    "a profile that turns them on": (
        "[profiles.work.features]\nhooks = true\n",
        {"home.config.toml": "[features]\ncodex_hooks = true\n"},
    ),
    "not a profile name": ("", {"my work.config.toml": "[features\n"}),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_HOOKS_ON))
def test_doctor_counts_runs_while_hooks_stay_on(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, ours = _installed(tmp_path)
    hooks.unlink()
    settings, files = _HOOKS_ON[shape]
    (codex_dir / "config.toml").write_text(
        settings + _toml_copy(ours) + _toml_copy(ours, "startup|resume"),
        encoding="utf-8",
    )
    for name, text in files.items():
        (codex_dir / name).write_text(text, encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert (
        "[FAIL] SessionStart starts the aelf-session-start-hook command 2 "
        "times" in out
    ), out
    assert "doctor counts no duplicate run" not in out, out


@pytest.mark.timeout(60)
def test_doctor_gives_setup_handler_no_line_of_its_own_while_hooks_are_off(
    tmp_path: Path,
) -> None:
    """Setup's own handlers in hooks.json get the one summary line, and a
    copy in another group gets its own."""
    codex_dir, hooks, ours = _installed(tmp_path)
    (codex_dir / "config.toml").write_text(
        "[features]\nhooks = false\n", encoding="utf-8",
    )
    _edit_hooks(hooks, lambda d: d["hooks"]["SessionStart"].append(
        _session_start_copy(ours),
    ))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 1, out
    assert "the Codex `hooks` feature is disabled" in out, out
    assert "times" not in out, out
    lines = [ln for ln in out.splitlines() if "doctor counts no run for it" in ln]
    assert len(lines) == 1, out
    assert lines[0].startswith(
        "[warn] SessionStart handler at `hooks.SessionStart[1].hooks[0]` names "
        "aelf-session-start-hook",
    ), out


def _state(key: str, enabled: str = "false") -> str:
    return f"[hooks.state.{json.dumps(key)}]\nenabled = {enabled}\n"


# `[hooks.state]` keys for the copy of setup's SessionStart handler, and
# whether Codex may skip it: the file, as an absolute path; `{alias}` is a
# symlink to the Codex home. `{file}` is the file that holds the copy.
_STATE_KEYS: dict[str, tuple[str, bool]] = {
    "its key": ("{file}:session_start:{g}:0", True),
    "its key through a symlinked home": ("{alias}/{name}:session_start:{g}:0", True),
    "its key with spaces Codex trims": ("  {file}:session_start:{g}:0　", True),
    "another handler's key": ("{file}:session_start:{g}:1", False),
    "another event's key": ("{file}:session_end:{g}:0", False),
    "a relative path": ("{name}:session_start:{g}:0", False),
    "another file": ("{other}/{name}:session_start:{g}:0", False),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("where", ["hooks.json", "config.toml"])
@pytest.mark.parametrize("shape", sorted(_STATE_KEYS))
def test_doctor_counts_no_run_for_a_handler_hooks_state_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, where: str,
) -> None:
    """Codex skips a handler whose `[hooks.state]` entry sets `enabled =
    false`, so its copy of setup's command is no run. Codex's key holds an
    absolute path, so a relative one names no handler, even from the Codex
    home."""
    codex_dir, hooks, ours = _installed(tmp_path)
    monkeypatch.chdir(codex_dir)
    alias = tmp_path / "alias"
    alias.symlink_to(codex_dir, target_is_directory=True)
    other = tmp_path / "other"
    other.mkdir()
    config = codex_dir / "config.toml"
    if where == "hooks.json":
        _edit_hooks(hooks, lambda d: d["hooks"]["SessionStart"].append(
            _session_start_copy(ours),
        ))
        source, g, copy = hooks, 1, _toml_copy("/bin/true")
    else:
        source, g, copy = config, 0, _toml_copy(ours)
    template, skipped = _STATE_KEYS[shape]
    key = template.format(
        file=source, alias=alias, other=other, name=source.name, g=g,
    )
    config.write_text(copy + _state(key), encoding="utf-8")
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    at = f"hooks.SessionStart[{g}].hooks[0]"
    if skipped:
        assert rc == 0, out
        assert "times" not in out, out
        assert any(
            ln.startswith(f"[warn] SessionStart handler at `{at}`")
            and "names aelf-session-start-hook, and doctor counts no run for "
            f"it, because the [hooks.state] entry {key!r} in {config} sets "
            "`enabled = false` for it, and Codex skips a disabled handler. "
            f"Handler: {ours}" in ln
            for ln in out.splitlines()
        ), out
    else:
        assert rc == 1, out
        assert "starts the aelf-session-start-hook command 2 times" in out


@pytest.mark.timeout(60)
def test_doctor_reads_the_handler_index_of_a_hooks_state_key(
    tmp_path: Path,
) -> None:
    """The copy is the second handler of its group, so its key ends in 1."""
    codex_dir, _hooks, ours = _installed(tmp_path)
    config = codex_dir / "config.toml"
    config.write_text(
        _toml_copy("/bin/true")
        + f'[[hooks.SessionStart.hooks]]\ntype = "command"\n'
        f"command = {json.dumps(ours)}\n"
        + _state(f"{config}:session_start:0:1"),
        encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert "times" not in out, out
    assert "`hooks.SessionStart[0].hooks[1]`" in out, out


@pytest.mark.timeout(60)
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_doctor_reads_hooks_state_in_a_profile_file(
    tmp_path: Path, enabled: str,
) -> None:
    """A profile file's entry applies when `--profile` selects it, so
    doctor can't tell whether Codex starts the handler."""
    codex_dir, hooks, ours = _installed(tmp_path)
    config = codex_dir / "config.toml"
    config.write_text(_toml_copy(ours), encoding="utf-8")
    profile = codex_dir / "work.config.toml"
    profile.write_text(
        _state(f"{config}:session_start:0:0", enabled), encoding="utf-8",
    )
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    if enabled == "true":
        assert rc == 1, out
        assert "starts the aelf-session-start-hook command 2 times" in out
        return
    assert rc == 0, out
    assert (
        f"in the profile file {profile}, which Codex reads when you pass "
        "`--profile work` sets `enabled = false` for it"
    ) in out, out


# --- names a shell spells ------------------------------------------------------

# Ways to write `aelf-session-start-hook` that sh, bash, and zsh turn back
# into the name by quote removal and line joining, and cmd's escape. `{p}`
# is setup's command, which holds the name.
_SPELLED = {
    "empty quotes": ("aelf-session", "aelf-sess''ion"),
    "double quotes": ("aelf-session-start-hook", 'aelf-"session"-start-hook'),
    "a backslash": ("aelf-session", "aelf-sess\\ion"),
    "a line continuation": ("start-hook", "start-ho\\\nok"),
    "a backtick pair": ("aelf-session", "aelf-sess``ion"),
    "cmd's caret": ("aelf-session", "aelf-sess^ion"),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_SPELLED))
def test_doctor_warns_about_a_name_spelled_with_shell_quoting(
    tmp_path: Path, shape: str,
) -> None:
    codex_dir, hooks, ours = _installed(tmp_path)
    old, new = _SPELLED[shape]
    spelled = ours.replace(old, new)
    assert "aelf-session-start-hook" not in spelled
    _edit_hooks(hooks, lambda d: d["hooks"]["SessionStart"].append(
        _session_start_copy(spelled),
    ))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert any(
        ln.startswith("[warn] SessionStart handler may also run "
                      "aelf-session-start-hook.")
        for ln in out.splitlines()
    ), out
    assert spelled.splitlines()[-1] in out, out


@pytest.mark.timeout(60)
def test_doctor_warns_about_a_spelled_name_outside_a_handler(
    tmp_path: Path,
) -> None:
    codex_dir, hooks, _ours = _installed(tmp_path)
    _edit_hooks(hooks, lambda d: d.__setitem__(
        "description", "runs aelf-st''op-hook",
    ))
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] The string at `description` in {hooks} names aelf-stop-hook"
    ) in out, out


# --- config.toml beyond [hooks] ----------------------------------------------

# config.toml keys outside [hooks] that start a program, and one that
# doesn't, each with the path doctor names and the program.
_CONFIG_STRINGS = {
    "notify": ('notify = ["/opt/bin/aelf-stop-hook"]\n', "notify[0]", "aelf-stop-hook"),
    "a profile's notify": (
        '[profiles.x]\nnotify = ["aelf-transcript-logger"]\n',
        "profiles.x.notify[0]", "aelf-transcript-logger",
    ),
    "an MCP server command": (
        '[mcp_servers.mem]\ncommand = "aelf-hook"\n',
        "mcp_servers.mem.command", "aelf-hook",
    ),
    "an MCP server argument": (
        '[mcp_servers.mem]\ncommand = "uvx"\nargs = ["--from", "x", "aelf-search-tool-hook"]\n',
        "mcp_servers.mem.args[2]", "aelf-search-tool-hook",
    ),
    "any key": ('[tui]\nfoo = "aelf-commit-ingest"\n', "tui.foo", "aelf-commit-ingest"),
    "a spelled name": ('notify = ["aelf-st\'\'op-hook"]\n', "notify[0]", "aelf-stop-hook"),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", sorted(_CONFIG_STRINGS))
@pytest.mark.parametrize("where", ["config.toml", "a profile file"])
def test_doctor_warns_about_a_program_name_anywhere_in_config_toml(
    tmp_path: Path, shape: str, where: str,
) -> None:
    codex_dir, _hooks, _ours = _installed(tmp_path)
    text, path, key = _CONFIG_STRINGS[shape]
    file = codex_dir / (
        "config.toml" if where == "config.toml" else "work.config.toml"
    )
    file.write_text(text, encoding="utf-8")
    label = str(file) if where == "config.toml" else f"the profile file {file}"
    rc, out = _doctor(codex_dir, tmp_path / "skills")
    assert rc == 0, out
    assert (
        f"[warn] The string at `{path}` in {label} names {key}, outside the "
        "handlers `aelf setup --host codex` writes"
    ) in out, out


