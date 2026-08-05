"""#1340: `bool` is a subclass of `int`, and TOML has a `true` literal.

`[hook_audit] max_bytes = true` passed a bare `isinstance(..., int)` and
`True <= 0` is `False`, so the value cleared both arms of the validation,
no "expected positive int" trace printed, and the config carried a cap of
`True` -- one byte. Every append then exceeded it and the audit log
rotated on every write, holding one record.

The one-line guard is the smaller half. The durable half is the census
below: `hook_audit` was the *last* TOML-backed numeric knob missing the
guard, and it was missing precisely because nothing checked. A convention
six modules follow and a seventh does not is not a convention, it is a
coin flip that landed well six times.
"""
from __future__ import annotations

import ast
import io
import pathlib
from typing import Any

import pytest

from aelfrice.hook_audit import AUDIT_DEFAULT_MAX_BYTES, load_hook_audit_config

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "aelfrice"


# ---------------------------------------------------------------------------
# The reported defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ["true", "false"])
def test_a_bool_max_bytes_is_rejected_with_a_trace(
    tmp_path: pathlib.Path, literal: str
) -> None:
    """Assert the trace as well as the value.

    The fallback value is reachable two ways -- a rejected value and an
    explicitly-written default -- so an equality-only assertion passes
    just as happily when the rejection never happened. The trace is what
    distinguishes them.
    """
    (tmp_path / ".aelfrice.toml").write_text(
        f"[hook_audit]\nmax_bytes = {literal}\n"
    )
    err = io.StringIO()

    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=err)

    assert cfg.max_bytes == AUDIT_DEFAULT_MAX_BYTES
    assert not isinstance(cfg.max_bytes, bool), (
        "a bool survived into the config: `max_bytes=True` is a 1-byte "
        "cap, so the audit log rotates on every append"
    )
    assert "expected positive int" in err.getvalue(), (
        f"rejected silently; the user gets the default and no reason: "
        f"{err.getvalue()!r}"
    )


def test_a_real_int_still_wins(tmp_path: pathlib.Path) -> None:
    """The distinguishing arm. Rejecting everything would satisfy the
    test above, so pin that a legitimate value still gets through and
    prints nothing."""
    (tmp_path / ".aelfrice.toml").write_text("[hook_audit]\nmax_bytes = 5000\n")
    err = io.StringIO()

    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=err)

    assert cfg.max_bytes == 5000
    assert err.getvalue() == ""


def test_the_explicit_default_stays_silent(tmp_path: pathlib.Path) -> None:
    """Writing the default explicitly is not a mistake and must not warn.
    This is the branch the added `bool` clause sits inside, so without it
    the guard could be satisfied by warning on everything."""
    (tmp_path / ".aelfrice.toml").write_text(
        f"[hook_audit]\nmax_bytes = {AUDIT_DEFAULT_MAX_BYTES}\n"
    )
    err = io.StringIO()

    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=err)

    assert cfg.max_bytes == AUDIT_DEFAULT_MAX_BYTES
    assert err.getvalue() == ""


# ---------------------------------------------------------------------------
# The census — closes the class rather than the instance
# ---------------------------------------------------------------------------


def _numeric_type_names(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    return {e.id for e in getattr(node, "elts", []) if isinstance(e, ast.Name)}


def _guards_bool_in_scope(scope: ast.AST, var: str) -> bool:
    """Whether anything in the enclosing function excludes `bool` for `var`.

    Scope is the whole function, not the one statement: `cadence` and
    `retrieval` both guard with a *preceding* early return rather than a
    sibling clause, and a statement-local check reports those as offenders.
    Both spellings are legitimate; only the absence is a defect.

    Stated limit, measured rather than assumed: because the check is
    scope-wide, a function that mentions `bool` *somewhere* satisfies it
    even if the load-bearing clause loses it. Deleting the `bool` clause
    from `hook_audit`'s outer condition leaves the inner one and the
    census stays green -- `test_a_bool_max_bytes_is_rejected_with_a_trace`
    is what goes red there. The census catches a *new* reader written with
    no bool handling at all, which is the shape #1340 actually shipped in;
    it is not a substitute for a behaviour test on each knob.
    """
    for node in ast.walk(scope):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            continue
        if ast.unparse(node.args[0]) != var:
            continue
        if "bool" in _numeric_type_names(node.args[1]):
            return True
    return False


def _toml_numeric_validations_admitting_bool() -> list[str]:
    """`["<file>:<line> <var>", ...]` for TOML readers that accept a bool.

    A validation counts as TOML-reachable when its enclosing function
    either parses TOML itself or reads out of a parsed `section` mapping.
    That is deliberately narrower than "every isinstance in the package":
    `session_ring`, `hook`'s ring readers, `telemetry` and `hook_tail`
    validate values they wrote themselves into JSON state, where `true`
    cannot appear in an int field unless aelfrice put it there -- a
    different defect with a different fix, and out of scope here.
    """
    offenders: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(fn)
            if "tomllib" not in body and "section" not in body:
                continue
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance"
                    and len(node.args) == 2
                ):
                    continue
                types = _numeric_type_names(node.args[1])
                if "int" not in types or "bool" in types:
                    continue
                var = ast.unparse(node.args[0])
                if _guards_bool_in_scope(fn, var):
                    continue
                rel = path.relative_to(SRC_ROOT.parent)
                offenders.append(f"{rel}:{node.lineno} {var}")
    return sorted(set(offenders))


def test_no_toml_numeric_knob_accepts_a_bool() -> None:
    """The convention, asserted instead of assumed.

    Six modules excluded `bool` and the seventh did not, which is how
    #1340 shipped. Keyed on the shape of the validation rather than a
    module list, because a list documents the modules that existed when
    it was written.
    """
    offenders = _toml_numeric_validations_admitting_bool()
    assert offenders == [], (
        "these read a numeric knob out of `.aelfrice.toml` and accept a "
        "bool, because `bool` subclasses `int` (#1340). TOML has a `true` "
        "literal, so the value arrives, passes, and silently becomes 1 or "
        "0:\n  " + "\n  ".join(offenders)
    )


def test_the_census_actually_scans_something() -> None:
    """A scan that finds nothing satisfies the assertion above.

    Without this, a bad glob or a renamed helper turns the guard green and
    it reads exactly like a clean tree -- which is the failure mode #1340
    is an instance of.
    """
    scanned = 0
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.unparse(fn)
                if "tomllib" in body or "section" in body:
                    scanned += 1
    assert scanned >= 20, (
        f"only {scanned} TOML-reading functions found; the scan is not "
        "reaching the package and the guard above is vacuous"
    )


def test_the_census_sees_an_unguarded_reader() -> None:
    """Mutation arm: the detector must flag the shape it exists to flag.

    Asserting `offenders == []` on the real tree proves nothing about the
    detector -- an always-empty predicate passes it. This drives the
    predicate over a synthetic reader with the #1340 shape.
    """
    tree = ast.parse(
        "import tomllib\n"
        "def load(section, default):\n"
        "    raw = section.get('n', default)\n"
        "    if not isinstance(raw, int) or raw <= 0:\n"
        "        raw = default\n"
        "    return raw\n"
    )
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    )
    assert not _guards_bool_in_scope(fn, "raw")

    guarded = ast.parse(
        "import tomllib\n"
        "def load(section, default):\n"
        "    raw = section.get('n', default)\n"
        "    if isinstance(raw, bool) or not isinstance(raw, int):\n"
        "        raw = default\n"
        "    return raw\n"
    )
    gfn = next(
        n for n in ast.walk(guarded) if isinstance(n, ast.FunctionDef)
    )
    assert _guards_bool_in_scope(gfn, "raw")


def test_a_preceding_early_return_counts_as_a_guard() -> None:
    """`cadence` and `retrieval` guard with an early return rather than a
    sibling clause. Both are correct; a statement-local check would report
    them as offenders and the census would be noise."""
    tree = ast.parse(
        "import tomllib\n"
        "def load(section):\n"
        "    value = section['k']\n"
        "    if isinstance(value, bool):\n"
        "        return None\n"
        "    if isinstance(value, (int, float)):\n"
        "        return float(value)\n"
        "    return None\n"
    )
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert _guards_bool_in_scope(fn, "value")


def test_state_readers_are_out_of_scope_on_purpose() -> None:
    """Records the disposition the issue asked for.

    `session_ring` validates values aelfrice itself wrote into JSON state,
    not values a user typed into `.aelfrice.toml`. A `true` in an int
    field there means aelfrice wrote one, which is a different defect with
    a different fix. It is named here so "not covered" is a decision on
    the record rather than an oversight, and so the census is not quietly
    widened later without one.
    """
    offenders = _toml_numeric_validations_admitting_bool()
    assert not any("session_ring" in o for o in offenders)
    assert not any("telemetry" in o for o in offenders)


def _unused(_: Any) -> None:  # pragma: no cover - typing shim
    return None
