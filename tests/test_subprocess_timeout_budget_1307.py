"""#1307: a subprocess-driven test must carry its own wall-clock budget.

`pyproject.toml` sets `timeout = 5` for the whole suite, sized for unit
and property tests. A test that spawns a child process is not that: it
pays interpreter startup, a `uv` resolve, and whatever the child does,
and under CI contention that routinely exceeds five seconds. When it
does, `pytest-timeout` reports a **timeout** — which reads as a hang, not
as slowness — and the next session spends its time hunting a deadlock
that is not there. That is exactly how the gate failure this issue was
filed from was diagnosed twice, wrongly.

The fix is per-test: `@pytest.mark.timeout(N)` on any test that reaches a
subprocess. The rule decays unless something enforces it, so this module
is the enforcement.

Why an AST walk rather than a grep
----------------------------------
`tests/test_test_termination_policy.py` carries `subprocess.run(['x'])`
inside `parametrize` **string literals** while spawning nothing at all. A
grep-based detector flags it and every future module that documents the
pattern it forbids. The walk below only counts a *call node*, so a
string that merely looks like one is invisible to it.

Transitive by design: most subprocess-driven tests never name
`subprocess` themselves — they call a module-level `_run(...)` /
`_git(...)` helper. A detector that only looked at the test function's
own body would report almost nothing and pass vacuously, which is the
failure mode #1319's guard hit from the other direction. So this
resolves helper calls within the module, to a fixed point.

Scope, stated rather than implied: this is a *static* check over
`tests/`. It cannot see a helper imported from another module, nor a
subprocess reached through a fixture. Both are recorded in
`_KNOWN_LIMITS` rather than left for a reader to discover.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent

_SUBPROCESS_CALLS = frozenset(
    {"run", "Popen", "check_output", "check_call", "call"}
)
"""`subprocess` entry points that start a child process."""

_KNOWN_LIMITS = """\
Not covered by this check: a helper imported from another test module, a
subprocess reached through a fixture, and `os.system` / `os.popen` (which
`tests/test_test_termination_policy.py` bans outright, so they cannot
appear). A test using one of those needs the marker on judgement.\
"""

_ALLOWLIST: frozenset[str] = frozenset()
"""Tests deliberately left on the suite default.

Empty, and that is the point: an exception recorded here is a decision
someone made, whereas an exception that merely never got flagged is an
accident. Add an entry as `"<relpath>::<test name>"` with a comment
saying why the default budget is right for it.
"""


def _module_subprocess_alias(tree: ast.Module) -> set[str]:
    """Names that refer to the `subprocess` module in this file."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "subprocess":
                    aliases.add(a.asname or "subprocess")
    return aliases


def _direct_spawners(tree: ast.Module, aliases: set[str]) -> set[str]:
    """Function names whose own body starts a child process."""
    out: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # `subprocess.run(...)` / `sp.run(...)`
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in aliases
                and f.attr in _SUBPROCESS_CALLS
            ):
                out.add(fn.name)
                break
            # `run(...)` after `from subprocess import run` — banned by
            # test_test_termination_policy, checked here so this module
            # does not silently depend on that one still existing.
            if isinstance(f, ast.Name) and f.id in _SUBPROCESS_CALLS:
                out.add(fn.name)
                break
    return out


def _call_graph(tree: ast.Module) -> dict[str, set[str]]:
    """`{function name: names it calls}`, within this module."""
    graph: dict[str, set[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        callees: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callees.add(node.func.id)
        graph[fn.name] = callees
    return graph


def _reaches_subprocess(tree: ast.Module) -> set[str]:
    """Functions that spawn, directly or through a same-module helper.

    Fixed-point closure, so a two-hop helper chain is caught. Terminates:
    the reachable set only grows and is bounded by the function count.
    """
    aliases = _module_subprocess_alias(tree)
    reaching = _direct_spawners(tree, aliases)
    graph = _call_graph(tree)
    changed = True
    while changed:
        changed = False
        for name, callees in graph.items():
            if name not in reaching and callees & reaching:
                reaching.add(name)
                changed = True
    return reaching


def _has_timeout_marker(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "timeout"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            return True
    return False


def unbudgeted_subprocess_tests() -> list[str]:
    """`["<relpath>::<test>", ...]` for spawners with no explicit budget."""
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - caught by collection
            continue
        reaching = _reaches_subprocess(tree)
        if not reaching:
            continue
        rel = path.relative_to(TESTS_ROOT.parent)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.name.startswith("test_"):
                continue
            if fn.name not in reaching:
                continue
            if _has_timeout_marker(fn):
                continue
            ident = f"{rel}::{fn.name}"
            if ident in _ALLOWLIST:
                continue
            offenders.append(ident)
    return offenders


@pytest.mark.timeout(60)
def test_every_subprocess_test_has_its_own_budget() -> None:
    """A spawning test must not run on the 5s unit-test default.

    Terminates by construction: a bounded walk over a finite file list,
    pure computation, no subprocess and no waiting of its own.
    """
    offenders = unbudgeted_subprocess_tests()
    assert offenders == [], (
        f"{len(offenders)} test(s) spawn a subprocess on the suite's 5s "
        "default, so contention reports as a hang rather than as "
        "slowness (#1307). Add @pytest.mark.timeout(N):\n  "
        + "\n  ".join(offenders)
        + f"\n\n{_KNOWN_LIMITS}"
    )


def test_the_detector_sees_through_a_helper() -> None:
    """Transitivity is the whole check; assert it rather than assume it.

    Without the fixed-point closure this module reports almost nothing,
    because subprocess-driven tests overwhelmingly call a `_run` helper
    rather than naming `subprocess` in their own body — a vacuous pass
    that looks identical to a clean tree.
    """
    tree = ast.parse(
        "import subprocess\n"
        "def _inner():\n"
        "    subprocess.run(['x'])\n"
        "def _outer():\n"
        "    _inner()\n"
        "def test_leaf():\n"
        "    _outer()\n"
    )
    assert _reaches_subprocess(tree) >= {"_inner", "_outer", "test_leaf"}


def test_the_detector_ignores_a_string_that_looks_like_a_call() -> None:
    """A grep would flag this; the AST walk must not.

    `tests/test_test_termination_policy.py` really does carry
    `subprocess.run(...)` inside parametrize literals while spawning
    nothing, so this is the live case, not a hypothetical one.
    """
    tree = ast.parse(
        "import pytest\n"
        "@pytest.mark.parametrize('src', [\"subprocess.run(['x'])\"])\n"
        "def test_leaf(src):\n"
        "    assert src\n"
    )
    assert _reaches_subprocess(tree) == set()


def test_an_aliased_import_is_still_seen() -> None:
    """`import subprocess as sp` must not launder a spawn past the check."""
    tree = ast.parse(
        "import subprocess as sp\n"
        "def test_leaf():\n"
        "    sp.run(['x'])\n"
    )
    assert _reaches_subprocess(tree) == {"test_leaf"}


def test_a_marker_on_the_test_satisfies_the_rule() -> None:
    """The marker is what clears a test — presence, not value."""
    tree = ast.parse(
        "import pytest\n"
        "import subprocess\n"
        "@pytest.mark.timeout(30)\n"
        "def test_leaf():\n"
        "    subprocess.run(['x'])\n"
    )
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    )
    assert _has_timeout_marker(fn)
