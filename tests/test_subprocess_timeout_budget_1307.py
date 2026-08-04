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
subprocess reached through a fixture, and `os.system` / `os.popen`, which
no gate in this repo bans -- `tests/test_test_termination_policy.py`
checks that blocking calls are bounded, not that these are absent. A test
using one of those needs the marker on judgement.\
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


def _is_timeout_mark(node: ast.expr) -> bool:
    """`pytest.mark.timeout` / `pytest.mark.timeout(N)`, either spelling."""
    node = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "timeout"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
    )


def _module_has_timeout_pytestmark(tree: ast.Module) -> bool:
    """A module-level `pytestmark` budget covering every test in the file.

    `pytest-timeout` resolves the budget with `item.get_closest_marker`, so
    a module `pytestmark` is a real budget and a decorator on the test
    *overrides* it. Reading only `decorator_list` therefore reports an
    already-budgeted test as an offender, and the only way to satisfy the
    report is to add a decorator that replaces the module's value with
    this rule's -- which is how three `tests/e2e/` tests briefly went from
    120s to 30s, under their own children's 60s `subprocess` timeouts.
    """
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets
        ):
            continue
        value = node.value
        if value is None:
            continue
        marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        if any(_is_timeout_mark(m) for m in marks):
            return True
    return False


def _class_marked_tests(tree: ast.Module) -> set[str]:
    """Test names under a `class Test...` carrying a timeout marker."""
    marked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_timeout_mark(d) for d in node.decorator_list):
            continue
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                marked.add(child.name)
    return marked


def _has_timeout_marker(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(_is_timeout_mark(dec) for dec in fn.decorator_list)


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
        if _module_has_timeout_pytestmark(tree):
            continue
        class_marked = _class_marked_tests(tree)
        rel = path.relative_to(TESTS_ROOT.parent)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.name.startswith("test_"):
                continue
            if fn.name not in reaching:
                continue
            if _has_timeout_marker(fn) or fn.name in class_marked:
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


def test_a_module_pytestmark_is_a_budget() -> None:
    """A module `pytestmark` budget must satisfy the rule, not trip it.

    `pytest-timeout` resolves with `item.get_closest_marker`, so a
    decorator on the test *replaces* the module value rather than adding
    to it. A gate that cannot see `pytestmark` reports an already-budgeted
    test as an offender, and the only way to clear the report is to write
    a decorator that lowers the budget to this rule's number. That is not
    hypothetical: it took three `tests/e2e/` tests from a deliberate 120s
    to 30s, under the 60s `subprocess` timeouts of their own children.
    """
    tree = ast.parse(
        "import pytest\n"
        "import subprocess\n"
        "pytestmark = pytest.mark.timeout(120)\n"
        "def test_leaf():\n"
        "    subprocess.run(['x'])\n"
    )
    assert _module_has_timeout_pytestmark(tree)
    # The list spelling is the other live form.
    listed = ast.parse(
        "import pytest\n"
        "pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120)]\n"
    )
    assert _module_has_timeout_pytestmark(listed)
    # A `pytestmark` that is not a timeout must not clear the rule.
    other = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.regression\n"
    )
    assert not _module_has_timeout_pytestmark(other)


def test_a_class_marker_is_a_budget() -> None:
    """Same reasoning one scope in: a marked `class Test...` covers its
    methods, so demanding a per-method decorator would lower those too."""
    tree = ast.parse(
        "import pytest\n"
        "import subprocess\n"
        "@pytest.mark.timeout(120)\n"
        "class TestThing:\n"
        "    def test_leaf(self):\n"
        "        subprocess.run(['x'])\n"
    )
    assert _class_marked_tests(tree) == {"test_leaf"}
    unmarked = ast.parse(
        "import pytest\n"
        "class TestThing:\n"
        "    def test_leaf(self):\n"
        "        pass\n"
    )
    assert _class_marked_tests(unmarked) == set()


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
