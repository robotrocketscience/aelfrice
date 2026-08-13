"""The perf gates must be reachable, and their own budgets must decide.

Two independent ways the latency benchmarks were unreachable before
#1160, both of which this module pins:

1. **The opt-in did not exist.** Four modules gate a latency assertion
   on a `_has_run_perf` helper reading `--run-perf`, but nothing
   registered that option — `test_bm25_index.py` carried a
   `pytest_addoption` stub, and pytest only calls that hook from
   `conftest.py` or an installed plugin, never from a plain test
   module. `pytest --run-perf` failed with `unrecognized arguments`
   (exit 4), so the tests were reachable only by editing the guard.

2. **The harness outranked the assertion.** `pyproject.toml` then set
   a global `timeout = 5` sized for unit tests. Every one of these
   tests asserts its own, larger wall-clock budget —
   `test_eigsolve_under_budget_n10k` allows 30s and measures 6.76s — so
   once the flag worked, pytest-timeout killed it at 5.0s and the test
   could never pass. A per-test `@pytest.mark.timeout` override
   restores the assertion as the judge.

   The base is 30 now (#1472), which does not retire the override or
   this test: the assertion below is `>`, not `>= 5`, so it holds
   against whatever the base becomes. That is the point — a guard
   written against the literal 5 would have gone green on the raise
   while the tests it protects went back to being decided by the
   harness.

Fixing only (1) ships a guaranteed-failing opt-in, which is why the
second test here is not merely cosmetic.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "tests"
_GUARD_HELPER = "_has_run_perf"


def _global_timeout() -> int:
    with (_REPO / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    timeout = config["tool"]["pytest"]["ini_options"]["timeout"]
    return int(timeout)


def _timeout_override(func: ast.FunctionDef, consts: dict[str, int]) -> int | None:
    """Resolve `@pytest.mark.timeout(N)` on `func`, if present."""
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if not (isinstance(target, ast.Attribute) and target.attr == "timeout"):
            continue
        if not decorator.args:
            return None
        arg = decorator.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
            return int(arg.value)
        if isinstance(arg, ast.Name):
            return consts.get(arg.id)
    return None


def _perf_gated_tests() -> list[tuple[str, str, int | None]]:
    """(file, test name, resolved timeout override) for every perf gate."""
    found: list[tuple[str, str, int | None]] = []
    for path in sorted(_TESTS.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if _GUARD_HELPER not in text:
            continue
        tree = ast.parse(text)
        consts = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            calls_guard = any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == _GUARD_HELPER
                for inner in ast.walk(node)
            )
            if calls_guard:
                found.append(
                    (path.name, node.name, _timeout_override(node, consts))
                )
    return found


def test_run_perf_option_is_registered(request: pytest.FixtureRequest) -> None:
    """Behavioural, not source-level: `getoption` raises if unregistered."""
    assert request.config.getoption("--run-perf") in (True, False)


def test_the_perf_gate_scan_is_not_vacuous() -> None:
    """Guard the guard: an empty scan would satisfy the test below."""
    gated = _perf_gated_tests()
    assert len(gated) >= 5, (
        f"expected at least the 5 known perf gates, found {len(gated)}: "
        f"{gated} — the scan has stopped matching and the budget assertion "
        f"below is now vacuous"
    )


def test_every_perf_test_outranks_the_global_timeout() -> None:
    """Each latency assertion, not the harness, must decide the outcome."""
    limit = _global_timeout()
    offenders = [
        (path, name, override)
        for path, name, override in _perf_gated_tests()
        if override is None or override <= limit
    ]
    assert not offenders, (
        f"these perf-gated tests are not exempt from the global "
        f"`timeout = {limit}` in pyproject.toml, so pytest-timeout can kill "
        f"them before their own wall-clock assertion is evaluated: "
        f"{offenders}. Add `@pytest.mark.timeout(N)` with N comfortably "
        f"above the budget the test asserts."
    )
