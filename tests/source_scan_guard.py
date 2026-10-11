"""#1746: a test that reads `src/aelfrice` as text carries `source_scan`.

mutmut runs the suite inside `mutants/`, where every file under
`src/aelfrice` is replaced by mutmut's mutated copy: each function copied
once per mutant, behind a trampoline. A test that reads the source as text
(an AST scan, a grep, `inspect.getsource`) reads that copy instead. It finds
hundreds of call sites where there were six, or a mutant that dropped the
`encoding=` it pins, and fails. mutmut runs the suite once unmutated before
any mutant and stops at the first failure, so one such test leaves every
mutant `not checked`.

The mutation run leaves these tests out with `-m "not source_scan"` (in
`[tool.mutmut]` in `pyproject.toml`). Nothing is lost by it: a test that
reads source text can only kill a mutant by noticing that the text changed,
not by noticing a change in behaviour.

This module is what keeps the marker honest. It audits every `open` of a
`.py` file under `src/aelfrice` while a test's setup, call, or teardown
runs, and fails that phase when the test does not carry the marker. It
also audits the import of each test module, where module-level code runs,
and fails collection when the module isn't marked as a whole.

It ignores the reads that are not a test reading the source: an import
(the loader reading the module, and the module's own code running as it is
imported), traceback, warning, and stack rendering reading
lines to print them, `runpy` executing a file, and Hypothesis sampling
constants from newly imported modules. `linecache` is cleared before each
phase, so a test that calls `inspect.getsource` is caught whether or not
another test read the same file first. The `functools` caches of the test
module, and of its classes' static and class methods, are cleared before
each test for the same reason.

What it can't see:

- a read in a subprocess;
- a scan memoised by hand, such as a module-level dict filled on first use;
- a scan cached outside the test's module and classes, such as in a
  module- or session-scoped fixture or a shared helper module.

A test that only reuses one of those caches is flagged only if it happens
to be the one that fills it, so whether it is flagged depends on test order.

`tests/conftest.py` imports the hooks below, which is what installs them.
"""
from __future__ import annotations

import functools
import inspect
import linecache
import os
import sys
from collections.abc import Generator
from pathlib import Path
from types import FrameType, ModuleType
from typing import Final, cast

import pytest

MARKER: Final[str] = "source_scan"

_REPO: Final[Path] = Path(__file__).resolve().parents[1]

#: The tree mutmut mutates, resolved, with a trailing separator.
SOURCE_ROOT: Final[str] = str(_REPO / "src" / "aelfrice") + os.sep

#: Where the reading test's own frame lives.
TESTS_ROOT: Final[str] = str(_REPO / "tests") + os.sep

#: Modules whose reads of a source file are not a test reading the source:
#: pytest rendering its own reports, traceback and warning display reading
#: lines to print them, `runpy` executing a file, and Hypothesis collecting
#: constants from each newly imported module to draw examples from. The
#: last one matters: it reads at random, so counting it made the verdict
#: depend on test order and marked property tests that read nothing.
_IGNORED_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "_pytest", "pluggy", "traceback", "warnings", "runpy", "hypothesis",
)

#: The import system reads a module's source in its loader's `get_code`,
#: and runs the module's own top-level code under `exec_module`. Both are an
#: import: `src/aelfrice/core_gate.py`, for one, digests its own source when
#: it is imported, and that read belongs to whichever test imports it first.
#: `loader.get_source`, `loader.get_data`, and `pkgutil.get_data` go through
#: the same loader without either frame, to hand the caller the text.
_IMPORT_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "importlib", "_frozen_importlib",
)
_IMPORT_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"get_code", "exec_module", "_load_unlocked"},
)

#: `inspect` functions that read source lines to render a stack, not to
#: hand a test the source: `inspect.stack()` and `inspect.trace()` reach
#: `findsource` through `getframeinfo`.
_IGNORED_INSPECT_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"getframeinfo", "stack", "trace", "getouterframes", "getinnerframes"},
)


class Audit:
    """Source files read during the phase being audited."""

    installed: bool = False
    active: bool = False
    reads: list[str] = []


def _is_source(path: object) -> str | None:
    """`path` resolved, when it names a `.py` file under `SOURCE_ROOT`."""
    # `open` audits what it was given: a str, bytes, a path-like object, or
    # a file descriptor, which names no path and is skipped.
    if isinstance(path, (str, bytes, os.PathLike)):
        text = os.fsdecode(cast("str | bytes | os.PathLike[str]", path))
    else:
        return None
    if not text.endswith(".py"):
        return None
    resolved = os.path.realpath(text)
    return resolved if resolved.startswith(SOURCE_ROOT) else None


@functools.lru_cache(maxsize=4096)
def _in_tests(filename: str) -> bool:
    return os.path.realpath(filename).startswith(TESTS_ROOT)


def _rendering(frame: FrameType | None) -> bool:
    """True when the read is not a test reading the source.

    Walks out from the `open` call to the first frame in `tests/`, which is
    the test, fixture, or helper that the read belongs to. On the way, an
    import (a loader's `get_code`) or a frame that renders, executes, or
    samples source means the read is one of those. Every test runs below
    pytest's own frames, so the walk has to stop at the test: reaching a
    pytest frame first means pytest itself read the file. A read with no
    test frame at all, such as one from a thread the test started, isn't
    the test's either.
    """
    while frame is not None:
        if _in_tests(frame.f_code.co_filename):
            return False
        module = frame.f_globals.get("__name__", "")
        if isinstance(module, str):
            if module.startswith(_IGNORED_MODULE_PREFIXES):
                return True
            if module.startswith(_IMPORT_MODULE_PREFIXES) and frame.f_code.co_name in _IMPORT_FUNCTIONS:
                return True
            if module == "inspect" and frame.f_code.co_name in _IGNORED_INSPECT_FUNCTIONS:
                return True
        frame = frame.f_back
    return True


def _hook(event: str, args: tuple[object, ...]) -> None:
    if event != "open" or not Audit.active or not args:
        return
    path = _is_source(args[0])
    if path is None:
        return
    here = inspect.currentframe()
    # The frame that called `open`, one above this hook.
    if not _rendering(here.f_back if here is not None else None):
        Audit.reads.append(path)


def install() -> None:
    """Add the audit hook once; an audit hook cannot be removed."""
    if not Audit.installed:
        sys.addaudithook(_hook)
        Audit.installed = True


def _names(reads: list[str]) -> str:
    names = ", ".join(str(Path(p).relative_to(_REPO)) for p in reads[:3])
    if len(reads) > 3:
        names += f" and {len(reads) - 3} more"
    return names


def audited(item: pytest.Item, phase: str) -> Generator[None, object, object]:
    """One test phase, failed when it read the source without the marker."""
    linecache.clearcache()
    Audit.reads = []
    Audit.active = True
    try:
        result = yield
    finally:
        Audit.active = False
    reads = sorted(set(Audit.reads))
    if reads and item.get_closest_marker(MARKER) is None:
        pytest.fail(
            f"this test's {phase} read {_names(reads)} as text but the test is not "
            f"marked `@pytest.mark.{MARKER}`. Inside mutmut's `mutants/` "
            f"tree that file is mutmut's mutated copy, so the test fails and "
            f"stops the mutation run (#1746). Mark the test, or its module "
            f"with `pytestmark = pytest.mark.{MARKER}`.",
            pytrace=False,
        )
    return result


def clear_module_caches(item: pytest.Item) -> None:
    """Empty every `functools` cache defined in the test's own module.

    A module that caches its source scan reads the source only in the
    first test that asks, and every later test reuses the result unseen.
    Inside `mutants/` each of those later tests reads the mutated copy
    when it runs alone, which is how mutmut runs it after `-m` has left the
    first one out. Clearing the cache before each test makes every test
    that uses it read the source itself, where the guard sees it.
    """
    module = getattr(item, "module", None)
    if not isinstance(module, ModuleType):
        return
    for value in list(vars(module).values()):
        if getattr(value, "__module__", None) != module.__name__:
            continue
        _clear(value)
        if isinstance(value, type):
            # A cached static or class method of a test class: the
            # descriptor holds the cached function as `__func__`.
            for member in list(vars(value).values()):
                _clear(getattr(member, "__func__", member))


def _clear(value: object) -> None:
    clear = getattr(value, "cache_clear", None)
    if callable(clear):
        clear()


def _module_marked(module: object) -> bool:
    """True when `module` sets `pytestmark` to, or to a list holding, the marker."""
    marks = getattr(module, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    return any(getattr(m, "name", None) == MARKER for m in cast("list[object]", marks))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Audit a test module's import, where module-level code runs.

    A module-level constant computed from the source is read once, at
    collection, and every test that uses it would pass the per-test audit.
    Such a module has to mark itself as a whole.
    """
    if not isinstance(collector, pytest.Module):
        return (yield)
    Audit.reads = []
    Audit.active = True
    try:
        report = yield
    finally:
        Audit.active = False
    if report.passed:
        failure = collection_failure(collector.nodeid, collector.obj, Audit.reads)
        if failure is not None:
            report.outcome = "failed"
            report.longrepr = failure
    return report


def collection_failure(nodeid: str, module: object, reads: list[str]) -> str | None:
    """Why importing a test module fails the guard, or None when it doesn't."""
    if not reads or _module_marked(module):
        return None
    return (
        f"importing {nodeid} read {_names(sorted(set(reads)))} as text, but "
        f"the module doesn't set `pytestmark = pytest.mark.{MARKER}`. Code "
        "that runs at import is shared by every test in the module, so the "
        "module has to be marked as a whole (#1746)."
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    clear_module_caches(item)
    return (yield from audited(item, "setup"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from audited(item, "call"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from audited(item, "teardown"))
