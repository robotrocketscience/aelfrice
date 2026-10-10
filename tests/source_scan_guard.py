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
ignores the reads that are not a test reading the source: the import
system loading a module, and traceback or stack rendering reading lines to
print them. `linecache` is cleared before each phase, so a test that calls
`inspect.getsource` is caught whether or not another test read the same
file first. A read in a subprocess is not seen.

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
from types import FrameType
from typing import Final, cast

import pytest

MARKER: Final[str] = "source_scan"

_REPO: Final[Path] = Path(__file__).resolve().parents[1]

#: The tree mutmut mutates, resolved, with a trailing separator.
SOURCE_ROOT: Final[str] = str(_REPO / "src" / "aelfrice") + os.sep

#: Where the reading test's own frame lives.
TESTS_ROOT: Final[str] = str(_REPO / "tests") + os.sep

#: Modules whose reads of a source file are not a test reading the source.
#: The import system loads modules, and pytest renders its own reports.
_IGNORED_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "importlib", "_frozen_importlib", "_pytest", "pluggy", "traceback",
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
    the test, fixture, or helper that the read belongs to. On the way, a
    frame of the import system or of stack rendering means the read is
    one of those. Every test runs below pytest's own frames, so the walk
    has to stop at the test: reaching a pytest frame first means pytest
    itself read the file.
    """
    while frame is not None:
        if _in_tests(frame.f_code.co_filename):
            return False
        module = frame.f_globals.get("__name__", "")
        if isinstance(module, str):
            if module.startswith(_IGNORED_MODULE_PREFIXES):
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
        names = ", ".join(str(Path(p).relative_to(_REPO)) for p in reads[:3])
        if len(reads) > 3:
            names += f" and {len(reads) - 3} more"
        pytest.fail(
            f"this test's {phase} read {names} as text but the test is not "
            f"marked `@pytest.mark.{MARKER}`. Inside mutmut's `mutants/` "
            f"tree that file is mutmut's mutated copy, so the test fails and "
            f"stops the mutation run (#1746). Mark the test, or its module "
            f"with `pytestmark = pytest.mark.{MARKER}`.",
            pytrace=False,
        )
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from audited(item, "setup"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from audited(item, "call"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from audited(item, "teardown"))
