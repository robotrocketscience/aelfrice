"""The guard that keeps `source_scan` on every test that reads the source (#1746).

mutmut's run leaves out `source_scan` tests, because inside `mutants/` they
read mutmut's mutated copies and fail, which stops the whole run. The guard
in `tests/source_scan_guard.py` fails any unmarked test that reads a file
under `src/aelfrice` as text. These tests pin what it counts as a read and
what it ignores, and that the suite and the mutation run are wired to it.

Every test here that reads the source on purpose carries the marker, or
the guard would fail it.
"""
from __future__ import annotations

import functools
import importlib.machinery
import inspect
import linecache
import tomllib
import traceback
import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from tests import conftest
from tests import source_scan_guard as guard

_REPO = Path(__file__).resolve().parents[1]
_ULID = _REPO / "src" / "aelfrice" / "ulid.py"


def _reads(action: Callable[[], object]) -> list[str]:
    """The source files the guard records while `action` runs."""
    previous_active, previous_reads = guard.Audit.active, guard.Audit.reads
    guard.Audit.reads = []
    guard.Audit.active = True
    try:
        action()
        return sorted({Path(p).name for p in guard.Audit.reads})
    finally:
        guard.Audit.active = previous_active
        guard.Audit.reads = previous_reads


class _Item:
    """The one thing the guard asks of a pytest item."""

    def __init__(self, marked: bool) -> None:
        self.marked = marked

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self.marked and name == guard.MARKER else None


def _phase(item: _Item, action: Callable[[], object]) -> None:
    """Run `action` as one audited test phase."""
    phase: Generator[None, object, object] = guard.audited(item, "call")  # type: ignore[arg-type]
    next(phase)
    action()
    with pytest.raises(StopIteration):
        phase.send(None)


@pytest.mark.source_scan
def test_reading_a_source_file_as_text_is_recorded() -> None:
    assert _reads(lambda: _ULID.read_text(encoding="utf-8")) == ["ulid.py"]


@pytest.mark.source_scan
def test_inspect_getsource_is_recorded() -> None:
    from aelfrice import ulid

    linecache.clearcache()
    assert _reads(lambda: inspect.getsource(ulid.make_generator)) == ["ulid.py"]


def test_importing_a_source_module_is_not_a_read() -> None:
    """`get_data` is how an import reads a `.py` file with no cached
    bytecode. `exec_module` alone would read the `.pyc` and prove nothing."""
    loader = importlib.machinery.SourceFileLoader("ulid_1746_copy", str(_ULID))
    assert _reads(lambda: loader.get_data(str(_ULID))) == []


def test_formatting_a_traceback_through_the_source_is_not_a_read() -> None:
    """`src` prints tracebacks in its hooks; that reads lines to show them."""
    from aelfrice import ulid

    def no_entropy(n: int) -> bytes:
        raise OSError(n)

    linecache.clearcache()
    with pytest.raises(OSError) as raised:
        ulid.make_generator(rand_source=no_entropy)()
    assert _reads(lambda: traceback.format_exception(raised.value)) == []


def test_a_file_outside_the_source_tree_is_not_a_read() -> None:
    assert _reads(lambda: Path(__file__).read_text(encoding="utf-8")) == []


@pytest.mark.source_scan
def test_an_unmarked_test_that_reads_the_source_fails() -> None:
    with pytest.raises(pytest.fail.Exception, match="ulid.py.*not marked"):
        _phase(_Item(marked=False), lambda: _ULID.read_text(encoding="utf-8"))


@pytest.mark.source_scan
def test_a_marked_test_that_reads_the_source_passes() -> None:
    _phase(_Item(marked=True), lambda: _ULID.read_text(encoding="utf-8"))


@pytest.mark.source_scan
def test_getsource_is_caught_even_when_another_test_read_the_file_first() -> None:
    """Each phase starts with an empty line cache, so the guard doesn't
    depend on which test happened to read a file first."""
    from aelfrice import ulid

    inspect.getsource(ulid.make_generator)  # warms linecache
    with pytest.raises(pytest.fail.Exception, match="ulid.py"):
        _phase(_Item(marked=False), lambda: inspect.getsource(ulid.make_generator))


def test_a_cache_in_the_test_module_is_cleared_before_each_test() -> None:
    """A cached source scan is redone, and seen, in every test that uses it."""
    module = types.ModuleType("tests.fake_scan_1746")
    exec(
        "import functools\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def scan():\n"
        "    return object()\n",
        module.__dict__,
    )
    foreign = functools.lru_cache(maxsize=None)(lambda: object())
    setattr(module, "foreign", foreign)
    scan: Any = module.__dict__["scan"]
    scan()
    foreign()

    class Item:
        def __init__(self, module: types.ModuleType) -> None:
            self.module = module

    guard.clear_module_caches(Item(module))  # type: ignore[arg-type]
    assert scan.cache_info().currsize == 0
    assert foreign.cache_info().currsize == 1, "only the test module's own caches"


def test_the_suite_runs_the_guard_on_every_phase() -> None:
    for hook in ("pytest_runtest_setup", "pytest_runtest_call", "pytest_runtest_teardown"):
        assert getattr(conftest, hook) is getattr(guard, hook), hook
    assert guard.Audit.installed


def test_the_mutation_run_leaves_marked_tests_out() -> None:
    config: dict[str, Any] = tomllib.loads(
        (_REPO / "pyproject.toml").read_text(encoding="utf-8"),
    )
    args = config["tool"]["mutmut"]["pytest_add_cli_args"]
    assert ["-m", f"not {guard.MARKER}"] in [args[i:i + 2] for i in range(len(args))]
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith(f"{guard.MARKER}:") for m in markers)
