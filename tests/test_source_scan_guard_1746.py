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


def test_importing_a_source_module_is_not_a_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """An import reads the `.py` in the loader's `get_code`. With no
    bytecode path it reads the source itself; with a cached `.pyc` it
    would read that instead and prove nothing."""
    import importlib._bootstrap_external as external  # pyright: ignore[reportMissingModuleSource]

    def no_cache(path: str) -> str:
        raise NotImplementedError(path)

    monkeypatch.setattr(external, "cache_from_source", no_cache)
    loader = importlib.machinery.SourceFileLoader("ulid_1746_copy", str(_ULID))
    assert _reads(lambda: loader.get_code("ulid_1746_copy")) == []


def test_a_module_reading_its_own_source_as_it_is_imported_is_not_a_read() -> None:
    """`core_gate` digests its own source at import. That read belongs to
    whichever test imports it first, so counting it made the verdict
    depend on order."""
    import importlib.util

    path = _ULID.with_name("core_gate.py")
    spec = importlib.util.spec_from_file_location("core_gate_1746_copy", str(path))
    assert spec is not None and spec.loader is not None
    loader = spec.loader
    module = importlib.util.module_from_spec(spec)
    linecache.clearcache()
    assert _reads(lambda: loader.exec_module(module)) == []


@pytest.mark.source_scan
def test_asking_a_loader_for_the_source_is_a_read() -> None:
    """`get_source` and `pkgutil.get_data` go through the import system's
    loader, but they hand the caller the text."""
    import pkgutil

    loader = importlib.machinery.SourceFileLoader("ulid_1746_copy", str(_ULID))
    assert _reads(lambda: loader.get_source("ulid_1746_copy")) == ["ulid.py"]
    assert _reads(lambda: pkgutil.get_data("aelfrice", "ulid.py")) == ["ulid.py"]


def test_rendering_the_stack_through_the_source_is_not_a_read() -> None:
    """`inspect.stack()` reads each frame's lines, `src` frames included."""
    from aelfrice import ulid

    def look(n: int) -> bytes:
        linecache.clearcache()
        stacks.append(_reads(inspect.stack))
        return bytes(n)

    stacks: list[list[str]] = []
    ulid.make_generator(rand_source=look)()
    assert stacks == [[]]


def test_hypothesis_sampling_the_source_is_not_a_read() -> None:
    """Hypothesis reads newly imported modules for constants, at random;
    counting it marked property tests and made the verdict order-dependent."""
    sampler = types.FunctionType(
        compile("lambda: open(path, encoding='utf-8').read()", "<sampler>", "eval"),
        {"__name__": "hypothesis.core", "path": str(_ULID)},
    )()
    assert _reads(sampler) == []


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


def test_only_python_files_count() -> None:
    assert guard._is_source(str(_ULID)) is not None  # pyright: ignore[reportPrivateUsage]
    assert guard._is_source(str(_ULID.with_name("ulid.txt"))) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.source_scan
def test_a_bytes_or_symlinked_path_is_still_a_read(tmp_path: Path) -> None:
    link = tmp_path / "pkg"
    link.symlink_to(_ULID.parent, target_is_directory=True)
    assert _reads(lambda: (link / "ulid.py").read_text(encoding="utf-8")) == ["ulid.py"]

    def read_bytes_path() -> None:
        with open(bytes(_ULID), "rb") as f:
            f.read()

    assert _reads(read_bytes_path) == ["ulid.py"]


def test_a_read_with_no_test_frame_is_not_the_tests() -> None:
    """A thread running library code reads with no test on its stack."""
    import threading

    def run() -> None:
        reader = threading.Thread(target=Path.read_text, args=(_ULID,))
        reader.start()
        reader.join(timeout=10)
        assert not reader.is_alive()

    assert _reads(run) == []


def test_executing_or_warning_through_the_source_is_not_a_read() -> None:
    """`runpy` executes a file, and showing a warning reads one line of it."""
    import runpy
    import warnings

    assert _reads(lambda: runpy.run_path(str(_ULID))) == []
    linecache.clearcache()
    assert _reads(lambda: warnings.formatwarning("m", UserWarning, str(_ULID), 1)) == []


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


_CACHED_CLASS = (
    "import functools\n"
    "class TestScan:\n"
    "    @staticmethod\n"
    "    @functools.lru_cache(maxsize=None)\n"
    "    def by_static():\n"
    "        return object()\n"
    "    @classmethod\n"
    "    @functools.lru_cache(maxsize=None)\n"
    "    def by_class(cls):\n"
    "        return object()\n"
)


class _ModuleItem(_Item):
    """An item that also has the module the guard clears caches in."""

    def __init__(self, marked: bool, module: types.ModuleType) -> None:
        super().__init__(marked)
        self.module = module


def _cached_class_module() -> tuple[types.ModuleType, Any]:
    module = types.ModuleType("tests.fake_class_scan_1746")
    exec(_CACHED_CLASS, module.__dict__)
    cls: Any = module.__dict__["TestScan"]
    cls.by_static()
    cls.by_class()
    return module, cls


def test_a_cache_on_a_test_class_is_cleared_too() -> None:
    module, cls = _cached_class_module()
    guard.clear_module_caches(_ModuleItem(False, module))  # type: ignore[arg-type]
    assert cls.by_static.cache_info().currsize == 0
    assert cls.__dict__["by_class"].__func__.cache_info().currsize == 0


@pytest.mark.source_scan
@pytest.mark.parametrize(
    "hook", ["pytest_runtest_setup", "pytest_runtest_call", "pytest_runtest_teardown"],
)
def test_every_phase_hook_audits_its_phase(hook: str) -> None:
    module, _ = _cached_class_module()
    phase: Generator[None, object, object] = getattr(guard, hook)(_ModuleItem(False, module))
    next(phase)
    _ULID.read_text(encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="ulid.py"):
        phase.send(None)


def test_only_setup_clears_the_caches_once_per_test() -> None:
    for hook, cleared in (
        ("pytest_runtest_setup", True),
        ("pytest_runtest_call", False),
        ("pytest_runtest_teardown", False),
    ):
        module, cls = _cached_class_module()
        phase: Generator[None, object, object] = getattr(guard, hook)(_ModuleItem(True, module))
        next(phase)
        assert (cls.by_static.cache_info().currsize == 0) is cleared, hook
        with pytest.raises(StopIteration):
            phase.send(None)


def test_a_module_that_reads_the_source_at_import_must_be_marked_whole() -> None:
    reads = [str(_ULID)]
    plain = types.ModuleType("tests.fake_import_scan_1746")
    failure = guard.collection_failure("tests/x.py", plain, reads)
    assert failure is not None and "src/aelfrice/ulid.py" in failure
    assert guard.collection_failure("tests/x.py", plain, []) is None
    for mark in (pytest.mark.source_scan, [pytest.mark.timeout(5), pytest.mark.source_scan]):
        marked = types.ModuleType("tests.fake_import_scan_1746")
        setattr(marked, "pytestmark", mark)
        assert guard.collection_failure("tests/x.py", marked, reads) is None
    other = types.ModuleType("tests.fake_import_scan_1746")
    setattr(other, "pytestmark", pytest.mark.timeout(5))
    assert guard.collection_failure("tests/x.py", other, reads) is not None


def test_the_suite_runs_the_guard_on_every_phase() -> None:
    for hook in (
        "pytest_runtest_setup", "pytest_runtest_call", "pytest_runtest_teardown",
        "pytest_make_collect_report",
    ):
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
