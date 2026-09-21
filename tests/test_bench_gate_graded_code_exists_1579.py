"""A bench gate must grade code that exists (#1579).

Three v2.0 bench gates — `dedup`, `enforcement`, `promotion_trigger` —
were authored against modules the package never shipped. Nothing caught
that, because the whole tier skips without `AELFRICE_CORPUS_ROOT` and the
corpus for those modules was never mounted. The cost landed at the release
cut instead of at authoring time: two of the three skipped on a guarded
`ModuleNotFoundError`, and the third — `aelfrice.dedup`, a module that
does import — sailed past its guard and raised `AttributeError` on the
first row, in the tier `docs/concepts/RELEASING.md` step 7 makes mandatory.

This check runs on every public CI pass, unmarked and corpus-free, and
fails at the moment a scaffold-before-code gate is committed. It pins two
things, because the `dedup` shape shows the first alone is not enough:

1. Every `aelfrice.*` module a bench-gate file imports has a file under
   `src/aelfrice/`.
2. Every attribute a bench-gate file reads off one of those imported
   modules exists on it. A module that imports is not a module that can
   be graded.

Scope is deliberately `tests/bench_gate/` only. The rest of the suite runs
on every CI pass, so a missing name there is a red test today; the bench
tier is the one place where it is not.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_GATE_DIR = Path(__file__).resolve().parent / "bench_gate"
PACKAGE_DIR = REPO_ROOT / "src" / "aelfrice"
PACKAGE = "aelfrice"


def _bench_gate_files() -> list[Path]:
    """Every Python file in the bench-gate tier, tests and helpers alike.

    Helpers are included because a gate's imports can sit in the module
    that builds its store rather than in the test file itself.
    """
    return sorted(BENCH_GATE_DIR.glob("*.py"))


def _module_path_exists(dotted: str) -> bool:
    """True when `dotted` has a file under `src/aelfrice/`.

    Resolved against the source tree rather than `find_spec`, so the
    answer is about what this repository ships and cannot be satisfied
    by some other `aelfrice` on the import path.
    """
    parts = dotted.split(".")
    assert parts[0] == PACKAGE
    rest = parts[1:]
    if not rest:
        return (PACKAGE_DIR / "__init__.py").is_file()
    base = PACKAGE_DIR.joinpath(*rest)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _has_member(module: ModuleType, name: str) -> bool:
    """True when `name` is an attribute of `module` or a submodule of it."""
    if hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module.__name__}.{name}")
    except ImportError:
        return False
    return True


class _Collector(ast.NodeVisitor):
    """Gather `aelfrice.*` imports and the attributes read off them.

    `bindings` maps a local name to the dotted module it is bound to.
    `members` is the set of (dotted module, member) pairs an import
    statement asserts exist. `rebound` holds local names the file later
    assigns to, whose attribute reads are therefore not module reads.
    """

    def __init__(self) -> None:
        self.modules: set[str] = set()
        self.members: set[tuple[str, str]] = set()
        self.bindings: dict[str, str] = {}
        self.rebound: set[str] = set()
        self.attributes: list[tuple[str, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name != PACKAGE and not alias.name.startswith(PACKAGE + "."):
                continue
            self.modules.add(alias.name)
            if alias.asname:
                self.bindings[alias.asname] = alias.name
            else:
                self.bindings[alias.name.split(".")[0]] = PACKAGE
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        # Relative imports are inside the tests package, not aelfrice.
        if node.level or (mod != PACKAGE and not mod.startswith(PACKAGE + ".")):
            self.generic_visit(node)
            return
        self.modules.add(mod)
        for alias in node.names:
            candidate = f"{mod}.{alias.name}"
            local = alias.asname or alias.name
            if _module_path_exists(candidate):
                # `from aelfrice import dedup` — a submodule import.
                self.modules.add(candidate)
                self.bindings[local] = candidate
            else:
                # `from aelfrice.models import Belief` — a symbol import.
                self.members.add((mod, alias.name))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        value = node.value
        if isinstance(value, ast.Name) and isinstance(node.ctx, ast.Load):
            self.attributes.append((value.id, node.attr))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.rebound.add(node.id)
        self.generic_visit(node)


def _collect(path: Path) -> _Collector:
    collector = _Collector()
    collector.visit(ast.parse(path.read_text(), filename=str(path)))
    return collector


@pytest.mark.parametrize(
    "path", _bench_gate_files(), ids=lambda p: p.name
)
def test_bench_gate_imports_a_module_that_exists(path: Path) -> None:
    """Every `aelfrice.*` module this bench-gate file imports is shipped."""
    missing = sorted(
        dotted
        for dotted in _collect(path).modules
        if not _module_path_exists(dotted)
    )
    assert not missing, (
        f"{path.name} bench-gates {missing}, which has no file under "
        f"src/aelfrice/. A gate cannot grade code that does not exist: "
        f"it either skips forever or errors at the release cut (#1579). "
        f"Ship the module first, or delete the gate and its corpus "
        f"scaffold."
    )


@pytest.mark.parametrize(
    "path", _bench_gate_files(), ids=lambda p: p.name
)
def test_bench_gate_grades_an_attribute_that_exists(path: Path) -> None:
    """Every name this bench-gate file reads off an `aelfrice` module exists.

    The `dedup` case is exactly this: `aelfrice.dedup` imports, so an
    existence check on the module passes, while `dedup.classify` — the
    function the gate actually graded — was never written.
    """
    collector = _collect(path)
    missing: list[str] = []

    for dotted, name in sorted(collector.members):
        if not _module_path_exists(dotted):
            continue  # Reported by the module check above.
        if not _has_member(importlib.import_module(dotted), name):
            missing.append(f"{dotted}.{name}")

    for local, attr in collector.attributes:
        dotted = collector.bindings.get(local)
        if dotted is None or local in collector.rebound:
            continue
        if not _module_path_exists(dotted):
            continue  # Reported by the module check above.
        if not _has_member(importlib.import_module(dotted), attr):
            missing.append(f"{dotted}.{attr}")

    assert not missing, (
        f"{path.name} bench-gates {sorted(set(missing))}, which the "
        f"module does not define. An importable module is not a gradable "
        f"one — this is the shape that made the retired `dedup` gate raise "
        f"AttributeError at the release cut instead of skipping (#1579). "
        f"Write the function first, or delete the gate."
    )


def test_the_retired_modules_stay_retired() -> None:
    """#1579's three modules are gone from the tier and the corpus scaffold.

    Re-adding a scaffold before its code is the failure this issue closed,
    and the two checks above only fire once such a gate names a missing
    name. This pins the specific three, whose dispositions are settled:
    `enforcement` H2 was dropped on security grounds, `promotion_trigger`
    graded the belief-sequence trigger #229 rejected (the ratified rule is
    explicit user acknowledgment, shipped via #550), and `dedup` has no
    `near-duplicate` band to grade against.
    """
    retired = ("dedup", "enforcement", "promotion_trigger")
    corpus_root = Path(__file__).resolve().parent / "corpus" / "v2_0"
    for module in retired:
        assert not (BENCH_GATE_DIR / f"test_{module}.py").exists(), (
            f"tests/bench_gate/test_{module}.py is back. #1579 retired it "
            f"because the code it grades does not exist."
        )
        assert not (corpus_root / module).exists(), (
            f"tests/corpus/v2_0/{module}/ is back. #1579 retired the "
            f"scaffold; do not re-mount rows before the code exists."
        )
