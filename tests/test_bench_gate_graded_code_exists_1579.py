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
   be graded. Shadowing is resolved per scope: a local `dedup = {...}`
   inside one helper must not disarm the check for the whole file.
   Whole dotted chains count, so `import aelfrice.dedup` followed by
   `aelfrice.dedup.classify(...)` is checked as `classify` on
   `aelfrice.dedup`; and decorators, annotations and parameter defaults
   are read in the scope that evaluates them, which is the enclosing one.
3. The three modules #1579 retired stay retired — but only while the
   function their gates called is still missing, so the documented
   revival route does not red.

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
PACKAGE_ROOT = REPO_ROOT / "src" / "aelfrice"
PACKAGE = "aelfrice"


def _bench_gate_files() -> list[Path]:
    """Every Python file in the bench-gate tier, tests and helpers alike.

    Helpers are included because a gate's imports can sit in the module
    that builds its store rather than in the test file itself.
    """
    return sorted(BENCH_GATE_DIR.rglob("*.py"))


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
        return (PACKAGE_ROOT / "__init__.py").is_file()
    base = PACKAGE_ROOT.joinpath(*rest)
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


SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _own_scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node inside `scope`'s own namespace, nested scopes excluded.

    A name assigned inside a nested function or comprehension binds there,
    not here, so the walk stops at each nested scope's boundary.
    """
    if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        pending: list[ast.AST] = list(scope.body)
    elif isinstance(scope, ast.Lambda):
        pending = [scope.body]
    elif isinstance(scope, ast.DictComp):
        pending = [scope.key, scope.value]
    else:  # ListComp / SetComp / GeneratorExp
        pending = [scope.elt]
    for generator in getattr(scope, "generators", []):
        pending.append(generator.target)
        pending.extend(generator.ifs)
    seen: list[ast.AST] = []
    while pending:
        node = pending.pop()
        seen.append(node)
        if isinstance(node, SCOPE_NODES):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return seen


def _shadowing_names(scope: ast.AST) -> tuple[set[str], set[str]]:
    """Names `scope` binds locally, and the names it rebinds via `global`.

    The first set is what shadows an imported module *inside* this scope;
    the second is what a nested scope writes back to module scope, which
    shadows the import everywhere.
    """
    local: set[str] = set()
    declared_global: set[str] = set()
    arguments = getattr(scope, "args", None)
    if isinstance(arguments, ast.arguments):
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            local.add(arg.arg)
        for extra in (arguments.vararg, arguments.kwarg):
            if extra is not None:
                local.add(extra.arg)
    for node in _own_scope_nodes(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local.add(node.id)
        elif isinstance(node, ast.Global):
            declared_global.update(node.names)
    # `global x; x = ...` writes the module name, so it does not shadow here.
    escaping = local & declared_global
    return local - declared_global, escaping


def _dotted_parts(node: ast.Attribute) -> tuple[str | None, list[str]]:
    """Split an attribute chain into its root `Name` and the attrs after it.

    `aelfrice.dedup.classify` reads back as `("aelfrice", ["dedup",
    "classify"])`. Recording only the innermost `Name.Attribute` pair was
    the first hole the #1579 review found: the chain's own value is an
    `Attribute`, not a `Name`, so the outer `.classify` was never seen and
    only the `aelfrice.dedup` prefix — which does exist — got checked.

    Returns `(None, [])` for a chain that does not bottom out in a name,
    such as `f().attr`, where nothing static can be resolved.
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None, []
    parts.reverse()
    return current.id, parts


def _outer_scope_expressions(scope: ast.AST) -> list[ast.AST]:
    """Sub-expressions of `scope` that evaluate in the ENCLOSING scope.

    A function's decorators, its return annotation, and its parameter
    annotations and defaults are evaluated where the `def` is written, not
    inside the body — so the function's own parameter names do not shadow
    anything in them. Applying the local scope to them was the second hole
    the #1579 review found, and it is only reachable when a parameter name
    collides with the module binding, as in `import aelfrice.dedup as
    dedup` plus `def gate(dedup=dedup.classify)`.

    A comprehension's first `iter` is the same shape: it is evaluated
    before the comprehension scope exists.
    """
    outer: list[ast.AST] = []
    outer.extend(getattr(scope, "decorator_list", None) or [])
    returns = getattr(scope, "returns", None)
    if returns is not None:
        outer.append(returns)
    arguments = getattr(scope, "args", None)
    if isinstance(arguments, ast.arguments):
        every_arg = (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        )
        for arg in every_arg:
            if arg is not None and arg.annotation is not None:
                outer.append(arg.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                outer.append(default)
    generators = getattr(scope, "generators", None)
    if generators:
        outer.append(generators[0].iter)
    return outer


class _Collector(ast.NodeVisitor):
    """Gather `aelfrice.*` imports and the attributes read off them.

    `bindings` maps a local name to the dotted module it is bound to.
    `members` is the set of (dotted module, member) pairs an import
    statement asserts exist. `attributes` holds each attribute chain read,
    as (root name, attrs after it, shadowed) — the whole dotted chain, so
    `aelfrice.dedup.classify` is resolved against `aelfrice.dedup` rather
    than stopping at the `aelfrice.dedup` prefix.

    The flag says whether the root name was shadowed by a local binding at
    that point — an unrelated `dedup = {...}` inside one function must not
    disarm the checks for the rest of the file (#1579 review), so
    shadowing is resolved per scope rather than per file.
    """

    def __init__(self) -> None:
        self.modules: set[str] = set()
        self.members: set[tuple[str, str]] = set()
        self.bindings: dict[str, str] = {}
        self.attributes: list[tuple[str, list[str], bool]] = []
        self._scopes: list[set[str]] = []

    def _shadowed(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Module, *SCOPE_NODES)):
            local, escaping = _shadowing_names(node)
            if escaping and self._scopes:
                # A `global` write lands in the module scope, not this one.
                self._scopes[0].update(escaping)
            # Decorators, annotations and defaults evaluate outside this
            # scope, so read them before its local names are in force. The
            # body walk below reaches them a second time, which is
            # harmless: the second reading can only mark them shadowed,
            # and a shadowed reading is discarded rather than reported.
            for expression in _outer_scope_expressions(node):
                self.visit(expression)
            self._scopes.append(local)
            try:
                super().visit(node)
            finally:
                self._scopes.pop()
            return
        super().visit(node)

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
        if isinstance(node.ctx, ast.Load):
            root, parts = _dotted_parts(node)
            if root is not None:
                self.attributes.append((root, parts, self._shadowed(root)))
        self.generic_visit(node)


def _collect_source(source: str, filename: str = "<source>") -> _Collector:
    collector = _Collector()
    collector.visit(ast.parse(source, filename=filename))
    return collector


def _collect(path: Path) -> _Collector:
    return _collect_source(path.read_text(), filename=str(path))


def _missing_graded_names(collector: _Collector) -> list[str]:
    """Every `aelfrice` name `collector` saw that the package does not define.

    Modules that do not exist at all are left out: the module check owns
    those, and reporting them twice would blame the wrong test.
    """
    missing: list[str] = []

    for dotted, name in sorted(collector.members):
        if not _module_path_exists(dotted):
            continue  # Reported by the module check above.
        if not _has_member(importlib.import_module(dotted), name):
            missing.append(f"{dotted}.{name}")

    for root, parts, shadowed in collector.attributes:
        dotted = collector.bindings.get(root)
        if dotted is None or shadowed:
            continue
        if not _module_path_exists(dotted):
            continue  # Reported by the module check above.
        # Absorb as much of the chain as still names a module, so
        # `aelfrice.dedup.classify` is checked as `classify` on
        # `aelfrice.dedup` and not as `dedup` on `aelfrice`.
        index = 0
        while index < len(parts) and _module_path_exists(f"{dotted}.{parts[index]}"):
            dotted = f"{dotted}.{parts[index]}"
            index += 1
        if index == len(parts):
            continue  # The whole chain is a module; the module check owns it.
        if not _has_member(importlib.import_module(dotted), parts[index]):
            missing.append(f"{dotted}.{parts[index]}")

    return sorted(set(missing))


@pytest.mark.parametrize(
    "path", _bench_gate_files(), ids=lambda p: p.name
)
def test_bench_gate_imports_a_module_that_exists(path: Path) -> None:
    """Every `aelfrice.*` module this bench-gate file imports is shipped.

    This fires on `from aelfrice.<missing> import X` and `import
    aelfrice.<missing>`. The `from aelfrice import <missing>` spelling —
    which is how all three retired gates were written — reads as a
    missing *member* of the package, and the attribute check below is
    what catches it.
    """
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
    missing = _missing_graded_names(_collect(path))
    assert not missing, (
        f"{path.name} bench-gates {missing}, which the "
        f"module does not define. An importable module is not a gradable "
        f"one — this is the shape that made the retired `dedup` gate raise "
        f"AttributeError at the release cut instead of skipping (#1579). "
        f"Write the function first, or delete the gate."
    )


# Retired module -> the function its deleted gate called. The pin below
# holds only while that function is missing, so a real revival (band
# defined, function written, rows re-mounted) retires the pin instead of
# tripping it.
RETIRED_GRADED_SYMBOL = {
    "dedup": "classify",
    "enforcement": "classify",
    "promotion_trigger": "decide",
}


def _graded_code_exists(module: str, symbol: str) -> bool:
    """True once `src/aelfrice/<module>.py` defines the graded function."""
    dotted = f"{PACKAGE}.{module}"
    if not _module_path_exists(dotted):
        return False
    return _has_member(importlib.import_module(dotted), symbol)


@pytest.mark.parametrize("module", sorted(RETIRED_GRADED_SYMBOL))
def test_the_retired_modules_stay_retired(module: str) -> None:
    """#1579's three modules stay gone until the code they grade exists.

    Re-adding a scaffold before its code is the failure this issue closed,
    and the two checks above only fire once such a gate names a missing
    name. This pins the specific three, whose dispositions are settled:
    `enforcement` H2 was dropped on security grounds, `promotion_trigger`
    graded the belief-sequence trigger #229 rejected (the ratified rule is
    explicit user acknowledgment, shipped via #550), and `dedup` has no
    `near-duplicate` band to grade against.

    The pin is conditional on purpose. `docs/design/dedup.md` prescribes a
    revival route — define the band, write `classify`, re-mount the rows —
    and an unconditional name ban would red that route and force deleting
    the guard to follow it. Once the graded function lands, the premise
    has expired and the two checks above take over.
    """
    symbol = RETIRED_GRADED_SYMBOL[module]
    if _graded_code_exists(module, symbol):
        pytest.skip(
            f"aelfrice.{module}.{symbol} now exists — #1579's premise for "
            f"this module has expired and the pin no longer applies."
        )
    corpus_root = Path(__file__).resolve().parent / "corpus" / "v2_0"
    assert not (BENCH_GATE_DIR / f"test_{module}.py").exists(), (
        f"tests/bench_gate/test_{module}.py is back, but "
        f"aelfrice.{module}.{symbol} still does not exist. #1579 retired "
        f"the gate because the code it grades is missing; write "
        f"{symbol}() first."
    )
    assert not (corpus_root / module).exists(), (
        f"tests/corpus/v2_0/{module}/ is back, but "
        f"aelfrice.{module}.{symbol} still does not exist. #1579 retired "
        f"the scaffold; do not re-mount rows before the code exists."
    )
