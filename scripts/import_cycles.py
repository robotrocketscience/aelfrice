#!/usr/bin/env python3
"""Report the import cycles in the first-party `aelfrice` package (#1631).

Usage:

    uv run python scripts/import_cycles.py
    uv run python scripts/import_cycles.py --ref github/main
    uv run python scripts/import_cycles.py --assert-acyclic aelfrice.hook
    uv run python scripts/import_cycles.py --dry-run

The script builds the first-party import graph from the AST of every module
under `src/aelfrice/`, computes its strongly connected components, and prints
each component with more than one member. A component of size two or more is
an import cycle.

Every import statement counts as an edge, wherever it sits: module scope,
inside a function, inside `try`, or under `if TYPE_CHECKING:`. A
function-local edge is still an edge. A cycle that survives only because each
edge is deferred fails the first time one of those edges moves to module
scope, so the script does not discount deferred imports.

How an import becomes an edge:

* `import aelfrice.a.b` and `from aelfrice.a.b import name` point at the most
  specific first-party module the dotted path names. If `name` is itself a
  module (`from aelfrice import cli`), the edge points at that module.
* Relative imports resolve against the importing module's package.
* The implicit edge from `aelfrice.pkg.mod` to `aelfrice.pkg` (Python runs a
  package's `__init__` before its submodules) is not drawn. Drawing it would
  report a cycle for every package whose `__init__` imports a submodule, which
  is how packages are built, not a defect.
* Dynamic imports are not drawn. `importlib.import_module(...)` takes a value
  this script cannot resolve statically, so an edge built that way, such as the
  lazy loader `aelfrice.hook._lazy` uses for its deferred retrieval names, is
  invisible here. A cycle that runs only through such an edge goes unreported.

Options:

* `--ref REF` reads the tree at a git ref through `git show`, so you can
  measure another branch without checking it out. Default: the working tree.
* `--assert-acyclic MODULE` exits 1 if MODULE is a member of any component of
  size two or more. Repeatable.
* `--dry-run` prints which tree and how many files it would scan, then exits
  0 without building the graph. The script never writes, so the flag exists
  only to satisfy the repo's reusable-script contract.

Exit status: 0 on success, 1 when an `--assert-acyclic` module is in a cycle,
2 when the tree cannot be read or a file does not parse.

`tests/test_import_cycles_1631.py` imports `build_graph` and
`strongly_connected_components` from this file, so the test and the published
figures share one producer.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

PACKAGE = "aelfrice"
SRC_PREFIX = "src/"


def _module_name(rel_path: str) -> str:
    """Map `src/aelfrice/a/b.py` to `aelfrice.a.b`, `__init__.py` to its package."""
    parts = rel_path[len(SRC_PREFIX):-len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_package(rel_path: str) -> bool:
    return rel_path.endswith("/__init__.py")


def read_sources_from_tree(root: Path) -> dict[str, str]:
    """Return {relative path: source} for every `.py` file in the working tree."""
    pkg_dir = root / SRC_PREFIX / PACKAGE
    out: dict[str, str] = {}
    for path in sorted(pkg_dir.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        out[rel] = path.read_text(encoding="utf-8")
    return out


def read_sources_from_ref(root: Path, ref: str) -> dict[str, str]:
    """Return {relative path: source} for every `.py` file at a git ref."""
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", SRC_PREFIX + PACKAGE],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out: dict[str, str] = {}
    for rel in sorted(p for p in listing if p.endswith(".py")):
        out[rel] = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return out


def _resolve(
    target: str, modules: frozenset[str]
) -> str | None:
    """Return the most specific first-party module that `target` names."""
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _relative_base(importer: str, is_pkg: bool, level: int) -> str:
    """The package a relative import of `level` dots resolves against."""
    parts = importer.split(".")
    if not is_pkg:
        parts = parts[:-1]
    drop = level - 1
    if drop:
        parts = parts[:-drop]
    return ".".join(parts)


def _imports_of(
    importer: str, is_pkg: bool, tree: ast.AST, modules: frozenset[str]
) -> set[str]:
    """Every first-party module `importer` imports, at any scope."""
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _resolve(alias.name, modules)
                if hit is not None:
                    edges.add(hit)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _relative_base(importer, is_pkg, node.level)
                origin = f"{base}.{node.module}" if node.module else base
            else:
                origin = node.module or ""
            if not (origin == PACKAGE or origin.startswith(PACKAGE + ".")):
                continue
            for alias in node.names:
                as_module = f"{origin}.{alias.name}"
                if as_module in modules:
                    edges.add(as_module)
                    continue
                hit = _resolve(origin, modules)
                if hit is not None:
                    edges.add(hit)
    edges.discard(importer)
    return edges


def build_graph(sources: Mapping[str, str]) -> dict[str, set[str]]:
    """Return the first-party import graph as {module: set of imported modules}.

    Raises `SyntaxError` if any source does not parse.
    """
    names = {rel: _module_name(rel) for rel in sources}
    modules = frozenset(names.values())
    graph: dict[str, set[str]] = {}
    for rel, src in sources.items():
        tree = ast.parse(src, filename=rel)
        graph[names[rel]] = _imports_of(
            names[rel], _is_package(rel), tree, modules
        )
    return graph


def strongly_connected_components(
    graph: Mapping[str, Iterable[str]],
) -> list[list[str]]:
    """Iterative Tarjan. Returns every component, each sorted, in sorted order.

    Iterative so the recursion depth does not grow with the graph, which keeps
    the bound fixed regardless of how long an import chain gets.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    adjacency = {node: sorted(graph[node]) for node in graph}

    for start in sorted(adjacency):
        if start in index:
            continue
        work: list[tuple[str, int]] = [(start, 0)]
        while work:
            node, pos = work.pop()
            if pos == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            neighbours = adjacency.get(node, [])
            descended = False
            while pos < len(neighbours):
                nxt = neighbours[pos]
                pos += 1
                if nxt not in adjacency:
                    continue
                if nxt not in index:
                    work.append((node, pos))
                    work.append((nxt, 0))
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return sorted(components)


def cycles(graph: Mapping[str, Iterable[str]]) -> list[list[str]]:
    """Every component with more than one member, largest first."""
    found = [c for c in strongly_connected_components(graph) if len(c) > 1]
    return sorted(found, key=lambda c: (-len(c), c))


def _short(module: str) -> str:
    return module[len(PACKAGE) + 1:] if module.startswith(PACKAGE + ".") else module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    _ = parser.add_argument(
        "--ref",
        help="git ref to read the tree from. Default: the working tree.",
    )
    _ = parser.add_argument(
        "--assert-acyclic",
        action="append",
        default=[],
        metavar="MODULE",
        help="exit 1 if MODULE is in a cycle. Repeatable.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be scanned and exit 0 without building the graph.",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent

    try:
        if args.ref:
            sources = read_sources_from_ref(root, args.ref)
        else:
            sources = read_sources_from_tree(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: cannot read the tree: {exc}", file=sys.stderr)
        return 2
    if not sources:
        print("error: no first-party sources found", file=sys.stderr)
        return 2

    where = args.ref or "working tree"
    if args.dry_run:
        print(f"would scan {len(sources)} files under {SRC_PREFIX}{PACKAGE} "
              f"at {where}")
        return 0

    try:
        graph = build_graph(sources)
    except SyntaxError as exc:
        print(f"error: {exc.filename}: {exc}", file=sys.stderr)
        return 2

    found = cycles(graph)
    print(f"{where}  modules: {len(graph)}  cycles: {len(found)}")
    for component in found:
        print(f"  size {len(component)}: "
              + ", ".join(_short(m) for m in component))

    in_cycle = {m for c in found for m in c}
    failed = False
    for module in args.assert_acyclic:
        if module not in graph:
            print(f"error: {module} is not a first-party module", file=sys.stderr)
            return 2
        if module in in_cycle:
            print(f"FAIL: {module} is in an import cycle", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
