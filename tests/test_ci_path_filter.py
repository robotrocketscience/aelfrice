"""The CI `code` paths-filter must cover every in-repo path the suite uses.

`.github/workflows/ci.yml` always runs so that `pytest (3.12)` and
`pytest (3.13)` — both required status checks — report on every PR,
including docs-only ones (#413/#427). Which PRs actually *run* the suite
is decided inside the job by a `dorny/paths-filter` glob list. A path the
suite depends on but the filter omits is therefore worse than an
uncovered path: the required check reports success from an `echo`,
having run nothing.

That is what #1160 found. `benchmarks/**` and `scripts/**` were absent
while `pythonpath = ["."]` made both importable and ~30 test modules
used them, so breaking either merged green.

#1786 (CI-07) found the same shape one level down: the filter named no
workflow file but its own, while tests across the suite read workflows,
and it named no document while tests read `README.md`, `SECURITY.md`
and pages under `docs/`. A PR that changed only one of those files ran
no test that reads it. The filter now carries `.github/**` and one exact
entry per file a test reads outside the package globs.

These tests **derive** the expected filter contents by scanning
`tests/` for what it actually imports and reads, rather than comparing
the filter against a hand-maintained list of names. A hand-maintained
list is the failure mode #1161 hit on the doctor side: its drift guard
compared a tuple against the same constants the tuple was copied from,
so it could only fail on a rename, never on the drift that occurred.

The read scan walks each module's AST for paths built from the module's
own location (`Path(__file__)`, `os.path.dirname(__file__)`). It follows
names, string constants, literal tuples and the loops over them,
`joinpath`, `Path(root, ...)`, `os.path.join` and `..`, so each spelling
the suite uses resolves the same as an inline path. It also follows a
repo module the test loads, by `importlib` spec, `import_module`,
`import ... as` or `from ... import`, one module deep: a path that
module binds, such as `cdf.REPO_ROOT`, is the test's own, and a call to
one of its functions that reads under its root, such as
`cdf.iter_files(roots)`, is a read the scan cannot resolve. A path built
on the repo root that the scan still cannot resolve, such as `REPO / p`
over `git ls-files` output, or a string built on the root, such as
`f"{_REPO}/docs/x.md"`, fails `test_every_repo_path_a_test_builds_is_resolved`
until its line carries a `# path-filter:` comment naming what it reads.
So a new test that reads a new file outside the filter through such a
path fails here, and an entry nobody reads any more fails the reverse
test. Outside the scan: a path handed to a subprocess or spelled
relative to the working directory, a module object the scan cannot trace
to its file (one a helper function returns), a module the loaded module
loads in turn, and a read a loaded module makes at import time.
"""
from __future__ import annotations

import ast
import functools
import io
import re
import shlex
import tokenize
from pathlib import Path
from typing import NamedTuple

import pytest

# The read scan follows a test's `from aelfrice... import` into
# `src/aelfrice` and parses that module, which is reading the source as
# text (#1746).
pytestmark = pytest.mark.source_scan

_REPO = Path(__file__).resolve().parents[1]
_CI_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"
_STAGING_GATE_WORKFLOW = _REPO / ".github" / "workflows" / "staging-gate.yml"

# A top-level directory holding Python the suite could depend on. Excludes
# directories with no .py at all (`CHANGELOG/`, and the untracked
# `telemetry/` a local run may leave behind), so the scan does not depend
# on working-tree state.
_IN_REPO_PACKAGES = frozenset(
    p.name
    for p in _REPO.iterdir()  # path-filter: none
    if p.is_dir() and not p.name.startswith(".") and any(p.rglob("*.py"))
)

# Exact entries the filter carries because they change what `uv sync`
# installs, not because a test reads them. They are the only exact
# entries that `e2e.yml` must mirror.
_BUILD_INPUTS = frozenset({"pyproject.toml", "uv.lock"})

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)

# The `release-docs-check` invocation in staging-gate.yml. Its arguments
# name the CHANGELOG paths that job checks on every PR; a directory the
# suite reads under one of them may stay out of the `code` filter.
_DUPES_INVOCATION_RE = re.compile(
    r"python3 scripts/check_changelog_dupes\.py ([^\n]*)"
)


def _filter_entries(key: str) -> list[str]:
    """Return the glob list under one key of the paths-filter."""
    lines = _CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == f"{key}:"]
    assert len(starts) == 1, (
        f"expected exactly one `{key}:` filter key in {_CI_WORKFLOW.name}, "
        f"found {len(starts)} — the parser below cannot be trusted, and a "
        f"parser that silently finds nothing turns every assertion in this "
        f"module green for free."
    )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    entries: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the {key} filter: {stripped!r}"
        entries.append(matched.group(1))
    assert entries, f"the `{key}` filter parsed empty"
    return entries


def _code_filter_globs() -> list[str]:
    """Return the glob list under the `code:` key of the paths-filter."""
    globs: list[str] = []
    for entry in _filter_entries("code"):
        # `_entry_covers` below knows two shapes: a whole directory
        # (`dir/**`) and an exact path. Any other wildcard would make it
        # guess, and a guess in a coverage predicate is a pass for free.
        # Globs a test reads every match of go under `tree_reads`.
        assert entry.endswith("/**") or "*" not in entry, (
            f"unsupported glob shape in the code filter: {entry!r}; use "
            f"'<dir>/**' for a directory or the exact path for a file"
        )
        globs.append(entry)
    return globs


def _entry_covers(entry: str, path: str) -> bool:
    """Whether one filter entry matches a changed file at `path`.

    `dir/**` covers everything under `dir`. An exact entry covers only
    that path: paths-filter matches with picomatch, and a bare directory
    name matches no file under it.
    """
    if entry.endswith("/**"):
        prefix = entry[:-3]
        return path == prefix or path.startswith(f"{prefix}/")
    return entry == path


def _is_covered(directory: str, globs: list[str]) -> bool:
    """Whether a glob in the filter covers files under `directory`.

    Directory-wide coverage, as distinct from an exact entry for one file
    inside the directory. `docs/user/CONFIG.md` in the filter does not
    make `docs` covered: a change to any other page under `docs/` still
    takes the `echo` branch, which is the property the docs-only tests
    below assert.
    """
    return any(
        g == f"{directory}/**" or (g.startswith(f"{directory}/") and "*" in g)
        for g in globs
    )


# --- what the suite reads, derived from its AST ---------------------------

# What the scan knows an expression may denote, as a set of tagged items:
# ("p", segments) is a repo path as its segments below the repo root, so
# the root itself is ("p", ()); ("s", text) is a string; ("c", items) is a
# literal tuple, list or set of such items; ("m", segments) is a module
# loaded from that repo file, ("spec", segments) the import spec for one,
# and ("fn", segments, name) a function imported from it that reads under
# the root; `_MAYBE` marks a binding or part the scan cannot follow. The
# empty set means the scan cannot tell at all. A set rather than one value
# lets a name bound in several places, or a loop over a tuple of paths,
# yield one path per binding.
_Value = frozenset[tuple]
_UNKNOWN: _Value = frozenset()
_MAYBE = ("?",)

# Calls that build a path from a base and string segments.
_CONSTRUCTORS = frozenset({"Path", "pathlib.Path", "os.path.join"})
# Calls whose result denotes what their one argument does.
_PASS_THROUGH = frozenset({
    "str", "os.fspath", "os.path.abspath", "os.path.realpath", "os.path.normpath",
    "list", "tuple", "sorted",
})
# Attribute steps that consume a path to build another one.
_PATH_STEPS = frozenset({"parent", "parents", "joinpath", "resolve", "absolute"})
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
# Ways to list a directory. On the repo root they read files that no
# path in the source names.
_LISTING_METHODS = frozenset({"glob", "rglob", "iterdir", "walk"})
_LISTING_CALLS = frozenset({"os.walk", "os.listdir", "os.scandir"})
# The comment that says what a line the scan cannot resolve reads.
_PRAGMA = "path-filter:"


def _dotted(node: ast.AST) -> str | None:
    """`os.path.join` for that attribute chain, `Path` for a bare name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _step(segments: tuple[str, ...], text: str) -> tuple[str, ...] | None:
    """`segments` joined with `text`; None for an absolute path or one above the root."""
    if text.startswith("/"):
        return None
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not segments:
                return None
            segments = segments[:-1]
        else:
            segments = (*segments, part)
    return segments


def _join(base: _Value, parts: list[_Value]) -> _Value:
    """Every path `base / part / part ...` may denote, marked when a part is unknown."""
    paths = {item[1] for item in base if item[0] == "p"}
    unknown = _MAYBE in base
    for part in parts:
        texts = [item[1] for item in part if item[0] == "s"]
        unknown = unknown or _MAYBE in part or not texts
        paths = {
            joined
            for segments in paths
            for text in texts
            if (joined := _step(segments, text)) is not None
        }
    return frozenset({("p", segments) for segments in paths} | ({_MAYBE} if unknown else set()))


def _up(value: _Value, levels: int) -> _Value:
    """The paths `levels` directories above each path in `value`, inside the repo."""
    return frozenset(
        ("p", item[1][:-levels])
        for item in value
        if item[0] == "p" and len(item[1]) >= levels
    ) | (frozenset({_MAYBE}) if _MAYBE in value else _UNKNOWN)


def _elements(value: _Value) -> _Value:
    """What a loop over `value` binds its target to."""
    found = frozenset(e for item in value if item[0] == "c" for e in item[1])
    known = value and _MAYBE not in value and all(item[0] == "c" for item in value)
    return found if known else found | {_MAYBE}


def _paths(value: _Value) -> set[tuple[str, ...]]:
    return {item[1] for item in value if item[0] == "p"}


def _resolve(node: ast.AST, names: dict[str, _Value], here: tuple[str, ...]) -> _Value:
    """What `node` may denote; `here` is the module's own repo-relative path."""
    if isinstance(node, ast.Constant):
        return frozenset({("s", node.value)}) if isinstance(node.value, str) else _UNKNOWN
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return frozenset({("p", here)})
        return names.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        elements = [_resolve(e, names, here) or frozenset({_MAYBE}) for e in node.elts]
        return frozenset({("c", frozenset().union(*elements))})
    if isinstance(node, ast.Attribute):
        modules = _modules(_resolve(node.value, names, here))
        if modules:
            # `cdf.REPO_ROOT`: what that module binds the name to.
            return frozenset().union(*(_module_exports(m).names.get(node.attr, _UNKNOWN) for m in modules))
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return _up(_resolve(node.value, names, here), 1)
    if isinstance(node, ast.Subscript):
        index = node.slice
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(index, ast.Constant)
            and type(index.value) is int
            and index.value >= 0
        ):
            return _up(_resolve(node.value.value, names, here), index.value + 1)
        return _UNKNOWN
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _join(_resolve(node.left, names, here), [_resolve(node.right, names, here)])
    if isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args):
        func = node.func
        args = [_resolve(a, names, here) for a in node.args]
        if isinstance(func, ast.Attribute) and func.attr in ("resolve", "absolute") and not args:
            return _resolve(func.value, names, here)
        if isinstance(func, ast.Attribute) and func.attr == "joinpath":
            return _join(_resolve(func.value, names, here), args)
        dotted = _dotted(func)
        if dotted in _CONSTRUCTORS and args:
            return _join(args[0], args[1:])
        if dotted == "os.path.dirname" and len(args) == 1:
            return _up(args[0], 1)
        if dotted in _PASS_THROUGH and len(args) == 1:
            return args[0]
        last = dotted.rsplit(".", 1)[-1] if dotted else None
        if last == "spec_from_file_location" and len(args) >= 2:
            return frozenset(("spec", segments) for segments in _paths(args[1]))
        if last == "module_from_spec" and len(args) == 1:
            return frozenset(("m", item[1]) for item in args[0] if item[0] == "spec")
        if last == "import_module" and len(node.args) == 1:
            return _imported(("", 0, None), here, args[0])
    return _UNKNOWN


def _modules(value: _Value) -> set[tuple[str, ...]]:
    return {item[1] for item in value if item[0] == "m"}


@functools.cache
def _module_file(dotted: str, level: int, here: tuple[str, ...]) -> tuple[str, ...] | None:
    """The repo file `import <dotted>` loads, from `here` for a relative import.

    An absolute name is looked up from the root and from `src/`, the two
    places `pythonpath` and the installed package put on `sys.path`.
    """
    parts = tuple(dotted.split(".")) if dotted else ()
    if not parts or level > len(here):
        return None
    bases = [here[:-level]] if level else [(), ("src",)]
    for base in bases:
        for candidate in ((*base, *parts[:-1], f"{parts[-1]}.py"), (*base, *parts, "__init__.py")):
            if _REPO.joinpath(*candidate).is_file():  # path-filter: none
                return candidate
    return None


def _imported(spec: tuple[str, int, str | None], here: tuple[str, ...], dotted: _Value = _UNKNOWN) -> _Value:
    """What an import binds: `from <module> import <name>`, or a module itself.

    `spec` is `(module, level, name)`, with `name` None for `import
    <module> as x`. `dotted`, for `importlib.import_module`, carries the
    module name as a string instead. A name the module binds to a path is
    that path; a name for a function that reads under the root is ("fn",
    ...); a submodule is ("m", ...).
    """
    module, level, name = spec
    # The importing module matters only to a relative import.
    where = here if level else ()
    found: set[tuple] = set()
    for text in [module] if not dotted else [item[1] for item in dotted if item[0] == "s"]:
        if name is None:
            target = _module_file(text, level, where)
            found |= {("m", target)} if target else set()
            continue
        sub = _module_file(f"{text}.{name}" if text else name, level, where)
        found |= {("m", sub)} if sub else set()
        base = _module_file(text, level, where)
        if base:
            exports = _module_exports(base)
            found |= exports.names.get(name, _UNKNOWN)
            found |= {("fn", base, name)} if name in exports.readers else set()
    return frozenset(found)


_Binding = tuple[str, "ast.AST | tuple[str, int, str | None] | None", bool]


def _bindings(nodes: list[ast.AST]) -> list[_Binding]:
    """`(name, value, is_loop)` per binding in one scope.

    `value` is None for a binding the scan cannot follow (a parameter, a
    tuple target, `with ... as`); it still makes the name local, so a
    module-level path of the same name does not leak in. For an import it
    is the `(module, level, name)` that `_imported` reads.
    """
    found: list[_Binding] = []
    followed: set[ast.AST] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            found += [
                (a.asname or a.name, (node.module or "", node.level, a.name), False)
                for a in node.names
                if a.name != "*"
            ]
        elif isinstance(node, ast.Import):
            found += [(a.asname, (a.name, 0, None), False) for a in node.names if a.asname]
        elif isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            found += [(t.id, node.value, False) for t in targets]
            followed.update(targets)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append((node.target.id, node.value, False))
            followed.add(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if isinstance(node.target, ast.Name):
                found.append((node.target.id, node.iter, True))
                followed.add(node.target)
        elif isinstance(node, ast.arg):
            found.append((node.arg, None, False))
    found += [
        (node.id, None, False)
        for node in nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node not in followed
    ]
    return found


def _bound_names(
    nodes: list[ast.AST], outer: dict[str, _Value], here: tuple[str, ...]
) -> dict[str, _Value]:
    """Each name bound in one scope, mapped to everything its bindings may denote.

    A name bound locally hides the enclosing scope's binding. The union
    over a name's bindings is iterated to a fixpoint, bounded by the
    number of bindings, so a name bound from another name resolves
    whatever order they appear in.
    """
    bindings = _bindings(nodes)
    local = {name for name, _, _ in bindings}
    names = {k: v for k, v in outer.items() if k not in local}
    # An import binds the same value on every round; resolve it once.
    imports = {id(node): _imported(node, here) for _, node, _ in bindings if isinstance(node, tuple)}
    for _ in range(len(bindings) + 1):
        changed = False
        for name, node, is_loop in bindings:
            if isinstance(node, tuple):
                value = imports[id(node)]
            else:
                value = _UNKNOWN if node is None else _resolve(node, names, here)
            if is_loop:
                value = _elements(value)
            value = value or frozenset({_MAYBE})
            merged = names.get(name, _UNKNOWN) | value
            if merged != names.get(name, _UNKNOWN):
                names[name] = merged
                changed = True
        if not changed:
            break
    return names


def _consumed(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether `node` is the base of a larger path expression, recorded instead."""
    parent = parents.get(node)
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
        return parent.left is node
    if isinstance(parent, ast.Attribute):
        return parent.value is node and parent.attr in _PATH_STEPS
    if isinstance(parent, ast.Call) and parent.args and parent.args[0] is node:
        dotted = _dotted(parent.func)
        return dotted in _CONSTRUCTORS or dotted in _PASS_THROUGH or dotted == "os.path.dirname"
    return False


def _base_of(node: ast.AST) -> ast.AST | None:
    """The expression a join builds on."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return node.left
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
            return node.func.value
        if _dotted(node.func) in _CONSTRUCTORS and node.args:
            return node.args[0]
    return None


class _Scan(NamedTuple):
    """What one test module reads, as far as its source says."""

    # Repo paths, files or directories, that the module builds or declares.
    reads: frozenset[str]
    # Globs declared in `# path-filter:` comments, such as `**/*.md`.
    patterns: frozenset[str]
    # The entries of `reads` that came from a comment rather than code.
    declared: frozenset[str]
    # Expressions on the repo root the scan cannot resolve, and comments
    # it cannot use, one line each.
    problems: tuple[str, ...]


_NO_SCAN = _Scan(frozenset(), frozenset(), frozenset(), ())


def _pragmas(source: str) -> dict[int, list[str]]:
    """The words after each `# path-filter:` comment, by line number."""
    found: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            body = token.string[1:].strip()
            if body.startswith(_PRAGMA):
                found[token.start[0]] = body[len(_PRAGMA) :].split()
    return found


def _lists_the_root(node: ast.AST, names: dict[str, _Value], here: tuple[str, ...]) -> bool:
    """Whether `node` lists the repo root itself: `_REPO.glob(...)`, `os.walk(_REPO)`."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if _dotted(func) in _LISTING_CALLS:
        return bool(node.args) and () in _paths(_resolve(node.args[0], names, here))
    if isinstance(func, ast.Attribute) and func.attr in _LISTING_METHODS:
        return () in _paths(_resolve(func.value, names, here))
    return False


def _flatten(value: _Value) -> _Value:
    """`value` with each literal tuple, list or set replaced by its elements."""
    items: set[tuple] = set()
    pending = list(value)
    while pending:
        item = pending.pop()
        if item[0] == "c":
            pending.extend(item[1])
        else:
            items.add(item)
    return frozenset(items)


def _builds_a_string_on_the_root(node: ast.AST, names: dict[str, _Value], here: tuple[str, ...]) -> bool:
    """Whether `node` builds a string from the repo root: `f"{_REPO}/docs/x.md"`.

    The scan follows path joins, not string operations, so it cannot say
    which file such a string names, as with `_REPO / name`. An f-string,
    `+`, `%`, `.format` and `str.join` count; `os.path.join` is a path
    constructor and resolves.
    """
    if isinstance(node, ast.JoinedStr):
        operands = [v.value for v in node.values if isinstance(v, ast.FormattedValue)]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        operands = [node.left, node.right]
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("format", "join")
        and _dotted(node.func) not in _CONSTRUCTORS
    ):
        operands = [*node.args, *(k.value for k in node.keywords)]
    else:
        return False
    return any(() in _paths(_flatten(_resolve(o, names, here))) for o in operands)


def _calls_a_reader(node: ast.AST, names: dict[str, _Value], here: tuple[str, ...]) -> bool:
    """Whether `node` calls another module's function that reads under the root.

    `cdf.iter_files(roots)` joins each root onto the module's own repo
    root, so the call reads files that no path in the caller names.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        modules = _modules(_resolve(func.value, names, here))
        return any(func.attr in _module_exports(m).readers for m in modules)
    if isinstance(func, ast.Name):
        return any(item[0] == "fn" for item in names.get(func.id, _UNKNOWN))
    return False


class _Analysis(NamedTuple):
    """Every path one module builds, before `# path-filter:` comments apply."""

    tree: ast.Module
    owner: dict[ast.AST, ast.AST | None]
    enclosing: dict[ast.AST, ast.AST | None]
    module_names: dict[str, _Value]
    found: frozenset[tuple[str, ...]]
    # Nodes that read under the root through a path the scan cannot name.
    unresolved: tuple[ast.AST, ...]


@functools.cache
def _parse(source: str) -> ast.Module:
    """`ast.parse` once per source; nothing here mutates a tree."""
    return ast.parse(source)


def _analyse(source: str, here: tuple[str, ...]) -> _Analysis:
    """Resolve every path expression in a module at `here`."""
    tree = _parse(source)
    # One pass over the tree: each node's parent, the scope that owns it
    # (None at module level) and each scope's enclosing scope. A function
    # node itself belongs to the scope that defines it.
    parents: dict[ast.AST, ast.AST] = {}
    owner: dict[ast.AST, ast.AST | None] = {}
    members: dict[ast.AST | None, list[ast.AST]] = {None: []}
    enclosing: dict[ast.AST, ast.AST | None] = {}
    pending: list[tuple[ast.AST, ast.AST | None]] = [(tree, None)]
    while pending:
        node, scope = pending.pop()
        owner[node] = scope
        members.setdefault(scope, []).append(node)
        inner = scope
        if isinstance(node, _SCOPES):
            enclosing[node] = scope
            inner = node
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            pending.append((child, inner))
    scopes: dict[ast.AST | None, dict[str, _Value]] = {
        None: _bound_names(members[None], {}, here)
    }

    def names_of(scope: ast.AST | None) -> dict[str, _Value]:
        if scope not in scopes:
            scopes[scope] = _bound_names(members.get(scope, []), names_of(enclosing[scope]), here)
        return scopes[scope]

    found: set[tuple[str, ...]] = set()
    unresolved: list[ast.AST] = []
    for node in owner:
        names = names_of(owner[node])
        if (
            _lists_the_root(node, names, here)
            or _calls_a_reader(node, names, here)
            or _builds_a_string_on_the_root(node, names, here)
        ):
            unresolved.append(node)
        if _consumed(node, parents):
            continue
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                continue
            if isinstance(parents.get(node), (ast.Assign, ast.AnnAssign)):
                continue
            value = names.get(node.id, _UNKNOWN)
        elif isinstance(node, (ast.BinOp, ast.Call, ast.Attribute, ast.Subscript)):
            value = _resolve(node, names, here)
        else:
            continue
        found |= _paths(value)
        if _paths(value) and _MAYBE not in value:
            continue
        current: ast.AST | None = node
        while (current := _base_of(current)) is not None:
            prefix = _paths(_resolve(current, names, here))
            if prefix:
                found |= prefix
                if () in prefix:
                    unresolved.append(node)
                break
    return _Analysis(tree, owner, enclosing, scopes[None], frozenset(found), tuple(unresolved))


class _Exports(NamedTuple):
    """What a module hands the test that loads it."""

    # Module-level names bound to a repo path, such as `REPO_ROOT`.
    names: dict[str, _Value]
    # Functions that read under the root through a path the scan cannot
    # resolve, such as `iter_files(roots)`, and those that call one.
    readers: frozenset[str]


_NO_EXPORTS = _Exports({}, frozenset())
_EXPORTS: dict[tuple[str, ...], _Exports] = {}
# The module whose exports are being computed, if any.
_EXPORTING: set[tuple[str, ...]] = set()


def _exports_of(source: str, here: tuple[str, ...]) -> _Exports:
    """The path-valued names and the reader functions of a module at `here`.

    A function is a reader when a node it owns reads under the root
    unresolved, with a comment or without one: the comment states what the
    line reads for that module's own arguments, and a caller passes its
    own. A function that calls a reader of the same module by its bare
    name is a reader too. Calls through `self`, and reads a module makes
    when it loads, are not followed.
    """
    analysis = _analyse(source, here)
    names = {n: v for n, v in analysis.module_names.items() if _paths(v)}
    functions: dict[str, ast.AST] = {}
    for node in analysis.tree.body:
        body = node.body if isinstance(node, ast.ClassDef) else [node]
        functions.update((f.name, f) for f in body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)))
    readers: set[str] = set()
    for node in analysis.unresolved:
        scope = analysis.owner[node]
        while scope is not None and analysis.enclosing[scope] is not None:
            scope = analysis.enclosing[scope]
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            readers.add(scope.name)
    calls = {
        name: {c.func.id for c in ast.walk(f) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        for name, f in functions.items()
    }
    for _ in range(len(calls) + 1):
        grown = {name for name, called in calls.items() if called & readers} - readers
        if not grown:
            break
        readers |= grown
    return _Exports(names, frozenset(readers))


def _module_exports(segments: tuple[str, ...]) -> _Exports:
    """`_exports_of` the repo module at `segments`, once per module.

    One module deep: while one module's exports are computed, the modules
    it loads export nothing. So a loaded module that reads through a
    third module is not followed, an import cycle ends, and what a module
    exports does not depend on which test reached it first.
    """
    if segments in _EXPORTS:
        return _EXPORTS[segments]
    path = _REPO.joinpath(*segments)  # path-filter: none
    if _EXPORTING or path.suffix != ".py" or not path.is_file():
        return _NO_EXPORTS
    source = path.read_text(encoding="utf-8")
    # Followed one module deep, a module builds paths only from its own
    # `__file__`, so one that never names it exports nothing.
    exports = _NO_EXPORTS
    if "__file__" in source:
        _EXPORTING.add(segments)
        try:
            exports = _exports_of(source, segments)
        finally:
            _EXPORTING.discard(segments)
    _EXPORTS[segments] = exports
    return exports


@functools.cache
def _exporting_stems() -> frozenset[str]:
    """The names of the repo modules that export a path or a reader.

    A module that names none of them, and not `__file__`, can reach no
    repo path the scan follows, so it is not parsed. A package's
    `__init__.py` goes by the package's name.
    """
    return frozenset(
        path.parent.name if path.stem == "__init__" else path.stem
        for package in sorted(_IN_REPO_PACKAGES)
        for path in sorted(_REPO.joinpath(package).rglob("*.py"))  # path-filter: none
        if _module_exports(path.relative_to(_REPO).parts) != _NO_EXPORTS
    )


def _scan_source(source: str, here: tuple[str, ...]) -> _Scan:
    """What a module at `here` reads through paths built from its own location.

    Names are scoped: a module-level binding is visible everywhere, a
    binding inside a function only inside that function, and a function's
    own binding of a name hides the module's. Without the split,
    `directory = tmp_path / "x"` in one test and `directory = _REPO /
    "CHANGELOG"` in another would read as the same name.

    Only the outermost expression of a chain is recorded, so `_REPO /
    "docs" / "user" / "CONFIG.md"` yields one path and not three. When
    the outermost join does not resolve (`_REPO / "docs" / name`), the
    longest resolvable prefix is recorded instead, since the test reads
    under it. When that prefix is the repo root itself, as in `_REPO /
    name` with `name` read from `git ls-files`, or when the module lists
    the root or builds a string on it (`f"{_REPO}/x"`, `str(_REPO) +
    "/x"`), the scan cannot say what is read. That line must carry a
    `# path-filter:` comment naming exact paths, `<dir>/**` or a glob, or
    `none`, and is otherwise reported as a problem. So is a comment on a
    line with nothing for it to declare.

    A path another module holds counts as this module's own. The scan
    follows a module the test loads from a repo file, through
    `importlib.util.spec_from_file_location`, `importlib.import_module`,
    `import ... as` or `from ... import`: `cdf.REPO_ROOT` is that module's
    root, and a call to one of its functions that reads under its root,
    such as `cdf.iter_files(roots)`, is a read the scan cannot resolve.
    """
    if "__file__" not in source and not any(stem in source for stem in _exporting_stems()):
        return _NO_SCAN
    analysis = _analyse(source, here)
    found, unresolved = analysis.found, analysis.unresolved
    reads = {"/".join(segments) for segments in found if segments}
    patterns: set[str] = set()
    declared: set[str] = set()
    problems: list[str] = []
    pragmas = _pragmas(source)
    used: set[int] = set()
    for node in unresolved:
        lines = [n for n in range(node.lineno, (node.end_lineno or node.lineno) + 1) if n in pragmas]
        if not lines:
            problems.append(f"line {node.lineno}: {ast.unparse(node)}")
        used.update(lines)
    for line, words in sorted(pragmas.items()):
        if line not in used:
            problems.append(f"line {line}: `# {_PRAGMA}` comment with no unresolved path on its line")
        elif not words or ("none" in words and len(words) > 1):
            problems.append(f"line {line}: `# {_PRAGMA}` needs paths, globs, or `none` alone")
        else:
            for word in words:
                if word == "none":
                    continue
                if word.endswith("/**") and "*" not in word[:-3]:
                    declared.add(word[:-3])
                elif "*" in word:
                    patterns.add(word)
                else:
                    declared.add(word)
    return _Scan(frozenset(reads | declared), frozenset(patterns), frozenset(declared), tuple(problems))


def _scan_module(path: Path) -> _Scan:
    """`_scan_source` over one file under the repo."""
    return _scan_source(path.read_text(encoding="utf-8"), path.relative_to(_REPO).parts)


@functools.cache
def _suite_scans() -> dict[str, _Scan]:
    """The scan of every module under `tests/`, by repo-relative path."""
    return {
        path.relative_to(_REPO).as_posix(): _scan_module(path)
        for path in sorted(_REPO.joinpath("tests").rglob("*.py"))
    }


def _missing_declared_paths(scans: dict[str, _Scan]) -> list[str]:
    """`<module>: <path>` for each declared path absent from the tree.

    `_suite_root_reads` drops absent paths, so a mistyped declaration
    would otherwise vanish without a word.
    """
    return sorted(
        f"{module}: {path}"
        for module, scan in scans.items()
        for path in scan.declared
        if not (_REPO / path).exists()  # path-filter: none
    )


def _suite_root_reads() -> dict[str, set[str]]:
    """Every repo-relative path `tests/` builds, mapped to the modules that build it.

    Paths absent from the tree are dropped: a test may spell a path to
    assert that it does not exist, and nothing can change under a path
    that is not there.
    """
    reads: dict[str, set[str]] = {}
    for module, scan in _suite_scans().items():
        for found in scan.reads:
            if (_REPO / found).exists():  # path-filter: none
                reads.setdefault(found, set()).add(module)
    return reads


def _suite_pattern_reads() -> dict[str, set[str]]:
    """Every glob a module declares it reads, mapped to the modules that declare it."""
    patterns: dict[str, set[str]] = {}
    for module, scan in _suite_scans().items():
        for pattern in scan.patterns:
            patterns.setdefault(pattern, set()).add(module)
    return patterns


def _suite_file_reads() -> dict[str, set[str]]:
    return {p: m for p, m in _suite_root_reads().items() if (_REPO / p).is_file()}  # path-filter: none


def _suite_directory_reads() -> dict[str, set[str]]:
    return {p: m for p, m in _suite_root_reads().items() if (_REPO / p).is_dir()}  # path-filter: none


def _suite_dependencies() -> set[str]:
    """Top-level in-repo packages that `tests/` imports or reads as code.

    A package counts when a test imports it, lists it, or opens a `.py`
    under it. A test that opens one document under a directory that also
    holds Python (`docs/assets/` has render scripts) does not make that
    directory a package dependency; the file tests below cover the page.
    """
    found: set[str] = set()
    for path in sorted(_REPO.joinpath("tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found |= {n for n in _IMPORT_RE.findall(text) if n in _IN_REPO_PACKAGES}
    code_reads = {
        p.split("/")[0]
        for p in _suite_root_reads()
        if (_REPO / p).is_dir() or p.endswith(".py")  # path-filter: none
    }
    return found | (code_reads & _IN_REPO_PACKAGES)


_TREE_READS_MATCH = "steps.filter.outputs.tree_reads == 'true'"


def _pytest_job_steps() -> list[list[str]]:
    """The lines of each step of ci.yml's first job, `pytest`."""
    steps: list[list[str]] = []
    for line in _CI_WORKFLOW.read_text(encoding="utf-8").splitlines():
        if re.match(r"\s{6}- ", line):
            steps.append([line])
        elif steps and (not line.strip() or line.startswith(" " * 8)):
            steps[-1].append(line)
        elif steps and line.strip() and not line.startswith(" " * 6):
            break
    assert len(steps) >= 5, f"parsed only {len(steps)} steps of the pytest job"
    return steps


def _tree_read_step_modules() -> list[str]:
    """The test modules the pytest job runs when only `tree_reads` matches.

    Read from the one step gated on `tree_reads == 'true'` whose command
    runs pytest, so a second such step, or a renamed output, fails here
    instead of leaving the modules unread.
    """
    runs = [
        step
        for step in _pytest_job_steps()
        if any(_TREE_READS_MATCH in line for line in step)
        and any(line.strip().startswith("run: uv run pytest ") for line in step)
    ]
    assert len(runs) == 1, (
        f"expected one pytest step gated on `tree_reads` in {_CI_WORKFLOW.name}, "
        f"found {len(runs)}"
    )
    command = next(line for line in runs[0] if line.strip().startswith("run: "))
    modules = re.findall(r"tests/\S+\.py\b", command)
    assert modules, f"the `tree_reads` step names no test module: {command.strip()!r}"
    return modules


def _release_docs_check_paths() -> list[str]:
    """The CHANGELOG paths `release-docs-check` scans on every PR."""
    found = _DUPES_INVOCATION_RE.search(_STAGING_GATE_WORKFLOW.read_text(encoding="utf-8"))
    assert found, (
        f"{_STAGING_GATE_WORKFLOW.name} no longer runs check_changelog_dupes.py; "
        f"the CHANGELOG directories the suite reads have lost their per-PR check "
        f"and must enter the `code` filter"
    )
    return shlex.split(found.group(1))


def _checked_by_release_docs_check(directory: str, scanned: list[str]) -> bool:
    """Whether `release-docs-check` scans `directory` or something under it."""
    return any(s == directory or s.startswith(f"{directory}/") for s in scanned)


def test_the_discovery_scan_is_not_vacuous() -> None:
    """Guard the guard: an empty scan would pass every test below."""
    deps = _suite_dependencies()
    assert {"src", "tests"} <= deps, (
        f"the scan found {sorted(deps)}, which is missing the packages the "
        f"suite unarguably uses — the regexes have stopped matching and the "
        f"coverage assertion below is now vacuous"
    )


def test_the_read_scan_is_not_vacuous() -> None:
    """The AST scan must see both spellings, or the drift tests pass for free.

    This module binds `_CI_WORKFLOW` to a path joined onto `_REPO` and
    reads it, which is the name-bound spelling. `test_docs_truth_1511.py`
    reads `docs/concepts/ARCHITECTURE.md` the same way. Both are asserted
    by name so that a scanner that stops following names fails here
    rather than turning every coverage assertion below green.
    """
    reads = _suite_root_reads()
    assert reads.get(".github/workflows/ci.yml", set()) >= {"tests/test_ci_path_filter.py"}, (
        f"the scan did not see this module reading ci.yml through `_CI_WORKFLOW`; "
        f"it found {sorted(reads)[:10]}..."
    )
    assert "docs/concepts/ARCHITECTURE.md" in reads, (
        "the scan did not see test_docs_truth_1511.py reading ARCHITECTURE.md"
    )
    assert "tests" in _suite_directory_reads(), (
        "the scan did not see any module reading the `tests` directory itself"
    )


def test_the_read_scan_credits_every_reader_of_an_entry() -> None:
    """The reverse test trusts this map, so a missed reader is a wrong verdict.

    `test_check_doc_defaults_1654.py` reads `docs/user/CONFIG.md` through
    a loop over a tuple of paths, and `test_pyright_ratchet_1503.py` reads
    `docs/concepts/RELEASING.md` the same way. A scan that credits only
    the inline readers would, once those stopped reading, tell the reverse
    test to drop an entry a looping test still needs.
    """
    reads = _suite_file_reads()
    for path, module in (
        ("docs/user/CONFIG.md", "tests/test_check_doc_defaults_1654.py"),
        ("docs/concepts/RELEASING.md", "tests/test_pyright_ratchet_1503.py"),
    ):
        assert module in reads.get(path, set()), (
            f"the scan does not credit {module} with reading {path}; it "
            f"credits {sorted(reads.get(path, set()))}"
        )


_PROBE_HEAD = (
    "import os\n"
    "from pathlib import Path\n"
    "_R = Path(__file__).resolve().parents[1]\n"
)

# One test module per spelling of a read of `docs/new.md`.
_SPELLINGS = {
    "inline": 'def test_x():\n    (Path(__file__).resolve().parents[1] / "docs" / "new.md").read_text()\n',
    "name_bound": 'def test_x():\n    (_R / "docs" / "new.md").read_text()\n',
    "string_constant": '_DOC = "docs/new.md"\ndef test_x():\n    (_R / _DOC).read_text()\n',
    "tuple_loop": (
        '_FILES = ("README.md", "docs/new.md")\n'
        "def test_x():\n    for rel in _FILES:\n        (_R / rel).read_text()\n"
    ),
    "comprehension": 'def test_x():\n    [(_R / rel).read_text() for rel in ["docs/new.md"]]\n',
    "joinpath": 'def test_x():\n    _R.joinpath("docs", "new.md").read_text()\n',
    "path_constructor": 'def test_x():\n    Path(_R, "docs", "new.md").read_text()\n',
    "os_path": (
        "def test_x():\n    open(os.path.join(os.path.dirname(os.path.dirname("
        'os.path.abspath(__file__))), "docs", "new.md")).read()\n'
    ),
    "dotdot": 'def test_x():\n    (Path(__file__).parent / ".." / "docs" / "new.md").read_text()\n',
    "function_local_root": (
        "def test_x():\n    root = Path(__file__).parent.parent\n"
        '    (root / "docs/new.md").read_text()\n'
    ),
}


@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_the_read_scan_resolves_each_spelling(spelling: str) -> None:
    """Each spelling in the suite of a read under the repo resolves to its path.

    The tuple loop is how `test_pyright_ratchet_1503.py` and
    `test_check_doc_defaults_1654.py` read; the others are each one
    rewrite away from the inline form. A spelling the scan misses lets a
    new test read a new file outside the filter with every test green.
    """
    scan = _scan_source(_PROBE_HEAD + _SPELLINGS[spelling], ("tests", "test_probe.py"))
    assert "docs/new.md" in scan.reads, f"{spelling}: the scan found {sorted(scan.reads)}"
    assert not scan.problems, scan.problems


def test_a_name_bound_twice_reads_under_both() -> None:
    """Either binding can reach the join, so both paths count as read."""
    source = _PROBE_HEAD + (
        "def test_x(flag):\n"
        '    base = _R / "other"\n'
        "    if flag:\n"
        '        base = _R / "docs"\n'
        '    (base / "new.md").read_text()\n'
    )
    reads = _scan_source(source, ("tests", "test_probe.py")).reads
    assert {"docs/new.md", "other/new.md"} <= reads, sorted(reads)


# One test module per way of reading under the root that the scan cannot
# name a path for.
_UNRESOLVED = {
    "git_ls_files": (
        "import subprocess\n"
        "def test_x():\n"
        '    out = subprocess.run(["git", "ls-files", "*.md"], cwd=_R, capture_output=True,\n'
        "                         text=True, check=True, timeout=30).stdout\n"
        "    for p in out.split():\n"
        "        (_R / p).read_text()\n"
    ),
    "parameter": "def test_x(name):\n    (_R / name).read_text()\n",
    "imported_name": "from elsewhere import NAME\ndef test_x():\n    (_R / NAME).read_text()\n",
    "partly_known_name": (
        "def test_x(other):\n"
        '    rel = "README.md"\n'
        "    if other:\n"
        "        rel = other\n"
        "    (_R / rel).read_text()\n"
    ),
    "tuple_with_unknown": (
        'def test_x():\n    for rel in ("README.md", os.environ["DOC"]):\n        (_R / rel).read_text()\n'
    ),
    "loop_over_partly_known": (
        '_NAMES = ("README.md",)\n'
        'if os.environ.get("DOCS"):\n'
        '    _NAMES = os.environ["DOCS"]\n'
        "def test_x():\n"
        "    for rel in _NAMES:\n"
        "        (_R / rel).read_text()\n"
    ),
    "root_glob": 'def test_x():\n    for p in sorted(_R.glob("*.md")):\n        p.read_text()\n',
    "root_walk": "def test_x():\n    for top, dirs, files in os.walk(_R):\n        print(top, files)\n",
    "fstring": 'def test_x():\n    open(f"{_R}/docs/new.md").read()\n',
    "str_concat": 'def test_x():\n    open(str(_R) + "/docs/new.md").read()\n',
    "percent": 'def test_x():\n    open("%s/docs/new.md" % (_R,)).read()\n',
    "format": 'def test_x():\n    open("{}/docs/new.md".format(_R)).read()\n',
    "os_sep_join": 'def test_x():\n    open(os.sep.join([str(_R), "docs", "new.md"])).read()\n',
}


@pytest.mark.parametrize("spelling", sorted(_UNRESOLVED))
def test_a_root_path_the_scan_cannot_resolve_is_a_problem(spelling: str) -> None:
    """Each of these reads files under the root that the drift tests cannot see."""
    scan = _scan_source(_PROBE_HEAD + _UNRESOLVED[spelling], ("tests", "test_probe.py"))
    assert scan.problems, f"{spelling}: no problem reported; reads {sorted(scan.reads)}"


# Probes that reach the root through another repo module, one module deep.
_CDF_HEAD = (
    "import importlib.util\n"
    "from pathlib import Path\n"
    "_R = Path(__file__).resolve().parents[1]\n"
    '_spec = importlib.util.spec_from_file_location("_cdf", _R / "scripts" / "check_derived_figures.py")\n'
    "cdf = importlib.util.module_from_spec(_spec)\n"
)


@pytest.mark.parametrize(
    "source",
    [
        'from tests.test_ci_path_filter import _REPO\ndef test_x():\n    (_REPO / "docs" / "new.md").read_text()\n',
        'import tests.test_ci_path_filter as f\ndef test_x():\n    (f._REPO / "docs" / "new.md").read_text()\n',
        _CDF_HEAD + 'def test_x():\n    (cdf.REPO_ROOT / "docs" / "new.md").read_text()\n',
    ],
    ids=["from_import", "import_as", "importlib_spec"],
)
def test_a_root_another_module_binds_resolves_as_the_module_own(source: str) -> None:
    """#1786: a root imported or read off a loaded module is the test's root.

    The first two probes name no `__file__`, so they also pin that such a
    module is parsed at all.
    """
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert "docs/new.md" in scan.reads, sorted(scan.reads)
    assert not scan.problems, scan.problems


@pytest.mark.parametrize(
    "source",
    [
        _CDF_HEAD + "def test_x():\n    cdf.iter_files(list(cdf.DEFAULT_ROOTS))\n",
        _CDF_HEAD + "def test_x():\n    for p in cdf.iter_files(['docs']):\n        p.read_text()\n",
        "from benchmarks.published_constants import figures\ndef test_x():\n    figures()\n",
    ],
    ids=["iter_files", "iter_files_loop", "imported_reader"],
)
def test_a_call_to_another_module_reader_is_a_problem(source: str) -> None:
    """#1786: `cdf.iter_files(...)` reads every page under its roots unseen.

    `iter_files` joins each root onto its module's own repo root, and
    `figures` calls a function that does the same, so neither call names
    the files it reads.
    """
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert scan.problems, f"no problem reported; reads {sorted(scan.reads)}"


def test_a_reader_call_that_declares_its_reads_is_not_a_problem() -> None:
    source = _CDF_HEAD + "def test_x():\n    cdf.iter_files(['docs'])  # path-filter: **/*.md\n"
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert not scan.problems, scan.problems
    assert scan.patterns == {"**/*.md"}


def test_an_unknown_name_under_a_directory_reads_the_directory() -> None:
    """Below the root, the resolvable prefix is the read and nothing is unresolved."""
    source = _PROBE_HEAD + 'def test_x(name):\n    (_R / "docs" / name).read_text()\n'
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert scan.reads == {"docs"}
    assert not scan.problems, scan.problems


def test_a_path_filter_comment_declares_what_the_line_reads() -> None:
    source = _PROBE_HEAD + (
        "def test_x(name, other, third):\n"
        "    (_R / name).read_text()  # path-filter: **/*.md\n"
        "    (_R / other).read_text()  # path-filter: docs/user/CONFIG.md docs/assets/**\n"
        "    (_R / third).exists()  # path-filter: none\n"
    )
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert not scan.problems, scan.problems
    assert scan.patterns == {"**/*.md"}
    assert scan.declared == {"docs/user/CONFIG.md", "docs/assets"}
    assert scan.declared <= scan.reads


def test_a_declared_path_that_does_not_exist_is_reported() -> None:
    source = _PROBE_HEAD + (
        "def test_x(name):\n"
        "    (_R / name).read_text()  # path-filter: docs/no-such-page.md README.md\n"
    )
    scan = _scan_source(source, ("tests", "test_probe.py"))
    assert _missing_declared_paths({"tests/test_probe.py": scan}) == [
        "tests/test_probe.py: docs/no-such-page.md"
    ]


@pytest.mark.parametrize(
    "line",
    [
        "    (_R / name).read_text()  # path-filter:\n",
        "    (_R / name).read_text()  # path-filter: none docs/x.md\n",
        '    (_R / "README.md").read_text()  # path-filter: README.md\n',
    ],
    ids=["empty", "none_with_paths", "nothing_to_declare"],
)
def test_a_path_filter_comment_the_scan_cannot_use_is_a_problem(line: str) -> None:
    """A malformed comment, or one that outlived its unresolved path."""
    scan = _scan_source(_PROBE_HEAD + "def test_x(name):\n" + line, ("tests", "test_probe.py"))
    assert scan.problems


def test_a_function_binding_hides_the_module_binding() -> None:
    """A local `directory` is not the module's `directory`."""
    source = _PROBE_HEAD + (
        '_DIR = _R / "CHANGELOG"\n'
        "def test_x(tmp_path):\n"
        '    _DIR = tmp_path / "x"\n'
        '    (_DIR / "a.md").read_text()\n'
    )
    assert _scan_source(source, ("tests", "test_probe.py")).reads == {"CHANGELOG"}


def test_an_exact_entry_covers_one_path_and_a_glob_covers_a_tree() -> None:
    """Pin the paths-filter semantics the coverage tests lean on.

    paths-filter matches with picomatch (`dot: true`), so `docs/**` matches
    every file under `docs/`, `docs/user/CONFIG.md` matches that file, and a
    bare `docs` matches no changed file at all. A predicate that let an
    exact entry prefix-match would call `docs/user/CONFIG.md` covered by a
    `docs` entry that runs nothing.
    """
    assert _entry_covers("docs/**", "docs/user/CONFIG.md")
    assert _entry_covers("docs/**", "docs")
    assert _entry_covers(".github/**", ".github/workflows/ci.yml")
    assert _entry_covers("docs/user/CONFIG.md", "docs/user/CONFIG.md")
    assert not _entry_covers("docs/user/CONFIG.md", "docs/user/CONFIG.md.bak")
    assert not _entry_covers("docs", "docs/user/CONFIG.md")
    assert not _entry_covers("docs/user", "docs/user/CONFIG.md")
    assert not _entry_covers("src/**", "scripts/x.py")


def test_every_repo_path_a_test_builds_is_resolved() -> None:
    """A read the scan cannot resolve is a read the drift tests cannot check.

    A join onto the repo root with an operand the scan cannot follow,
    such as `REPO / p` over `git ls-files` output or `_REPO / path` over
    a parameter, or a listing of the root itself, reads files that no
    path in the source names. Such a line must say what it reads in a
    `# path-filter:` comment: exact paths, `<dir>/**`, a glob such as
    `**/*.md`, or `none` when it reads nothing the filter must cover,
    such as a stat of a path the scan already found. The declared paths
    then take part in every check here as if the code had spelled them.
    """
    scans = _suite_scans()
    problems = {module: list(scan.problems) for module, scan in scans.items() if scan.problems}
    assert not problems, (
        f"paths on the repo root that the scan cannot resolve, or "
        f"`# {_PRAGMA}` comments it cannot use: {problems}. Spell the path "
        f"so the scan resolves it, or say on the line what it reads."
    )
    missing = _missing_declared_paths(scans)
    assert not missing, f"`# {_PRAGMA}` comments name paths that do not exist: {missing}"


def test_code_filter_covers_every_in_repo_package_the_suite_uses() -> None:
    globs = _code_filter_globs()
    uncovered = sorted(d for d in _suite_dependencies() if not _is_covered(d, globs))
    assert not uncovered, (
        f"{uncovered} are used by tests/ but absent from the `code` filter in "
        f"{_CI_WORKFLOW.name}. A PR touching only those paths takes the `echo` "
        f"branch and reports a passing pytest matrix without running the tests "
        f"that cover them. Add '<dir>/**' to the filter."
    )


def test_every_file_the_suite_reads_is_in_the_code_filter() -> None:
    """#1786 CI-07: a file a test opens must run the suite when it changes.

    Derived, not listed: the set of files comes from the AST scan of
    `tests/`, so a new test that reads a new file outside the filter
    fails here on the PR that adds it, with the file and the module named.
    """
    globs = _code_filter_globs()
    missing = {
        path: sorted(modules)
        for path, modules in _suite_file_reads().items()
        if not any(_entry_covers(g, path) for g in globs)
    }
    assert not missing, (
        f"files read by tests/ but absent from the `code` filter in "
        f"{_CI_WORKFLOW.name}: {missing}. A PR that changes only such a file "
        f"takes the `echo` branch and never runs the test that reads it. Add "
        f"the exact path to the filter."
    )


def test_every_directory_the_suite_reads_is_in_the_code_filter() -> None:
    """A directory a test lists must be under a `dir/**` glob.

    One exception, by ruling (#413/#427): the CHANGELOG directories. A
    fragment-only PR stays off the suite, and `release-docs-check` in
    staging-gate.yml scans those same directories on every PR. The
    exception is read from that job's arguments, so removing the job or
    a directory from it puts the directory back under this assertion.
    """
    globs = _code_filter_globs()
    exempt = _release_docs_check_paths()
    # A bare directory name as an exact entry would pass `_entry_covers`
    # here while matching no file under it in paths-filter. The reverse
    # test below rejects such an entry, since no test reads it as a file.
    uncovered = {
        path: sorted(modules)
        for path, modules in _suite_directory_reads().items()
        if not any(_entry_covers(g, path) for g in globs)
        and not _checked_by_release_docs_check(path, exempt)
    }
    assert not uncovered, (
        f"directories listed by tests/ but not under a '<dir>/**' glob in "
        f"{_CI_WORKFLOW.name}: {uncovered}. An exact entry covers one file; "
        f"a test that globs a directory needs the directory."
    )


def test_the_release_docs_exemption_is_narrow() -> None:
    """Only a directory the dupes check names, or a parent of one, is exempt."""
    scanned = ["CHANGELOG/*.md", "CHANGELOG.md", "CHANGELOG/unreleased"]
    assert _checked_by_release_docs_check("CHANGELOG", scanned)
    assert _checked_by_release_docs_check("CHANGELOG/unreleased", scanned)
    assert not _checked_by_release_docs_check("CHANGELOG/unreleased/sub", scanned)
    assert not _checked_by_release_docs_check("CHANGELOG.md", ["CHANGELOG/*.md"])
    assert not _checked_by_release_docs_check("docs", scanned)
    assert not _checked_by_release_docs_check("C", scanned)
    assert not _checked_by_release_docs_check("CHANGELOG", [])


def test_every_exact_entry_in_the_code_filter_is_read_by_a_test() -> None:
    """The reverse of the file test: no entry outlives the test that needed it.

    An exact entry nobody reads runs the suite for nothing on every PR
    that touches the file, and the list silently becomes the
    hand-maintained kind. Build inputs are the exception: they change
    what `uv sync` installs, whether or not a test reads them.
    """
    reads = _suite_file_reads()
    stale = sorted(
        g
        for g in _code_filter_globs()
        if not g.endswith("/**") and g not in _BUILD_INPUTS and g not in reads
    )
    assert not stale, (
        f"exact entries in the `code` filter of {_CI_WORKFLOW.name} that no "
        f"test reads: {stale}. Remove them, or move the test that read them "
        f"back under tests/."
    )


def test_every_glob_the_suite_reads_runs_its_readers_when_a_match_changes() -> None:
    """#1786 CI-07: a test that reads every match of a glob runs on any match.

    `test_docs_cross_file_anchors_1511.py` reads every tracked Markdown
    file, and the derived-figure tests read every page under `docs/` and
    `CHANGELOG/` through `cdf.iter_files`. Listing them all under `code`
    would put the full matrix on every docs-only PR (#413/#427), and
    leaving them out let a PR break an anchor, or leave a fence open, in
    an unlisted page with the required check green. So the globs sit under
    the filter's `tree_reads` key, and a PR that matches one but no `code`
    path runs exactly the modules that declare one. Both lists come from
    the modules' `# path-filter:` comments.
    """
    patterns = _suite_pattern_reads()
    assert "**/*.md" in patterns, (
        f"the scan found no module declaring `**/*.md`; found {patterns}. The "
        f"anchor test lost its comment, or the scan stopped reading comments."
    )
    globs = _filter_entries("tree_reads")
    missing = sorted(set(patterns) - set(globs))
    assert not missing, (
        f"globs declared by tests/ but absent from the `tree_reads` filter in "
        f"{_CI_WORKFLOW.name}: {missing}"
    )
    stale = sorted(set(globs) - set(patterns))
    assert not stale, (
        f"`tree_reads` entries in {_CI_WORKFLOW.name} that no test declares: "
        f"{stale}. A PR matching only those runs tests that read nothing it changed."
    )
    readers = sorted(set().union(*patterns.values()))
    step = sorted(_tree_read_step_modules())
    assert step == readers, (
        f"the `tree_reads` step runs {step}, but the modules that declare a "
        f"glob are {readers}. A module left out never runs on the PR that "
        f"changes only the files it reads."
    )


def test_a_tree_reads_match_installs_first_and_skips_the_echo() -> None:
    """The partial run needs uv and the lockfile, and is not a docs-only pass."""
    steps = _pytest_job_steps()
    installs = [
        step for step in steps
        if any("astral-sh/setup-uv@" in line or line.strip() == "run: uv sync --frozen --group dev --extra archive" for line in step)
    ]
    assert len(installs) == 2, f"expected the setup-uv and uv sync steps, found {len(installs)}"
    for step in installs:
        assert _TREE_READS_MATCH in step[0], (
            f"an install step does not run on a `tree_reads` match, so the "
            f"partial run finds no uv: {step[0].strip()!r}"
        )
    echoes = [step for step in steps if any(line.strip().startswith("run: echo ") for line in step)]
    assert len(echoes) == 1, f"expected one echo step, found {len(echoes)}"
    assert "steps.filter.outputs.tree_reads != 'true'" in echoes[0][0], (
        f"the docs-only echo also fires on a `tree_reads` match: {echoes[0][0].strip()!r}"
    )


def test_benchmarks_and_scripts_are_in_the_code_filter() -> None:
    """Pin the #1160 regression by name, not only by derivation."""
    globs = _code_filter_globs()
    for directory in ("benchmarks", "scripts"):
        assert _is_covered(directory, globs), (
            f"'{directory}/**' dropped out of the code filter; this is the "
            f"#1160 defect returning"
        )


def test_the_build_inputs_are_in_the_code_filter() -> None:
    """`uv sync --frozen` installs from these files, so a change to one runs the suite.

    No test reads `uv.lock`, so the read scan cannot demand it, and the
    reverse test exempts the build inputs rather than requiring them.
    Without this assertion, deleting the entry failed nothing.
    """
    missing = sorted(_BUILD_INPUTS - set(_code_filter_globs()))
    assert not missing, (
        f"{missing} dropped out of the code filter in {_CI_WORKFLOW.name}; a "
        f"PR that changes only the dependency set takes the `echo` branch and "
        f"never runs the suite against what it installs"
    )


def test_workflow_changes_run_the_suite() -> None:
    """Pin #1786 CI-07 by name: `.github/**`, not one workflow file.

    The filter used to name `.github/workflows/ci.yml` alone, while the
    merge-train, mutation, publish and staging-gate workflows each have
    tests that read them. A PR that changed one of those reported a
    passing pytest matrix having run none of them.
    """
    globs = _code_filter_globs()
    assert ".github/**" in globs, (
        "'.github/**' dropped out of the code filter; a workflow-only PR "
        "skips every test that reads a workflow"
    )
    workflow_reads = sorted(p for p in _suite_file_reads() if p.startswith(".github/workflows/"))
    assert len(workflow_reads) >= 5, (
        f"expected the suite to read several workflow files, found {workflow_reads}; "
        f"the read scan has stopped seeing them"
    )


def test_docs_only_changes_still_skip_the_suite() -> None:
    """`docs/` and `CHANGELOG/` carry no directory-wide glob.

    The filter names the exact pages tests read (the test above derives
    them), and nothing wider: a PR that changes any other page, or adds a
    CHANGELOG fragment, does not run the suite (#413/#427). A Markdown
    change runs only the `tree_reads` modules. Widen to `docs/**` only on
    an operator ruling, and move this expectation first.
    """
    globs = _code_filter_globs()
    for directory in ("docs", "CHANGELOG"):
        assert not _is_covered(directory, globs), (
            f"'{directory}' gained a directory-wide glob in the code filter, "
            f"so docs-only PRs now run the full matrix, reversing #413/#427. "
            f"Intentional? The exact files tests read are already listed."
        )


# --- e2e trigger coverage (#1420 §1) -------------------------------------

_E2E_WORKFLOW = _REPO / ".github" / "workflows" / "e2e.yml"
_TESTING_STRATEGY = _REPO / "docs" / "design" / "testing-strategy.md"


def _block_range(lines: list[str], start: int) -> range:
    """Indices of the block nested under `lines[start]`.

    Index range rather than a list of line *contents*: two sibling blocks
    can hold byte-identical lines (`branches: [main]` appears under both
    triggers in e2e.yml), and membership-testing on content would match
    the wrong one.
    """
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.strip().startswith("#"):
            if len(line) - len(line.lstrip()) <= indent:
                break
        end += 1
    return range(start + 1, end)


def _sole_key(lines: list[str], within: range, key: str, where: str) -> int:
    """The single index in `within` whose stripped content is `key`."""
    hits = [i for i in within if lines[i].strip() == key]
    assert len(hits) == 1, (
        f"expected exactly one `{key}` {where} in {_E2E_WORKFLOW.name}, found "
        f"{len(hits)} — a parser that finds the wrong block, or none, turns "
        f"every assertion below green for free"
    )
    return hits[0]


def _e2e_pull_request_paths() -> list[str]:
    """The glob list under `on.pull_request.paths` in e2e.yml.

    Walks `on:` -> `pull_request:` -> `paths:` by indentation rather than
    grepping the file for `paths:`. The difference is load-bearing twice
    over. A file-wide search cannot tell a `paths:` under `pull_request:`
    from one under `push:`, so it would (a) bind these assertions to the
    wrong trigger without saying so, and (b) fail — with a message about
    parser trust — the moment someone legitimately adds a `push: paths:`,
    which is a confusing signal for an unrelated edit.

    Asserted non-empty for the same reason `_code_filter_globs` is: a
    parser that quietly finds nothing turns every assertion below green
    for free.
    """
    lines = _E2E_WORKFLOW.read_text(encoding="utf-8").splitlines()

    on_key = _sole_key(lines, range(len(lines)), "on:", "top-level key")
    pr_key = _sole_key(lines, _block_range(lines, on_key), "pull_request:", "trigger under `on:`")
    start = _sole_key(lines, _block_range(lines, pr_key), "paths:", "key under `on.pull_request`")
    indent = len(lines[start]) - len(lines[start].lstrip())
    globs: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the e2e paths list: {stripped!r}"
        globs.append(matched.group(1))
    assert globs, "the e2e paths list parsed empty"
    return globs


def test_e2e_runs_on_code_prs_without_an_opt_in_label() -> None:
    """#1420 §1: e2e must not require a label to run on a PR.

    Before this, the job carried
    `if: github.event_name == 'push' || contains(...labels..., 'e2e')`, so an
    installed-package regression — which the unit suite structurally cannot
    see, because it imports from the source tree — was caught on `main` after
    merge instead of on the PR. Asserting the absence of the label condition
    is the whole point; a test that only checked the workflow parses would
    have passed in the broken state.
    """
    lines = _E2E_WORKFLOW.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)

    # The invariant, not one spelling of its violation: the job carries no
    # `if:` at all. Asserting only the absence of the old label expression
    # lets any re-gate through — `if: github.event_name == 'push'`, a
    # differently-quoted `contains(...)`, or a `vars.`-driven toggle would
    # each fully negate AC1 while keeping that string absent.
    jobs_key = _sole_key(lines, range(len(lines)), "jobs:", "top-level key")
    e2e_job = _sole_key(lines, _block_range(lines, jobs_key), "e2e:", "job under `jobs:`")
    job_body = _block_range(lines, e2e_job)
    job_indent = min(
        (len(lines[i]) - len(lines[i].lstrip()) for i in job_body if lines[i].strip()),
        default=0,
    )
    gating = [
        lines[i].strip()
        for i in job_body
        if lines[i].strip().startswith("if:")
        and len(lines[i]) - len(lines[i].lstrip()) == job_indent
    ]
    assert not gating, (
        f"the `e2e` job is conditionally gated again ({gating}) — AC1 says it "
        f"runs on any code PR, and *any* job-level `if:` can negate that. "
        f"Trigger-level `paths` is where docs-only PRs are excluded."
    )
    assert "labels.*.name, 'e2e'" not in text, (
        "e2e is gated on an opt-in label again — a PR without the label gets "
        "no end-to-end coverage and the regression lands on main"
    )
    assert "types: [opened, synchronize, reopened]" in text, (
        "e2e's pull_request trigger must include `opened`; with only "
        "[labeled, synchronize, reopened] a freshly-opened PR never fires it"
    )


def test_e2e_paths_cover_the_ci_code_filter() -> None:
    """The two path lists must agree, or e2e silently under-triggers.

    `e2e.yml` says to keep its `paths` in sync with ci.yml's `code` filter.
    A comment cannot enforce that: a path added to ci.yml and forgotten here
    means a PR that changes installed behaviour runs the unit suite and skips
    e2e, with nothing red to show for it.

    Entries under `.github/` are excluded from both sides: ci.yml carries
    `.github/**` because unit tests read the workflows, and e2e names only
    its own file, since a workflow cannot change the installed package. So
    are ci.yml's exact entries other than the build inputs: they exist
    because a unit test reads the file, and e2e tests the installed package,
    which a page under `docs/` cannot change. The test below holds the two
    documents that state this mirror to naming each exclusion.
    """
    ci_globs = {
        g
        for g in _code_filter_globs()
        if not g.startswith(".github/") and (g.endswith("/**") or g in _BUILD_INPUTS)
    }
    e2e_globs = {g for g in _e2e_pull_request_paths() if not g.startswith(".github/")}
    missing = sorted(ci_globs - e2e_globs)
    assert not missing, (
        f"{missing} are in ci.yml's `code` filter but not in e2e.yml's "
        f"`paths`. A PR touching only those runs pytest but not e2e."
    )


def test_the_e2e_mirror_docs_name_each_glob_e2e_leaves_out() -> None:
    """The e2e.yml header and testing-strategy.md say e2e mirrors the `code` filter.

    The comparison above drops `.github/` entries, so a directory glob or
    build input that e2e leaves out on purpose passes it unseen. Each
    document that states the mirror must name every such entry, or the
    mirror it states does not hold.
    """
    e2e = set(_e2e_pull_request_paths())
    omitted = sorted(
        g for g in _code_filter_globs() if (g.endswith("/**") or g in _BUILD_INPUTS) and g not in e2e
    )
    lines = _E2E_WORKFLOW.read_text(encoding="utf-8").splitlines()
    header = "\n".join(lines[: _sole_key(lines, range(len(lines)), "on:", "top-level key")])
    for name, text in (
        (_E2E_WORKFLOW.name, header),
        (_TESTING_STRATEGY.name, _TESTING_STRATEGY.read_text(encoding="utf-8")),
    ):
        unnamed = [g for g in omitted if f"`{g}`" not in text]
        assert not unnamed, (
            f"{name} says e2e's paths mirror ci.yml's `code` filter, but e2e "
            f"leaves out {unnamed} and {name} does not name them"
        )


def test_e2e_still_skips_docs_only_prs() -> None:
    """#1420 §1 AC2, which the subset check above cannot express.

    `test_e2e_paths_cover_the_ci_code_filter` asserts `ci - e2e == {}` — e2e's
    list is a *superset* of ci's. A superset assertion is blind to additions,
    so appending `docs/**` (or `'**'`) to the e2e trigger keeps every other
    test in this module green while putting the 3-leg install matrix on every
    docs-only and CHANGELOG-only PR. That is AC2 reversed, with nothing red to
    show for it.

    The mirror of `test_docs_only_changes_still_skip_the_suite`, which is the
    same guard on the ci.yml side and the reason that side cannot drift.
    """
    globs = _e2e_pull_request_paths()
    for directory in ("docs", "CHANGELOG"):
        assert not _is_covered(directory, globs), (
            f"'{directory}' entered e2e's trigger paths, so docs-only PRs now "
            f"run the install matrix — reversing #1420 §1 AC2. Intentional? "
            f"Confirm the installed-package path can actually regress from a "
            f"prose change first."
        )


def test_e2e_is_path_filtered_at_the_trigger_not_inside_the_job() -> None:
    """Pin *why* the trigger-level filter is safe here.

    A path-filtered **required** context never reports on a PR it skips and
    leaves that PR permanently pending — which is exactly why ci.yml has no
    `paths` and filters inside the job instead. e2e is not a required context,
    so filtering at the trigger is fine and cheaper. If e2e is ever promoted
    to required, this must move inside the job first.
    """
    ci_text = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert "dorny/paths-filter" in ci_text, (
        "ci.yml stopped filtering inside the job; if it gained a trigger-level "
        "`paths` it would leave docs-only PRs permanently pending on the "
        "required pytest contexts"
    )
    assert "dorny/paths-filter" not in _E2E_WORKFLOW.read_text(encoding="utf-8")
