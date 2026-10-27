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
own location (`Path(__file__).resolve().parents[1] / "docs" / ...`),
following names bound to such paths, so a test that first binds the repo
root to a name and then joins onto it is seen the same as one that spells
the path inline. A new test that reads a new file outside the filter
fails `test_every_file_the_suite_reads_is_in_the_code_filter`; an entry
nobody reads any more fails the reverse test beside it.
"""
from __future__ import annotations

import ast
import functools
import re
import shlex
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CI_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"
_STAGING_GATE_WORKFLOW = _REPO / ".github" / "workflows" / "staging-gate.yml"

# A top-level directory holding Python the suite could depend on. Excludes
# directories with no .py at all (`CHANGELOG/`, and the untracked
# `telemetry/` a local run may leave behind), so the scan does not depend
# on working-tree state.
_IN_REPO_PACKAGES = frozenset(
    p.name
    for p in _REPO.iterdir()
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


def _code_filter_globs() -> list[str]:
    """Return the glob list under the `code:` key of the paths-filter."""
    lines = _CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "code:"]
    assert len(starts) == 1, (
        f"expected exactly one `code:` filter key in {_CI_WORKFLOW.name}, "
        f"found {len(starts)} — the parser below cannot be trusted, and a "
        f"parser that silently finds nothing turns every assertion in this "
        f"module green for free."
    )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    globs: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the code filter: {stripped!r}"
        entry = matched.group(1)
        # `_entry_covers` below knows two shapes: a whole directory
        # (`dir/**`) and an exact path. Any other wildcard would make it
        # guess, and a guess in a coverage predicate is a pass for free.
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

# A path value under resolution: how many `.parent` steps above the test
# file it starts, and the literal segments joined onto that. The repo root
# is the value whose depth equals the file's depth below the repo and whose
# segments are empty.
_PathValue = tuple[int, tuple[str, ...]]


def _is_path_of_file(node: ast.AST) -> bool:
    """`Path(__file__)` or `pathlib.Path(__file__)`."""
    return (
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Path")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Path")
        )
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "__file__"
    )


def _resolve(node: ast.AST, names: dict[str, _PathValue]) -> _PathValue | None:
    """The path `node` denotes relative to the test file, or None."""
    if _is_path_of_file(node):
        return (0, ())
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("resolve", "absolute"):
            if not node.args and not node.keywords:
                return _resolve(func.value, names)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        inner = _resolve(node.value, names)
        if inner is None:
            return None
        depth, segments = inner
        return (depth, segments[:-1]) if segments else (depth + 1, ())
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "parents":
            inner = _resolve(value.value, names)
            index = node.slice
            if (
                inner is not None
                and not inner[1]
                and isinstance(index, ast.Constant)
                and isinstance(index.value, int)
            ):
                return (inner[0] + index.value + 1, ())
        return None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve(node.left, names)
        right = node.right
        if left is None or not isinstance(right, ast.Constant):
            return None
        if not isinstance(right.value, str) or right.value.startswith("/"):
            return None
        parts = tuple(p for p in right.value.split("/") if p and p != ".")
        return (left[0], left[1] + parts)
    return None


def _bound_names(
    nodes: list[ast.AST], outer: dict[str, _PathValue]
) -> dict[str, _PathValue]:
    """Names in `nodes` bound to a path value, starting from `outer`.

    Iterated to a fixpoint, bounded by the number of assignments, so a
    name bound from another name resolves whatever order they appear in.
    """
    names = dict(outer)
    assigns = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assigns) + 1):
        changed = False
        for node in assigns:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            if node.value is None:
                continue
            value = _resolve(node.value, names)
            if value is not None and names.get(targets[0].id) != value:
                names[targets[0].id] = value
                changed = True
        if not changed:
            break
    return names


def _scan_module(path: Path) -> set[str]:
    """Repo-relative paths that `path` builds from its own location.

    Names are scoped: a module-level binding is visible everywhere, a
    binding inside a function only inside that function. Without the
    split, `directory = tmp_path / "x"` in one test and `directory =
    _REPO / "CHANGELOG"` in another would read as the same name.

    Only the outermost join of a chain is recorded, so `_REPO / "docs" /
    "user" / "CONFIG.md"` yields one path and not three. When the
    outermost join does not resolve (`_REPO / "docs" / name`), the longest
    resolvable prefix is recorded instead, since the test reads under it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # One pass over the tree: each node's parent, and the function that
    # owns it (None at module level). The function node itself belongs to
    # the scope that defines it.
    parents: dict[ast.AST, ast.AST] = {}
    owner: dict[ast.AST, ast.AST | None] = {}
    members: dict[ast.AST | None, list[ast.AST]] = {None: []}
    pending: list[tuple[ast.AST, ast.AST | None]] = [(tree, None)]
    while pending:
        node, function = pending.pop()
        owner[node] = function
        members.setdefault(function, []).append(node)
        inner = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            pending.append((child, inner))
    module_names = _bound_names(members[None], {})
    scopes = {
        function: _bound_names(nodes, module_names)
        for function, nodes in members.items()
        if function is not None
    }
    file_depth = len(path.relative_to(_REPO).parts)
    found: set[str] = set()

    def names_for(node: ast.AST) -> dict[str, _PathValue]:
        function = owner[node]
        return module_names if function is None else scopes[function]

    def record(value: _PathValue | None) -> None:
        if value is not None and value[0] == file_depth and value[1]:
            found.add("/".join(value[1]))

    for node in owner:
        parent = parents.get(node)
        joined_further = (
            isinstance(parent, ast.BinOp)
            and isinstance(parent.op, ast.Div)
            and parent.left is node
        )
        if joined_further:
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            names = names_for(node)
            current: ast.AST = node
            value = _resolve(current, names)
            while value is None and isinstance(current, ast.BinOp):
                if not isinstance(current.op, ast.Div):
                    break
                current = current.left
                value = _resolve(current, names)
            record(value)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                continue
            record(names_for(node).get(node.id))
    return found


@functools.cache
def _suite_root_reads() -> dict[str, set[str]]:
    """Every repo-relative path `tests/` builds, mapped to the modules that build it.

    Paths absent from the tree are dropped: a test may spell a path to
    assert that it does not exist, and nothing can change under a path
    that is not there.
    """
    reads: dict[str, set[str]] = {}
    for path in sorted(_REPO.joinpath("tests").rglob("*.py")):
        for found in _scan_module(path):
            if (_REPO / found).exists():
                reads.setdefault(found, set()).add(path.relative_to(_REPO).as_posix())
    return reads


def _suite_file_reads() -> dict[str, set[str]]:
    return {p: m for p, m in _suite_root_reads().items() if (_REPO / p).is_file()}


def _suite_directory_reads() -> dict[str, set[str]]:
    return {p: m for p, m in _suite_root_reads().items() if (_REPO / p).is_dir()}


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
        if (_REPO / p).is_dir() or p.endswith(".py")
    }
    return found | (code_reads & _IN_REPO_PACKAGES)


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


def test_benchmarks_and_scripts_are_in_the_code_filter() -> None:
    """Pin the #1160 regression by name, not only by derivation."""
    globs = _code_filter_globs()
    for directory in ("benchmarks", "scripts"):
        assert _is_covered(directory, globs), (
            f"'{directory}/**' dropped out of the code filter; this is the "
            f"#1160 defect returning"
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
    CHANGELOG fragment, still takes the `echo` branch (#413/#427). Widen
    to `docs/**` only on an operator ruling, and move this expectation
    first.
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

    Each list may legitimately reference its *own* workflow file, so those are
    excluded from the comparison rather than the assertion being loosened.
    So are ci.yml's exact entries other than the build inputs: they exist
    because a unit test reads the file, and e2e tests the installed package,
    which a page under `docs/` cannot change.
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
