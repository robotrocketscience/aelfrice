#!/usr/bin/env python3
"""#1605, #1632 — what a per-PR mutation run should actually mutate.

`.github/workflows/mutation.yml` scopes mutmut with `only_mutate`, whose
entries are FILE paths. So touching one line of a file puts the whole file
in the mutant set. On `cli.py` — 11k lines — that exceeded the job's
60-minute budget and was cancelled with no report. The scope is narrowed in
two passes.

## Pass 1: files (#1605)

Drop files whose diff cannot produce a mutant. This decides that by
**comparing ASTs with docstrings stripped**, not by classifying lines. A
line heuristic has to answer "is this line inside a docstring", which needs
a parser anyway, and it still says nothing about a change that only moves
code around. Two files whose stripped ASTs are equal differ by comments,
docstrings, blank lines, or formatting — none of which mutmut can mutate —
so the diff between them introduces no mutant.

## Pass 2: functions (#1632)

A real change to one function of `cli.py` still put the whole file in scope,
and the cost is in mutant *generation*, not only in running them: mutmut
3.8.0 writes every mutant of a function as a full copy of that function, so
generating a whole large file such as `cli.py` takes minutes and produces
tens of thousands of mutants. With one function in scope it takes seconds.
Filtering mutant names after generation (`mutmut run <glob>`) leaves that
cost in place, and the mutants it filters out are reported as `not checked`.

So the restriction happens before generation. mutmut mutates only two kinds
of function: a module-level `def`, and a `def` directly in the body of a
module-level class. Everything nested inside one of those is mutated as part
of it, and nothing else is mutated at all. Those are the *units* here. A unit
is in scope when its line span (decorators included) intersects the PR's
diff hunks **and** its docstring-stripped AST differs from the same-named
unit on the base side. The second condition is pass 1 applied per function:
a function that only moved, or only had a comment edited, carries no new
mutant. A name defined more than once in the file, on either side, cannot
be paired with its base version, so it is never called unchanged.

Every out-of-scope unit then gets mutmut's documented
`# pragma: no mutate block` on its header line, in the CI checkout only. A
comment changes no line number and no AST, and the rewritten file is
re-parsed and compared to the original to prove it. Each annotated function
is skipped whole by mutmut's mutation visitor, so it costs no generation
time.

Changed lines outside every unit — module-level statements, class bodies,
and methods of nested classes — carry no mutant under mutmut, so they are
reported in the summary rather than mutated. Nothing is dropped silently:
`--summary` writes every in-scope function and every skipped file, function,
and line range with the reason.

## Failing open

Failing open is deliberate throughout. A file that will not parse, that
cannot be read out of the base commit, or whose rewrite does not round-trip
is mutated whole: the cost of mutating code needlessly is runner time, and
the cost of skipping it wrongly is an unmeasured mutant.

This lives in a script rather than in the workflow's inline Python because
the workflow cannot be tested in CI — PyYAML is not importable there
(#1436) — while a script can be. `tests/test_mutation_scope.py` tests the
file pass, and `tests/test_mutation_function_scope_1632.py` tests the
function pass.

## Usage

    scripts/mutation_scope.py --base <sha> --head <sha> --dry-run
    scripts/mutation_scope.py --base <sha> --head <sha> \\
        --write-pragmas --summary scope.md

Prints one path per line on stdout: the changed files under `src/aelfrice/`
that hold at least one in-scope function. Prints nothing when none do,
which the caller reads as "skip the run". `--dry-run` prints the per-file
and per-function verdicts on stderr and writes nothing. `--write-pragmas`
annotates the out-of-scope functions in the working tree, and is meant for a
throwaway CI checkout: it refuses to run unless the `CI` environment
variable is `true`, which GitHub Actions sets, or `--allow-src-rewrite` is
passed. `--summary PATH` writes the Markdown report.

Exits 0 when the scope was computed, 2 on a git failure, and 3 when
`--write-pragmas` is refused. An empty scope is a result, not a failure.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import copy
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

#: Only these are mutated; matches the workflow's own pathspec.
PATHSPEC: Final[str] = "src/aelfrice/*.py"

#: mutmut's documented pragma for skipping a whole compound statement.
PRAGMA_BLOCK: Final[str] = "# pragma: no mutate block"

#: mutmut's documented pragma for skipping one line.
PRAGMA_LINE: Final[str] = "# pragma: no mutate"

#: The one decorator form mutmut still mutates (mutmut 3.8.0,
#: `MutationVisitor._skip_node_and_children`); any other decorator, or
#: more than one, makes it skip the function entirely.
MUTATED_DECORATORS: Final[frozenset[str]] = frozenset(
    {"staticmethod", "classmethod"},
)

#: Function names mutmut never mutates, whatever their decorators or
#: position. mutmut 3.8.0, `mutmut/mutation/file_mutation.py:37`
#: (`NEVER_MUTATE_FUNCTION_NAMES`), checked by
#: `MutationVisitor._skip_node_and_children` at line 307.
MUTMUT_SKIPPED_NAMES: Final[frozenset[str]] = frozenset(
    {"__getattribute__", "__setattr__", "__new__"},
)

#: The summary's reason for each kind of unit mutmut skips.
SKIP_NAMED: Final[str] = "mutmut skips this method"
SKIP_DECORATED: Final[str] = "mutmut does not mutate decorated functions"

#: `mutmut`'s separator between class and method in a mangled name.
CLASS_SEPARATOR: Final[str] = "ǁ"

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class _StripDocstrings(ast.NodeTransformer):
    """Remove every docstring, so a docstring edit compares equal.

    A docstring is the first statement of a module, class, or function
    when it is a bare string expression. Removing it rather than blanking
    it keeps the comparison insensitive to its content *and* its
    presence, which matters because adding a docstring to a function is
    as inert, for mutation purposes, as editing one.
    """

    def _strip(self, node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # Never leave a body empty; that is a syntax error on reparse
            # and would change the shape of the tree being compared.
            node.body = body[1:] or [ast.Pass()]  # type: ignore[attr-defined]
        self.generic_visit(node)
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def normalised_ast(source: str) -> str | None:
    """`source` as a docstring-free AST dump, or None if it will not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    stripped = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(stripped)
    # `include_attributes=False` is the default and is load-bearing: line
    # numbers move when a comment is added above a statement, so a dump
    # carrying them would call every comment change mutable.
    return ast.dump(stripped)


def has_mutable_change(before: str | None, after: str | None) -> bool:
    """True when the change between two file versions can carry a mutant.

    `None` means the version could not be read — a file added by this
    diff, or deleted from it. Both are mutable by construction: an added
    file is all-new code, and a deleted one is not in the scope anyway
    because the caller filters deletions out before asking.
    """
    if before is None or after is None:
        return True
    a, b = normalised_ast(before), normalised_ast(after)
    if a is None or b is None:
        return True  # unparseable on either side: fail open
    return a != b


# --- pass 2: functions (#1632) ----------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One function mutmut mutates as a whole, with its nested code."""

    qualname: str
    start: int
    end: int
    #: Why mutmut never mutates this unit, or None when it does.
    skip_reason: str | None
    node: _FunctionNode = field(compare=False, repr=False)

    @property
    def skipped_by_mutmut(self) -> bool:
        """True when mutmut generates no mutant for this unit at all."""
        return self.skip_reason is not None

    @property
    def key(self) -> str:
        """The mangled name mutmut gives this unit's mutants.

        `x_name` for a module-level function and `xǁClassǁname` for a
        method, followed in the mutant name by `__mutmut_<n>`.
        """
        cls, _, name = self.qualname.rpartition(".")
        if cls:
            return f"x{CLASS_SEPARATOR}{cls}{CLASS_SEPARATOR}{name}"
        return f"x_{name}"


def _skip_reason(node: _FunctionNode) -> str | None:
    """Why mutmut never mutates `node`, or None when it does.

    mutmut skips a function by name before it looks at decorators, and
    never mutates a decorated function, with one exception.
    """
    if node.name in MUTMUT_SKIPPED_NAMES:
        return SKIP_NAMED
    decorators = node.decorator_list
    if not decorators:
        return None
    if len(decorators) == 1:
        only = decorators[0]
        if isinstance(only, ast.Name) and only.id in MUTATED_DECORATORS:
            return None
    return SKIP_DECORATED


def _unit(node: _FunctionNode, qualname: str) -> Unit:
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    end = node.end_lineno
    assert end is not None, "ast.parse always sets end_lineno"
    return Unit(qualname, first, end, _skip_reason(node), node)


def mutation_units(tree: ast.Module) -> list[Unit]:
    """Every function mutmut can mutate as a unit, in source order.

    Module-level functions, and methods directly in a module-level class.
    A function nested in either is part of its parent unit; a method of a
    nested class is no unit at all, because mutmut does not mutate it.
    """
    units: list[Unit] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(_unit(node, node.name))
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    units.append(_unit(member, f"{node.name}.{member.name}"))
    return units


def touched_lines(diff_text: str) -> set[int]:
    """Head-side line numbers a `git diff -U0` touches.

    An added or modified line is touched itself. A pure deletion has no
    head-side line, so the two lines it fell between are both touched:
    which of the two functions lost the code is not knowable from the
    hunk alone, and over-including one costs only runner time.
    """
    touched: set[int] = set()
    for line in diff_text.splitlines():
        match = _HUNK.match(line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        if count == 0:
            touched.update((start, start + 1))
        else:
            touched.update(range(start, start + count))
    return touched


def _unit_dump(node: _FunctionNode) -> str:
    stripped = _StripDocstrings().visit(copy.deepcopy(node))
    return ast.dump(stripped)


@dataclass
class FileScope:
    """The per-function verdict for one changed file."""

    path: str
    in_scope: list[Unit] = field(default_factory=lambda: [])
    #: Touched, but the same as on the base side once docstrings,
    #: comments, and position are set aside: moved or reworded only.
    unchanged: list[Unit] = field(default_factory=lambda: [])
    #: Changed code-bearing lines that lie in no unit.
    outside: list[int] = field(default_factory=lambda: [])
    #: No base version was compared, so every in-scope unit is new code
    #: rather than a change to existing code.
    added: bool = False

    @property
    def mutated(self) -> list[Unit]:
        """In-scope units mutmut actually generates mutants for."""
        return [u for u in self.in_scope if not u.skipped_by_mutmut]


def _code_bearing(line: str) -> bool:
    text = line.strip()
    return bool(text) and not text.startswith("#")


def classify(
    path: str, before: str | None, after: str, touched: set[int],
) -> FileScope | None:
    """Which functions of `after` the diff changed.

    Returns None when `after` will not parse, which the caller reads as
    "mutate the whole file". A `before` that is missing or unparseable
    leaves every touched unit in scope.
    """
    try:
        head_tree = ast.parse(after)
    except SyntaxError:
        return None
    base_units: list[Unit] = []
    if before is not None:
        try:
            base_units = mutation_units(ast.parse(before))
        except SyntaxError:
            pass
    head_units = mutation_units(head_tree)
    # A name defined more than once on either side cannot be paired with
    # its base version: with two `dup`s, a change that makes the second
    # equal the first would match the first and read as unchanged. Such
    # a unit is never called unchanged, so it stays in scope if touched.
    ambiguous = {
        name
        for units in (base_units, head_units)
        for name, count in Counter(u.qualname for u in units).items()
        if count > 1
    }
    base_dumps = {
        u.qualname: _unit_dump(u.node)
        for u in base_units if u.qualname not in ambiguous
    }

    result = FileScope(path, added=before is None)
    covered: set[int] = set()
    for unit in head_units:
        span = range(unit.start, unit.end + 1)
        covered.update(span)
        if touched.isdisjoint(span):
            continue
        if base_dumps.get(unit.qualname) == _unit_dump(unit.node):
            result.unchanged.append(unit)
        else:
            result.in_scope.append(unit)

    lines = after.splitlines()
    result.outside = sorted(
        n for n in touched - covered
        if 1 <= n <= len(lines) and _code_bearing(lines[n - 1])
    )
    return result


def _header_colon(
    tokens: list[tokenize.TokenInfo],
    starts: list[tuple[int, int]],
    node: _FunctionNode,
) -> tuple[int, int] | None:
    """(row, col) just past the `:` that ends `node`'s header.

    Only a comment, a newline, and an indent can sit between the header
    colon and the first body statement, so it is the first `:` found
    walking back from the body. `starts` is sorted, so the walk begins at
    the body rather than at the top of the file: on an 11k-line module, a
    linear scan per function is quadratic in practice.
    """
    begin = (node.lineno, node.col_offset)
    body = (node.body[0].lineno, node.body[0].col_offset)
    index = bisect.bisect_left(starts, body) - 1
    while index >= 0 and tokens[index].start >= begin:
        tok = tokens[index]
        if tok.type == tokenize.OP and tok.string == ":":
            return tok.end
        index -= 1
    return None


def exclude_units(source: str, keep: set[str]) -> tuple[str, list[str]]:
    """Annotate every unit not in `keep` so mutmut skips it.

    Returns the rewritten source and the qualnames that could not be
    annotated. Units mutmut already skips are left alone. Raises
    ValueError when the rewrite does not round-trip to the same AST,
    which the caller treats as "mutate the whole file".
    """
    tree = ast.parse(source)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    starts = [tok.start for tok in tokens]
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    failed: list[str] = []
    for unit in mutation_units(tree):
        if unit.qualname in keep or unit.skipped_by_mutmut:
            continue
        colon = _header_colon(tokens, starts, unit.node)
        if colon is None:
            failed.append(unit.qualname)
            continue
        row, col = colon
        if unit.node.body[0].lineno > row:
            # An indented body: the comment after the colon is the
            # block's header comment, which is where mutmut reads
            # `no mutate block`. Inserting it before any existing
            # comment keeps it the first `no mutate` mutmut parses.
            edits.append((row, col, f"  {PRAGMA_BLOCK}"))
        elif unit.start == row == unit.end:
            # `def f(): return x` on one line. mutmut reads a trailing
            # comment on a one-line suite as a line pragma for the line
            # the `def` starts on, which here is the whole function.
            end = len(lines[row - 1].rstrip("\r\n"))
            edits.append((row, end, f"  {PRAGMA_LINE}"))
        else:
            failed.append(unit.qualname)
    for row, col, text in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + text + line[col:]
    rewritten = "".join(lines)
    if ast.dump(ast.parse(rewritten)) != ast.dump(tree):
        raise ValueError("annotated source does not round-trip")
    return rewritten, failed


# --- git --------------------------------------------------------------------


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _blob(sha: str, path: str) -> str | None:
    try:
        return _git("show", f"{sha}:{path}")
    except RuntimeError:
        return None


@dataclass(frozen=True)
class Change:
    """One changed file: its head path, and its path on the base side.

    `base_path` differs from `path` when git paired the file with one it
    renamed; it is the path the base blob is read from.
    """

    path: str
    base_path: str


def changed_files(base: str, head: str) -> list[Change]:
    """Changed, non-deleted files under the pathspec, in the PR's own diff.

    Three dots, not two, for the reason the workflow already documents:
    `A..B` is the difference between two trees and would include commits
    merged into the base since the branch point.

    Renames are detected (`-M`), so a moved file is compared with its old
    self. Without that, the base blob is looked up at the new path, is
    missing, and every function of a merely renamed file looks new.
    """
    out = _git(
        "diff", "--name-status", "-M", "--diff-filter=d", f"{base}...{head}",
        "--", PATHSPEC,
    )
    changes: list[Change] = []
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[0].startswith(("R", "C")):
            changes.append(Change(fields[2], fields[1]))
        elif len(fields) == 2:
            changes.append(Change(fields[1], fields[1]))
    return changes


def _file_diff(base: str, head: str, change: Change) -> str:
    """The `-U0` diff of one file, old path against new across a rename.

    Naming both paths, with rename detection on, makes git diff the
    renamed file against its old content rather than against nothing.
    """
    paths = dict.fromkeys((change.base_path, change.path))
    return _git(
        "diff", "-U0", "-M", "--no-color", f"{base}...{head}", "--", *paths,
    )


def scope(base: str, head: str, *, explain: bool = False) -> list[str]:
    """Pass 1 alone: the changed files that carry a mutable change."""
    keep: list[str] = []
    for change in changed_files(base, head):
        mutable = has_mutable_change(
            _blob(base, change.base_path), _blob(head, change.path),
        )
        if explain:
            verdict = "mutable" if mutable else "comments/docstrings only"
            print(f"  {change.path}: {verdict}", file=sys.stderr)
        if mutable:
            keep.append(change.path)
    return keep


# --- both passes, and the report --------------------------------------------


@dataclass
class Report:
    """Everything the run decided, for stdout and the step summary."""

    #: Files to put in `only_mutate`, in diff order.
    files: list[str] = field(default_factory=lambda: [])
    scopes: list[FileScope] = field(default_factory=lambda: [])
    #: (path, reason) for each changed file left out of `only_mutate`.
    skipped_files: list[tuple[str, str]] = field(default_factory=lambda: [])
    #: (path, reason) for each file mutated whole.
    whole_files: list[tuple[str, str]] = field(default_factory=lambda: [])
    #: (path, qualname) for out-of-scope units that are mutated anyway.
    not_excluded: list[tuple[str, str]] = field(default_factory=lambda: [])


def function_scope(base: str, head: str, *, write: bool) -> Report:
    """Run both passes over the PR's diff.

    With `write`, annotate the working tree. Units are matched by name,
    not line, because a pull-request checkout is the merge commit, whose
    line numbers can differ from the head commit the diff was taken on.
    """
    report = Report()
    for change in changed_files(base, head):
        path = change.path
        before, after = _blob(base, change.base_path), _blob(head, path)
        if not has_mutable_change(before, after):
            report.skipped_files.append(
                (path, "comments, docstrings, or formatting only"),
            )
            continue
        if after is None:
            report.whole_files.append((path, "head version unreadable"))
            report.files.append(path)
            continue
        verdict = classify(
            path, before, after, touched_lines(_file_diff(base, head, change)),
        )
        if verdict is None:
            report.whole_files.append((path, "head version does not parse"))
            report.files.append(path)
            continue
        report.scopes.append(verdict)
        if not verdict.mutated:
            report.skipped_files.append(
                (path, "no changed function that mutmut mutates"),
            )
            continue
        report.files.append(path)
        if not write:
            continue
        target = Path(path)
        try:
            rewritten, failed = exclude_units(
                target.read_text(encoding="utf-8"),
                {u.qualname for u in verdict.in_scope},
            )
        except (OSError, SyntaxError, ValueError) as exc:
            report.whole_files.append((path, f"could not annotate: {exc}"))
            continue
        target.write_text(rewritten, encoding="utf-8")
        report.not_excluded.extend((path, name) for name in failed)
    return report


def _ranges(numbers: list[int]) -> str:
    """`[3, 4, 5, 9]` as `3-5, 9`."""
    spans: list[str] = []
    start = prev = None
    for n in [*numbers, None]:
        if n is not None and prev is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            spans.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ", ".join(spans)


def render_summary(report: Report) -> str:
    """The Markdown step summary. Every exclusion carries its reason."""
    out = ["### Mutation scope", ""]
    out.append(
        "Only the functions this PR changed are mutated. mutmut mutates "
        "module-level functions and methods of module-level classes, "
        "including any code nested inside them.",
    )
    out.append("")
    mutated = [(s.path, u) for s in report.scopes for u in s.mutated]
    if mutated:
        out += ["| file | function | mutant names |", "| --- | --- | --- |"]
        out += [
            f"| `{p}` | `{u.qualname}` | `{u.key}__mutmut_*` |"
            for p, u in mutated
        ]
    else:
        out.append("No changed function is mutated.")
    out.append("")
    notes: list[str] = []
    for path, reason in report.whole_files:
        notes.append(f"- `{path}`: mutated whole: {reason}.")
    for path, reason in report.skipped_files:
        notes.append(f"- `{path}`: not mutated: {reason}.")
    for scope_ in report.scopes:
        # Only a unit whose own span the diff touched is in `in_scope`,
        # so an untouched decorated function is never listed here. In an
        # added file every unit is new, and is called added, not changed.
        verb = "added" if scope_.added else "changed"
        for u in scope_.in_scope:
            if u.skip_reason is not None:
                notes.append(
                    f"- `{scope_.path}` `{u.qualname}`: {verb}, but not "
                    f"mutated: {u.skip_reason}.",
                )
        for u in scope_.unchanged:
            notes.append(
                f"- `{scope_.path}` `{u.qualname}`: not mutated: only "
                "moved, or only comments or docstrings changed.",
            )
        if scope_.outside:
            notes.append(
                f"- `{scope_.path}` lines {_ranges(scope_.outside)}: not "
                "mutated: outside every function mutmut mutates (module "
                "level, class body, or a nested class).",
            )
    for path, name in report.not_excluded:
        notes.append(
            f"- `{path}` `{name}`: mutated although unchanged: its header "
            "could not be annotated.",
        )
    if notes:
        out += ["Left out or widened, with the reason:", "", *notes, ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True, help="base commit SHA")
    parser.add_argument("--head", required=True, help="head commit SHA")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="print per-file and per-function verdicts on stderr; write nothing",
    )
    mode.add_argument(
        "--write-pragmas", action="store_true",
        help="annotate out-of-scope functions in the working tree",
    )
    parser.add_argument(
        "--allow-src-rewrite", action="store_true",
        help="let --write-pragmas edit src/ outside CI (CI=true allows it)",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="write the Markdown scope report to this path",
    )
    args = parser.parse_args(argv)

    if (
        args.write_pragmas
        and os.environ.get("CI") != "true"
        and not args.allow_src_rewrite
    ):
        print(
            "mutation_scope: --write-pragmas edits files under src/ in place "
            "and is meant for a throwaway CI checkout. Refusing: CI is not "
            "\"true\". Pass --allow-src-rewrite to run it here anyway.",
            file=sys.stderr,
        )
        return 3

    try:
        report = function_scope(args.base, args.head, write=args.write_pragmas)
    except RuntimeError as exc:
        print(f"mutation_scope: {exc}", file=sys.stderr)
        return 2

    summary = render_summary(report)
    if args.dry_run:
        print(summary, file=sys.stderr)
    if args.summary is not None:
        args.summary.write_text(summary + "\n", encoding="utf-8")
    for path in report.files:
        print(path)
    if not report.files:
        print(
            "mutation_scope: no changed function carries a mutable change",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
