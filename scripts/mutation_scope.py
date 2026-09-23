#!/usr/bin/env python3
"""#1605 — which changed files a mutation run should actually mutate.

`.github/workflows/mutation.yml` scopes mutmut with `only_mutate`, whose
entries are FILE paths. So touching one line of a file puts the whole file
in the mutant set. A comment-only edit to a large module therefore costs
the same as rewriting it, and on `cli.py` — 11k lines — that exceeded the
job's 60-minute budget and was cancelled with no report (#1605).

The fix is to drop files whose diff cannot produce a mutant. This decides
that by **comparing ASTs with docstrings stripped**, not by classifying
lines. A line heuristic has to answer "is this line inside a docstring",
which needs a parser anyway, and it still says nothing about a change that
only moves code around. Two files whose stripped ASTs are equal differ by
comments, docstrings, blank lines, or formatting — none of which mutmut can
mutate — so the diff between them introduces no mutant.

Failing open is deliberate. A file that will not parse on either side, or
that cannot be read out of the base commit, is reported as mutable: the
cost of mutating a file needlessly is runner time, and the cost of skipping
one wrongly is an unmeasured mutant, so the asymmetry decides it.

This lives in a script rather than in the workflow's inline Python because
the workflow cannot be tested in CI — PyYAML is not importable there
(#1436) — while a script can be, and `tests/test_mutation_scope.py` does.

## Usage

    scripts/mutation_scope.py --base <sha> --head <sha>
    scripts/mutation_scope.py --base <sha> --head <sha> --dry-run

Prints one path per line on stdout: the changed files under
`src/aelfrice/` that carry a mutable change. Prints nothing when none do,
which the caller reads as "skip the run". `--dry-run` adds a per-file
verdict on stderr and changes no output on stdout.

Exits 0 when the scope was computed, 2 on a git failure. An empty scope is
a result, not a failure.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from typing import Final

#: Only these are mutated; matches the workflow's own pathspec.
PATHSPEC: Final[str] = "src/aelfrice/*.py"


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


def changed_files(base: str, head: str) -> list[str]:
    """Changed, non-deleted files under the pathspec, in the PR's own diff.

    Three dots, not two, for the reason the workflow already documents:
    `A..B` is the difference between two trees and would include commits
    merged into the base since the branch point.
    """
    out = _git(
        "diff", "--name-only", "--diff-filter=d", f"{base}...{head}",
        "--", PATHSPEC,
    )
    return [line for line in out.splitlines() if line.strip()]


def scope(base: str, head: str, *, explain: bool = False) -> list[str]:
    """The subset of changed files that carry a mutable change."""
    keep: list[str] = []
    for path in changed_files(base, head):
        mutable = has_mutable_change(_blob(base, path), _blob(head, path))
        if explain:
            verdict = "mutable" if mutable else "comments/docstrings only"
            print(f"  {path}: {verdict}", file=sys.stderr)
        if mutable:
            keep.append(path)
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit SHA")
    parser.add_argument("--head", required=True, help="head commit SHA")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="also print a per-file verdict on stderr",
    )
    args = parser.parse_args(argv)

    try:
        paths = scope(args.base, args.head, explain=args.dry_run)
    except RuntimeError as exc:
        print(f"mutation_scope: {exc}", file=sys.stderr)
        return 2

    for path in paths:
        print(path)
    if not paths:
        print(
            "mutation_scope: no changed file carries a mutable change",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
