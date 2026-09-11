"""#1469 — the store-free half of the published-figure gate.

Not every published figure needs a corpus. A shipped bound, a pattern-bank
size, a length guard: these are properties of the code, they are quoted in
release notes and in source comments as numbers, and they go stale the moment
someone edits the constant without editing the prose. That class is
re-derivable anywhere the package imports, which means public CI can hold it to
a hard diff rather than to an advisory.

This module is the producer for that class. It imports the shipped modules and
reports what the constants actually are — never a literal transcribed from the
prose, which would make the check a tautology. Compare
`benchmarks/stop_prompt_block_bounds.py`, whose figures cannot be re-derived
without a real belief store and whose markers therefore carry `corpus=`.

Each key is a figure that ships in prose somewhere. Adding a key here is only
half a change: the figure it re-derives has to gain a marker naming it, or the
key guards nothing.

Usage:

    uv run python benchmarks/published_constants.py            # human-readable
    uv run python benchmarks/published_constants.py --emit-figures   # JSON

`--emit-figures` is the protocol `scripts/check_derived_figures.py` speaks: a
flat JSON object of key -> value on stdout, and nothing else on stdout.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def _top_level_test_count(rel_path: str) -> int:
    """How many top-level `test_` functions a test module declares.

    Parsed rather than grepped: a `def test_` inside a string or a comment is
    not a test, and this number is published as an enumeration count in a
    release note. Top-level only, because the file it is used on declares no
    test classes and a class would change what the count means.
    """
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def figures() -> dict[str, Any]:
    """Every store-free published figure, re-derived from the shipped code."""
    from aelfrice import hook, sentiment_feedback

    return {
        # #1442 — the Stop-block bounds. The *distribution* they were chosen
        # off needs a store and lives in stop_prompt_block_bounds.py; the
        # shipped limits themselves are code and belong here.
        "stop_prompt_max_items": hook.STOP_PROMPT_MAX_ITEMS,
        "stop_prompt_max_content": hook.STOP_PROMPT_MAX_CONTENT,
        # #193 — the sentiment pattern banks, published as "12 positive and 12
        # negative" in the module docstring and in the v2.0 spec. Written in
        # digits there so the gate can see them: a figure spelled out in words
        # is one no scanner can bind a marker to.
        # Read through `getattr` rather than as attributes: the banks are
        # module-private, and reaching into them by name is a pyright-strict
        # error. Re-declaring them public to satisfy a benchmark would widen
        # the module's API for the convenience of its measurer.
        "sentiment_positive_patterns": len(
            getattr(sentiment_feedback, "_POSITIVE_PATTERNS")
        ),
        "sentiment_negative_patterns": len(
            getattr(sentiment_feedback, "_NEGATIVE_PATTERNS")
        ),
        "sentiment_max_prompt_chars": sentiment_feedback.MAX_PROMPT_CHARS,
        # #1436 — the mutation count the #1451 review disputed: the entry
        # published "thirteen" over a list that enumerated twelve. An
        # enumeration count is not a constant, but it is still code-derived:
        # each listed mutation is one top-level test in this file, so the
        # count re-derives here and a fourteenth test turns the entry red
        # rather than leaving the number to drift.
        "ci_manual_dispatch_mutations": _top_level_test_count(
            "tests/test_ci_manual_dispatch.py"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--emit-figures",
        action="store_true",
        help="emit a flat JSON object of key -> value on stdout and nothing else",
    )
    args = ap.parse_args(argv)

    values = figures()
    if args.emit_figures:
        json.dump(values, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    width = max(len(k) for k in values)
    for key in sorted(values):
        print(f"{key:<{width}}  {values[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
