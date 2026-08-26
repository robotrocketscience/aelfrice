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
import json
import sys
from typing import Any


def figures() -> dict[str, Any]:
    """Every store-free published figure, re-derived from the shipped code."""
    from aelfrice import hook, sentiment_feedback

    return {
        # #1442 — the Stop-block bounds. The *distribution* they were chosen
        # off needs a store and lives in stop_prompt_block_bounds.py; the
        # shipped limits themselves are code and belong here.
        "stop_prompt_max_items": hook.STOP_PROMPT_MAX_ITEMS,
        "stop_prompt_max_content": hook.STOP_PROMPT_MAX_CONTENT,
        # #193 — the sentiment pattern banks, quoted as "twelve positive and
        # twelve negative" in the module docstring and in the v2.0 spec.
        "sentiment_positive_patterns": len(sentiment_feedback._POSITIVE_PATTERNS),
        "sentiment_negative_patterns": len(sentiment_feedback._NEGATIVE_PATTERNS),
        "sentiment_max_prompt_chars": sentiment_feedback.MAX_PROMPT_CHARS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
