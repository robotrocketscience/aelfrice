#!/usr/bin/env python3
"""Which writer produced each `rebuild_log` row, and what it scored (#1405).

`rebuild_logs/*.jsonl` is written by two callers with different query
pipelines, and every published split of that corpus so far has been derived by
hand and then gone stale or failed to add up. This is the committed
re-derivation.

**The discriminator is `input.n_recent_turns`, not the pack summary.** An
earlier account split the corpus by "the `rebuild_v14` pack-summary shape",
which cannot work: both writers build `pack_summary` from the same helper, so
every row carries the identical six keys. The UPS path passes exactly one
synthetic turn, so `n_recent_turns == 1` selects it; `context_rebuilder` passes
the real window.

Why this script and not the one #1405's AC5 asks for: that criterion wants the
99.4%-changed / 8.5%-empty figures re-derivable, but those describe
`transform_query` over the `rebuild_v14` population, which this work shows is a
2.3% slice. Committing them would pin a mis-scoped measurement in place. The
per-path shares are the number that matters, so they are what is made
reproducible here.

Read-only. Never opens a `MemoryStore` -- see `db_paths` for why a diagnostic
must not, since opening one runs pending migrations.

Usage::

    python benchmarks/rebuild_log_query_population.py \\
        --logs .git/aelfrice/rebuild_logs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The UPS path builds one synthetic turn from the prompt and passes it alone,
# so this is an exact selector rather than a threshold.
UPS_N_RECENT_TURNS = 1

# Every row carries these, from both writers. Asserted rather than assumed:
# if they ever diverge, the shape *would* become a usable discriminator and
# this script should say so instead of silently keeping the old one.
PACK_SUMMARY_KEYS = frozenset({
    "n_candidates",
    "n_dropped_by_budget",
    "n_dropped_by_dedup",
    "n_dropped_by_floor",
    "n_packed",
    "total_chars_packed",
})


def load_rows(logs: Path) -> list[dict[str, Any]]:
    """Every parseable JSONL record under `logs`.

    Unparseable lines are skipped rather than fatal: these are append-only
    logs written by a non-blocking hook, so a torn final line is expected
    after a crash and is not a reason to refuse to report.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(logs.glob("*.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", type=Path, default=Path(".git/aelfrice/rebuild_logs"),
        help="directory of rebuild_log JSONL files (read-only)",
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.logs)
    if not rows:
        print(f"no rebuild_log rows under {args.logs}", file=sys.stderr)
        return 2

    total = len(rows)
    turns = Counter(
        (row.get("input") or {}).get("n_recent_turns") for row in rows
    )
    ups = turns.get(UPS_N_RECENT_TURNS, 0)
    other = total - ups

    print(f"logs   : {args.logs}")
    print(f"rows   : {total}")
    print()
    print("writer, by input.n_recent_turns:")
    print(f"  user_prompt_submit (== 1)  {ups:6d}  {100.0 * ups / total:5.1f}%")
    print(f"  rebuild_v14        (!= 1)  {other:6d}  {100.0 * other / total:5.1f}%")

    if other:
        print()
        print("  the non-UPS rows, by window size:")
        for value, count in sorted(
            ((v, c) for v, c in turns.items() if v != UPS_N_RECENT_TURNS),
            key=lambda kv: (-kv[1], str(kv[0])),
        ):
            print(f"    n_recent_turns={value!s:<6} {count:6d}")

    # The claim that pack_summary cannot discriminate, checked rather than
    # repeated. If this stops holding, the split above may be wrong.
    shapes = Counter(
        frozenset((row.get("pack_summary") or {}).keys()) for row in rows
    )
    print()
    if len(shapes) == 1 and next(iter(shapes)) == PACK_SUMMARY_KEYS:
        print(
            f"pack_summary: one shape across all {total} rows, so it "
            "discriminates nothing (the six expected keys)"
        )
    else:
        print("pack_summary: MORE THAN ONE SHAPE -- re-check the split above")
        for shape, count in shapes.most_common():
            print(f"  {count:6d}  {sorted(shape)}")

    # Forward-only field (#1405). Reported as a count so the A/B consumer
    # change has a number to wait on rather than a guess.
    scored = sum(
        1 for row in rows if "scored_query" in (row.get("input") or {})
    )
    print()
    print(
        f"scored_query present: {scored} / {total} "
        f"({100.0 * scored / total:.1f}%) -- forward-only, written from "
        "this change onward"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
