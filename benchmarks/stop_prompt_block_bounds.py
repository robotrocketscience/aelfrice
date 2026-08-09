"""#1442 — re-derive the Stop-block size distribution that sets the two bounds.

`hook._format_stop_prompt` writes to stderr once per assistant turn and, before
#1442, bounded neither the number of candidates nor the length of any one of
them. `STOP_PROMPT_MAX_ITEMS` and `STOP_PROMPT_MAX_CONTENT` are set off the
distribution this script produces, so per the project rule the published
figures ship with the script that re-derives them.

## What it measures

For every `session_id` in a store, the exact population
`hook._collect_lock_candidates` would return — session-scoped, not
`lock_level=user`, and correction-class **or** passing `detect_directive` — and
renders it through the real `_format_stop_prompt`.

It reports the distribution under two populations:

* `pre_1315`  — correction-class only, the population before #1315 widened
  candidacy.
* `post_1315` — the shipped population, correction-class ∪ directives.

Both, because the obvious reading of the size tail is that #1315's widening
caused it. It did not: the maximum is byte-identical under either population,
since the worst session's candidates are all correction-class. Anyone
re-attributing a volume regression to a candidacy change should run both arms
before drawing that conclusion.

## Read-only

Opens each store through a `mode=ro` SQLite URI rather than `MemoryStore`,
because `MemoryStore.__init__` runs DDL, pending migrations and the #1314
open-time expiry sweep — opening a live store to measure it would write to it.

Usage:

    uv run python benchmarks/stop_prompt_block_bounds.py [DB ...]
    uv run python benchmarks/stop_prompt_block_bounds.py --json out.json

With no DB arguments it sweeps every `*/.git/aelfrice/memory.db` under
`~/projects`, which is where this project's repo-local stores live.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sqlite3
import sys
from typing import Any

from aelfrice import hook
from aelfrice.directive_detector import detect_directive
from aelfrice.hook import _format_stop_prompt
from aelfrice.models import BELIEF_CORRECTION, LOCK_NONE, Belief

# Mirrors `hook._STOP_PROMPT_AGENT_ORIGINS`. Duplicated as a literal so a
# change to the hook's frozenset shows up here as a divergence rather than
# being silently absorbed into the measurement.
AGENT_ORIGINS = frozenset({"agent_inferred", "agent_remembered"})


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def at(frac: float) -> int:
        return ordered[min(n - 1, int(frac * n))]

    return {
        "p50": at(0.50), "p75": at(0.75), "p90": at(0.90),
        "p95": at(0.95), "p99": at(0.99), "max": ordered[-1],
    }


def _rows(db: str) -> list[tuple[str, str, str, str, str]]:
    """Candidate rows in the order production sees them: `rowid DESC`.

    The `ORDER BY` is load-bearing, not tidiness. `_collect_lock_candidates`
    walks `MemoryStore.list_belief_ids_newest_first`, which is
    `ORDER BY rowid DESC`, and `_format_stop_prompt` caps by taking the
    **head** — so the bounded figures are a function of which 20 rows
    arrive first. With no `ORDER BY` SQLite scans `rowid ASC`, i.e.
    oldest-first, and every bounded percentile measures a slice production
    never renders (max 10,218 rather than 11,388 on this repo's store).
    Sorting by `id` instead measures content-hash order, the superseded
    design this bound exists to avoid.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            (str(r[0]), str(r[1] or ""), str(r[2] or ""), str(r[3] or ""), str(r[4] or ""))
            for r in con.execute(
                "SELECT id, content, type, origin, session_id FROM beliefs "
                "WHERE valid_to IS NULL AND session_id IS NOT NULL "
                "AND (lock_level IS NULL OR lock_level != 'user') "
                "ORDER BY rowid DESC"
            )
        ]
    finally:
        con.close()


def _as_belief(bid: str, content: str, btype: str, origin: str, sid: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=btype, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-01-01T00:00:00Z", last_retrieved_at=None,
        session_id=sid, origin=origin,
    )


def _render_unbounded(candidates: list[Belief]) -> str:
    """`_format_stop_prompt` with both #1442 bounds lifted.

    This is what the renderer did before #1442, and it is how the "before"
    figures in the changelog stay re-derivable. It raises the real constants
    rather than reimplementing the old renderer, so it cannot drift from
    what actually ships — a hand-copied old body would keep reporting the
    2026 shape forever.
    """
    items, content = hook.STOP_PROMPT_MAX_ITEMS, hook.STOP_PROMPT_MAX_CONTENT
    hook.STOP_PROMPT_MAX_ITEMS = 10**9        # type: ignore[misc]
    hook.STOP_PROMPT_MAX_CONTENT = 10**9      # type: ignore[misc]
    try:
        return _format_stop_prompt(candidates)
    finally:
        hook.STOP_PROMPT_MAX_ITEMS = items    # type: ignore[misc]
        hook.STOP_PROMPT_MAX_CONTENT = content  # type: ignore[misc]


def measure(dbs: list[str]) -> dict[str, Any]:
    pre: dict[str, list[Belief]] = collections.defaultdict(list)
    post: dict[str, list[Belief]] = collections.defaultdict(list)
    lengths: list[int] = []
    active = 0
    skipped: list[str] = []

    for db in dbs:
        try:
            rows = _rows(db)
        except sqlite3.Error as exc:            # schema drift on old stores
            skipped.append(f"{db}: {exc}")
            continue
        for bid, content, btype, origin, sid in rows:
            active += 1
            correction_class = btype == BELIEF_CORRECTION or origin in AGENT_ORIGINS
            if not (correction_class or detect_directive(content)):
                continue
            b = _as_belief(bid, content, btype, origin, sid)
            key = f"{db}::{sid}"
            post[key].append(b)
            if correction_class:
                pre[key].append(b)
            lengths.append(len(content))

    def arm(groups: dict[str, list[Belief]]) -> dict[str, Any]:
        counts = [len(v) for v in groups.values()]
        return {
            "sessions": len(groups),
            "candidates_per_session": _percentiles(counts),
            "rendered_bytes_bounded": _percentiles(
                [len(_format_stop_prompt(v)) for v in groups.values()]
            ),
            "rendered_bytes_unbounded": _percentiles(
                [len(_render_unbounded(v)) for v in groups.values()]
            ),
        }

    return {
        "stores": len(dbs),
        "skipped": skipped,
        "active_unlocked_session_scoped_beliefs": active,
        "candidate_content_chars": _percentiles(lengths),
        "candidates_over_1000_chars": sum(1 for x in lengths if x > 1000),
        "candidates_total": len(lengths),
        "pre_1315": arm(pre),
        "post_1315": arm(post),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dbs", nargs="*", help="store paths; default sweeps ~/projects")
    ap.add_argument("--json", dest="json_out", help="also write the report here")
    args = ap.parse_args(argv)

    dbs = args.dbs or sorted(
        glob.glob(os.path.expanduser("~/projects/*/.git/aelfrice/memory.db"))
    )
    if not dbs:
        print("no stores found", file=sys.stderr)
        return 1

    report = measure(dbs)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
