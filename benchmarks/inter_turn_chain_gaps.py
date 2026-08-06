"""#1364 — how many inter-turn DERIVED_FROM edges span a skipped turn.

`ingest_jsonl` anchored its chain on `_ingest_turn_ids`, which returns
**newly-inserted** ids. A turn whose sentences all corroborated existing
beliefs therefore returned `[]`, was skipped, and left `last_per_session`
pointing at the turn before it — so the next turn linked across it and
asserted a derivation the transcript does not support.

This measures the footprint on a real store.

## Why `ingest_log` and not `beliefs`

`beliefs.created_at` cannot see this. A corroborating turn creates no
belief row — the corroborated belief keeps the `created_at` of the turn
that first inserted it — so a corroboration-only turn contributes no
timestamp of its own to the belief table and is invisible there. The
turn boundary only exists in `ingest_log`, which carries one row per
sentence with the turn's `ts` whether the sentence inserted or
corroborated.

Measuring this off `beliefs` gives a plausible-looking number that
counts something else. It was the first thing tried here.

## What the buckets mean

A turn is classified from its own log rows:

* `has-new` — at least one sentence first appears at this turn.
* `corroboration-only` — every sentence resolves to a belief whose
  earliest log row is an *earlier* turn. **These are the turns this
  defect skips.**
* `no-belief` — the turn produced no belief at all (`persist=False`,
  noise-filtered).

An edge is then attributed by what it spans. Only the
`corroboration-present` bucket is attributable to #1364. Edges spanning
only `has-new` turns are a different thing — `last_per_session` is
per-invocation, so a session ingested across several `ingest_jsonl`
calls legitimately restarts its chain at each one. That bucket is
reported rather than folded in, because folding it in would overstate
this defect by roughly 2.3x.

Usage:

    uv run python -m benchmarks.inter_turn_chain_gaps --store <path>

`--store` is opened **read-only** through `sqlite3`, never through
`MemoryStore` — opening a store runs migrations and a lifecycle sweep,
which is a write.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import Counter, defaultdict

CORROBORATION_ONLY = "corroboration-only"
HAS_NEW = "has-new"
NO_BELIEF = "no-belief"


def measure(store_path: str) -> dict[str, object]:
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        rows = list(conn.execute(
            "SELECT session_id, ts, derived_belief_ids FROM ingest_log "
            "WHERE source_kind = 'transcript' AND session_id IS NOT NULL "
            "ORDER BY ts"
        ))
        edges = conn.execute(
            "SELECT src, dst FROM edges WHERE type = 'DERIVED_FROM'"
        ).fetchall()
    finally:
        conn.close()

    first_seen: dict[str, tuple[str, str]] = {}
    turn_beliefs: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen_turns: set[tuple[str, str]] = set()
    for sid, ts, blob in rows:
        key = (str(sid), str(ts))
        seen_turns.add(key)
        try:
            ids = json.loads(blob) if blob else []
        except (TypeError, ValueError):
            ids = []
        if not isinstance(ids, list):
            continue
        for b in ids:
            if isinstance(b, str):
                turn_beliefs[key].append(b)
                first_seen.setdefault(b, key)

    kind: dict[tuple[str, str], str] = {}
    for key, beliefs in turn_beliefs.items():
        kind[key] = (
            HAS_NEW if any(first_seen[b] == key for b in beliefs)
            else CORROBORATION_ONLY
        )
    for key in seen_turns:
        kind.setdefault(key, NO_BELIEF)

    by_session: dict[str, list[str]] = defaultdict(list)
    for sid, ts in kind:
        by_session[sid].append(ts)
    ordered = {k: sorted(v) for k, v in by_session.items()}

    inter = 0
    spanned = 0
    causes: Counter[str] = Counter()
    for src, dst in edges:
        a = first_seen.get(str(src))
        b = first_seen.get(str(dst))
        if a is None or b is None or a[0] != b[0] or a[1] == b[1]:
            continue
        inter += 1
        lo, hi = sorted((a[1], b[1]))
        arr = ordered[a[0]]
        between = arr[bisect.bisect_right(arr, lo):bisect.bisect_left(arr, hi)]
        if not between:
            continue
        spanned += 1
        kinds = {kind[(a[0], t)] for t in between}
        causes[
            "corroboration-present" if CORROBORATION_ONLY in kinds
            else "other-only"
        ] += 1

    turn_kinds = Counter(kind.values())
    return {
        "transcript_log_rows": len(rows),
        "turns_total": len(kind),
        "turns_by_kind": dict(turn_kinds),
        "inter_turn_same_session_edges": inter,
        "spanning_at_least_one_turn": spanned,
        "attributable_to_1364": causes["corroboration-present"],
        "spanning_only_other_kinds": causes["other-only"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#1364 inter-turn chain gaps")
    ap.add_argument("--store", required=True, help="path to a memory.db")
    args = ap.parse_args(argv)

    result = measure(args.store)
    for k, v in result.items():
        print(f"{k:<34} {v}")

    inter = int(result["inter_turn_same_session_edges"])  # type: ignore[arg-type]
    if inter:
        att = int(result["attributable_to_1364"])  # type: ignore[arg-type]
        print()
        print(f"attributable share: {att}/{inter} = {100 * att / inter:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
