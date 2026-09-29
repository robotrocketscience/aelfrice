"""Count how many transcript beliefs a per-turn anchor reaches (#1602).

This produces the #1602 AC1 figures (AC6: no figure without a producer in
the tree). It reads one aelfrice `memory.db` and prints JSON with:

- `transcript_rows`: `ingest_log` rows with `source_kind = 'transcript'`.
- `rows_with_turn_identity`: those rows that carry both `session_id` and
  `ts`, the two halves of a turn's identity. On a store written before
  #1602 this is an upper bound: a turn ingested without its own
  timestamp got one from the clock, and nothing on the row records which
  happened. The format cannot tell them apart either, because aelfrice's
  own transcript logger also writes `+00:00` timestamps. Through
  `ingest_jsonl`, the fallback fires only for a line that has no `ts`.
- `distinct_turns` and `distinct_sessions`: over those rows.
- `beliefs_reachable`: distinct beliefs derived from those rows, which is
  the population a turn anchor could cover.
- `rows_with_turn_sha`: rows that carry the turn fingerprint, so their
  beliefs get a turn anchor. It is 0 on a store written before #1602.
- `anchors_by_kind`: `belief_documents` rows split into `turn` (the URI
  has a turn fragment) and `label_only`.
- `distinct_doc_uris`: the count the issue reported as 4.

The store opens read-only (`mode=ro`), because a normal
`MemoryStore.open` runs DDL and migrations. Copy the store together with
its `-wal` and `-shm` sidecars first when a live process holds it.

Usage:

    uv run python scripts/turn_anchor_census.py --db <path-to>/memory.db

Exits 2 when the file is missing or has no `ingest_log` table.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from aelfrice.derivation import META_TURN_SHA

def census(db_path: Path) -> dict[str, object]:
    """Return the #1602 AC1 counts for the store at `db_path`."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT session_id, ts, raw_meta, derived_belief_ids "
            "FROM ingest_log WHERE source_kind = 'transcript'"
        ).fetchall()
        anchors = conn.execute(
            "SELECT doc_uri FROM belief_documents"
        ).fetchall()
    finally:
        conn.close()

    turns: set[tuple[str, str]] = set()
    beliefs: set[str] = set()
    with_identity = 0
    with_sha = 0
    for session_id, ts, raw_meta, derived in rows:
        if session_id and ts:
            with_identity += 1
            turns.add((session_id, ts))
            for bid in json.loads(derived) if derived else []:
                beliefs.add(bid)
        meta = json.loads(raw_meta) if raw_meta else None
        if isinstance(meta, dict) and meta.get(META_TURN_SHA):
            with_sha += 1

    uris = [r[0] for r in anchors]
    turn_anchors = sum(1 for u in uris if "#" in u)
    return {
        "transcript_rows": len(rows),
        "rows_with_turn_identity": with_identity,
        "distinct_turns": len(turns),
        "distinct_sessions": len({s for s, _ in turns}),
        "beliefs_reachable": len(beliefs),
        "rows_with_turn_sha": with_sha,
        "anchors_by_kind": {
            "turn": turn_anchors,
            "label_only": len(uris) - turn_anchors,
        },
        "distinct_doc_uris": len(set(uris)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args(argv)
    db: Path = args.db
    if not db.is_file():
        print(f"turn_anchor_census: {db} not found", file=sys.stderr)
        return 2
    try:
        result = census(db)
    except sqlite3.OperationalError as exc:
        print(f"turn_anchor_census: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
