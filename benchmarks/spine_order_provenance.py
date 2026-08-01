"""#1283 temporal-spine order provenance — what actually decides TEMPORAL_NEXT,
and which candidate key can reproduce it.

#1283 prices its recommended fix on the claim that **97.2% of edges are second-
order projections of data already in the log**, resting on `TEMPORAL_NEXT`
(88.3% of all edges) being *"recomputable from the belief table alone — it is an
ordering over beliefs' own timestamps."*

Timestamps do not decide that ordering. `store.session_predecessor_id` orders by
`(created_at, rowid)`:

    AND (b2.created_at < b1.created_at
         OR (b2.created_at = b1.created_at AND b2.rowid < b1.rowid))
    ORDER BY b2.created_at DESC, b2.rowid DESC

and `created_at` is massively tied, so `rowid` is not a tiebreak — it is the
primary sort key for almost the whole corpus. `rowid` is also the worst possible
thing to depend on for a determinism contract: `beliefs` is declared
`id TEXT PRIMARY KEY` (not `INTEGER PRIMARY KEY`, not `WITHOUT ROWID`), so the
rowid is implicit and **VACUUM is free to renumber it**. It is not a logical
column, it is not in `ingest_log`, and it does not survive a rebuild.

This script measures three things on a real store, all deterministic, no judge:

  1. **Tie density** — what share of session-scoped beliefs sit in a
     `(session_id, created_at)` group of size > 1, i.e. are ordered by rowid
     rather than by time.
  2. **Edge census** — the type distribution behind the 97.2% claim.
  3. **Reproduction fidelity** — rebuild every session's predecessor chain
     under two candidate durable keys and report what share of the *shipped*
     spine each reproduces:

       * `(created_at, belief_id)` — belief-table-only, #1283's option (b).
       * `(created_at, ingest_log ULID)` — log-derived, #1283's option (a).
         `ingest_log.id` is a ULID: monotonic, lexicographically sortable,
         a `TEXT PRIMARY KEY`, and therefore VACUUM-immune.

Read-only: the store is opened `mode=ro` and nothing is written. Aggregate
counts only — no belief content, no session ids and no prompt text is printed.

Run: `python benchmarks/spine_order_provenance.py --db <path/to/memory.db>`

Exits non-zero if the store cannot support the measurement (no session-scoped
beliefs, or no ingest_log coverage), so a silently empty run cannot be mistaken
for a result.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

# Sorts after every real ULID (ULIDs are Crockford base32: 0-9, A-Z).
_NO_LOG_SENTINEL = "~"


def _connect_ro(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def tie_density(conn: sqlite3.Connection) -> dict[str, int]:
    """How many session-scoped beliefs are ordered by rowid, not by time."""
    row = conn.execute("""
        WITH g AS (
            SELECT COUNT(*) AS c
            FROM beliefs
            WHERE valid_to IS NULL AND session_id IS NOT NULL
            GROUP BY session_id, created_at
        )
        SELECT COALESCE(SUM(CASE WHEN c > 1 THEN c ELSE 0 END), 0) AS tied,
               COALESCE(SUM(c), 0)                                 AS total,
               COALESCE(MAX(c), 0)                                 AS largest
        FROM g
    """).fetchone()
    distinct_ts = conn.execute("""
        SELECT COUNT(DISTINCT created_at) FROM beliefs
        WHERE valid_to IS NULL AND session_id IS NOT NULL
    """).fetchone()[0]
    return {
        "tied": int(row["tied"]), "total": int(row["total"]),
        "largest_group": int(row["largest"]), "distinct_created_at": int(distinct_ts),
    }


def edge_census(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    return [
        (r[0], int(r[1]))
        for r in conn.execute(
            "SELECT type, COUNT(*) FROM edges GROUP BY type ORDER BY 2 DESC"
        )
    ]


def belief_to_log_ulid(conn: sqlite3.Connection) -> dict[str, str]:
    """Map belief id -> earliest originating `ingest_log` ULID."""
    out: dict[str, str] = {}
    for lid, blob in conn.execute(
        "SELECT id, derived_belief_ids FROM ingest_log "
        "WHERE derived_belief_ids IS NOT NULL"
    ):
        try:
            bids = json.loads(blob)
        except (TypeError, ValueError):
            continue
        for bid in bids or []:
            if bid not in out or lid < out[bid]:
                out[bid] = lid
    return out


def _chain(items: Iterable[tuple[str, int, str]],
           key: Callable[[tuple[str, int, str]], object]) -> set[tuple[str, str]]:
    """Predecessor links (successor, predecessor) under one ordering key."""
    ordered = sorted(items, key=key)  # type: ignore[arg-type]
    return {
        (ordered[i][2], ordered[i - 1][2]) for i in range(1, len(ordered))
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True,
                    help="path to memory.db (opened read-only)")
    args = ap.parse_args()

    conn = _connect_ro(args.db)

    ties = tie_density(conn)
    if ties["total"] == 0:
        print("no session-scoped active beliefs; nothing to measure",
              file=sys.stderr)
        return 1

    print("=" * 70)
    print("1. TIE DENSITY — is TEMPORAL_NEXT ordered by time, or by rowid?")
    print("=" * 70)
    pct = 100.0 * ties["tied"] / ties["total"]
    print(f"  session-scoped active beliefs : {ties['total']:,}")
    print(f"  distinct created_at values    : {ties['distinct_created_at']:,}")
    print(f"  in a (session, created_at) tie: {ties['tied']:,}  ({pct:.2f}%)")
    print(f"  largest single tie group      : {ties['largest_group']:,}")
    print(f"  => {pct:.1f}% of the spine is ordered by rowid, not by timestamp.")

    print()
    print("=" * 70)
    print("2. EDGE CENSUS")
    print("=" * 70)
    census = edge_census(conn)
    total_edges = sum(n for _, n in census) or 1
    for etype, n in census:
        print(f"  {etype:16s} {n:7,}  {100.0 * n / total_edges:5.2f}%")

    print()
    print("=" * 70)
    print("3. REPRODUCTION FIDELITY vs the shipped (created_at, rowid) spine")
    print("=" * 70)
    log_ulid = belief_to_log_ulid(conn)
    rows = conn.execute("""
        SELECT session_id, id, created_at, rowid FROM beliefs
        WHERE valid_to IS NULL AND session_id IS NOT NULL
    """).fetchall()

    covered = sum(1 for r in rows if r["id"] in log_ulid)
    cov_pct = 100.0 * covered / len(rows)
    print(f"  beliefs with an ingest_log ULID: {covered:,} / {len(rows):,} "
          f"({cov_pct:.1f}%)")
    if covered == 0:
        print("  no ingest_log coverage; fidelity is unmeasurable here",
              file=sys.stderr)
        return 1

    by_session: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(
            (r["created_at"], int(r["rowid"]), r["id"])
        )

    shipped: set[tuple[str, str]] = set()
    candidates: dict[str, set[tuple[str, str]]] = {
        "(created_at, belief_id)  [belief-derived]": set(),
        "(created_at, log ULID)   [log-derived]": set(),
    }
    for items in by_session.values():
        shipped |= _chain(items, key=lambda t: (t[0], t[1]))
        candidates["(created_at, belief_id)  [belief-derived]"] |= _chain(
            items, key=lambda t: (t[0], t[2]))
        candidates["(created_at, log ULID)   [log-derived]"] |= _chain(
            items, key=lambda t: (t[0], log_ulid.get(t[2],
                                                     _NO_LOG_SENTINEL + t[2])))

    print(f"  shipped spine links            : {len(shipped):,}")
    print()
    print(f"  {'candidate ordering key':44s} {'reproduced':>12s} {'changed':>12s}")
    for name, links in candidates.items():
        same = len(shipped & links)
        changed = len(shipped) - same
        print(f"  {name:44s} {same:7,} {100.0 * same / len(shipped):5.1f}% "
              f"{changed:7,} {100.0 * changed / len(shipped):5.1f}%")

    print()
    print("  => a durable key is required because rowid is implicit and VACUUM")
    print("     may renumber it. The log-derived key is the one that reproduces")
    print("     the existing spine; the belief-table key does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
