"""Characterise the ULID-prefix date clusters in `ingest_log` (#1283).

#1283's AC2 keys a deterministic spine recompute on
`(created_at, ingest_log ULID)`, and one of its stated constraints is to
**refuse to key on migration-synth ULIDs** — rows whose 48-bit ULID
prefix is migration wall-clock rather than anything about the content.
The obvious detector is "exclude the big ULID-prefix date cluster".

**That detector is unsafe, and this script is why.** The store has two
large clusters and they are opposite cases:

* `2026-04-29` — the #263 legacy log synthesis. 20,852 rows stamped
  `source_kind=legacy_unknown`, minted across **202 distinct millisecond
  prefixes inside a 201 ms window**, of which only **5 are
  session-scoped** — so it is inert for the spine on this store, by
  accident of when the migration ran rather than by design. Its ordering
  is `beliefs.rowid` relabelled, and excluding it is correct. The date
  bucket holds one extra, unrelated row, which is the first hint that
  the date is the wrong key.
* `2026-07-07` — 72,411 rows, **51.8% of the entire log**, minted across
  **9,699 distinct millisecond prefixes** over 19h10m,
  99.9% `source_kind=transcript`, and overwhelmingly session-scoped.
  That is a bulk backfill of roughly two months of transcripts, not a
  synth event.

A date-cluster detector cannot tell them apart, and applied to
`2026-07-07` it would drop **half the log** — 20,095 session-scoped
beliefs, the bulk of the spine population. It also over-selects on
`2026-04-29`: the synth burst is 201 **milliseconds** wide, while the
date bucket spans 5h47m and sweeps in an unrelated same-day row.

**The exclusion predicate is `source_kind = 'legacy_unknown'`.** That is
a stated, durable column rather than a threshold, and it is exact in
both directions here: all 20,852 `legacy_unknown` rows fall inside the
201 ms window, and every row inside that window is `legacy_unknown`. It
needs no tuning, cannot drift as the log grows, and does not depend on
the ULID prefix it exists to distrust. Minting density (rows per
distinct millisecond prefix: 103 vs 7.5) corroborates the split but is
not what a rule should key on.

The backfill carries a second warning of its own. Its ULID prefix is
*backfill processing time*, so it is not a wall-clock proxy for the
content: the median row is minted **14.1 days** after the turn it
records, and 66.4% are more than a week late. Any rule that reads the
prefix as "when this happened" is wrong for half the log. As an
*ordering* key it survives — ULID order agrees with content-`ts` order
on 98.94% of adjacent pairs — but those are different claims and only
the second one holds.

Read-only in the strong sense: opens a `mode=ro` SQLite connection
rather than a `MemoryStore`, because constructing one runs open-time
DDL, pending one-shot migrations and the #1314 lock-expiry sweep, and a
diagnostic must not mutate what it inspects.

Usage::

    uv run python benchmarks/ingest_log_ulid_clusters.py [PATH_TO_DB]

Exit status: 0 on success, 2 if the store is missing.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Final

_DEFAULT_DB: Final[Path] = Path(".git/aelfrice/memory.db")

# Crockford base32, the ULID alphabet. Lexicographic order on the encoded
# string is numeric order on the decoded value, which is what lets the
# spine sort on the raw id.
_CROCKFORD: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# How many clusters to profile, largest first.
_TOP_N: Final[int] = 4

# The #263 legacy log synthesis stamps its rows with this source_kind.
# It is the exclusion predicate — a durable column, exact in both
# directions on this store — and the density figure below is only
# corroboration.
_SYNTH_SOURCE_KIND: Final[str] = "legacy_unknown"


def _ulid_ms(ulid: str) -> int | None:
    """Decode a ULID's 48-bit millisecond prefix, or None if malformed."""
    value = 0
    for ch in ulid[:10].upper():
        idx = _CROCKFORD.find(ch)
        if idx < 0:
            return None
        value = value * 32 + idx
    return value


def _parse_ts(raw: object) -> dt.datetime | None:
    """Parse an `ingest_log.ts`, normalising to UTC. None if unparseable.

    The column carries both `...Z` and `...+00:00` spellings, and some
    legacy rows carry a non-UTC offset, so this cannot assume a shape.
    """
    text = str(raw).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[idx]


def main(argv: list[str]) -> int:
    db = Path(argv[1]) if len(argv) > 1 else _DEFAULT_DB
    if not db.exists():
        print(f"no store at {db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT rowid, id, ts, source_kind, derived_belief_ids "
            "FROM ingest_log ORDER BY rowid"
        ).fetchall()
        session_scoped = {
            str(r[0])
            for r in conn.execute(
                "SELECT id FROM beliefs WHERE valid_to IS NULL "
                "AND session_id IS NOT NULL AND session_id != ''"
            )
        }
    finally:
        conn.close()

    print(f"store            : {db}")
    print(f"ingest_log rows  : {len(rows):,}")
    print(f"session-scoped   : {len(session_scoped):,} active beliefs")

    by_day: dict[str, list[sqlite3.Row]] = collections.defaultdict(list)
    minted: dict[int, int] = {}
    unparseable = 0
    for row in rows:
        ms = _ulid_ms(str(row["id"]))
        if ms is None:
            unparseable += 1
            continue
        minted[row["rowid"]] = ms
        stamp = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
        by_day[stamp.date().isoformat()].append(row)
    if unparseable:
        print(f"unparseable ids  : {unparseable:,}")

    print("\n--- ULID-prefix date clusters, largest first ---")
    for day, members in sorted(
        by_day.items(), key=lambda kv: -len(kv[1])
    )[:_TOP_N]:
        _profile(day, members, minted, session_scoped, len(rows))
    return 0


def _profile(
    day: str,
    members: list[sqlite3.Row],
    minted: dict[int, int],
    session_scoped: set[str],
    total_rows: int,
) -> None:
    stamps = [minted[r["rowid"]] for r in members]
    lo, hi = min(stamps), max(stamps)
    distinct_ms = len(set(stamps))
    density = len(members) / distinct_ms if distinct_ms else 0.0

    print(f"\n{day}: {len(members):,} rows ({100*len(members)/total_rows:.1f}% of log)")
    print(
        f"  minted           : {dt.datetime.fromtimestamp(lo/1000, dt.timezone.utc)}"
        f" .. {dt.datetime.fromtimestamp(hi/1000, dt.timezone.utc)}"
        f"  ({(hi-lo)/1000:,.0f}s)"
    )
    # Rows per distinct millisecond prefix. A machine loop and a working
    # day separate cleanly here — the synth burst runs at ~103, the
    # busiest genuine day at ~7.5 — but this is reported as corroboration
    # and deliberately has no threshold constant: the whole finding is
    # that the rule should key on `source_kind`, and a named cut-off
    # sitting here would invite exactly the heuristic being argued
    # against.
    print(
        f"  distinct ms      : {distinct_ms:,}"
        f"   density {density:,.1f} rows/ms-prefix"
    )
    kinds = collections.Counter(str(r["source_kind"]) for r in members)
    print(f"  source_kind      : {dict(kinds.most_common(3))}")

    # ULID order vs rowid order. 100% is expected of any single-writer
    # sequential ingest, so on its own it distinguishes nothing — it is
    # only damning for the synth cluster, where the source SELECT had no
    # ORDER BY and rowid order is therefore all the ULID encodes.
    by_ulid = sorted(members, key=lambda r: str(r["id"]))
    ascending = sum(
        1 for a, b in zip(by_ulid, by_ulid[1:]) if a["rowid"] < b["rowid"]
    )
    pairs = max(1, len(by_ulid) - 1)
    print(f"  ULID order==rowid: {ascending:,}/{pairs:,} ({100*ascending/pairs:.2f}%)")

    covered: set[str] = set()
    for row in members:
        raw = row["derived_belief_ids"]
        if not raw:
            continue
        try:
            covered.update(str(b) for b in (json.loads(raw) or []))
        except (ValueError, TypeError):
            continue
    in_spine = len(covered & session_scoped)
    print(
        f"  beliefs referenced: {len(covered):,}"
        f"   session-scoped: {in_spine:,}"
        f" ({100*in_spine/len(covered):.1f}%)" if covered else
        "  beliefs referenced: 0"
    )

    # Lag = how far the mint time sits after the content it records.
    lags = sorted(
        (minted[r["rowid"]] / 1000 - parsed.timestamp())
        for r in members
        if (parsed := _parse_ts(r["ts"])) is not None
    )
    if lags:
        print(
            f"  mint-minus-content: p50 {_percentile(lags, 0.5)/86400:,.1f}d"
            f"   p90 {_percentile(lags, 0.9)/86400:,.1f}d"
            f"   max {lags[-1]/86400:,.1f}d"
        )
        stale = sum(1 for lag in lags if lag > 7 * 86400)
        print(
            f"  older than 7 days : {stale:,} ({100*stale/len(lags):.1f}%)"
        )

    # Does ULID order track content order? This is the only claim AC2
    # actually needs from a backfill cluster — the prefix being a poor
    # wall-clock proxy does not by itself disqualify it as a sort key.
    timed = [
        (str(r["id"]), parsed)
        for r in by_ulid
        if (parsed := _parse_ts(r["ts"])) is not None
    ]
    if len(timed) > 1:
        agree = sum(1 for a, b in zip(timed, timed[1:]) if a[1] <= b[1])
        span = len(timed) - 1
        print(
            f"  ULID order==ts    : {agree:,}/{span:,} ({100*agree/span:.2f}%)"
        )

    synth = kinds.most_common(1)[0][0] == _SYNTH_SOURCE_KIND
    if synth:
        verdict = (
            f"SYNTH ({_SYNTH_SOURCE_KIND}) — exclude by source_kind, not by "
            f"date; density {density:,.0f} rows/ms corroborates"
        )
    else:
        verdict = (
            "REAL MINT — do NOT exclude. The prefix is honest wall clock "
            "even where the content it records is old, so it is a sound "
            "sort key and an unsound timestamp."
        )
    print(f"  verdict          : {verdict}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
