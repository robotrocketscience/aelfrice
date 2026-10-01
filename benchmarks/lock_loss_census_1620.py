"""#1620 — re-derive the lock-loss figures from one or more stores.

#1620 reported its counts from ad-hoc probes. AC5 (#1469: no figure
without a producer) asks for the census that re-derives them. This is
that census. It embeds no figure of its own: every number comes from
running it against a store.

## What each figure means and why

* `lock_level.active` and `lock_level.all` — the `lock_level`
  distribution over active beliefs (`valid_to IS NULL`) and over every
  row, reported separately. A cleanup that retires rows changes the
  second and leaves the first alone, so a "1 of N" figure is
  meaningless until it says which N. Both are always reported.
* `aelf_commands` — beliefs whose content, after leading whitespace, is
  an aelfrice command: it starts with `/aelf:`, `aelf `, or
  `uv run aelf`. These are instructions to the tool that the capture
  path stored as claims about the world. This is the AC2 population.
* `self_ingestion` — beliefs whose content contains `<belief id=`, which
  is aelfrice re-admitting its own rendered injection block. This is
  the AC4 population.
* `speculative_active` — active beliefs with the speculative origin.
  `aelf lock` on text that already exists as one of these reported
  success without locking (H3), so this is the population that hole
  could reach.
* `family` — with `--pattern REGEX`, every belief whose content matches
  it, case-insensitively. Each row carries its origin, lock level,
  `valid_to`, posterior mean `alpha / (alpha + beta)`, and its
  corroboration count from `belief_corroborations`. This is how you
  check whether repeated restatements of one instruction ever bound.
* `family_lock_history` — `feedback_history` rows with source
  `lock:unlock` or `lock:expire` for the family's beliefs. A family
  member that is unlocked today may have been locked and then unlocked
  on purpose; this is where that shows.

## Why more than one store

Stores are per repository: the store path resolves through the git
common dir. A lock requested in a session in one repository lands in
that repository's store and nowhere else. Read against one store, such
a lock looks lost. Pass every store with `--store` and the `cross_store`
section lists the family across all of them, with the stores where a
member is locked and active.

Usage:

    uv run python -m benchmarks.lock_loss_census_1620 \\
        --store <path> [--store <path> ...] [--pattern REGEX]

Every `--store` is opened **read-only** through `sqlite3` with a
`mode=ro` URI, never through `MemoryStore`: opening a store runs
migrations and a lifecycle sweep, which is a write. Content is never
printed beyond its first 80 characters. Output is JSON with sorted keys
and every list in a fixed order, so two runs over the same stores print
the same bytes.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from typing import Final, cast

from aelfrice.models import FEEDBACK_SOURCE_LOCK_EXPIRE, LOCK_USER, ORIGIN_SPECULATIVE

CONTENT_PREVIEW_CHARS: Final[int] = 80
COMMAND_PREFIXES: Final[tuple[str, ...]] = ("/aelf:", "aelf ", "uv run aelf")
SELF_INGESTION_MARKER: Final[str] = "<belief id="
LOCK_HISTORY_SOURCES: Final[tuple[str, ...]] = (
    "lock:unlock",
    FEEDBACK_SOURCE_LOCK_EXPIRE,
)

Row = dict[str, object]


def is_aelf_command(content: str) -> bool:
    return content.lstrip().startswith(COMMAND_PREFIXES)


def is_self_ingestion(content: str) -> bool:
    return SELF_INGESTION_MARKER in content


def _preview(content: str) -> str:
    return content[:CONTENT_PREVIEW_CHARS]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _posterior_mean(alpha: float, beta: float) -> float | None:
    total = alpha + beta
    return round(alpha / total, 6) if total > 0 else None


def _sort_key(row: Row) -> tuple[str, str]:
    return (str(row["created_at"]), str(row["id"]))


def measure(store_path: str, pattern: re.Pattern[str] | None) -> dict[str, object]:
    conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = 1")
        cols = _columns(conn, "beliefs")
        valid_to = "valid_to" if "valid_to" in cols else "NULL"
        origin = "origin" if "origin" in cols else "'unknown'"
        beliefs = conn.execute(
            f"SELECT id, content, alpha, beta, lock_level, created_at, "
            f"{origin}, {valid_to} FROM beliefs ORDER BY created_at, id"
        ).fetchall()
        corroborations: Counter[str] = Counter()
        if _columns(conn, "belief_corroborations"):
            for bid, n in conn.execute(
                "SELECT belief_id, COUNT(*) FROM belief_corroborations "
                "GROUP BY belief_id"
            ):
                corroborations[str(bid)] = int(n)
        history = []
        if _columns(conn, "feedback_history"):
            marks = ",".join("?" * len(LOCK_HISTORY_SOURCES))
            history = conn.execute(
                f"SELECT belief_id, source, valence, created_at "
                f"FROM feedback_history WHERE source IN ({marks}) "
                f"ORDER BY created_at, id",
                LOCK_HISTORY_SOURCES,
            ).fetchall()
    finally:
        conn.close()

    level_active: Counter[str] = Counter()
    level_all: Counter[str] = Counter()
    commands: list[Row] = []
    self_ingested: list[Row] = []
    speculative: list[Row] = []
    family: list[Row] = []
    for bid, content, alpha, beta, lock, created, orig, vto in beliefs:
        text = str(content)
        active = vto is None
        level_all[str(lock)] += 1
        if active:
            level_active[str(lock)] += 1
        row: Row = {
            "id": str(bid),
            "origin": str(orig),
            "lock_level": str(lock),
            "created_at": str(created),
            "valid_to": None if vto is None else str(vto),
            "content_prefix": _preview(text),
        }
        if is_aelf_command(text):
            commands.append(row)
        if is_self_ingestion(text):
            self_ingested.append(row)
        if active and str(orig) == ORIGIN_SPECULATIVE:
            speculative.append(row)
        if pattern is not None and pattern.search(text):
            family.append({
                **row,
                "posterior_mean": _posterior_mean(float(alpha), float(beta)),
                "corroborations": corroborations[str(bid)],
            })

    family_ids = {str(r["id"]) for r in family}
    lock_history: list[Row] = [
        {
            "belief_id": str(bid),
            "source": str(src),
            "valence": float(val),
            "created_at": str(ts),
        }
        for bid, src, val, ts in history
        if str(bid) in family_ids
    ]

    def _counted(rows: list[Row]) -> dict[str, object]:
        rows = sorted(rows, key=_sort_key)
        return {
            "count": len(rows),
            "active": sum(1 for r in rows if r["valid_to"] is None),
            "rows": rows,
        }

    result: dict[str, object] = {
        "beliefs_active": sum(level_active.values()),
        "beliefs_all": sum(level_all.values()),
        "lock_level": {
            "active": dict(sorted(level_active.items())),
            "all": dict(sorted(level_all.items())),
        },
        "aelf_commands": _counted(commands),
        "self_ingestion": _counted(self_ingested),
        "speculative_active": _counted(speculative),
    }
    if pattern is not None:
        result["family"] = _counted(family)
        result["family_lock_history"] = lock_history
    return result


def cross_store(per_store: dict[str, dict[str, object]]) -> dict[str, object]:
    """Merge the pattern family across stores, tagging each row with its store."""
    rows: list[Row] = []
    for store, result in per_store.items():
        fam = result.get("family")
        if not isinstance(fam, dict):
            continue
        family_rows = cast("list[Row]", cast("dict[str, object]", fam)["rows"])
        for row in family_rows:
            rows.append({**row, "store": store})
    rows.sort(key=lambda r: (str(r["created_at"]), str(r["store"]), str(r["id"])))
    locked = sorted({
        str(r["store"]) for r in rows
        if r["lock_level"] == LOCK_USER and r["valid_to"] is None
    })
    return {
        "family_count": len(rows),
        "stores_with_active_lock": locked,
        "rows": rows,
    }


def run(stores: list[str], pattern_text: str | None) -> dict[str, object]:
    pattern = (
        re.compile(pattern_text, re.IGNORECASE) if pattern_text is not None else None
    )
    per_store = {path: measure(path, pattern) for path in stores}
    out: dict[str, object] = {
        "pattern": pattern_text,
        "stores": per_store,
    }
    if pattern is not None:
        out["cross_store"] = cross_store(per_store)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#1620 lock-loss census")
    ap.add_argument(
        "--store", required=True, action="append",
        help="path to a memory.db; repeat for more than one store",
    )
    ap.add_argument(
        "--pattern", default=None,
        help="case-insensitive regex selecting one instruction's family",
    )
    args = ap.parse_args(argv)
    stores = list(dict.fromkeys(args.store))
    print(json.dumps(run(stores, args.pattern), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
