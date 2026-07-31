#!/usr/bin/env python3
"""Compute the current replay-soak green-streak (#403 C).

Reads the append-only JSONL status file `.replay-soak-status.json` and
prints the count of consecutive green rows from the tail — green meaning
`replay_full_equality_result == "pass"` and `mismatched + derived_orphan
== 0` — counting **one per distinct commit**, so that repeated entries for
an unchanged `main` do not accumulate a streak (#1239).

Used by `.github/workflows/replay-soak-gate.yml` to gate
`#264`-touching merges. The check name produced is
`consecutive-green ≥ 7 commits`. Per the 2026-05-04
ratification on #403, ≥7 is the threshold; a streak ≥ 7 → exit 0;
otherwise exit 1.

Exit codes:
  0  streak ≥ threshold (default 7) — PR may merge w.r.t. the soak gate
  1  streak < threshold — PR blocked
  2  malformed status file (any JSONL parse error or missing field)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def streak(rows: list[dict]) -> int:  # type: ignore[type-arg]
    """Count consecutive green rows from the tail, one per distinct commit.

    Consecutive rows recording the same `sha` collapse to one. The soak is
    deterministic — the same tree replayed against the same corpus yields the
    same result — so a second entry for an unchanged `main` repeats a
    measurement rather than adding one. Counting rows instead of commits let
    an idle week manufacture the threshold: `main` did not advance between
    2026-07-22 and 2026-07-29, and the cron recorded `018eb88a` on seven
    consecutive days, which satisfied "7 consecutive green" on its own (#1239).

    A row whose `sha` is absent — missing, `null`, or empty — counts as its
    own measurement rather than collapsing into its neighbour, because absent
    provenance is not evidence of sameness. An empty string is as much a
    non-answer as a missing key, so both take that branch. The cron has always
    written a real `sha`, so this only affects hand-edited or pre-schema rows.
    """
    n = 0
    prev_sha: str | None = None
    for row in reversed(rows):
        if row.get("replay_full_equality_result") != "pass":
            break
        if int(row.get("mismatched", 0)) + int(row.get("derived_orphan", 0)) != 0:
            break
        sha = row.get("sha")
        if not sha or sha != prev_sha:
            n += 1
        prev_sha = sha
    return n


def load_rows(path: Path) -> list[dict]:  # type: ignore[type-arg]
    rows: list[dict] = []  # type: ignore[type-arg]
    if not path.exists():
        return rows
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{lineno}: invalid JSON — {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{lineno}: row must be a JSON object")
            rows.append(obj)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path(".replay-soak-status.json"),
        help="Path to the append-only JSONL status file.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help="Minimum consecutive green commits required (default 7).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print the streak number only; suppress prose.",
    )
    args = parser.parse_args()

    try:
        rows = load_rows(args.status_file)
    except SystemExit:
        # SystemExit raised by load_rows already prints the reason.
        return 2

    n = streak(rows)
    if args.quiet:
        print(n)
    else:
        print(
            f"replay-soak streak: {n} consecutive green commits "
            f"(threshold ≥ {args.threshold})"
        )

    return 0 if n >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
