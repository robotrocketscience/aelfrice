#!/usr/bin/env python3
"""Compute the current replay-soak green-streak (#403 C).

Reads the append-only JSONL status file `.replay-soak-status.json` and
prints the count of consecutive rows from the tail where
`mismatched + derived_orphan == 0` (i.e. `replay_full_equality_result
== "pass"`).

Used by `.github/workflows/replay-soak-gate.yml` to gate
`#264`-touching merges. The required-check name produced is
`replay-soak / consecutive-green ≥ 7d`. Per the 2026-05-04 ratification
on #403, ≥7 is the threshold; a streak ≥ 7 → exit 0; otherwise exit 1.

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
    n = 0
    for row in reversed(rows):
        if row.get("replay_full_equality_result") != "pass":
            break
        if int(row.get("mismatched", 0)) + int(row.get("derived_orphan", 0)) != 0:
            break
        n += 1
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
        help="Minimum consecutive green rows required (default 7).",
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
        print(f"replay-soak streak: {n} consecutive green (threshold ≥ {args.threshold})")

    return 0 if n >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
