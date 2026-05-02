#!/usr/bin/env python3
"""Summarise a per-session rebuild_log JSONL.

Phase-1c companion to the #288 phase-1a rebuild diagnostic log. Reads
one or more `<session_id>.jsonl` files and prints:

    * total rebuild invocations
    * pack rate (n_packed / n_candidates) distribution: mean, p50, p90
    * drop-reason histogram (e.g. ``below_floor:0.40``,
      ``content_hash_collision_with:<id>``, ``budget``)
    * for packed rows, the rank distribution (1, 2, 3, ...) — surfaces
      whether the rebuilder typically packs the top candidate or
      something further down
    * count of truncated session-files

Usage:

    python scripts/audit_rebuild_log.py <path-to-jsonl-or-dir> [...]

Exit codes:
    0   summary printed
    1   no readable input
    2   usage error

Reads only — never modifies the log files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def _iter_records(path: Path) -> Iterable[dict]:
    """Yield decoded JSON records from a JSONL file. Skips malformed
    lines (the writer is fail-soft so partial lines on a crash are
    expected)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"audit_rebuild_log: cannot read {path}: {exc}",
            file=sys.stderr,
        )
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _collect_paths(targets: list[Path]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        if t.is_dir():
            out.extend(sorted(t.glob("*.jsonl")))
        elif t.is_file():
            out.append(t)
        else:
            print(
                f"audit_rebuild_log: skipping {t}: not a file or dir",
                file=sys.stderr,
            )
    return out


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _summarise(records: list[dict]) -> dict:
    n_records = 0
    n_truncated = 0
    pack_rates: list[float] = []
    drop_reasons: Counter[str] = Counter()
    packed_ranks: Counter[int] = Counter()
    n_no_pack_summary = 0

    for r in records:
        if r.get("truncated") is True:
            n_truncated += 1
            continue
        n_records += 1
        pack = r.get("pack_summary")
        if isinstance(pack, dict):
            n_cand = pack.get("n_candidates", 0) or 0
            n_packed = pack.get("n_packed", 0) or 0
            if n_cand > 0:
                pack_rates.append(n_packed / n_cand)
        else:
            n_no_pack_summary += 1
        candidates = r.get("candidates")
        if not isinstance(candidates, list):
            continue
        for c in candidates:
            if not isinstance(c, dict):
                continue
            decision = c.get("decision")
            if decision == "dropped":
                reason = c.get("reason") or "<unspecified>"
                drop_reasons[str(reason)] += 1
            elif decision == "packed":
                rank = c.get("rank")
                if isinstance(rank, int):
                    packed_ranks[rank] += 1

    return {
        "n_records": n_records,
        "n_truncated": n_truncated,
        "n_no_pack_summary": n_no_pack_summary,
        "pack_rate_mean": (
            sum(pack_rates) / len(pack_rates) if pack_rates else 0.0
        ),
        "pack_rate_p50": _percentile(pack_rates, 50),
        "pack_rate_p90": _percentile(pack_rates, 90),
        "drop_reasons": drop_reasons,
        "packed_ranks": packed_ranks,
    }


def _bucketise_drop_reasons(reasons: Counter[str]) -> Counter[str]:
    """Group reasons by their semantic prefix.

    A reason looks like ``below_floor:0.40`` or
    ``content_hash_collision_with:abc123``. The float / id tail varies
    per call, so the histogram is dominated by uniques unless we
    bucket on the part before ``:``. Reasons without a ``:`` are kept
    as-is.
    """
    out: Counter[str] = Counter()
    for reason, n in reasons.items():
        head = reason.split(":", 1)[0] if ":" in reason else reason
        out[head] += n
    return out


def _print_report(summary: dict, *, paths_read: int) -> None:
    print(f"rebuild_log audit — {paths_read} file(s)")
    print(f"  records:       {summary['n_records']}")
    if summary["n_truncated"]:
        print(
            f"  truncated:     {summary['n_truncated']} "
            "(session(s) hit the 5 MB cap)"
        )
    if summary["n_no_pack_summary"]:
        print(
            f"  malformed:     {summary['n_no_pack_summary']} "
            "(record had no pack_summary)"
        )
    print()
    print("pack rate (n_packed / n_candidates):")
    print(f"  mean: {summary['pack_rate_mean']:.3f}")
    print(f"  p50:  {summary['pack_rate_p50']:.3f}")
    print(f"  p90:  {summary['pack_rate_p90']:.3f}")
    print()
    print("drop-reason histogram (bucketed by prefix):")
    bucketed = _bucketise_drop_reasons(summary["drop_reasons"])
    if not bucketed:
        print("  (none)")
    else:
        for reason, n in bucketed.most_common():
            print(f"  {n:>6}  {reason}")
    print()
    print("packed-rank distribution (where in the candidate list "
          "the packed row was):")
    if not summary["packed_ranks"]:
        print("  (none)")
    else:
        for rank in sorted(summary["packed_ranks"]):
            print(f"  rank {rank:>2}: {summary['packed_ranks'][rank]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise rebuild_log JSONL files (phase-1c for #288). "
            "Reads only; never modifies the input."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=(
            "One or more JSONL files, or directories containing "
            "*.jsonl session-log files."
        ),
    )
    args = parser.parse_args(argv)

    paths = _collect_paths(args.paths)
    if not paths:
        print(
            "audit_rebuild_log: no readable JSONL inputs",
            file=sys.stderr,
        )
        return 1

    records: list[dict] = []
    for p in paths:
        records.extend(_iter_records(p))

    summary = _summarise(records)
    _print_report(summary, paths_read=len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
