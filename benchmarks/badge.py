"""Compute the README reproducibility-badge text from a canonical bench JSON.

The nightly `bench-canonical` cron writes a merged report under
`benchmarks/results/v2.0.0-cron-<date>.json`. This module reads that
report and produces the one-line badge text that the workflow splices
between the `<!-- bench-canonical-badge:start -->` and
`<!-- bench-canonical-badge:end -->` markers in README.md.

Issue: #477.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def _count_invocations(headline_cut: Mapping[str, list]) -> int:
    return sum(len(v) for v in headline_cut.values())


def _count_ok(results: Mapping[str, Mapping[str, Mapping]]) -> int:
    ok = 0
    for by_sub in results.values():
        for entry in by_sub.values():
            if isinstance(entry, Mapping) and entry.get("_status") == "ok":
                ok += 1
    return ok


def compute_badge_text(report_path: Path, *, today: str | None = None) -> str:
    """Render the badge line from the report at *report_path*.

    `today` defaults to UTC `YYYY-MM-DD`; the parameter exists so tests
    can pin a date.
    """
    data = json.loads(Path(report_path).read_text())
    total = _count_invocations(data["headline_cut"])
    ok = _count_ok(data["results"])
    if today is None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    icon = "✅" if ok == total and total > 0 else "⚠️"
    return f"reproducibility: {icon} {ok}/{total} ok · last run {today}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="path to merged canonical JSON")
    parser.add_argument("--today", default=None, help="override UTC date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    sys.stdout.write(compute_badge_text(args.report, today=args.today) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
