"""#1407 — the BM25 sidecar rebuild rate, counted rather than inferred.

#1380's cost case is `cold_cost x cold_rate`. After the 2026-08-06
re-derivation `cold_cost` is known (2.89 s cold first-fire at 44,668 beliefs)
and `cold_rate` was not: the only estimate was a latency proxy (a fire is
"cold" if `latency_ms >= 1000`) that yielded 8.5% but cannot attribute any
individual slow fire to a rebuild rather than to SQLite lock contention, a cold
page cache, or an unrelated stall. The operator ruling of 2026-08-06 ~18:00Z is
that #1380 must not be decided on the proxy.

This reads the `sidecar_outcome` field that #1407 added and reports the direct
counts.

## Three states, not a boolean

`fresh` / `incremental` / `full_rebuild`. Collapsing the middle one is what made
#1199's 86.2% ("sidecar not fresh") and the 8.5% proxy ("fire was slow") look
contradictory when they measure different events: since #1199 shipped the
incremental path, a stale sidecar no longer implies a full rebuild.

**`full_rebuild` is the rate #1380 is priced on.** `incremental` is cheap.

## Rows predating the field

A row with no `sidecar_outcome` key is **excluded and counted**, never treated
as `fresh` — every fire logged before #1407 shipped lacks the key, and folding
those into the denominator as cache hits would drive the measured rebuild rate
toward zero purely as a function of how long the log has existed. The excluded
count is printed on every run; if it dominates, the answer is "not enough data
yet", not a low rate.

Usage:
    uv run python benchmarks/sidecar_rebuild_rate.py [AUDIT_LOG ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

OUTCOMES = ("fresh", "incremental", "full_rebuild")


def _default_logs() -> list[Path]:
    from aelfrice.db_paths import _git_common_dir

    git_dir = _git_common_dir()
    if git_dir is None:
        raise SystemExit("not in a git work-tree; pass audit log paths")
    d = git_dir / "aelfrice"
    return sorted(d.glob("hook_audit.jsonl*"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    args = ap.parse_args()

    logs = args.logs or _default_logs()
    logs = [p for p in logs if p.is_file()]
    if not logs:
        print("no audit logs found", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    missing = 0
    non_ups = 0

    for path in logs:
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("hook") != "user_prompt_submit":
                non_ups += 1
                continue
            outcome = rec.get("sidecar_outcome")
            if outcome is None:
                # Pre-#1407 row, or a fire that did no index work. Both are
                # "not measured" and must stay out of the denominator.
                missing += 1
                continue
            if outcome not in OUTCOMES:
                unknown[str(outcome)] += 1
                continue
            counts[str(outcome)] += 1

    scored = sum(counts.values())

    print("#1407 — BM25 sidecar outcome per user_prompt_submit fire")
    for p in logs:
        print(f"  log                            {p}")
    print(f"  fires with an outcome (scored) {scored}")
    print(f"  fires with no outcome key      {missing}   <- excluded, NOT 'fresh'")
    if unknown:
        print(f"  unrecognised outcome values    {dict(unknown)}  <- vocabulary drift")
    print()

    if scored == 0:
        print("  NO MEASUREMENT YET. Every user_prompt_submit row predates the")
        print("  sidecar_outcome field (or did no index work). Let the log")
        print("  accumulate before pricing #1380 — do not read this as a low")
        print("  rebuild rate.")
        return 0

    for name in OUTCOMES:
        n = counts[name]
        print(f"  {name:<14} {n:>6}  {n / scored:>7.2%}")
    print()
    rebuild_rate = counts["full_rebuild"] / scored
    print(f"  FULL-REBUILD RATE  {counts['full_rebuild']}/{scored} = {rebuild_rate:.2%}")
    print("  (this is the cold_rate term in #1380's cold_cost x cold_rate)")

    if missing > scored:
        print()
        print(f"  CAUTION: {missing} excluded rows against {scored} scored. The")
        print("  sample is dominated by pre-field fires; treat the rate as")
        print("  provisional until the scored count is the larger of the two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
