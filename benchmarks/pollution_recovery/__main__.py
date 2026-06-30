"""CLI for the pollution-recovery benchmark (#1011 doc-chunk signal/noise).

    python -m benchmarks.pollution_recovery [--fixtures PATH] [-k K] [--json]

Reports, per retrieval regime, whether user-stated facts survive a store
flooded with keyword-overlapping document chunks. Pass
`--origin-tier-rerank` to score a candidate fix instead of the baseline
(no-op unless `retrieve()` grows that kwarg).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.pollution_recovery.run import (
    evaluate,
    load_cases,
    regime_recall,
)


def _default_fixtures() -> Path:
    return Path(__file__).parent / "fixtures" / "default.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.pollution_recovery",
    )
    parser.add_argument("--fixtures", default=None)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--origin-tier-rerank",
        action="store_true",
        help="score with the (candidate) origin-tier rerank lane on",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    fixtures = Path(args.fixtures) if args.fixtures else _default_fixtures()
    cases = load_cases(fixtures)
    report = evaluate(
        cases, k=args.k, use_origin_tier_rerank=args.origin_tier_rerank
    )
    by_regime = regime_recall(report)

    if args.json:
        print(json.dumps({"report": asdict(report), "regime_recall": by_regime}))
        return 0

    print(f"pollution-recovery  (n={report.n_cases}, k={report.k})")
    print(f"  fact_recall@{report.k} : {report.fact_recall_at_k:.2f}")
    print(f"  mean_fact_rank   : {report.mean_fact_rank:.1f}")
    print(f"  chunk_share@{report.k} : {report.chunk_share_at_k:.2f}")
    print("  per-regime recall:")
    for regime, rec in by_regime.items():
        print(f"    {regime:8} {rec:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
