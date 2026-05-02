"""Command-line entry point for the posterior_ranking benchmark.

Run with `python -m benchmarks.posterior_ranking [flags]` from a source
checkout. Argument surface mirrors the prior `aelf bench
posterior-residual` subcommand removed in #342.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from benchmarks.posterior_ranking import run as _pr_run


def _default_fixtures() -> Path:
    return (
        Path(__file__).parent / "fixtures" / "default.jsonl"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.posterior_ranking",
    )
    parser.add_argument("--fixtures", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mrr-threshold", type=float, default=0.05)
    parser.add_argument("--ece-threshold", type=float, default=0.10)
    parser.add_argument("--json", dest="json_out", action="store_true")
    parser.add_argument(
        "--heat-kernel",
        action="store_true",
        help="enable heat-kernel composition in retrieve() (slice 2 of #151)",
    )
    args = parser.parse_args(argv)

    fixtures_path = Path(args.fixtures) if args.fixtures else _default_fixtures()
    result = _pr_run.run(
        fixtures_path,
        n_seeds=args.seeds,
        mrr_threshold=args.mrr_threshold,
        ece_threshold=args.ece_threshold,
        heat_kernel=args.heat_kernel,
    )

    if args.json_out:
        print(json.dumps({
            "mrr": asdict(result["mrr"]),
            "ece": asdict(result["ece"]),
            "overall_pass": result["overall_pass"],
        }, indent=2))
    else:
        mrr = result["mrr"]
        ece = result["ece"]
        print(
            f"posterior-residual eval\n"
            f"  MRR uplift:  {mrr.mean_uplift:+.4f}  "
            f"(±2σ: [{mrr.uplift_lo:+.4f}, {mrr.uplift_hi:+.4f}])  "
            f"threshold={mrr.pass_threshold:+.2f}  "
            f"{'PASS' if mrr.passed else 'FAIL'}\n"
            f"  ECE:         {ece.ece:.4f}  "
            f"threshold={ece.pass_threshold:.2f}  "
            f"n={ece.n_total}  "
            f"{'PASS' if ece.passed else 'FAIL'}\n"
            f"  overall:     {'PASS' if result['overall_pass'] else 'FAIL'}"
        )

    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
