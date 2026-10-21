"""Run the nightly canonical bench and decide the job's result (audit CI-01).

Usage:
  python3 scripts/bench_canonical_step.py run --out PATH [--dry-run] -- CMD [ARG...]
  python3 scripts/bench_canonical_step.py verdict --rc RC --bandcheck OUTCOME [--dry-run]

`run` writes `out=PATH` to `$GITHUB_OUTPUT` before CMD starts, runs CMD,
then writes `rc=N`. It exits 0 when CMD exited 0 or 2 and left a
non-empty file at PATH, and 1 otherwise. `--dry-run` prints the command
and the output lines without running or writing anything.

`verdict` exits 0 only when RC is 0 or 2 and OUTCOME, the band-check
step's `outcome`, is `success`. Anything else exits 1 with one
`::error::` line per reason. `--dry-run` prints the reasons and exits 0.

`aelf bench all` exits 2 when an adapter's dataset is missing
(`benchmarks/run.py`, `main_all`). That run still finished and wrote its
JSON, and `benchmarks/tolerance.py` reads the missing adapters as SKIP.
The workflow used to run the bench under `set -euo pipefail` and write
`out` after it, so exit 2 ended the step before `out` existed. Every
step gated on `out` was skipped, including the band check, and the job
reported success on 40 consecutive nights without comparing a band.
Writing `out` first and recording the exit code removes that path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 0: every adapter ran. 2: at least one adapter's data was missing and
# none errored; the JSON holds the rest. Any other code is an error.
FINISHED_RCS: frozenset[int] = frozenset({0, 2})


def _emit(lines: list[str], *, dry_run: bool) -> None:
    """Append `key=value` lines to `$GITHUB_OUTPUT`, or print them."""
    target = os.environ.get("GITHUB_OUTPUT", "")
    if dry_run or not target:
        for line in lines:
            print(f"[output] {line}")
        return
    with open(target, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def run_step(out: Path, cmd: list[str], *, dry_run: bool = False) -> int:
    """Record `out`, run `cmd`, record its exit code, and judge the run."""
    if not cmd:
        print("::error::bench_canonical_step run: no command given after --", file=sys.stderr)
        return 1
    _emit([f"out={out}"], dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        _emit(["rc=<exit code>"], dry_run=True)
        return 0
    rc = subprocess.run(cmd, check=False).returncode
    _emit([f"rc={rc}"], dry_run=False)
    if rc not in FINISHED_RCS:
        print(f"::error::bench command exited {rc}; only 0 and 2 mean the run finished")
        return 1
    if not out.is_file() or out.stat().st_size == 0:
        print(f"::error::bench command exited {rc} but wrote no JSON at {out}")
        return 1
    if rc == 2:
        print("::notice::bench finished with at least one adapter skipped for missing data")
    return 0


def verdict_reasons(rc: str, bandcheck: str) -> list[str]:
    """Every reason the job must fail. Empty means the job passes."""
    reasons: list[str] = []
    try:
        rc_int: int | None = int(rc)
    except ValueError:
        rc_int = None
    if rc_int is None:
        reasons.append(f"bench exit code was not recorded (rc={rc!r})")
    elif rc_int not in FINISHED_RCS:
        reasons.append(f"bench exited {rc_int}; only 0 and 2 mean the run finished")
    if bandcheck == "failure":
        reasons.append(
            "band-check failed: a metric is out of band or nothing was measured "
            "vs benchmarks/results/v2.0.0.json"
        )
    elif bandcheck != "success":
        reasons.append(f"band-check did not run (outcome={bandcheck!r})")
    return reasons


def verdict_step(rc: str, bandcheck: str, *, dry_run: bool = False) -> int:
    reasons = verdict_reasons(rc, bandcheck)
    for reason in reasons:
        print(f"::error::{reason}")
    if not reasons:
        print(f"bench-canonical ok (rc={rc}, band-check={bandcheck})")
        return 0
    if dry_run:
        print("[dry-run] the job would fail")
        return 0
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_canonical_step.py",
        description="Run the nightly canonical bench and decide the job's result.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="record out, run the bench, record rc")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("cmd", nargs=argparse.REMAINDER)
    verdict = sub.add_parser("verdict", help="fail on a bad rc or a band-check that did not pass")
    verdict.add_argument("--rc", required=True)
    verdict.add_argument("--bandcheck", required=True)
    verdict.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        return run_step(args.out, cmd, dry_run=args.dry_run)
    return verdict_step(args.rc, args.bandcheck, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
