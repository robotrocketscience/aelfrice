#!/usr/bin/env python3
"""Run a figure producer N times and fail unless every run is byte-identical.

## Why this exists

`scripts/check_derived_figures.py` runs each store-free producer **once** per
CI job and compares its output against the prose. That gate cannot tell a
producer that is wrong from a producer that is unstable: a run whose figures
move between invocations would red the gate on one draw and pass it on the
next, and the published number would be whichever draw the author happened to
take. A review of #1547 reported exactly that shape -- a published cell
reading one set of values on a clean tree and another set later.

So the property this script holds down is not "the figures are correct" but
"the producer is a function of the tree". It runs the producer `--runs` times,
hashes stdout each time, and exits non-zero unless every hash agrees, naming
the keys that differ when they do not.

## Usage

    uv run python scripts/soak_producer_figures.py \\
        benchmarks/injection_budget_bytes.py --runs 25 --jobs 4

    uv run python scripts/soak_producer_figures.py \\
        benchmarks/injection_budget_bytes.py --runs 25 --in-process

    uv run python scripts/soak_producer_figures.py --dry-run   # plan only

`--jobs` runs that many producers at once. Concurrency is part of the test:
the shapes that move between runs -- a wall clock read, a subprocess timeout,
a temp path collision -- are the shapes a loaded machine exposes and an idle
one hides.

Every child runs under the same environment policy the gate uses
(`check_derived_figures.producer_env`), which is what keeps a soak from
passing on bytecode the tree no longer contains. The policy is imported rather
than restated so the two cannot drift.

## Two populations, and the default one cannot see the other

The default arm starts each run from a cold interpreter, which is what
`check_derived_figures.py` does and therefore what the gate's own stability
depends on. It also means every module-level cache in the producer and in the
package under it is empty at the start of every run, so a figure that moves
because an earlier call in the *same* process left state behind -- a memoised
resolver, a lazily-built index, a module global a caller rebound -- is
invisible to it by construction. A review of #1559 reported a committed test
failing on run 4 of 11 while a 28-run subprocess soak reported 28 of 28
identical, which is exactly that shape of report.

`--in-process` is the other population: the producer is imported once and
`figures()` is called `--runs` times in one interpreter, each result hashed and
required to agree. The producer must expose `figures()`; the default arm only
needs `--emit-figures` on a command line. `--jobs` is refused here, because
`figures()` chdirs and two of them in one process would be measuring each
other.

It is still not the whole population -- a pytest session that has already run
other tests carries state this script never creates -- and that is stated
rather than papered over. What it does close is the half that does not need
pytest to reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = REPO_ROOT / "scripts" / "check_derived_figures.py"
DEFAULT_PRODUCER = "benchmarks/injection_budget_bytes.py"

# A producer that publishes figures is re-run by the gate on every PR, so a
# soak that is cheaper than the gate is not measuring the thing that ships.
DEFAULT_RUNS = 25
RUN_TIMEOUT_S = 600


def _gate_module() -> Any:
    """`scripts/check_derived_figures.py`, loaded for its env policy."""
    spec = importlib.util.spec_from_file_location("_cdf_soak", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_once(producer: Path, env: dict[str, str]) -> tuple[str, str]:
    """One producer run: (sha256 of stdout, stdout)."""
    proc = subprocess.run(
        [sys.executable, str(producer), "--emit-figures"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"{producer} exited {proc.returncode}: {proc.stderr.strip()[:2000]}"
        )
    return hashlib.sha256(proc.stdout.encode()).hexdigest(), proc.stdout


def _load_producer(producer: Path) -> Any:
    """Import the producer by path, for the in-process arm.

    By path and not by package name: `benchmarks/` is not on the install path,
    which is how every other consumer of these modules reaches them.
    """
    spec = importlib.util.spec_from_file_location("_soak_producer", producer)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{producer} is not importable as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_once_in_process(module: Any) -> tuple[str, str]:
    """One in-process run: (sha256 of the serialised figures, that blob).

    `sort_keys` so a dict whose insertion order moved but whose content did not
    is not reported as a divergence -- the property is the figures, not the
    order a dict happens to carry them in. `default=str` so a producer that
    emits something JSON does not know about fails on the comparison rather
    than on the serialiser.
    """
    figures = getattr(module, "figures", None)
    if figures is None:
        raise SystemExit(
            f"{module.__name__} exposes no figures(); --in-process needs one "
            "(the default arm runs the producer's --emit-figures instead)"
        )
    blob = json.dumps(figures(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest(), blob


def _differing_keys(first: str, other: str) -> list[str]:
    """Top-level keys whose values differ, for the failure message."""
    try:
        a, b = json.loads(first), json.loads(other)
    except json.JSONDecodeError:
        return ["<stdout is not JSON>"]
    if not isinstance(a, dict) or not isinstance(b, dict):
        return ["<stdout is not a JSON object>"]
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", nargs="?", default=DEFAULT_PRODUCER)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "import the producer once and call figures() --runs times in this "
            "interpreter, instead of running it as a subprocess per run"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit 0 without running the producer",
    )
    args = parser.parse_args(argv)

    producer = (REPO_ROOT / args.producer).resolve()
    if not producer.is_file():
        print(f"no such producer: {args.producer}", file=sys.stderr)
        return 2
    if args.runs < 2 or args.jobs < 1:
        print("--runs must be at least 2 and --jobs at least 1", file=sys.stderr)
        return 2
    if args.in_process and args.jobs != 1:
        print(
            "--in-process runs one at a time: figures() chdirs, so two of them "
            "in one interpreter would be measuring each other",
            file=sys.stderr,
        )
        return 2

    where = "in this interpreter" if args.in_process else f"{args.jobs} at a time"
    plan = (
        f"{args.runs} runs of {args.producer}, {where}, "
        "each hashed and required to agree"
    )
    if args.dry_run:
        print(f"dry run: would perform {plan}")
        return 0

    print(f"soaking: {plan}", flush=True)
    if args.in_process:
        module = _load_producer(producer)
        results = [_run_once_in_process(module) for _ in range(args.runs)]
    else:
        with tempfile.TemporaryDirectory(prefix="soak-figures-pyc-") as pyc:
            env = _gate_module().producer_env(pyc)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results = list(
                    pool.map(lambda _: _run_once(producer, env), range(args.runs))
                )

    digests = [digest for digest, _ in results]
    first = digests[0]
    divergent = [i for i, digest in enumerate(digests) if digest != first]
    if divergent:
        print(
            f"FAIL: {len(divergent)} of {args.runs} runs diverged from run 0 "
            f"({first[:16]})",
            file=sys.stderr,
        )
        for i in divergent[:3]:
            keys = _differing_keys(results[0][1], results[i][1])
            print(
                f"  run {i} ({digests[i][:16]}) differs in: "
                f"{', '.join(keys[:12]) or '<whitespace only>'}",
                file=sys.stderr,
            )
        return 1
    print(f"OK: {args.runs}/{args.runs} runs identical, sha256 {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
