#!/usr/bin/env python3
"""Fail when a file's pyright error count rises above its baseline (#1503).

`RELEASING.md` step 6 told the releaser to run `uv run pyright src/` with the
comment `# strict`, implying it passes. It emitted 991 errors, and **no
workflow anywhere ran pyright**, so the tick was self-reported and unenforced.
Three other documents repeated the claim. The cost is on the record:
`CHANGELOG/v4.md` notes a `NameError` that reached `main` behind it.

The operator ruling of 2026-08-19 is to drive the count to zero and gate it in
CI. That is multi-session work — 991 errors over 77 files — so the ratchet
lands first. Without it, a burn-down is a leaky bucket: nothing stops a new
error arriving in `cli.py` while someone is clearing `store.py`.

## Per file, not a single total

A repo-wide total lets a fix in one module pay for a regression in another,
and the two are unrelated changes by unrelated authors. The baseline is a
mapping of file to count, and the check fails when **any** file rises. A file
that improves is a green diff; the baseline is then updated downward, which is
the ratchet turning.

## Scope: `src/`, and the wider surface is stated rather than implied

This gates `pyright src/` — the command the documents name, over the code that
ships. `pyproject.toml` sets `include = ["src/aelfrice", "tests"]`, and honouring
that raises the count to 6,938 over 462 files, because the test suite is
almost entirely unannotated. Gating the wider surface today would freeze a
number seven times larger and make every test edit fight the ratchet.

So `tests/` is deliberately **not** gated, and this docstring says so rather
than letting `pyright` in a CI job name imply full coverage. Extending the
ratchet to `tests/` is a separate decision with its own cost.

Usage:
    python3 scripts/check_pyright_baseline.py            # check
    python3 scripts/check_pyright_baseline.py --update   # rewrite the baseline
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = REPO / "pyright_baseline.json"
TARGET = "src/"


def measure() -> dict[str, int]:
    """Per-file error counts from a pyright run over `TARGET`."""
    proc = subprocess.run(
        ["uv", "run", "pyright", TARGET, "--outputjson"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=1800,
    )
    # pyright exits non-zero whenever it reports an error, which is the normal
    # state here, so the exit code says nothing. Parse failure is the real
    # error, and it must not read as "zero errors".
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        sys.stderr.write(
            "could not parse pyright --outputjson. stdout head:\n"
            f"{proc.stdout[:2000]}\nstderr head:\n{proc.stderr[:2000]}\n"
        )
        raise SystemExit(2) from None

    counts: Counter[str] = Counter()
    for d in payload.get("generalDiagnostics", []):
        if d.get("severity") != "error":
            continue
        try:
            rel = pathlib.Path(d["file"]).resolve().relative_to(REPO).as_posix()
        except (KeyError, ValueError):
            rel = str(d.get("file", "<unknown>"))
        counts[rel] += 1
    return dict(counts)


def load_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        sys.stderr.write(f"no baseline at {BASELINE}; run with --update\n")
        raise SystemExit(2)
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data.get("files", {}).items()}


def write_baseline(counts: dict[str, int]) -> None:
    BASELINE.write_text(
        json.dumps(
            {
                "_comment": (
                    "Per-file pyright error counts for `pyright src/` (#1503). "
                    "A file may only go DOWN. Regenerate with "
                    "`python3 scripts/check_pyright_baseline.py --update` and "
                    "commit the drop alongside the fix. tests/ is not gated — "
                    "see the script docstring."
                ),
                "target": TARGET,
                "total": sum(counts.values()),
                "files": dict(sorted(counts.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    args = ap.parse_args(argv)

    counts = measure()

    if args.update:
        write_baseline(counts)
        print(f"baseline written: {sum(counts.values())} errors over {len(counts)} files")
        return 0

    baseline = load_baseline()
    if not baseline:
        sys.stderr.write("baseline is empty — the check would be vacuous\n")
        return 2

    regressions = sorted(
        (f, baseline.get(f, 0), n)
        for f, n in counts.items()
        if n > baseline.get(f, 0)
    )
    improvements = sorted(
        (f, baseline[f], counts.get(f, 0))
        for f in baseline
        if counts.get(f, 0) < baseline[f]
    )

    total_now, total_was = sum(counts.values()), sum(baseline.values())
    print(f"pyright {TARGET}: {total_now} errors (baseline {total_was})")

    for f, was, now in improvements:
        print(f"  improved  {f}: {was} -> {now}")

    if regressions:
        print("\nnew pyright errors:", file=sys.stderr)
        for f, was, now in regressions:
            print(f"  {f}: {was} -> {now}", file=sys.stderr)
        print(
            "\nEach file may only go down. Fix the new errors, or — if they are "
            "genuinely pre-existing and newly surfaced (a pyright upgrade, a "
            "dependency's stubs changing) — say so in the PR body and run "
            "`python3 scripts/check_pyright_baseline.py --update`.",
            file=sys.stderr,
        )
        return 1

    if improvements:
        print(
            "\nFile(s) improved. Run "
            "`python3 scripts/check_pyright_baseline.py --update` and commit "
            "the lowered baseline, or the ratchet does not turn."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
