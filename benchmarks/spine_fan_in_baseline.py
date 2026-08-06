"""#1356 — re-derive the temporal-spine fan-in baseline and the corrected share.

The ratified constraint on #1283 AC4 is that the count of `TEMPORAL_NEXT`
successors carrying more than one predecessor edge is **non-increasing**. Before
#1356 that constraint lived only as prose — in `spine_recompute`'s docstring and
in `aelf spine verify`'s printed output — with no baseline committed anywhere,
so nothing could assert it. This script produces the baseline and compares
against it, via the shipped `fan_in_regressed_against` predicate.

It also re-derives `reproduced_share` under the corrected denominator. That
number is published, so per the project rule it ships with the script that
re-derives it.

## The denominator correction

A chain gives every successor exactly one predecessor, so on a successor with
fan-in 2 the recompute can only ever reproduce one of the two shipped edges —
the other is a guaranteed miss no key can avoid. Leaving those edges in the
denominator charges the candidate key for a writer defect it cannot express.

Both edges are dropped, not just the missed one. Dropping only the miss would
move numerator and denominator by different amounts and *inflate* the share
rather than correct it.

## Provenance of the published figures

Against the full shipped set this store measures **93.69%** (39,335 / 41,984);
under the corrected denominator it reports **94.86%** (38,789 / 40,892). Those
two come from one `spine_divergence()` call on one store, which is the only
pair it is meaningful to subtract — and the change is the correction, not a
movement in fidelity, so do not compare them silently either.

The **93.68%** published in #1336 is a *different* measurement: 39,280 / 41,929,
on a snapshot with 55 fewer shipped edges. It is not this store's before-figure
and must not be paired with the 94.86% as though it were.

Usage:

    uv run python -m benchmarks.spine_fan_in_baseline --store <path> [--write]

`--store` is required and is opened **read-only**: `MemoryStore.__init__` runs
migrations and a lifecycle sweep, so opening a live store is otherwise a write.
Exit status is 1 when the fan-in surplus has grown, so a caller can gate on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aelfrice.spine_recompute import fan_in_regressed_against, spine_divergence
from aelfrice.store import MemoryStore

BASELINE_PATH = Path(__file__).with_name("spine_fan_in_baseline.json")


def measure(store_path: str) -> dict[str, Any]:
    """Open `store_path` read-only and return the divergence figures.

    Read-only rather than copied. The copy existed because
    ``MemoryStore.__init__`` runs migrations and a lifecycle sweep, so
    opening a live store is a write — but ``read_only=True`` is the
    supported answer to that and is what ``aelf spine verify`` itself
    uses (``cli.py``), so the two now read the store the same way.

    Copying was also silently lossy. ``shutil.copy`` takes the main
    database file alone, and the store runs in WAL mode: every commit
    since the last checkpoint lives in ``-wal``, which was not copied.
    On a live store the script could therefore measure a valid but stale
    snapshot — and report a fan-in count below the baseline, printing
    "OK — non-increasing" off figures that were never the store's.
    """
    store = MemoryStore(store_path, read_only=True)
    try:
        d = spine_divergence(store)
    finally:
        store.close()

    return {
        "n_shipped": d.n_shipped,
        "n_recomputed": d.n_recomputed,
        "n_reproduced": d.n_reproduced,
        "n_recomputed_only": d.n_recomputed_only,
        "n_fan_in_successors": d.n_fan_in_successors,
        "n_eligible_shipped": d.n_eligible_shipped,
        "n_eligible_reproduced": d.n_eligible_reproduced,
        "reproduced_share": round(d.reproduced_share, 6),
        "missing_touching_no_log": d.missing_touching_no_log,
        "missing_fan_in": d.missing_fan_in,
        "missing_other": d.missing_other,
    }


def load_baseline() -> dict[str, Any]:
    with BASELINE_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, help="path to a memory.db")
    ap.add_argument(
        "--write",
        action="store_true",
        help="overwrite the committed baseline with the measured figures",
    )
    args = ap.parse_args(argv)

    measured = measure(args.store)
    for k, v in measured.items():
        print(f"{k:<26} {v}")

    baseline = load_baseline()
    observed = measured["n_fan_in_successors"]
    recorded = baseline["figures"]["n_fan_in_successors"]
    print()
    print(f"fan-in successors: observed {observed} vs baseline {recorded}")
    # Call the shipped predicate rather than restating `observed > recorded`
    # here. It is the one the tests exercise, and a second copy of the
    # comparison is a second place for the equal case to be got wrong —
    # which is the whole content of the rule (equal is not a regression).
    regressed = fan_in_regressed_against(observed, recorded)
    if regressed:
        print("REGRESSED — the fan-in surplus grew; the writer defect widened.")
    else:
        print("OK — non-increasing.")

    if args.write:
        baseline["figures"] = measured
        with BASELINE_PATH.open("w") as f:
            json.dump(baseline, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {BASELINE_PATH}")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
