"""#1356 — re-derive the temporal-spine fan-in baseline and the corrected share.

The ratified constraint on #1283 AC4 is that the count of `TEMPORAL_NEXT`
successors carrying more than one predecessor edge is **non-increasing**. Before
#1356 that constraint lived only as prose — in `spine_recompute`'s docstring and
in `aelf spine verify`'s printed output — with no baseline committed anywhere,
so nothing could assert it. This script produces the baseline, and
`fan_in_regressed_against` is what compares to it.

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

`reproduced_share` was published as **93.68%** against the full shipped set.
Under the corrected denominator the same store reports **94.86%**. The change is
the correction, not a movement in fidelity — do not compare the two silently.

Usage:

    uv run python -m benchmarks.spine_fan_in_baseline --store <path> [--write]

`--store` is required and is **copied** before opening: `MemoryStore.__init__`
runs migrations and a lifecycle sweep, so opening a live store is a write. The
copy is what gets opened.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from aelfrice.spine_recompute import spine_divergence
from aelfrice.store import MemoryStore

BASELINE_PATH = Path(__file__).with_name("spine_fan_in_baseline.json")


def measure(store_path: str) -> dict[str, Any]:
    """Open a COPY of `store_path` and return the divergence figures."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "spine_baseline.db"
        shutil.copy(store_path, copy)
        store = MemoryStore(str(copy))
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
    if observed > recorded:
        print("REGRESSED — the fan-in surplus grew; the writer defect widened.")
    else:
        print("OK — non-increasing.")

    if args.write:
        baseline["figures"] = measured
        with BASELINE_PATH.open("w") as f:
            json.dump(baseline, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
