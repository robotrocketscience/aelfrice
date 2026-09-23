#!/usr/bin/env python3
"""#1546 K3 — the A/A noise replicate for the budget discriminability census.

`docs/design/budget_discriminability_k0_preregistration.md` builds the census
noise floor as

    NF = max(1.96 * sqrt(0.25 / N), 9.5pp, A/A_band)

and records that the third term is missing. This script supplies it. `A/A_band`
is the spread the census statistic shows when nothing that should matter has
changed: the corpus, the gold sets, the grid, the lane set, and every belief's
content are held fixed, and only a detail that no correct instrument may read
is varied.

## Why the perturbation is ingest order, and not a re-run

The census is deterministic by construction, and
`scripts/budget_discriminability_census.py --check` proves it on every run by
hashing two reports. A replicate that re-runs the same inputs therefore
measures exactly zero by construction, and such a zero says nothing about the
instrument.

The perturbation is **insertion order**. Each labelled query's
beliefs are re-ingested under a seeded permutation, so every belief keeps its
id, content, type, origin, posterior and timestamp, and only the `beliefs`
rowids change. Rowid is a real tie-break key in this codebase — the temporal
spine orders by rowid rather than by time — so an instrument that reads it
would move under this perturbation while nothing a measurement may depend on
has moved.

That choice was fixed before any band was computed, and the artifact that
records it is `docs/design/budget_discriminability_aa_result.md` — the K0
pre-registration names the missing term but not how to measure it, so read
"fixed in advance" as a claim about the order of work rather than as a
citation. The reason to write it down at all: picking a perturbation after
seeing a band is picking the one that yields the narrowest band. Do not
substitute a different one without saying why this one was wrong.

## What is reused, and why

The replicate never reimplements EC. It hands each permuted population to
`budget_discriminability_census.report(queries=...)`, so the lanes, the grid,
the cost functions, `measure_query`, the exclusion rules, the block-identity
check and the containment check are the census's own. A replicate that
reimplements the statistic measures its own reimplementation.

Replicate 0 is the census's committed order, unpermuted, so its EC row must
equal the published K0 figures. Replicates 1..S are the seeded permutations.

## The cells, and the band

A *cell* is one EC value the census publishes: a lane total, or a lane crossed
with a corpus. For each cell the band is `max(EC) - min(EC)` over the
replicates, in percentage points. `aa_band_pp` is the maximum over cells, which
is the value `NF` would take.

The band is a sample range, so it is downward biased: more replicates can only
widen it, never narrow it. Read a reported band as a lower bound on the
instrument's spread.

## Usage

    uv run python scripts/budget_discriminability_aa_replicate.py
    uv run python scripts/budget_discriminability_aa_replicate.py --dry-run
    uv run python scripts/budget_discriminability_aa_replicate.py --check
    uv run python scripts/budget_discriminability_aa_replicate.py --emit-figures

`--dry-run` prints the plan and opens no store. `--check` runs the whole
replicate twice at the pinned seed and exits non-zero unless both reports hash
identically (#605). `--emit-figures` is the flat-JSON protocol
`scripts/check_derived_figures.py` speaks. Every mode exits non-zero on
failure.

## Stores, and determinism

Every store is `:memory:` and lives for one query, because the census's own
`_open_store` builds it. The replicate never opens a user store, never reads
`AELFRICE_CORPUS_ROOT`, and never touches the lab corpus (#1456). Importing the
census clears every `AELFRICE_*` variable from this process, and this module
asserts that afterwards rather than clearing a second time, so a census that
stops clearing fails here instead of being covered here.

There is no embedding and no unseeded randomness. `BASE_SEED` is pinned in this
file and is printed in every report.

## The result this can legitimately report, and the control that earns it

A band of 0.0pp is a possible and useful outcome, but only when the instrument
could have reported otherwise. `tests/test_budget_aa_replicate.py` drives the
replicate against a fake `retrieval.retrieve` whose output genuinely depends on
insertion order and requires the band to go non-zero. Without that arm a zero
here is a print statement, not a measurement.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CENSUS_PATH: Final[Path] = REPO_ROOT / "scripts" / "budget_discriminability_census.py"
CENSUS_MODULE: Final[str] = "budget_discriminability_census"


def _load_census() -> Any:
    """Import the census as a module, the `tests/test_budget_census.py` way.

    Importing it has two side effects this module depends on: it puts
    `src/` on `sys.path`, and it clears every `AELFRICE_*` variable from the
    process before `aelfrice` is imported.
    """
    if CENSUS_MODULE in sys.modules:
        return sys.modules[CENSUS_MODULE]
    spec = importlib.util.spec_from_file_location(CENSUS_MODULE, str(CENSUS_PATH))
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot import the census from {CENSUS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[CENSUS_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


census: Final[Any] = _load_census()

_STRAY_ENV: Final[list[str]] = sorted(
    k for k in os.environ if k.startswith("AELFRICE_")
)
if _STRAY_ENV:  # pragma: no cover - the census clears these at import
    raise RuntimeError(
        "the census no longer clears AELFRICE_* at import, so this replicate "
        f"would measure a different lane set: {_STRAY_ENV}"
    )

# --- The knobs, fixed before any band was computed ---------------------

# Pinned, printed in every report, and never derived from a clock, a path or
# an environment variable (#605).
BASE_SEED: Final[int] = 1546

# 8 permuted replicates plus the unpermuted reference. The band is a sample
# range and is monotone non-decreasing in the replicate count, so this default
# can only understate it, never flatter it. It is chosen against the
# 300-second producer timeout `scripts/check_derived_figures.py` enforces: a
# replicate costs one `census.report()` (measured at 0.85s) plus its share of
# the order-sensitivity sweep, and 9 replicates measured 56 seconds on this
# machine against 105 seconds at 16. Raise it with --seeds when you want a
# tighter lower bound; never lower it after seeing a band.
DEFAULT_SEEDS: Final[int] = 8

# The reference replicate: the corpora in their committed order. Its EC row
# must reproduce the published K0 figures, which is what ties this replicate
# to the census it claims to be replicating.
REFERENCE_REPLICATE: Final[int] = 0


def permutation_seed(replicate: int, corpus: str, qid: str) -> int:
    """The seed for one query's insertion-order permutation.

    Derived by SHA-256 from the pinned base seed and the query's identity, so
    it is stable across processes, platforms and `PYTHONHASHSEED` values, and
    two different queries never share a permutation by accident.
    """
    payload = f"{BASE_SEED}:{replicate}:{corpus}:{qid}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def permuted_corpora(
    queries: list[Any], replicate: int
) -> list[Any]:
    """One replicate's population: the same queries, re-ingested in a new order.

    Only `beliefs` moves. `corpus`, `qid`, `query` and `gold_ids` are carried
    across untouched, so the gold join, the grid and the lane set are identical
    in every replicate and the only difference downstream is the rowid each
    belief is assigned by `census._open_store`.
    """
    if replicate == REFERENCE_REPLICATE:
        return list(queries)
    out: list[Any] = []
    for q in queries:
        beliefs = list(q.beliefs)
        random.Random(permutation_seed(replicate, q.corpus, q.qid)).shuffle(
            beliefs
        )
        out.append(dataclasses.replace(q, beliefs=tuple(beliefs)))
    return out


def insertion_order(queries: list[Any]) -> dict[str, tuple[str, ...]]:
    """The belief-id insertion order per query, which is the rowid order.

    `census._open_store` inserts `q.beliefs` in sequence into a fresh
    `:memory:` store, so position in this tuple is the belief's rowid minus
    one. Reported so a reader can check the perturbation was not inert.
    """
    return {
        f"{q.corpus}/{q.qid}": tuple(b.id for b in q.beliefs) for q in queries
    }


# --- The EC cells ------------------------------------------------------


def ec_cells(rep: dict[str, Any]) -> dict[str, float | None]:
    """Every EC value one census report publishes, keyed by cell.

    Read off the census's own report rather than recomputed, so a change to how
    the census aggregates EC reaches this replicate instead of being silently
    diverged from.
    """
    cells: dict[str, float | None] = {}
    for lane_name, row in rep["lanes"].items():
        cells[lane_name] = row["ec_pp"]
        for corpus, by in row["by_corpus"].items():
            cells[f"{lane_name}/{corpus}"] = by["ec_pp"]
    return cells


def spread(values: list[float]) -> float:
    """`max - min`, rounded the way the census rounds a percentage point."""
    return round(max(values) - min(values), 4)


def cell_band(values: list[float | None]) -> float | None:
    """One cell's band, or `None` when any replicate had no statistic.

    Kill criterion K-1 is "no statistic", not "a zero band", and the
    difference is the whole point: a cell where `ec_pp` is `None` on some
    replicate has `N = 0` there, and folding that in as 0.0 would let an
    unmeasurable cell narrow the noise floor. Extracted so the `None`
    branch can be tested — it cannot arise on the committed corpora, so
    inline it was mutation-transparent.
    """
    if any(v is None for v in values):
        return None
    return spread([float(v) for v in values])  # type: ignore[arg-type]


def n_drift(ns: list[int]) -> int:
    """How far `N` moved across replicates; non-zero is a violation.

    The replicates must share a population or their EC values are not
    comparable. Extracted for the same reason as `cell_band`: `N` holds
    at 7 on the committed corpora, so an inline check that could never
    fire was indistinguishable from one that was disabled.
    """
    return max(ns) - min(ns)


def band_over_cells(bands: dict[str, float]) -> float | None:
    """The band `NF` would take: the WIDEST cell, not the narrowest.

    Extracted so it can be tested against cells with different spreads.
    Inline, on a corpus where every cell bands at 0.0, `max` and `min`
    return the same number and no test could tell them apart — which is
    the shape a noise floor must never be allowed to take, since `min`
    understates the floor and understating it is what lets a later
    verdict clear a band it should not have.

    `None` when no cell is bandable: no statistic, rather than a zero.
    """
    return max(bands.values()) if bands else None


# --- The order-sensitivity diagnostic ----------------------------------


def arm_outputs(
    queries_by_replicate: list[list[Any]],
) -> tuple[int, int, list[str]]:
    """Does any single grid arm's output move under the permutation at all?

    The band answers "does the statistic move". This answers the sharper
    question one level down: does anything the statistic is computed from move.
    A band of zero over a statistic whose inputs never moved is a different
    finding from a band of zero over inputs that moved and cancelled, and only
    this pass can tell the two apart.

    Returns `(cells_that_varied, cells_examined, examples)`, over every
    lane x query x grid cell.
    """
    lanes = census.lanes()
    varied: list[str] = []
    examined = 0
    per_replicate_queries = [
        {f"{q.corpus}/{q.qid}": q for q in qs} for qs in queries_by_replicate
    ]
    keys = sorted(per_replicate_queries[0])
    # The store depends only on `(key, replicate)`, but the arm it is probed
    # under depends on `(lane, budget, sub)`. Opening inside the arm loops
    # therefore rebuilt the same store once per arm: lanes x budgets x
    # sub-budgets = 90 identical opens per `(key, replicate)` pair, and
    # `MemoryStore(":memory:")` runs the full open-time DDL, migration and
    # origin backfill every time. Hoisting the open above the arm loops takes
    # the census from 4,140 opens to 46 and the sweep from ~12s to ~1s per
    # replicate, with byte-identical output -- `examined` counts cells, and
    # reordering the loops changes only the order `varied` is appended in,
    # which the return already normalises with `sorted(...)`.
    #
    # Reuse is sound because `census._retrieve` is read-only with respect to
    # the store; `budget_discriminability_census.measure_query` already relies
    # on that, reusing one store across its probe and all 15 grid arms.
    for key in keys:
        stores = [census._open_store(m[key]) for m in per_replicate_queries]
        try:
            for lane in lanes:
                for mult in census.BUDGET_MULTIPLIERS:
                    budget = census.arm_budget_for(lane.budget, mult)
                    for sub in census.L25_SUBBUDGETS:
                        seen: set[tuple[str, ...]] = set()
                        for store, mapping in zip(stores, per_replicate_queries):
                            q = mapping[key]
                            hits = census._retrieve(
                                store,
                                lane,
                                q.query,
                                token_budget=budget,
                                l25_token_subbudget=sub,
                            )
                            seen.add(tuple(b.id for b in hits))
                        examined += 1
                        if len(seen) > 1:
                            varied.append(
                                f"{lane.name}/{key} budget={budget} sub={sub}: "
                                f"{len(seen)} distinct outputs"
                            )
        finally:
            for store in stores:
                store.close()
    return len(varied), examined, sorted(varied)[:10]


# --- The replicate -----------------------------------------------------


def replicate(
    seeds: int = DEFAULT_SEEDS, *, order_sensitivity_sweep: bool = True
) -> dict[str, Any]:
    """Run the census once per replicate and return the whole A/A report.

    `order_sensitivity_sweep` runs the per-arm diagnostic above, which costs
    one `retrieve()` per lane x query x grid cell per replicate — 4s of an
    11s run here, against about 7s for the band. Turning it off
    changes nothing about how the band is computed; a report produced with it
    off carries `arm_sweep: False`, and `figures()` then OMITS the sweep keys
    rather than zero-filling them, so a skipped diagnostic can never reach a
    published figure.

    Note the asymmetry, which is deliberate rather than an oversight: this
    parameter defaults to True, so a library caller gets the complete
    measurement, while the CLI's `--sweep` defaults to OFF, so the
    derived-figures gate is not charged for a diagnostic none of its markers
    cover. Pass `--sweep` to reproduce the published sweep figures.
    """
    if seeds < 1:
        raise ValueError(f"seeds must be at least 1, got {seeds}")
    base = census.corpora()
    populations = [
        permuted_corpora(base, r) for r in range(REFERENCE_REPLICATE, seeds + 1)
    ]

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for index, population in enumerate(populations):
        rep = census.report(queries=population)
        violations.extend(
            f"replicate {index}: {v}" for v in rep["violations"]
        )
        rows.append(
            {
                "replicate": index,
                "permuted": index != REFERENCE_REPLICATE,
                "n": rep["n"],
                "cells": ec_cells(rep),
            }
        )

    # The band is computed whatever the census reported, and only a cell with
    # no statistic at all is dropped from it. Gating the band on a clean
    # violation list would be wrong here and not merely cautious: on these
    # corpora every pool prices below every budget in the grid, so any
    # discordant footprint also trips the census's block-identity check. A
    # band that is only computed when the violation list is empty is therefore
    # a band that can only ever report zero — which is the exact defect this
    # replicate exists to rule out. Violations stay in the report and still
    # fail the run; they do not silence the measurement.
    cell_names = sorted(rows[0]["cells"])
    bands: dict[str, float] = {}
    unbandable: list[str] = []
    for name in cell_names:
        values = [row["cells"].get(name) for row in rows]
        band = cell_band(values)
        if band is None:
            # Kill criterion K-1: no statistic, rather than a zero band.
            unbandable.append(name)
            violations.append(
                f"cell {name} has ec_pp None on at least one replicate, so "
                "N is 0 there and the cell contributes no statistic to band "
                "with; it is excluded from aa_band_pp"
            )
            continue
        bands[name] = band

    ns = [int(r["n"]) for r in rows]
    n_band = n_drift(ns)
    if n_band:
        violations.append(
            "the permutation moved N from "
            f"{min(ns)} to {max(ns)}, so the replicates do not share a "
            "population and their EC values are not comparable"
        )

    orders = [insertion_order(p) for p in populations]
    distinct_orders = {
        key: len({o[key] for o in orders}) for key in orders[0]
    }
    inert = sorted(k for k, v in distinct_orders.items() if v == 1)

    if order_sensitivity_sweep:
        varied_cells, examined_cells, examples = arm_outputs(populations)
    else:
        varied_cells, examined_cells, examples = None, None, []

    aa_band = band_over_cells(bands)
    return {
        "instrument": "scripts/budget_discriminability_aa_replicate.py",
        "replicates_of": "scripts/budget_discriminability_census.py",
        "pre_registration": (
            "docs/design/budget_discriminability_k0_preregistration.md"
        ),
        "statistic": (
            "A/A_band = max over EC cells of (max - min) across replicates, "
            "in percentage points; the third term of NF"
        ),
        "perturbation": (
            "seeded permutation of each labelled query's belief insertion "
            "order, which moves every rowid and nothing else"
        ),
        "base_seed": BASE_SEED,
        "seeds": seeds,
        "replicates": len(rows),
        "reference_replicate": REFERENCE_REPLICATE,
        "n": ns[0],
        "n_band": n_band,
        "cells": cell_names,
        "unbandable_cells": unbandable,
        "rows": rows,
        "bands": bands,
        "aa_band_pp": aa_band,
        "distinct_insertion_orders": distinct_orders,
        "queries_with_one_insertion_order": inert,
        "arm_sweep": order_sensitivity_sweep,
        "arm_cells_examined": examined_cells,
        "arm_cells_order_sensitive": varied_cells,
        "arm_cells_order_sensitive_examples": examples,
        "resolvers": census.resolvers(),
        "violations": violations,
    }


def figures(rep: dict[str, Any] | None = None) -> dict[str, Any]:
    """The flat key -> value object `--emit-figures` publishes."""
    rep = rep if rep is not None else replicate()
    out: dict[str, Any] = {
        "aa_band_pp": rep["aa_band_pp"],
        "aa_base_seed": rep["base_seed"],
        "aa_seeds": rep["seeds"],
        "aa_replicates": rep["replicates"],
        "aa_cells": len(rep["cells"]),
        "aa_n": rep["n"],
        "aa_n_band": rep["n_band"],
        "aa_violations": len(rep["violations"]),
        "aa_arm_sweep": rep["arm_sweep"],
        "aa_queries_with_one_insertion_order": len(
            rep["queries_with_one_insertion_order"]
        ),
    }
    # The sweep keys are OMITTED rather than zero-filled when the sweep did
    # not run. An earlier revision raised instead, which had the same intent
    # — a skipped diagnostic must never reach a published figure — and made
    # the cheap path unusable for `--emit-figures`, so the derived-figures
    # gate had to pay for a diagnostic none of its markers cover.
    # Omitting keeps the guarantee: there is no key to read, rather than a
    # key holding a number nothing measured. `aa_arm_sweep` above says which
    # run this was, so a consumer cannot mistake absence for zero.
    if rep["arm_sweep"]:
        out["aa_arm_cells_examined"] = rep["arm_cells_examined"]
        out["aa_arm_cells_order_sensitive"] = rep["arm_cells_order_sensitive"]
    for lane in census.lanes():
        out[f"aa_band_pp.{lane.name}"] = rep["bands"].get(lane.name)
    return out


# --- Output ------------------------------------------------------------


def render_text(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("#1546 K3 — A/A noise replicate for the budget census")
    lines.append(f"instrument      {rep['instrument']}")
    lines.append(f"replicates      {rep['replicates_of']}")
    lines.append(f"pre-registration {rep['pre_registration']}")
    lines.append(f"statistic       {rep['statistic']}")
    lines.append(f"perturbation    {rep['perturbation']}")
    lines.append(
        f"base seed       {rep['base_seed']} "
        f"({rep['seeds']} permuted replicates plus the committed order)"
    )
    lines.append("")
    lines.append("resolvers (env-first; a stray AELFRICE_* changes the lane set)")
    for key in sorted(rep["resolvers"]):
        lines.append(f"  {key:<32} {rep['resolvers'][key]!r}")
    lines.append("")
    lines.append("EC per replicate, in percentage points")
    header = "  replicate  N   " + "  ".join(
        f"{name}" for name in rep["cells"]
    )
    lines.append(header)
    for row in rep["rows"]:
        tag = "permuted" if row["permuted"] else "committed"
        values = "  ".join(str(row["cells"][name]) for name in rep["cells"])
        lines.append(
            f"  {row['replicate']:<9} {row['n']:<3} {values}   [{tag}]"
        )
    lines.append("")
    lines.append("band per cell (max - min across replicates)")
    for name in rep["cells"]:
        lines.append(f"  {name:<40} {rep['bands'].get(name)}")
    lines.append("")
    lines.append("the perturbation, checked rather than assumed")
    for key in sorted(rep["distinct_insertion_orders"]):
        lines.append(
            f"  {key:<40} {rep['distinct_insertion_orders'][key]} distinct "
            "insertion orders"
        )
    if rep["queries_with_one_insertion_order"]:
        lines.append(
            "  INERT on: "
            + ", ".join(rep["queries_with_one_insertion_order"])
        )
    lines.append("")
    if rep["arm_sweep"]:
        lines.append(
            f"grid arms whose retrieved output moved: "
            f"{rep['arm_cells_order_sensitive']} of {rep['arm_cells_examined']}"
        )
    else:
        lines.append("grid arms whose retrieved output moved: not swept")
    for example in rep["arm_cells_order_sensitive_examples"]:
        lines.append(f"  {example}")
    lines.append("")
    lines.append(f"N                {rep['n']} (band {rep['n_band']})")
    lines.append(f"A/A_band         {rep['aa_band_pp']}pp")
    lines.append(f"violations       {len(rep['violations'])}")
    for v in rep["violations"]:
        lines.append(f"  VIOLATION {v}")
    return "\n".join(lines) + "\n"


def render_dry_run(seeds: int) -> str:
    base = census.corpora()
    lanes = census.lanes()
    cells = len(census.BUDGET_MULTIPLIERS) * len(census.L25_SUBBUDGETS)
    lines = [
        "#1546 K3 A/A replicate — dry run, nothing retrieved, no store opened",
        "",
        f"base seed                {BASE_SEED}",
        f"permuted replicates      {seeds}",
        f"reference replicate      {REFERENCE_REPLICATE} (committed order)",
        f"census report() calls    {seeds + 1}",
        f"labelled queries         {len(base)}",
        f"lanes                    {len(lanes)}",
        f"grid cells per query     {cells}",
        (
            "order-sensitivity retrieves "
            f"{len(lanes) * len(base) * cells * (seeds + 1)}"
        ),
        "",
        "perturbation: a seeded permutation of each labelled query's belief",
        "insertion order. Content, ids, gold sets, the grid and the lane set",
        "are held fixed; only the rowid census._open_store assigns moves.",
    ]
    return "\n".join(lines) + "\n"


def _stable_hash(rep: dict[str, Any]) -> str:
    payload = json.dumps(rep, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            "number of permuted replicates on top of the committed order "
            f"(default {DEFAULT_SEEDS}); the band is monotone non-decreasing "
            "in this count"
        ),
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "also run the per-arm order-sensitivity diagnostic. It costs one "
            "retrieve() per lane x query x grid cell per replicate — 4s of an "
            "11s run here — while the band itself takes about 7s. Off by "
            "default so the derived-figures gate, whose markers cover only "
            "the band, does not pay for it"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without opening a store",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="run twice and exit non-zero unless both reports hash identically",
    )
    ap.add_argument(
        "--emit-figures",
        action="store_true",
        help="emit a flat JSON object of key -> value on stdout and nothing else",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit the whole report as JSON instead of text",
    )
    args = ap.parse_args(argv)

    if args.seeds < 1:
        print("FAIL: --seeds must be at least 1", file=sys.stderr)
        return 1

    if args.dry_run:
        sys.stdout.write(render_dry_run(args.seeds))
        return 0

    if args.check:
        first = replicate(args.seeds, order_sensitivity_sweep=args.sweep)
        second = replicate(args.seeds, order_sensitivity_sweep=args.sweep)
        h1, h2 = _stable_hash(first), _stable_hash(second)
        if h1 != h2:
            print(
                f"FAIL: the A/A replicate is not deterministic ({h1} != {h2})",
                file=sys.stderr,
            )
            return 1
        if first["violations"]:
            for v in first["violations"]:
                print(f"FAIL: {v}", file=sys.stderr)
            return 2
        print(f"OK: byte-identical across two runs, sha256 {h1}")
        return 0

    rep = replicate(args.seeds, order_sensitivity_sweep=args.sweep)
    if args.emit_figures:
        json.dump(figures(rep), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 2 if rep["violations"] else 0
    if args.json:
        json.dump(rep, sys.stdout, sort_keys=True, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(rep))
    return 2 if rep["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
