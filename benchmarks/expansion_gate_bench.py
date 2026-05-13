"""Adaptive expansion-gate latency micro-bench (#741 acceptance).

Replays a labelled fixture of broad / narrow prompts against the
public ``retrieve()`` surface in four cells:

    (gate-on, bfs-on)   (gate-on, bfs-off)
    (gate-off, bfs-on)  (gate-off, bfs-off)

The gate axis is driven by ``AELFRICE_NO_EXPANSION_GATE``; the BFS
axis is driven by the ``bfs_enabled`` kwarg on ``retrieve()``. For
each cell × label, the harness reports wall-clock p50 / p95 in
milliseconds and the count of calls where ``LaneTelemetry.expansion_
gate_skipped_bfs`` fired (a sanity-check on the gate actually doing
something on broad prompts in the ``(gate-on, bfs-on)`` cell).

Acceptance bullet from #741: broad-prompt p95 in ``(gate-on, bfs-on)``
must beat broad-prompt p95 in ``(gate-off, bfs-on)`` by >= 30%. The
narrow-prompt p50 in the same two cells must not regress.

Fixture format: JSONL, one row per line:

    {"prompt": "<text>", "label": "broad"}
    {"prompt": "<text>", "label": "narrow"}

Output: a single JSON file at ``<out-dir>/expansion_gate_bench.json``.
The harness seeds the multi-hop corpus from
``aelfrice.benchmark.seed_multihop_corpus`` so BFS and HRR have real
edges to walk; small corpus on purpose, latency-not-recall is the
load-bearing metric.

Usage:

    uv run python -m benchmarks.expansion_gate_bench \\
        --fixture benchmarks/fixtures/expansion_gate_stub.jsonl \\
        --out benchmarks/results/<run-id>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from aelfrice import __version__ as AELFRICE_VERSION
from aelfrice.benchmark import seed_multihop_corpus
from aelfrice.expansion_gate import (
    ENV_FORCE_EXPANSION,
    ENV_NO_EXPANSION_GATE,
)
from aelfrice.retrieval import last_lane_telemetry, retrieve
from aelfrice.store import MemoryStore


FIXTURE_LABELS = ("broad", "narrow")
CELL_KEYS = (
    ("gate-on", "bfs-on"),
    ("gate-on", "bfs-off"),
    ("gate-off", "bfs-on"),
    ("gate-off", "bfs-off"),
)


@dataclass(frozen=True)
class FixtureRow:
    prompt: str
    label: str  # "broad" | "narrow"


@dataclass
class CellLabelStats:
    n: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    gate_skipped_bfs_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int]:
        d = asdict(self)
        d.pop("latencies_ms")
        return d  # type: ignore[return-value]


@dataclass
class BenchReport:
    run_id: str
    aelfrice_version: str
    fixture_path: str
    fixture_size: int
    fixture_broad: int
    fixture_narrow: int
    started_at: str
    finished_at: str
    cells: dict[str, dict[str, dict[str, float | int]]]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "aelfrice_version": self.aelfrice_version,
            "fixture_path": self.fixture_path,
            "fixture_size": self.fixture_size,
            "fixture_broad": self.fixture_broad,
            "fixture_narrow": self.fixture_narrow,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cells": self.cells,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def load_fixture(path: Path) -> list[FixtureRow]:
    rows: list[FixtureRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno} not valid JSON: {exc}"
                ) from exc
            prompt = obj.get("prompt")
            label = obj.get("label")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{lineno} missing/empty 'prompt'")
            if label not in FIXTURE_LABELS:
                raise ValueError(
                    f"{path}:{lineno} label must be one of "
                    f"{FIXTURE_LABELS!r}, got {label!r}"
                )
            rows.append(FixtureRow(prompt=prompt, label=label))
    return rows


@contextmanager
def _gate_env(gate_on: bool) -> Iterator[None]:
    """Toggle the expansion-gate via env. gate_on=True leaves the
    gate resolver at default; gate_on=False sets
    ``AELFRICE_NO_EXPANSION_GATE=1`` to disable it. Restores the
    prior env on exit."""
    prior_no_gate = os.environ.get(ENV_NO_EXPANSION_GATE)
    prior_force = os.environ.get(ENV_FORCE_EXPANSION)
    # Always clear FORCE_EXPANSION so it never contaminates a cell.
    if prior_force is not None:
        del os.environ[ENV_FORCE_EXPANSION]
    if gate_on:
        if prior_no_gate is not None:
            del os.environ[ENV_NO_EXPANSION_GATE]
    else:
        os.environ[ENV_NO_EXPANSION_GATE] = "1"
    try:
        yield
    finally:
        if prior_no_gate is None:
            os.environ.pop(ENV_NO_EXPANSION_GATE, None)
        else:
            os.environ[ENV_NO_EXPANSION_GATE] = prior_no_gate
        if prior_force is not None:
            os.environ[ENV_FORCE_EXPANSION] = prior_force


def run_cell(
    store: MemoryStore,
    rows: list[FixtureRow],
    *,
    gate_on: bool,
    bfs_on: bool,
) -> dict[str, CellLabelStats]:
    stats: dict[str, CellLabelStats] = {
        label: CellLabelStats() for label in FIXTURE_LABELS
    }
    with _gate_env(gate_on):
        for row in rows:
            t0 = time.perf_counter()
            retrieve(store, row.prompt, bfs_enabled=bfs_on)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tel = last_lane_telemetry()
            s = stats[row.label]
            s.n += 1
            s.latencies_ms.append(dt_ms)
            if tel.expansion_gate_skipped_bfs:
                s.gate_skipped_bfs_count += 1
    for s in stats.values():
        s.p50_ms = _percentile(s.latencies_ms, 0.50)
        s.p95_ms = _percentile(s.latencies_ms, 0.95)
    return stats


def _cell_name(gate_axis: str, bfs_axis: str) -> str:
    return f"{gate_axis}__{bfs_axis}"


def run(fixture_path: Path, out_dir: Path, run_id: str) -> Path:
    rows = load_fixture(fixture_path)
    if not rows:
        raise SystemExit(f"fixture {fixture_path} is empty")

    broad_n = sum(1 for r in rows if r.label == "broad")
    narrow_n = sum(1 for r in rows if r.label == "narrow")

    store = MemoryStore(":memory:")
    seed_multihop_corpus(store)

    started_at = datetime.now(timezone.utc).isoformat()
    cells: dict[str, dict[str, dict[str, float | int]]] = {}
    for gate_axis, bfs_axis in CELL_KEYS:
        gate_on = gate_axis == "gate-on"
        bfs_on = bfs_axis == "bfs-on"
        cell_stats = run_cell(store, rows, gate_on=gate_on, bfs_on=bfs_on)
        cells[_cell_name(gate_axis, bfs_axis)] = {
            label: s.to_dict() for label, s in cell_stats.items()
        }
    finished_at = datetime.now(timezone.utc).isoformat()

    report = BenchReport(
        run_id=run_id,
        aelfrice_version=AELFRICE_VERSION,
        fixture_path=str(fixture_path),
        fixture_size=len(rows),
        fixture_broad=broad_n,
        fixture_narrow=narrow_n,
        started_at=started_at,
        finished_at=finished_at,
        cells=cells,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expansion_gate_bench.json"
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="expansion_gate_bench",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Path to JSONL fixture of {prompt, label} rows.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; writes <out>/expansion_gate_bench.json.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier embedded in the output JSON. "
        "Default: UTC timestamp.",
    )
    args = parser.parse_args(argv)

    if not args.fixture.is_file():
        parser.error(f"fixture not found: {args.fixture}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    out_path = run(args.fixture, args.out, run_id)
    print(f"wrote {out_path}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
