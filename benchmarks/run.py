"""`aelf bench all` dispatcher.

Subprocesses each adapter at the canonical headline cut (full per the
2026-05-06 ratification on #437), parses each adapter's JSON output,
and merges into a single schema-v2 results file. The dispatcher does
not touch adapter internals — every adapter keeps its own argparse and
its own `--output PATH` write path; the dispatcher is the loop, the
JSON merge, and the canonical-vs-cron filename split.

Spec: docs/v2_reproducibility_harness.md (ratified 2026-05-06).
Issue: #437.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HARNESS_VERSION = "1"
SCHEMA_VERSION = 2

# Per the 2026-05-06 ratification: "full benchmarks (no sized cut)".
# Each entry is one subprocess invocation. Adapters that produce a
# single-value scope flag (MAB --split, StructMemEval --task) get
# multiple invocations; the merge step folds them under the adapter
# name in the final JSON.
@dataclass(frozen=True)
class AdapterInvocation:
    adapter: str           # logical name in the merged JSON ("mab", ...)
    sub_key: str | None    # None = single-invocation; else key under adapter ("Conflict_Resolution", "location")
    module: str            # python -m target ("benchmarks.mab_adapter")
    args: tuple[str, ...]  # adapter-specific scope flags (excluding --output)

    @property
    def label(self) -> str:
        return self.adapter if self.sub_key is None else f"{self.adapter}/{self.sub_key}"


CANONICAL_INVOCATIONS: tuple[AdapterInvocation, ...] = (
    # MAB: 4 splits, full each.
    AdapterInvocation("mab", "Conflict_Resolution",
                      "benchmarks.mab_adapter",
                      ("--split", "Conflict_Resolution")),
    AdapterInvocation("mab", "Test_Time_Learning",
                      "benchmarks.mab_adapter",
                      ("--split", "Test_Time_Learning")),
    AdapterInvocation("mab", "Long_Range_Understanding",
                      "benchmarks.mab_adapter",
                      ("--split", "Long_Range_Understanding")),
    AdapterInvocation("mab", "Accurate_Retrieval",
                      "benchmarks.mab_adapter",
                      ("--split", "Accurate_Retrieval")),
    # LoCoMo: full (10 conversations).
    AdapterInvocation("locomo", None,
                      "benchmarks.locomo_adapter", ()),
    # LongMemEval: full dataset (override — spec recommended oracle subset).
    AdapterInvocation("longmemeval", None,
                      "benchmarks.longmemeval_adapter", ()),
    # StructMemEval: 4 tasks, --bench big each (override — spec recommended small).
    AdapterInvocation("structmemeval", "location",
                      "benchmarks.structmemeval_adapter",
                      ("--task", "location", "--bench", "big")),
    AdapterInvocation("structmemeval", "accounting",
                      "benchmarks.structmemeval_adapter",
                      ("--task", "accounting", "--bench", "big")),
    AdapterInvocation("structmemeval", "recommendations",
                      "benchmarks.structmemeval_adapter",
                      ("--task", "recommendations", "--bench", "big")),
    AdapterInvocation("structmemeval", "tree",
                      "benchmarks.structmemeval_adapter",
                      ("--task", "tree", "--bench", "big")),
    # AMA-Bench: full 208 episodes.
    AdapterInvocation("amabench", None,
                      "benchmarks.amabench_adapter", ()),
)


# Smoke invocations are a separate, smaller registry the PR-CI tier
# uses. Cap is ≤2 minutes wall-clock total.
SMOKE_INVOCATIONS: tuple[AdapterInvocation, ...] = (
    AdapterInvocation("mab", "Conflict_Resolution",
                      "benchmarks.mab_adapter",
                      ("--split", "Conflict_Resolution", "--rows", "5", "--subset", "5")),
    AdapterInvocation("amabench", None,
                      "benchmarks.amabench_adapter",
                      ("--max-episodes", "5")),
)


@dataclass
class InvocationResult:
    invocation: AdapterInvocation
    status: str                       # "ok" | "skipped_data_missing" | "error"
    elapsed_sec: float
    output: dict[str, Any] | None     # parsed --output JSON when status=="ok"
    error_message: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _aelfrice_version() -> str:
    try:
        from aelfrice import __version__ as v
        return v
    except Exception:
        return "unknown"


def run_invocation(
    inv: AdapterInvocation,
    *,
    runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
    tmp_root: Path | None = None,
) -> InvocationResult:
    """Subprocess one adapter and parse its --output JSON.

    `runner` is injectable so tests can stub the subprocess call without
    actually running a benchmark.
    """
    if runner is None:
        runner = _default_runner
    if tmp_root is None:
        tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f"{inv.adapter}_{inv.sub_key or 'all'}.json"
    cmd = [sys.executable, "-m", inv.module, *inv.args, "--output", str(out_path)]
    start = time.monotonic()
    try:
        proc = runner(cmd, out_path)
    except Exception as exc:  # noqa: BLE001 — surface any subprocess crash
        return InvocationResult(
            invocation=inv, status="error", elapsed_sec=time.monotonic() - start,
            output=None, error_message=f"runner crashed: {exc!r}",
        )
    elapsed = time.monotonic() - start
    # Adapter exit-code contract per the 2026-05-06 ratification:
    #   0 → ok                  1 → error
    #   2 → skipped_data_missing  (e.g. /tmp/LoCoMo not present)
    if proc.returncode == 2:
        return InvocationResult(
            invocation=inv, status="skipped_data_missing", elapsed_sec=elapsed,
            output=None, error_message=(proc.stderr or proc.stdout or "").strip()[:500],
        )
    if proc.returncode != 0:
        return InvocationResult(
            invocation=inv, status="error", elapsed_sec=elapsed,
            output=None, error_message=(proc.stderr or proc.stdout or "").strip()[:500],
        )
    if not out_path.exists():
        return InvocationResult(
            invocation=inv, status="error", elapsed_sec=elapsed,
            output=None,
            error_message=f"adapter exited 0 but did not write {out_path}",
        )
    try:
        with out_path.open() as f:
            parsed = json.load(f)
    except json.JSONDecodeError as exc:
        return InvocationResult(
            invocation=inv, status="error", elapsed_sec=elapsed,
            output=None, error_message=f"output JSON parse failed: {exc}",
        )
    return InvocationResult(
        invocation=inv, status="ok", elapsed_sec=elapsed, output=parsed,
    )


def _default_runner(cmd: list[str], out_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# Per-row detail fields stripped from the canonical output before
# write. Keep them only in the in-memory output / sidecar; the
# canonical JSON should hold summary metrics that change rarely, not
# 6,000+ per-question rows that bloat the file to 37MB and dominate
# every diff. Re-add a field to the keep set if a tolerance band
# needs to read it directly.
_DETAIL_FIELDS_TO_STRIP: frozenset[str] = frozenset({
    "per_question",
})


def _strip_detail(output: Any) -> Any:
    """Recursively remove per-row detail fields. Returns a new structure."""
    if isinstance(output, dict):
        return {
            k: _strip_detail(v)
            for k, v in output.items()
            if k not in _DETAIL_FIELDS_TO_STRIP
        }
    if isinstance(output, list):
        return [_strip_detail(v) for v in output]
    return output


def _merge(results: list[InvocationResult]) -> dict[str, dict[str, Any]]:
    """Fold per-invocation outputs into adapter-keyed map.

    Single-invocation adapters: results[adapter] = output.
    Multi-invocation adapters: results[adapter][sub_key] = output.
    `per_question` (and similar per-row lists) are stripped — see
    _DETAIL_FIELDS_TO_STRIP for rationale.
    """
    merged: dict[str, dict[str, Any]] = {}
    for r in results:
        adapter = r.invocation.adapter
        bucket = merged.setdefault(adapter, {})
        # Dispatcher-level metadata uses underscore prefix so
        # tolerance.check_report skips it during band walks. Only
        # `output`'s leaves should land on the band-check path.
        payload: dict[str, Any] = {
            "_status": r.status,
            "_elapsed_sec": round(r.elapsed_sec, 3),
        }
        if r.output is not None:
            payload["output"] = _strip_detail(r.output)
        if r.error_message:
            payload["_error_message"] = r.error_message
        if r.invocation.sub_key is None:
            # Single-invocation: payload becomes the adapter's record directly
            # (but keep the bucket dict shape so multi-invocation rows don't
            # have to special-case "is this a leaf or a sub-key map?").
            bucket["_"] = payload
        else:
            bucket[r.invocation.sub_key] = payload
    return merged


def _headline_cut_for(invocations: tuple[AdapterInvocation, ...]) -> dict[str, Any]:
    """Reproduce the canonical headline-cut declaration from the registry.

    This is what gets written into the JSON's headline_cut field so a
    later regression check can verify `--canonical` matched the cut on
    record.
    """
    cut: dict[str, list[str] | dict[str, Any]] = {}
    for inv in invocations:
        cut.setdefault(inv.adapter, []).append(  # type: ignore[union-attr]
            {"sub_key": inv.sub_key, "args": list(inv.args)}
        )
    return cut


def build_report(
    results: list[InvocationResult],
    *,
    label: str,
    invocations_used: tuple[AdapterInvocation, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "captured_at_utc": _utc_now_iso(),
        "git_commit": _git_commit(),
        "aelfrice_version": _aelfrice_version(),
        "harness_version": HARNESS_VERSION,
        "headline_cut": _headline_cut_for(invocations_used),
        "results": _merge(results),
    }


def _validate_canonical_cut(invocations_used: tuple[AdapterInvocation, ...]) -> None:
    """Refuse `--canonical` writes when the run did not match the registry.

    Prevents accidental overwrite of the canonical artifact with a
    partial run (e.g. `--adapters mab` followed by `--canonical`).
    """
    if invocations_used != CANONICAL_INVOCATIONS:
        raise SystemExit(
            "refusing to write --canonical: invocations did not match "
            "CANONICAL_INVOCATIONS. Run `aelf bench all` with no "
            "--adapters override (or fix the registry)."
        )


def main_all(
    *,
    out_path: Path,
    canonical: bool,
    adapters: tuple[str, ...] | None = None,
    smoke: bool = False,
    runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
    tmp_root: Path | None = None,
) -> int:
    """Entry point for `aelf bench all`.

    Returns process exit code (0 ok, 1 if any adapter status was
    "error", 2 if any was "skipped_data_missing" and none was "error").
    """
    invocations = SMOKE_INVOCATIONS if smoke else CANONICAL_INVOCATIONS
    if adapters is not None:
        # Capture the available-adapters list BEFORE filtering, so the
        # error message can enumerate valid options. Building it from
        # the post-filter `invocations` produces an empty list whenever
        # the filter matched nothing — exactly when the user needs the
        # enumeration most.
        available = sorted({i.adapter for i in invocations})
        invocations = tuple(i for i in invocations if i.adapter in adapters)
        if not invocations:
            raise SystemExit(
                f"no adapters matched filter {adapters!r}; "
                "available: " + ", ".join(available)
            )
    if canonical:
        # Cut-mismatch refusal applies whether or not --adapters was passed.
        _validate_canonical_cut(invocations)

    results: list[InvocationResult] = []
    for inv in invocations:
        print(f"[bench] {inv.label}: running…", flush=True)
        r = run_invocation(inv, runner=runner, tmp_root=tmp_root)
        print(f"[bench] {inv.label}: {r.status} in {r.elapsed_sec:.1f}s", flush=True)
        results.append(r)

    label = "v2.0.0 canonical" if canonical else f"v2.0.0 cron {_utc_now_iso()}"
    if smoke:
        label = "v2.0.0 smoke"
    report = build_report(results, label=label, invocations_used=invocations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[bench] wrote {out_path}", flush=True)

    if any(r.status == "error" for r in results):
        return 1
    if any(r.status == "skipped_data_missing" for r in results):
        return 2
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="aelf bench all dispatcher (also reachable as `aelf bench all`)",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to write the merged JSON report.")
    parser.add_argument("--canonical", action="store_true",
                        help="Assert run matches canonical headline cut.")
    parser.add_argument("--adapters", default=None,
                        help="Comma-separated adapter filter (mab,locomo,...)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the smoke invocations instead of canonical.")
    args = parser.parse_args()
    adapter_filter = (
        tuple(a.strip() for a in args.adapters.split(",") if a.strip())
        if args.adapters else None
    )
    return main_all(
        out_path=args.out, canonical=args.canonical,
        adapters=adapter_filter, smoke=args.smoke,
    )


if __name__ == "__main__":
    raise SystemExit(_cli())
