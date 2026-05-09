#!/usr/bin/env python3
"""Summarise a per-session rebuild_log JSONL, or run the calibration
harness against a public synthetic corpus.

Two modes:

1. **Audit (default).** Phase-1c companion to the #288 phase-1a
   rebuild diagnostic log. Reads one or more ``<session_id>.jsonl``
   files and prints:

       * total rebuild invocations
       * pack rate (n_packed / n_candidates) distribution: mean, p50, p90
       * drop-reason histogram (e.g. ``below_floor:0.40``,
         ``content_hash_collision_with:<id>``, ``budget``)
       * packed-rank distribution
       * count of truncated session-files

   Usage:

       python scripts/audit_rebuild_log.py <path-to-jsonl-or-dir> [...]

2. **Calibrate (#365 R1, opt-in via ``--calibrate-corpus``).** Runs
   the relevance-calibration harness against a public synthetic
   corpus (default
   ``benchmarks/posterior_ranking/fixtures/default.jsonl``) and emits
   P@K / ROC-AUC / Spearman ρ. No live-data dependency. Establishes
   the harness for the close-the-loop relevance-calibration loop
   ratified at #317 (operator approval 2026-05-03).

   Usage:

       python scripts/audit_rebuild_log.py --calibrate-corpus \\
           [path/to/corpus.jsonl] [--k 10] [--seed 0]

Exit codes:
    0   summary or report printed
    1   no readable input / corpus
    2   usage error

Reads only — never modifies the input files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def _iter_records(path: Path) -> Iterable[dict]:
    """Yield decoded JSON records from a JSONL file. Skips malformed
    lines (the writer is fail-soft so partial lines on a crash are
    expected)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"audit_rebuild_log: cannot read {path}: {exc}",
            file=sys.stderr,
        )
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _collect_paths(targets: list[Path]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        if t.is_dir():
            out.extend(sorted(t.glob("*.jsonl")))
        elif t.is_file():
            out.append(t)
        else:
            print(
                f"audit_rebuild_log: skipping {t}: not a file or dir",
                file=sys.stderr,
            )
    return out


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _summarise(records: list[dict]) -> dict:
    n_records = 0
    n_truncated = 0
    pack_rates: list[float] = []
    drop_reasons: Counter[str] = Counter()
    packed_ranks: Counter[int] = Counter()
    n_no_pack_summary = 0

    for r in records:
        if r.get("truncated") is True:
            n_truncated += 1
            continue
        n_records += 1
        pack = r.get("pack_summary")
        if isinstance(pack, dict):
            n_cand = pack.get("n_candidates", 0) or 0
            n_packed = pack.get("n_packed", 0) or 0
            if n_cand > 0:
                pack_rates.append(n_packed / n_cand)
        else:
            n_no_pack_summary += 1
        candidates = r.get("candidates")
        if not isinstance(candidates, list):
            continue
        for c in candidates:
            if not isinstance(c, dict):
                continue
            decision = c.get("decision")
            if decision == "dropped":
                reason = c.get("reason") or "<unspecified>"
                drop_reasons[str(reason)] += 1
            elif decision == "packed":
                rank = c.get("rank")
                if isinstance(rank, int):
                    packed_ranks[rank] += 1

    return {
        "n_records": n_records,
        "n_truncated": n_truncated,
        "n_no_pack_summary": n_no_pack_summary,
        "pack_rate_mean": (
            sum(pack_rates) / len(pack_rates) if pack_rates else 0.0
        ),
        "pack_rate_p50": _percentile(pack_rates, 50),
        "pack_rate_p90": _percentile(pack_rates, 90),
        "drop_reasons": drop_reasons,
        "packed_ranks": packed_ranks,
    }


def _bucketise_drop_reasons(reasons: Counter[str]) -> Counter[str]:
    """Group reasons by their semantic prefix.

    A reason looks like ``below_floor:0.40`` or
    ``content_hash_collision_with:abc123``. The float / id tail varies
    per call, so the histogram is dominated by uniques unless we
    bucket on the part before ``:``. Reasons without a ``:`` are kept
    as-is.
    """
    out: Counter[str] = Counter()
    for reason, n in reasons.items():
        head = reason.split(":", 1)[0] if ":" in reason else reason
        out[head] += n
    return out


def _print_report(summary: dict, *, paths_read: int) -> None:
    print(f"rebuild_log audit — {paths_read} file(s)")
    print(f"  records:       {summary['n_records']}")
    if summary["n_truncated"]:
        print(
            f"  truncated:     {summary['n_truncated']} "
            "(session(s) hit the 5 MB cap)"
        )
    if summary["n_no_pack_summary"]:
        print(
            f"  malformed:     {summary['n_no_pack_summary']} "
            "(record had no pack_summary)"
        )
    print()
    print("pack rate (n_packed / n_candidates):")
    print(f"  mean: {summary['pack_rate_mean']:.3f}")
    print(f"  p50:  {summary['pack_rate_p50']:.3f}")
    print(f"  p90:  {summary['pack_rate_p90']:.3f}")
    print()
    print("drop-reason histogram (bucketed by prefix):")
    bucketed = _bucketise_drop_reasons(summary["drop_reasons"])
    if not bucketed:
        print("  (none)")
    else:
        for reason, n in bucketed.most_common():
            print(f"  {n:>6}  {reason}")
    print()
    print("packed-rank distribution (where in the candidate list "
          "the packed row was):")
    if not summary["packed_ranks"]:
        print("  (none)")
    else:
        for rank in sorted(summary["packed_ranks"]):
            print(f"  rank {rank:>2}: {summary['packed_ranks'][rank]}")


# Calibration mode delegates to ``aelfrice.eval_harness`` (#365 R4 lift).
# The script's CLI keeps its existing flags, exit codes, and stderr
# wording; only the implementation moved.
def _load_aelfrice_eval_harness():
    """Lazy import so audit mode keeps working without the wheel."""
    from aelfrice import eval_harness  # noqa: PLC0415
    return eval_harness


def _calibration_defaults() -> tuple[Path, int, int]:
    eh = _load_aelfrice_eval_harness()
    return eh.DEFAULT_CALIBRATION_CORPUS, eh.DEFAULT_K, eh.DEFAULT_SEED


def _load_calibration_fixtures(path: Path) -> list[dict]:
    """Back-compat alias for ``aelfrice.eval_harness.load_calibration_fixtures``."""
    return _load_aelfrice_eval_harness().load_calibration_fixtures(path)


def _run_calibration(corpus_path: Path, k: int, seed: int) -> int:
    """Run the #365 R1 calibration harness; print report or error."""
    eh = _load_aelfrice_eval_harness()
    if not corpus_path.is_file():
        print(
            f"audit_rebuild_log: calibration corpus not found: "
            f"{corpus_path}",
            file=sys.stderr,
        )
        return 1

    fixtures = eh.load_calibration_fixtures(corpus_path)
    if not fixtures:
        print(
            f"audit_rebuild_log: corpus is empty: {corpus_path}",
            file=sys.stderr,
        )
        return 1

    report = eh.run_calibration_on_fixtures(fixtures, k=k, seed=seed)
    sys.stdout.write(
        eh.format_calibration_report(
            report, corpus_path=corpus_path, seed=seed,
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    default_corpus, default_k, default_seed = _calibration_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Summarise rebuild_log JSONL files (phase-1c for #288), or "
            "run the relevance-calibration harness on a synthetic "
            "corpus (--calibrate-corpus, #365 R1)."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "One or more JSONL files, or directories containing "
            "*.jsonl session-log files. Required in audit mode; "
            "ignored in calibration mode."
        ),
    )
    parser.add_argument(
        "--calibrate-corpus",
        nargs="?",
        const=default_corpus,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Run the #365 R1 calibration harness against the supplied "
            "synthetic corpus (defaults to "
            f"{default_corpus.name} when no path is given)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=default_k,
        help=(
            "K for P@K in calibration mode "
            f"(default {default_k})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help=(
            "Deterministic seed for noise-belief shuffle in "
            f"calibration mode (default {default_seed})."
        ),
    )
    args = parser.parse_args(argv)

    if args.calibrate_corpus is not None:
        if args.k <= 0:
            print(
                "audit_rebuild_log: --k must be positive",
                file=sys.stderr,
            )
            return 2
        return _run_calibration(args.calibrate_corpus, args.k, args.seed)

    if not args.paths:
        parser.error("paths required in audit mode")

    paths = _collect_paths(args.paths)
    if not paths:
        print(
            "audit_rebuild_log: no readable JSONL inputs",
            file=sys.stderr,
        )
        return 1

    records: list[dict] = []
    for p in paths:
        records.extend(_iter_records(p))

    summary = _summarise(records)
    _print_report(summary, paths_read=len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
