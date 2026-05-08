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
import random
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from aelfrice.calibration_metrics import CalibrationReport


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


DEFAULT_CALIBRATION_CORPUS = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "posterior_ranking"
    / "fixtures"
    / "default.jsonl"
)
DEFAULT_CALIBRATION_K = 10
DEFAULT_CALIBRATION_SEED = 0


def _load_calibration_fixtures(path: Path) -> list[dict]:
    """Load a JSONL calibration corpus.

    Each line must decode to a dict carrying ``id``, ``query``,
    ``known_belief_content``, and ``noise_belief_contents`` (a list).
    Fixtures missing any required key, or with malformed JSON, are
    silently skipped — same fail-soft posture as the audit reader.
    """
    out: list[dict] = []
    text = path.read_text(encoding="utf-8")
    required = ("id", "query", "known_belief_content", "noise_belief_contents")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if not all(key in row for key in required):
            continue
        if not isinstance(row["noise_belief_contents"], list):
            continue
        out.append(row)
    return out


def _build_calibration_store(fixture: dict, seed: int):
    """Build a fresh in-memory ``MemoryStore`` for one fixture.

    Imports `aelfrice` lazily and inserts one belief per content row
    (one relevant + N noise). The noise order is shuffled with the
    supplied seed so AUC / ρ aggregates are deterministic across
    reruns at fixed seed (per the #365 ship gate).
    """
    from aelfrice.models import (  # noqa: PLC0415
        BELIEF_FACTUAL,
        LOCK_NONE,
        Belief,
    )
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(":memory:")
    fid = str(fixture["id"])
    known_content = str(fixture["known_belief_content"])
    noise_contents = list(fixture["noise_belief_contents"])

    def make_belief(bid: str, content: str) -> Belief:
        return Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=0.5,
            beta=0.5,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        )

    rng = random.Random(seed)
    rng.shuffle(noise_contents)

    store.insert_belief(make_belief(f"{fid}_known", known_content))
    for i, nc in enumerate(noise_contents):
        store.insert_belief(make_belief(f"{fid}_noise_{i}", nc))

    return store


def _run_calibration(
    corpus_path: Path, k: int, seed: int,
) -> int:
    """Run the #365 R1 calibration harness against a synthetic corpus.

    For each fixture in the corpus, build an in-memory store with one
    relevant belief and N noise beliefs, run ``aelfrice.retrieval.retrieve``
    against the query, and observe the rank of the relevant belief.
    Aggregate (rank-as-score, is_relevant) observations across queries
    and report P@K / ROC-AUC / Spearman ρ.
    """
    # Imported lazily so the audit-mode CLI does not need the package
    # installed, only the calibrate path does.
    from aelfrice.calibration_metrics import (  # noqa: PLC0415
        CalibrationReport,
        precision_at_k,
        roc_auc,
        spearman_rho,
    )
    from aelfrice.retrieval import retrieve  # noqa: PLC0415

    if not corpus_path.is_file():
        print(
            f"audit_rebuild_log: calibration corpus not found: "
            f"{corpus_path}",
            file=sys.stderr,
        )
        return 1

    fixtures = _load_calibration_fixtures(corpus_path)
    if not fixtures:
        print(
            f"audit_rebuild_log: corpus is empty: {corpus_path}",
            file=sys.stderr,
        )
        return 1

    p_at_k_values: list[float] = []
    n_truncated = 0
    pooled_scores: list[float] = []
    pooled_labels: list[bool] = []

    for fx in fixtures:
        store = _build_calibration_store(fx, seed)
        query = str(fx["query"])
        known_content = str(fx["known_belief_content"])
        n_candidates = 1 + len(fx["noise_belief_contents"])  # type: ignore[arg-type]

        results = retrieve(
            store,
            query,
            l1_limit=max(k, n_candidates),
            entity_index_enabled=False,
            bfs_enabled=False,
            posterior_weight=None,
        )

        relevance_top_k = [b.content == known_content for b in results]
        if len(relevance_top_k) < k:
            n_truncated += 1
        p_at_k_values.append(precision_at_k(relevance_top_k, k))

        # Rank-as-score for AUC / ρ: top of the list maps to the
        # highest score so AUC reads "score increases with relevance"
        # naturally. Candidates the retriever did not surface get
        # score 0 (lower than any returned), preserving the corpus's
        # full positive/negative balance in the pooled observations.
        for rank_idx, belief in enumerate(results):
            pooled_scores.append(float(len(results) - rank_idx))
            pooled_labels.append(belief.content == known_content)
        retrieved_contents = {b.content for b in results}
        for noise_content in fx["noise_belief_contents"]:  # type: ignore[union-attr]
            if noise_content not in retrieved_contents:
                pooled_scores.append(0.0)
                pooled_labels.append(False)
        if known_content not in retrieved_contents:
            pooled_scores.append(0.0)
            pooled_labels.append(True)

        store.close()

    avg_p_at_k = (
        sum(p_at_k_values) / len(p_at_k_values) if p_at_k_values else 0.0
    )
    auc = roc_auc(pooled_scores, pooled_labels)
    rho = spearman_rho(
        pooled_scores, [1.0 if x else 0.0 for x in pooled_labels],
    )

    report = CalibrationReport(
        p_at_k=avg_p_at_k,
        k=k,
        n_queries=len(fixtures),
        n_truncated_queries=n_truncated,
        roc_auc=auc,
        spearman_rho=rho,
        n_observations=len(pooled_scores),
    )
    _print_calibration_report(report, corpus_path=corpus_path, seed=seed)
    return 0


def _format_optional_float(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a (undefined)"


def _print_calibration_report(
    report: CalibrationReport, *, corpus_path: Path, seed: int,
) -> None:
    print(f"calibration harness — corpus {corpus_path.name}")
    print(f"  n_queries:    {report.n_queries}")
    print(f"  n_obs:        {report.n_observations}")
    print(f"  seed:         {seed}")
    if report.n_truncated_queries:
        print(
            f"  truncated:    {report.n_truncated_queries} "
            f"(query returned <{report.k} candidates)"
        )
    print()
    print(f"P@{report.k}:        {report.p_at_k:.4f}")
    print(f"ROC-AUC:      {_format_optional_float(report.roc_auc)}")
    print(f"Spearman ρ:   {_format_optional_float(report.spearman_rho)}")


def main(argv: list[str] | None = None) -> int:
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
        const=DEFAULT_CALIBRATION_CORPUS,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Run the #365 R1 calibration harness against the supplied "
            "synthetic corpus (defaults to "
            f"{DEFAULT_CALIBRATION_CORPUS.name} when no path is given)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_CALIBRATION_K,
        help=(
            "K for P@K in calibration mode "
            f"(default {DEFAULT_CALIBRATION_K})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CALIBRATION_SEED,
        help=(
            "Deterministic seed for noise-belief shuffle in "
            f"calibration mode (default {DEFAULT_CALIBRATION_SEED})."
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
