"""Cohen's-κ inter-rater agreement for eval-judge calibration (#687).

Computes pairwise inter-judge κ across N≥3 independent judge runs over
the same `(expected, actual)` pairs, plus a judge-vs-baseline κ where
the baseline is the zero-LLM substring-exact-match path. Emits the
`judge_kappa.json` artifact specified in `docs/BENCHMARKS.md
§Eval-judge calibration` (PR #700).

Inputs are judge-response JSONL files in the shape written by
`benchmarks.context-rebuilder.judges.llm_judge.write_judge_requests`
plus the operator's dispatch step — one `{"turn_idx": int,
"matched": bool, "rationale": str}` row per line. The baseline is
the same JSONL shape: typically the operator runs
`benchmarks.qa_scoring.score_substring_exact_match` against the
fixture's `(expected, actual)` pairs and writes `matched =
score > 0` per row.

Calibrated verdict:

- ``inter_judge_kappa.min ≥ 0.70`` (min across all run-pairs, not mean)
- ``hot_start_fidelity_mean ≥ 0.80``
- ``n_runs ≥ 3``

All three must hold. The judge-vs-baseline κ is reported but does
NOT participate in the calibrated verdict — a high value means the
judge isn't earning its API cost; a low value is expected (the judge
exists to catch semantic-match cases substring misses).

Pure stdlib — no numpy / scipy dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Mapping

# Calibration thresholds. Ratified 2026-05-11 (#687) — inter-judge κ
# at "substantial agreement" on the Landis-Koch scale; hot-start
# fidelity per #592 AC.
INTER_JUDGE_KAPPA_THRESHOLD: float = 0.70
HOT_START_FIDELITY_THRESHOLD: float = 0.80
MIN_RUNS: int = 3


@dataclass(frozen=True)
class JudgeRun:
    """One judge run loaded from a `judge_responses.jsonl` file.

    `verdicts` maps `turn_idx -> matched`. Run id is the file's stem
    so a directory layout like `judge_1.jsonl` / `judge_2.jsonl`
    becomes `run_1` / `run_2` in the artifact's per-pair κ keys.
    """
    run_id: str
    verdicts: Mapping[int, bool]


@dataclass
class KappaReport:
    """Output of `compute_kappa_report`.

    All κ fields are populated even when calibration fails — operators
    inspect the report to diagnose which run-pair or which fidelity
    sample drove the rejection.
    """
    run_id: str
    n_runs: int
    n_pairs: int
    judge_model: str
    baseline: str
    inter_judge_kappa: dict[str, float] = field(default_factory=dict)
    judge_vs_baseline_kappa: dict[str, float] = field(default_factory=dict)
    per_run_hot_start_fidelity: list[float] = field(default_factory=list)
    hot_start_fidelity_mean: float = 0.0
    calibrated: bool = False
    failure_reasons: list[str] = field(default_factory=list)


def cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Cohen's κ on two binary rater vectors of equal length.

    Returns 1.0 when both raters are identical (covers the "everyone
    agrees the same way" case where p_e=1 and the textbook formula
    is 0/0); 0.0 on empty input. Otherwise the standard
    `(p_o - p_e) / (1 - p_e)` formula on the 2x2 confusion matrix.

    Vectors must be the same length; ValueError otherwise.
    """
    n = len(rater_a)
    if n != len(rater_b):
        raise ValueError(
            f"rater vectors differ in length: {n} vs {len(rater_b)}"
        )
    if n == 0:
        return 0.0
    if rater_a == rater_b:
        # Perfect agreement — handle here so the p_e=1 edge case
        # below doesn't bite. p_o=1, p_e ≤ 1; κ collapses to 1.0
        # by convention.
        return 1.0
    # Confusion matrix counts.
    a = sum(1 for x, y in zip(rater_a, rater_b) if x and y)
    b = sum(1 for x, y in zip(rater_a, rater_b) if x and not y)
    c = sum(1 for x, y in zip(rater_a, rater_b) if not x and y)
    d = sum(1 for x, y in zip(rater_a, rater_b) if not x and not y)
    p_o = (a + d) / n
    # Marginal probabilities.
    p_yes_a = (a + b) / n
    p_yes_b = (a + c) / n
    p_no_a = 1.0 - p_yes_a
    p_no_b = 1.0 - p_yes_b
    p_e = p_yes_a * p_yes_b + p_no_a * p_no_b
    if p_e >= 1.0:
        # Both raters always voted the same single class but their
        # vectors still differ — impossible algebraically, but guard
        # against floating-point edge.
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def load_judge_run(path: Path) -> JudgeRun:
    """Read a `judge_responses.jsonl`-shaped file into a JudgeRun.

    Skips malformed lines (consistent with `read_judge_responses` in
    `llm_judge.py`). Run id is the file stem.
    """
    verdicts: dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            turn_idx = obj.get("turn_idx")
            matched = obj.get("matched")
            if not isinstance(turn_idx, int) or not isinstance(matched, bool):
                continue
            verdicts[turn_idx] = matched
    return JudgeRun(run_id=path.stem, verdicts=verdicts)


def _aligned_vectors(
    runs: list[JudgeRun],
    baseline: JudgeRun | None,
) -> tuple[list[int], list[list[bool]], list[bool] | None]:
    """Return the turn_idx subset present in every run (and baseline
    if provided), plus aligned bool vectors per run.

    κ requires the same item set across raters. The fixture's
    deduplicated pair count drives the eligible turn_idx universe;
    any turn missing from any rater is dropped.
    """
    if not runs:
        return [], [], None
    common: set[int] = set(runs[0].verdicts.keys())
    for r in runs[1:]:
        common &= set(r.verdicts.keys())
    if baseline is not None:
        common &= set(baseline.verdicts.keys())
    aligned_turns = sorted(common)
    run_vecs = [[r.verdicts[t] for t in aligned_turns] for r in runs]
    baseline_vec = (
        [baseline.verdicts[t] for t in aligned_turns]
        if baseline is not None else None
    )
    return aligned_turns, run_vecs, baseline_vec


def compute_kappa_report(
    runs: list[JudgeRun],
    baseline: JudgeRun | None,
    *,
    run_id: str,
    judge_model: str = "<judge-model>",
    baseline_name: str = "score_substring_exact_match",
) -> KappaReport:
    """Build the κ report from N judge runs and an optional baseline.

    The report's `calibrated` boolean encodes the gate from
    `docs/BENCHMARKS.md §Eval-judge calibration`:

    - inter_judge_kappa.min ≥ 0.70
    - hot_start_fidelity_mean ≥ 0.80
    - len(runs) ≥ 3

    Any failing condition records a string in `failure_reasons`.
    """
    aligned_turns, run_vecs, baseline_vec = _aligned_vectors(runs, baseline)
    n_pairs = len(aligned_turns)
    report = KappaReport(
        run_id=run_id,
        n_runs=len(runs),
        n_pairs=n_pairs,
        judge_model=judge_model,
        baseline=baseline_name,
    )

    # Pairwise inter-judge κ (run_i_vs_run_j for every distinct pair).
    pair_kappas: list[float] = []
    for i, j in combinations(range(len(runs)), 2):
        key = f"{runs[i].run_id}_vs_{runs[j].run_id}"
        k = cohens_kappa(run_vecs[i], run_vecs[j])
        report.inter_judge_kappa[key] = round(k, 4)
        pair_kappas.append(k)
    if pair_kappas:
        report.inter_judge_kappa["mean"] = round(
            sum(pair_kappas) / len(pair_kappas), 4
        )
        report.inter_judge_kappa["min"] = round(min(pair_kappas), 4)

    # Judge-vs-baseline κ — one entry per run, plus mean. Reported
    # only; does not feed `calibrated`.
    if baseline_vec is not None:
        baseline_kappas: list[float] = []
        for i, run in enumerate(runs):
            k = cohens_kappa(run_vecs[i], baseline_vec)
            report.judge_vs_baseline_kappa[run.run_id] = round(k, 4)
            baseline_kappas.append(k)
        if baseline_kappas:
            report.judge_vs_baseline_kappa["mean"] = round(
                sum(baseline_kappas) / len(baseline_kappas), 4
            )

    # Per-run hot-start fidelity = matched / n_pairs.
    for vec in run_vecs:
        fidelity = sum(1 for v in vec if v) / len(vec) if vec else 0.0
        report.per_run_hot_start_fidelity.append(round(fidelity, 4))
    if report.per_run_hot_start_fidelity:
        report.hot_start_fidelity_mean = round(
            sum(report.per_run_hot_start_fidelity)
            / len(report.per_run_hot_start_fidelity),
            4,
        )

    # Calibrated verdict.
    reasons: list[str] = []
    if report.n_runs < MIN_RUNS:
        reasons.append(
            f"n_runs={report.n_runs} < required {MIN_RUNS}"
        )
    min_pair_kappa = report.inter_judge_kappa.get("min")
    if min_pair_kappa is None:
        reasons.append("no run pairs to compute inter-judge κ")
    elif min_pair_kappa < INTER_JUDGE_KAPPA_THRESHOLD:
        reasons.append(
            f"inter_judge_kappa.min={min_pair_kappa} "
            f"< threshold {INTER_JUDGE_KAPPA_THRESHOLD}"
        )
    if report.hot_start_fidelity_mean < HOT_START_FIDELITY_THRESHOLD:
        reasons.append(
            f"hot_start_fidelity_mean={report.hot_start_fidelity_mean} "
            f"< threshold {HOT_START_FIDELITY_THRESHOLD}"
        )
    report.failure_reasons = reasons
    report.calibrated = not reasons

    return report


def report_to_json(report: KappaReport) -> dict:
    """Project a KappaReport into the artifact-contract JSON shape
    documented in `docs/BENCHMARKS.md §Eval-judge calibration`.

    Keys match the documented schema exactly so downstream tooling
    (CI gate, release-prep checklist) can rely on stable field names.
    """
    return {
        "run_id": report.run_id,
        "n_runs": report.n_runs,
        "n_pairs": report.n_pairs,
        "judge_model": report.judge_model,
        "baseline": report.baseline,
        "inter_judge_kappa": dict(report.inter_judge_kappa),
        "judge_vs_baseline_kappa": dict(report.judge_vs_baseline_kappa),
        "per_run_hot_start_fidelity": list(report.per_run_hot_start_fidelity),
        "hot_start_fidelity_mean": report.hot_start_fidelity_mean,
        "calibrated": report.calibrated,
        "failure_reasons": list(report.failure_reasons),
    }


def _cli(argv: list[str] | None = None) -> int:
    """`python -m benchmarks.context_rebuilder.kappa` entry point.

    Returns 0 when the report's `calibrated` boolean is True, 1
    otherwise — operators wire this into CI gates by exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.context_rebuilder.kappa",
        description="Compute Cohen's-κ inter-judge agreement (#687).",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Paths to N≥3 judge_responses.jsonl-shaped files.",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to a baseline jsonl (same shape) for judge-vs-baseline κ. "
             "Report-only — does not affect the calibrated verdict.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Write judge_kappa.json to this path. Parent dir created if missing.",
    )
    parser.add_argument(
        "--run-id", default="kappa_run",
        help="Run identifier stamped into the report JSON.",
    )
    parser.add_argument(
        "--judge-model", default="<judge-model>",
        help="Judge model identifier for the report (descriptive only).",
    )
    args = parser.parse_args(argv)

    runs = [load_judge_run(Path(p)) for p in args.runs]
    baseline = load_judge_run(Path(args.baseline)) if args.baseline else None
    report = compute_kappa_report(
        runs, baseline,
        run_id=args.run_id,
        judge_model=args.judge_model,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report_to_json(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report_to_json(report), indent=2, ensure_ascii=False))
    return 0 if report.calibrated else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
