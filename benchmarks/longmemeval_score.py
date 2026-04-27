"""LongMemEval scoring: compute accuracy from judge results.

Reads predictions + GT + judge verdicts, reports per-category accuracy.

Usage:
    uv run python benchmarks/longmemeval_score.py \
        /tmp/longmemeval_preds.json /tmp/longmemeval_gt.json /tmp/longmemeval_judge.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: longmemeval_score.py <preds.json> <gt.json> <judge.json>")
        sys.exit(1)

    preds_path: Path = Path(sys.argv[1])
    gt_path: Path = Path(sys.argv[2])
    judge_path: Path = Path(sys.argv[3])

    with preds_path.open("r", encoding="utf-8") as f:
        preds: list[dict[str, object]] = json.load(f)
    with gt_path.open("r", encoding="utf-8") as f:
        gt: list[dict[str, object]] = json.load(f)
    with judge_path.open("r", encoding="utf-8") as f:
        judges: list[dict[str, object]] = json.load(f)

    # Build lookups
    gt_by_id: dict[str, dict[str, object]] = {str(g["question_id"]): g for g in gt}
    judge_by_id: dict[str, dict[str, object]] = {str(j["question_id"]): j for j in judges}

    # Score
    correct_by_type: Counter[str] = Counter()
    total_by_type: Counter[str] = Counter()

    for pred in preds:
        qid: str = str(pred["question_id"])
        gt_entry: dict[str, object] | None = gt_by_id.get(qid)
        judge_entry: dict[str, object] | None = judge_by_id.get(qid)

        if gt_entry is None or judge_entry is None:
            continue

        qtype: str = str(gt_entry.get("question_type", "unknown"))
        verdict: str = str(judge_entry.get("verdict", "incorrect")).lower()
        is_correct: bool = verdict in ("correct", "yes", "true", "1")

        total_by_type[qtype] += 1
        if is_correct:
            correct_by_type[qtype] += 1

    # Report
    print("=" * 60)
    print("LongMemEval Results (Opus binary judge)")
    print("=" * 60)
    print()

    total_correct: int = 0
    total_count: int = 0

    for qtype in sorted(total_by_type.keys()):
        c: int = correct_by_type[qtype]
        t: int = total_by_type[qtype]
        pct: float = c / t * 100 if t > 0 else 0.0
        print(f"  {qtype:35s}  {c:3d}/{t:3d}  ({pct:5.1f}%)")
        total_correct += c
        total_count += t

    overall: float = total_correct / total_count * 100 if total_count > 0 else 0.0
    print(f"\n  {'OVERALL':35s}  {total_correct:3d}/{total_count:3d}  ({overall:5.1f}%)")
    print()
    print("Reference: GPT-4o + LongMemEval_S pipeline = 60.6%")
    print("Note: Judge is Opus (non-standard; paper uses GPT-4o)")


if __name__ == "__main__":
    main()
