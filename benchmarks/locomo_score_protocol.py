"""Protocol-correct LoCoMo scoring.

Reads predictions JSON and ground truth JSON (separate files).
Handles category 5 forced-choice scoring per the original protocol.

Usage:
    uv run python benchmarks/locomo_score_protocol.py <predictions.json> <ground_truth.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.locomo_adapter import (
    CATEGORY_NAMES,
    f1_multi_hop,
    f1_score_single,
)


def _score_cat5(prediction: str, gt: dict[str, object]) -> float:
    """Score category 5 using forced-choice protocol.

    The ground truth has cat5_a, cat5_b, and cat5_correct.
    If model picks the letter corresponding to 'Not mentioned',
    score is 1.0. Otherwise 0.0.
    """
    pred_lower: str = prediction.strip().lower()
    correct_letter: str = str(gt.get("cat5_correct", ""))

    # Direct letter answer: "a" or "b" or "(a)" or "(b)"
    if len(pred_lower) <= 3:
        cleaned: str = pred_lower.replace("(", "").replace(")", "").strip()
        if cleaned == correct_letter:
            return 1.0
        if cleaned in ("a", "b"):
            return 0.0

    # Check if the prediction contains "not mentioned"
    if "not mentioned" in pred_lower or "no information available" in pred_lower:
        return 1.0

    # Check if prediction matches the correct option text
    correct_text: str = str(gt.get(f"cat5_{correct_letter}", "")).lower()
    if correct_text and correct_text in pred_lower:
        return 1.0

    return 0.0


def score_predictions(
    predictions_path: str,
    ground_truth_path: str,
    output_path: str | None = None,
) -> None:
    """Score predictions against ground truth."""
    with Path(predictions_path).open("r", encoding="utf-8") as f:
        preds: list[dict[str, object]] = json.load(f)

    with Path(ground_truth_path).open("r", encoding="utf-8") as f:
        gt_list: list[dict[str, object]] = json.load(f)

    # Index ground truth by id
    gt_by_id: dict[int, dict[str, object]] = {int(g["id"]): g for g in gt_list}  # type: ignore[arg-type]

    total_qa: int = 0
    total_f1: float = 0.0
    category_scores: dict[int, list[float]] = {}
    category_counts: dict[int, int] = {}
    scored_items: list[dict[str, object]] = []

    for pred_item in preds:
        item_id: int = int(pred_item["id"])  # type: ignore[arg-type]
        prediction: str = str(pred_item.get("llm_prediction", ""))
        gt: dict[str, object] = gt_by_id[item_id]

        category: int = int(gt["category"])  # type: ignore[arg-type]
        answer: str = str(gt.get("answer", ""))

        # Score based on category
        if category == 5:
            f1: float = _score_cat5(prediction, gt)
        elif category == 1:
            f1 = f1_multi_hop(prediction, answer)
        elif category == 3:
            answer = answer.split(";")[0].strip()
            f1 = f1_score_single(prediction, answer)
        else:
            f1 = f1_score_single(prediction, answer)

        total_qa += 1
        total_f1 += f1

        if category not in category_scores:
            category_scores[category] = []
            category_counts[category] = 0
        category_scores[category].append(f1)
        category_counts[category] += 1

        scored_items.append({
            "id": item_id,
            "question": str(gt.get("question", "")),
            "answer": answer,
            "category": category,
            "prediction": prediction[:500],
            "f1": round(f1, 4),
        })

    overall_f1: float = total_f1 / total_qa if total_qa > 0 else 0.0

    print(f"\n{'='*60}")
    print("LoCoMo Benchmark Results (protocol-correct)")
    print(f"{'='*60}")
    print(f"Total QA pairs:    {total_qa}")
    print(f"Overall F1:        {overall_f1:.4f} ({overall_f1*100:.1f}%)")
    print()
    print("Per-category F1:")
    for cat in sorted(category_scores.keys()):
        name: str = CATEGORY_NAMES.get(cat, "unknown")
        count: int = category_counts.get(cat, 0)
        scores: list[float] = category_scores[cat]
        cat_f1: float = sum(scores) / len(scores) if scores else 0.0
        print(f"  {cat}. {name:12s}  {cat_f1:.4f} ({cat_f1*100:.1f}%)  n={count}")
    print()
    print("Reference baselines (from LoCoMo paper):")
    print("  Human:                    87.9%")
    print("  GPT-4-turbo (128K):       51.6%")
    print("  RAG (DRAGON+gpt-3.5):     43.3%")
    print(f"{'='*60}")

    if output_path:
        out_data: dict[str, object] = {
            "mode": "protocol-correct, aelfrice retrieval + Opus reader",
            "overall_f1": round(overall_f1, 4),
            "total_qa": total_qa,
            "category_f1": {
                str(cat): round(sum(s) / len(s), 4)
                for cat, s in sorted(category_scores.items())
            },
            "per_question": scored_items,
        }
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nDetailed results written to {output_path}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Score LoCoMo predictions (protocol-correct)",
    )
    parser.add_argument("predictions", help="Path to predictions JSON")
    parser.add_argument("ground_truth", help="Path to ground truth JSON")
    parser.add_argument("--output", default=None, help="Write detailed results JSON")
    args: argparse.Namespace = parser.parse_args()
    score_predictions(args.predictions, args.ground_truth, args.output)


if __name__ == "__main__":
    main()
