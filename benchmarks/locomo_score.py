"""LoCoMo scoring from predictions JSON.

Reads predictions (with LLM-generated answers) and computes F1 scores
using LoCoMo's exact evaluation methodology.

Usage:
    uv run python benchmarks/locomo_score.py <predictions.json> [--output results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from benchmarks.locomo_adapter import (
    CATEGORY_NAMES,
    score_qa,
)


def score_predictions(predictions_path: str, output_path: str | None = None) -> None:
    """Score predictions and print results."""
    path: Path = Path(predictions_path)
    with path.open("r", encoding="utf-8") as f:
        items: list[dict[str, object]] = json.load(f)

    total_qa: int = 0
    total_f1: float = 0.0
    category_scores: dict[int, list[float]] = {}
    category_counts: dict[int, int] = {}
    scored_items: list[dict[str, object]] = []

    for item in items:
        question: str = str(item["question"])
        answer: str = str(item.get("answer", ""))
        category: int = int(item["category"])  # type: ignore[arg-type]
        prediction: str = str(item.get("llm_prediction", item.get("prediction", "")))

        f1: float = score_qa(prediction, answer, category)

        total_qa += 1
        total_f1 += f1

        if category not in category_scores:
            category_scores[category] = []
            category_counts[category] = 0
        category_scores[category].append(f1)
        category_counts[category] += 1

        scored_items.append({
            "question": question,
            "answer": answer,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "prediction": prediction[:500],
            "f1": round(f1, 4),
        })

    overall_f1: float = total_f1 / total_qa if total_qa > 0 else 0.0

    print(f"\n{'='*60}")
    print("LoCoMo Benchmark Results (retrieval + LLM)")
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
    print("Reference baselines:")
    print("  Filesystem+grep (Letta):  74.0%")
    print("  EverMemOS (SOTA):         92.3%")
    print(f"{'='*60}")

    if output_path:
        out_data: dict[str, object] = {
            "mode": "retrieval + LLM",
            "overall_f1": round(overall_f1, 4),
            "total_qa": total_qa,
            "category_f1": {
                str(cat): round(sum(scores) / len(scores), 4)
                for cat, scores in sorted(category_scores.items())
            },
            "per_question": scored_items,
        }
        out_path: Path = Path(output_path)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nDetailed results written to {output_path}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Score LoCoMo predictions",
    )
    parser.add_argument("predictions", help="Path to predictions JSON")
    parser.add_argument("--output", default=None, help="Write detailed results JSON")
    args: argparse.Namespace = parser.parse_args()
    score_predictions(args.predictions, args.output)


if __name__ == "__main__":
    main()
