"""MAB LLM reader: generates answers from retrieval context.

Takes a retrieval JSON file (question + context pairs) and sends each
to an LLM reader to generate answers. Writes predictions file compatible
with exp5_score.py.

Usage:
    uv run python benchmarks/mab_reader.py /tmp/exp6_temporal.json --model opus
    uv run python benchmarks/mab_reader.py /tmp/exp6_temporal.json --model haiku
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTCONSOLIDATION_PROMPT: Final[str] = (
    "Pretend you are a knowledge management system. Each fact in the "
    "knowledge pool is provided with a serial number at the beginning, "
    "and the newer fact has larger serial number. You need to solve the "
    "conflicts of facts in the knowledge pool by finding the newest fact "
    "with larger serial number. You need to answer a question based on "
    "this rule. You should give a very concise answer without saying "
    "other words for the question **only** from the knowledge pool you "
    "have memorized rather than the real facts in real world."
)

MODEL_MAP: Final[dict[str, str]] = {
    "opus": "claude-opus-4-20250514",
    "haiku": "claude-haiku-4-5-20251001",
}


def read_question(
    client: anthropic.Anthropic,
    question: str,
    context: str,
    model_id: str,
) -> str:
    """Send question + context to LLM reader, return concise answer."""
    if not context.strip():
        return "unknown"

    user_msg: str = (
        f"Knowledge pool:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely with only the answer, no explanation."
    )

    response = client.messages.create(
        model=model_id,
        max_tokens=100,
        system=FACTCONSOLIDATION_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    block: object = response.content[0]
    text: str = str(getattr(block, "text", "unknown")).strip()
    return text


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MAB LLM reader: generate answers from retrieval context",
    )
    parser.add_argument(
        "retrieval_file",
        help="Path to retrieval JSON (from mab_entity_index_adapter.py)",
    )
    parser.add_argument(
        "--model", default="opus", choices=list(MODEL_MAP.keys()),
        help="Reader model (default: opus)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output predictions path (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Print progress every N questions",
    )
    args: argparse.Namespace = parser.parse_args()

    retrieval_path: Path = Path(args.retrieval_file)
    with retrieval_path.open("r", encoding="utf-8") as f:
        items: list[dict[str, object]] = json.load(f)

    model_id: str = MODEL_MAP[args.model]
    print(f"Reader: {args.model} ({model_id})")
    print(f"Questions: {len(items)}")

    # Output path
    if args.output:
        out_path: Path = Path(args.output)
    else:
        stem: str = retrieval_path.stem
        out_path = retrieval_path.with_name(f"{stem}_preds_{args.model}.json")

    client: anthropic.Anthropic = anthropic.Anthropic()

    predictions: list[dict[str, object]] = []
    t0: float = time.monotonic()

    for i, item in enumerate(items):
        question: str = str(item["question"])
        context: str = str(item["context"])
        qid: int = int(item["id"])  # type: ignore[arg-type]

        answer: str = read_question(client, question, context, model_id)
        predictions.append({"id": qid, "llm_prediction": answer})

        if (i + 1) % args.batch_size == 0:
            elapsed: float = time.monotonic() - t0
            rate: float = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(items)}] {rate:.1f} q/s - last: {answer[:60]}")

    elapsed = time.monotonic() - t0
    print(f"\nDone: {len(items)} questions in {elapsed:.1f}s ({len(items)/elapsed:.1f} q/s)")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
