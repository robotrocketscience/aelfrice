"""LoCoMo answer generation from retrieved context.

Reads the retrieval results JSON from locomo_adapter.py --retrieve-only,
generates answers for each question from context, and writes predictions.

This script is designed to be run by a Claude Code subagent (model=haiku)
which generates answers natively without needing an API key.

Usage (called by subagent, not directly):
    Read the retrieval JSON, generate answers, write predictions JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_answers(retrieval_path: str, output_path: str) -> None:
    """Read retrieval results and write a predictions file.

    This function is a no-op placeholder. The actual answer generation
    happens in the subagent prompt, which reads the retrieval JSON,
    generates answers using its native LLM capability, and writes
    the predictions JSON.
    """
    path: Path = Path(retrieval_path)
    with path.open("r", encoding="utf-8") as f:
        items: list[dict[str, object]] = json.load(f)

    print(f"Loaded {len(items)} questions from {retrieval_path}")
    print(f"Output will be written to {output_path}")
    print("This script is a schema reference. Run via subagent for actual generation.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: locomo_generate.py <retrieval.json> <predictions.json>")
        sys.exit(1)
    generate_answers(sys.argv[1], sys.argv[2])
