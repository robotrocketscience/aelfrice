"""Verify a retrieval file contains no ground truth contamination.

This is the gatekeeper. If this script fails, the benchmark run is
INVALID. No exceptions. Fix the adapter and re-run.

Usage:
    uv run python benchmarks/verify_clean.py <retrieval_file.json>
    uv run python benchmarks/verify_clean.py --all /tmp/benchmark_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

BANNED_KEYS: Final[frozenset[str]] = frozenset({
    # Direct answer fields
    "answer",
    "answers",
    "answer_raw",
    "reference_answer",
    "ground_truth",
    "gt",
    "gold",
    "target",
    "expected",
    "solution",
    "correct",
    "correct_answer",
    "label",
    # Score fields (should not be in retrieval output)
    "score",
    "f1",
    "exact_match",
    "substring_exact_match",
    "accuracy",
    "rouge",
    "bleu",
    # Judgment fields
    "is_correct",
    "judgment",
    "eval_score",
})


def verify_file(path: str) -> bool:
    """Check a single file for contamination.

    Returns True if clean, False if contaminated.
    """
    file_path: Path = Path(path)
    if not file_path.exists():
        print(f"FILE NOT FOUND: {path}")
        return False

    with file_path.open("r", encoding="utf-8") as f:
        data: list[dict[str, object]] = json.load(f)

    if not data:
        print(f"EMPTY FILE: {path}")
        return False

    # Collect all keys across all items
    all_keys: set[str] = set()
    for item in data:
        all_keys.update(item.keys())

    # Check for banned keys
    leaked: set[str] = all_keys & BANNED_KEYS
    if leaked:
        print(f"CONTAMINATION DETECTED: {path}")
        print(f"  Banned keys found: {sorted(leaked)}")
        print(f"  All keys present:  {sorted(all_keys)}")
        print(f"  Items in file:     {len(data)}")
        print()
        print("  VERDICT: INVALID")
        print("  The retrieval file contains ground truth or scoring data.")
        print("  Any results derived from this file are automatically 0%.")
        print("  Fix the adapter to write answers to a separate _gt.json file,")
        print("  then re-run the entire benchmark from scratch.")
        return False

    # Additional check: scan values for suspiciously short items
    # that might be answer strings smuggled in via differently-named keys
    suspicious_keys: list[str] = []
    for key in all_keys:
        if key in ("id", "question", "context", "retrieved_context",
                    "question_id", "question_type", "question_date",
                    "source", "row_idx", "q_idx", "case_id", "task",
                    "domain", "task_type", "episode_id", "qa_type",
                    "qa_type_name", "question_uuid",
                    "num_beliefs", "retrieval_latency_ms"):
            continue  # Known safe keys
        # Flag any unknown key for manual review
        suspicious_keys.append(key)

    if suspicious_keys:
        print(f"WARNING: {path}")
        print(f"  Unknown keys found: {sorted(suspicious_keys)}")
        print("  These are not in the banned list but should be reviewed.")
        print("  Verify they do not contain answer information.")
        print()

    print(f"CLEAN: {path}")
    print(f"  Keys: {sorted(all_keys)}")
    print(f"  Items: {len(data)}")
    return True


def main() -> None:
    """Verify one or more retrieval files."""
    if len(sys.argv) < 2:
        print("Usage: verify_clean.py <file.json> [<file2.json> ...]")
        print("       verify_clean.py --all /tmp/benchmark_*.json")
        sys.exit(1)

    files: list[str] = sys.argv[1:]
    all_clean: bool = True

    for path in files:
        if path == "--all":
            continue
        clean: bool = verify_file(path)
        if not clean:
            all_clean = False
        print()

    if all_clean:
        print("ALL FILES CLEAN")
        sys.exit(0)
    else:
        print("CONTAMINATION FOUND. Results are INVALID.")
        sys.exit(1)


if __name__ == "__main__":
    main()
