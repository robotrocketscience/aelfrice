"""Deterministic QA correctness scorers for benchmark adapters.

Pure stdlib leaf module. Mirrors the MemoryAgentBench paper's scoring
functions (no Porter stemming) so adapters that currently emit only
retrieval statistics can fold in answer-correctness without an LLM
judge in the canonical dispatcher path.

Spec: #507 acceptance criteria. The exposed surfaces are:

- `normalize_answer(s)` — lowercase, drop punctuation, drop articles,
  collapse whitespace.
- `score_exact_match(prediction, ground_truth)` — normalized strings
  must be identical. Returns 1.0 / 0.0.
- `score_substring_exact_match(prediction, ground_truth)` — normalized
  ground truth must appear inside normalized prediction. Returns
  1.0 / 0.0.
- `score_f1(prediction, ground_truth)` — token-level F1 on normalized
  whitespace splits. Returns float in [0, 1].
- `score_multi_answer(prediction, ground_truths)` — best-of across a
  list of acceptable ground truths. Returns the three metrics keyed by
  name.

Determinism: every function is a pure transformation over its inputs.
Two invocations on identical inputs produce identical outputs by
construction (no RNG, no hashing, no time, no I/O).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Final

_ARTICLE_RE: Final[re.Pattern[str]] = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE: Final[dict[int, str | None]] = str.maketrans(
    "", "", string.punctuation,
)


def normalize_answer(s: str) -> str:
    """Normalize: lowercase, strip punctuation, drop articles, collapse spaces."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def score_exact_match(prediction: str, ground_truth: str) -> float:
    """Normalized strings must be identical."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def score_substring_exact_match(prediction: str, ground_truth: str) -> float:
    """Normalized ground truth appears as a substring of normalized prediction."""
    norm_gt: str = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt in normalize_answer(prediction) else 0.0


def score_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 on normalized whitespace splits."""
    pred_tokens: list[str] = normalize_answer(prediction).split()
    gt_tokens: list[str] = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: Counter[str] = Counter(pred_tokens) & Counter(gt_tokens)
    num_common: int = sum(common.values())
    if num_common == 0:
        return 0.0
    precision: float = num_common / len(pred_tokens)
    recall: float = num_common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_multi_answer(
    prediction: str,
    ground_truths: list[str],
) -> dict[str, float]:
    """Best-of EM / SEM / F1 across a list of acceptable ground truths.

    Empty list → all three metrics are 0.0.
    """
    best_em: float = 0.0
    best_sem: float = 0.0
    best_f1: float = 0.0
    for gt in ground_truths:
        if score_exact_match(prediction, gt) > best_em:
            best_em = 1.0
        if score_substring_exact_match(prediction, gt) > best_sem:
            best_sem = 1.0
        f1: float = score_f1(prediction, gt)
        if f1 > best_f1:
            best_f1 = f1
    return {
        "exact_match": best_em,
        "substring_exact_match": best_sem,
        "f1": best_f1,
    }
