"""Bench gate for #374 — H1 directive detection re-entry.

Per `docs/design/v2_enforcement.md` § H1, H1 unblocks for implementation only when
the candidate detector hits ≥80% precision and ≥60% recall on ≥200 labeled
coding prompts. This test scores `aelfrice.directive_detector.detect_directive`
against the lab-side corpus and asserts the gate.

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset (public CI), when the
`directive_detection/` module dir is missing, or when the corpus has fewer
than 200 rows (the gate requires a 200-row floor before it can fire).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

PRECISION_GATE = 0.80
RECALL_GATE = 0.60
MIN_ROWS = 200

# Deterministic train/test partition for the validity guard below. Keyed on the
# row id so the split is stable across runs and independent of file order.
TRAIN_BUCKET_CEILING = 60

_HEAD_WORD = re.compile(r"^\s*[-*\d.)\s]*([A-Za-z']+)")


def _bucket(row_id: str) -> int:
    return int(hashlib.sha1(row_id.encode()).hexdigest(), 16) % 100


def _head_word(prompt: str) -> str:
    match = _HEAD_WORD.match(prompt)
    return match.group(1).lower() if match else ""


@pytest.mark.bench_gated
def test_directive_detection_gate(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "directive_detection")

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"directive_detection corpus has {len(rows)} rows; gate requires "
            f"≥{MIN_ROWS} per docs/design/v2_enforcement.md § H1"
        )

    from aelfrice.directive_detector import detect_directive

    tp = fp = fn = tn = 0
    for row in rows:
        actual = row["label"] == "directive"
        predicted = detect_directive(row["prompt"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif (not predicted) and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    assert precision >= PRECISION_GATE, (
        f"directive_detection precision {precision:.3f} below "
        f"{PRECISION_GATE} gate (TP={tp}, FP={fp}, FN={fn}, TN={tn}, "
        f"n={len(rows)}); H1 stays deferred per docs/design/v2_enforcement.md § H1"
    )
    assert recall >= RECALL_GATE, (
        f"directive_detection recall {recall:.3f} below {RECALL_GATE} gate "
        f"(TP={tp}, FP={fp}, FN={fn}, TN={tn}, n={len(rows)}); "
        f"H1 stays deferred per docs/design/v2_enforcement.md § H1"
    )


@pytest.mark.bench_gated
def test_directive_corpus_defeats_a_first_token_baseline(
    aelfrice_corpus_root: Path,
) -> None:
    """The gate is only evidence if a degenerate classifier cannot clear it.

    A detector that clears P≥0.80 / R≥0.60 is supposed to have learned the
    distinction between a durable rule and a one-shot task. It has only shown
    that if a classifier which *cannot* represent that distinction fails.

    The weakest such classifier: read the first word of the prompt, look up the
    majority label for that word among the training rows, and answer with it.
    One token, no grammar, no notion of mood, attribution, or durability. If it
    clears the gate on held-out rows, then the corpus separates its two classes
    by opening vocabulary, and any detector keyed on head position scores free
    precision that will not survive contact with real prompts.

    This guard fails when that happens. It is a statement about the corpus, not
    about `detect_directive`.
    """
    rows = load_corpus_module(aelfrice_corpus_root, "directive_detection")

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"directive_detection corpus has {len(rows)} rows; the validity "
            f"guard needs the same ≥{MIN_ROWS} floor as the gate it guards"
        )

    train = [r for r in rows if _bucket(r["id"]) < TRAIN_BUCKET_CEILING]
    held_out = [r for r in rows if _bucket(r["id"]) >= TRAIN_BUCKET_CEILING]

    table: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in train:
        table[_head_word(row["prompt"])][row["label"]] += 1

    def predict(prompt: str) -> bool:
        counts = table.get(_head_word(prompt))
        if not counts:
            return False
        directive = counts["directive"]
        return directive > sum(counts.values()) - directive

    tp = fp = fn = 0
    for row in held_out:
        actual = row["label"] == "directive"
        predicted = predict(row["prompt"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif actual:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    assert not (precision >= PRECISION_GATE and recall >= RECALL_GATE), (
        f"a first-token-only classifier clears the H1 gate on held-out rows "
        f"(P={precision:.3f}, R={recall:.3f}, TP={tp}, FP={fp}, FN={fn}, "
        f"train={len(train)}, held_out={len(held_out)}). The corpus separates "
        f"its classes by opening vocabulary, so clearing the gate is not "
        f"evidence that a detector distinguishes durable rules from one-shot "
        f"tasks. Fix the corpus before reading anything into the gate: see "
        f"docs/design/v2_directive_detection.md § Gate validity."
    )
