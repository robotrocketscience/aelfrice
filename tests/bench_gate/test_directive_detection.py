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

# The guard sweeps this many salted partitions rather than asserting on one.
# A single partition decides precision on ~34 positive predictions, whose 95%
# interval straddles the gate — so one draw can clear it on an unchanged corpus
# and read as "corpus fixed" (#1349). Measured over K=200 on the corpora that
# exist: v0.1 lets the baseline clear on 196/200 partitions, v0.1+v0.2 on 0/200
# with a maximum precision of 0.754. The two populations are far enough apart
# that the exact K is not delicate.
#
# One property of the "clears on any of K" rule to hold on to before changing
# this number: it is monotone in K. The salts are `s0..s{K-1}`, so raising K
# only adds candidates — the guard can become redder, never greener. The
# v0.1+v0.2 result is therefore a K=200 statement, and its margin (worst
# partition 0.754 against a 0.800 gate) is the max of K draws rather than a
# property of the distribution; the median separation quoted above does not
# bound it. A session that raises K and sees the union corpus go red should
# read that as the tail being sampled further out, report the new count, and
# not lower K back to hide it.
PARTITION_SWEEP_K = 200

_HEAD_WORD = re.compile(r"^\s*[-*\d.)\s]*([A-Za-z']+)")


def _bucket(row_id: str, salt: str = "") -> int:
    return int(hashlib.sha1((salt + row_id).encode()).hexdigest(), 16) % 100


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

    It sweeps `PARTITION_SWEEP_K` salted partitions and fails if the baseline
    clears the gate on **any** of them. One partition is not enough evidence:
    precision there rests on ~34 positive predictions and its 95% interval
    straddles the gate, so a re-partition alone can turn a single-draw guard
    green while the head-word correlation is untouched (#1349). Failing on any
    partition is the conservative direction — a false alarm costs a corpus
    inspection, a false clearance blesses an overfit detector.
    """
    rows = load_corpus_module(aelfrice_corpus_root, "directive_detection")

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"directive_detection corpus has {len(rows)} rows; the validity "
            f"guard needs the same ≥{MIN_ROWS} floor as the gate it guards"
        )

    def score(salt: str) -> tuple[float, float, int, int, int]:
        train, held_out = [], []
        for row in rows:
            bucket = _bucket(row["id"], salt)
            (train if bucket < TRAIN_BUCKET_CEILING else held_out).append(row)

        table: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for row in train:
            table[_head_word(row["prompt"])][row["label"]] += 1

        tp = fp = fn = 0
        for row in held_out:
            counts = table.get(_head_word(row["prompt"]))
            directive = counts["directive"] if counts else 0
            predicted = bool(counts) and directive > sum(counts.values()) - directive
            actual = row["label"] == "directive"
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif actual:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return precision, recall, tp, fp, fn

    results = [score(f"s{i}") for i in range(PARTITION_SWEEP_K)]
    clearing = [
        (i, p, r)
        for i, (p, r, _, _, _) in enumerate(results)
        if p >= PRECISION_GATE and r >= RECALL_GATE
    ]
    # Report a partition that actually cleared, not the globally
    # highest-precision one: `max` by precision alone can name a partition whose
    # recall is under the floor, i.e. one that did not clear, offered as the
    # evidence that something did. Only fall back to the global max when nothing
    # cleared, where the message does not render anyway.
    cleared_idx = {i for i, _, _ in clearing}
    worst = max(
        (t for i, t in enumerate(results) if i in cleared_idx),
        key=lambda t: t[0],
        default=max(results, key=lambda t: t[0]),
    )

    assert not clearing, (
        f"a first-token-only classifier clears the H1 gate on "
        f"{len(clearing)}/{PARTITION_SWEEP_K} salted partitions of this corpus "
        f"(n={len(rows)} rows). Worst partition: P={worst[0]:.3f}, R={worst[1]:.3f} "
        f"(TP={worst[2]}, FP={worst[3]}, FN={worst[4]}). The corpus separates its "
        f"classes by opening vocabulary, so clearing the gate is not evidence that "
        f"a detector distinguishes durable rules from one-shot tasks. Fix the "
        f"corpus before reading anything into the gate: see "
        f"docs/design/v2_directive_detection.md § Gate validity."
    )
