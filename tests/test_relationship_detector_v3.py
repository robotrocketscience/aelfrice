"""Integration tests for the v3 value-comparison gate (#422).

Pins the contract that ``use_value_comparison=True`` activates the
typed-slot gate ahead of the residual-overlap floor, and that
``use_value_comparison=False`` (the default) preserves v1 behaviour
byte-for-byte.
"""
from __future__ import annotations

import pytest

from aelfrice.relationship_detector import (
    LABEL_CONTRADICTS,
    LABEL_UNRELATED,
    analyze,
    classify,
)


def test_v1_default_unchanged_for_paraphrase_pair() -> None:
    """The case that motivated #422: numeric mismatch with low token
    overlap. v1 misses it (this is the documented failure mode)."""
    a = "alpha = 0.5 prior"
    b = "alpha = 1.0 in config"
    # v1 (default) — falls below residual_overlap floor → unrelated.
    assert classify(a, b) == LABEL_UNRELATED


def test_v3_catches_numeric_paraphrase_contradiction() -> None:
    a = "alpha = 0.5 prior"
    b = "alpha = 1.0 in config"
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.label == LABEL_CONTRADICTS
    assert verdict.score == 1.0
    assert "value_comparison" in verdict.rationale
    assert "numeric" in verdict.rationale


def test_v3_catches_enum_paraphrase_contradiction() -> None:
    a = "the pipeline runs synchronous on hot path"
    b = "async execution model used here"
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.label == LABEL_CONTRADICTS
    assert "execution_mode" in verdict.rationale


def test_v3_unrelated_pair_still_unrelated() -> None:
    """No slots, no overlap → still ``unrelated`` even with flag on."""
    a = "totally unrelated text here"
    b = "something else entirely"
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.label == LABEL_UNRELATED


def test_v3_alias_pair_does_not_falsely_contradict() -> None:
    """``sync`` and ``synchronous`` are aliases, not opposites."""
    a = "use sync mode"
    b = "synchronous everywhere"
    # v3 should NOT emit contradicts. With short texts there's also
    # too little residual overlap for v1 to call it ``refines``, so
    # ``unrelated`` is the expected verdict here.
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.label != LABEL_CONTRADICTS


def test_v3_within_tolerance_numeric_does_not_falsely_contradict() -> None:
    """0.5 vs 0.502 → within tolerance → no slot conflict → falls
    through to v1 modality pass."""
    a = "alpha is 0.5"
    b = "alpha is 0.502"
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.label != LABEL_CONTRADICTS


def test_classify_passes_flag_through() -> None:
    """``classify`` is the bench-gate entry; the kwarg must reach
    ``analyze`` unmodified. Use a pair where v1 returns the floor
    verdict (``unrelated``) to make the flag-on flip unambiguous."""
    a = "alpha = 0.5 prior"
    b = "alpha = 1.0 in config"
    assert classify(a, b) == LABEL_UNRELATED
    assert classify(a, b, use_value_comparison=True) == LABEL_CONTRADICTS


def test_v3_score_pinned_at_one_when_slot_fires() -> None:
    """Pinning score=1.0 lets the auto-emit policy in #422 acceptance
    #3 use a single threshold for slot-fire vs modality-fire."""
    a = "full backup nightly"
    b = "incremental backup nightly"
    verdict = analyze(a, b, use_value_comparison=True)
    assert verdict.score == 1.0


def test_v3_does_not_run_when_flag_off() -> None:
    """Negative test: v1 path stays untouched. We assert the rationale
    string never carries the v3 prefix when the flag is off."""
    a = "alpha = 0.5 in synchronous mode"
    b = "alpha = 1.0 in async mode"
    verdict = analyze(a, b, use_value_comparison=False)
    assert "value_comparison" not in verdict.rationale
