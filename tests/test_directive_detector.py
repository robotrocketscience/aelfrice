"""Sanity tests for `aelfrice.directive_detector` (#374 H1 candidate).

These are NOT the bench gate — those live in `tests/bench_gate/` and need
the lab corpus. These cover the four spec-mandated filter behaviors so the
detector's branching logic is exercised in public CI.
"""
from __future__ import annotations

import pytest

from aelfrice.directive_detector import detect_directive


@pytest.mark.parametrize(
    "text",
    [
        "never push directly to main",
        "always sign your commits",
        "must use uv for python deps",
        "don't add co-authorship to commits",
        "do not commit large data files",
        "avoid bypassing pre-push hooks",
        "before merging, run the staging gate",
        "unless the user asks, do not force-push",
    ],
)
def test_directive_positives(text: str) -> None:
    assert detect_directive(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what does the staging gate check?",
        "should we rebase or merge here?",
        "I never push to main when I'm tired",
        "I always check git status because the worktrees confuse me",
        "maybe we should never use force-push",
        "I think we must rebase",
        "the deploy ran fine",
        "looks good",
    ],
)
def test_directive_negatives(text: str) -> None:
    assert detect_directive(text) is False
