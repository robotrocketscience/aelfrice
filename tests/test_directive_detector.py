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


# Path A: head-position coding-task prefix short-circuits to False even when
# a downstream imperative verb would otherwise fire. Each row embeds a verb
# from the imperative bank ("never", "ensure", "must", "only", "avoid", …)
# so the test would pass under the old detector if Path A weren't applied —
# making the test load-bearing for the new branch.
@pytest.mark.parametrize(
    "text",
    [
        "Refactor X so it never blocks",
        "Add a test that ensures the gate fires",
        "Implement the parser so it must reject empty input",
        "Write a guard that always returns False on the empty case",
        "Create a wrapper that should not propagate exceptions",
        "Update the README so it only mentions the public API",
        "Fix the pre-push hook to avoid bypassing on rebase",
        "Make the worker shutdown ensure no half-flushed batches",
        "Build a fixture that requires the v0.1 corpus path",
        "Remove the dead branch before merging",
        "Rename _emit so it cannot collide with _emit_core",
        "Extract the helper unless the call site needs inlining",
        "Merge the two threads after the gate passes",
        "Split the test so each case must check exactly one signal",
        "Move the docstring before the type annotations",
        "Delete the cache whenever the schema bumps",
    ],
)
def test_directive_coding_task_prefix_short_circuits(text: str) -> None:
    assert detect_directive(text) is False


# Regression: leading deontic anchors and durable rules where the head verb
# is NOT in the coding-task bank still classify True. The prefix filter is
# case-insensitive but positional, and only fires on the head verb.
@pytest.mark.parametrize(
    "text",
    [
        "always update the changelog before tagging",
        "never delete a worktree without releasing the claim first",
        "must rename the temp file before commit",
        "only merge after the gate passes",
        "before merging, ensure CI is green",
    ],
)
def test_directive_prefix_filter_does_not_swallow_rules(text: str) -> None:
    assert detect_directive(text) is True
