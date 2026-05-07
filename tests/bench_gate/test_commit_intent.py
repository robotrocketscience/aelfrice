"""Unit tests for the commit-intent classifier (#438 Path A1)."""

from __future__ import annotations

import pytest

from tests.bench_gate._commit_intent import classify_commit_intent


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Fix #1234: handle empty input", "fix"),
        ("fixes a bug in the parser", "fix"),
        ("Fixed regression introduced in 1.4", "fix"),
        ("Hotfix for production crash", "fix"),
        ("patch the broken header", "fix"),
        ("Correct misleading docstring on Model._do_update", "correction"),
        ("Corrects wrong return-type claim", "correction"),
        ("Typo in error message", "correction"),
        ("docstring was inaccurate; correction here", "correction"),
        ("Revert \"feature: add foo\"", "revert-of-error"),
        ("revert: rollback bad migration", "revert-of-error"),
        ("undo the over-eager refactor", "revert-of-error"),
        ("Backed out commit abc123", "revert-of-error"),
    ],
)
def test_classifies_corrective_messages(message: str, expected: str) -> None:
    assert classify_commit_intent(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "feat: add new endpoint",
        "Add docstring for X",
        "Refactor handler into helper",
        "Update dependencies",
        "Bump version to 1.5.0",
        "Initial commit",
        "",
    ],
)
def test_returns_none_for_non_corrective(message: str) -> None:
    assert classify_commit_intent(message) is None


def test_precedence_revert_over_fix() -> None:
    assert (
        classify_commit_intent("Revert: this commit was wrong, rollback the fix")
        == "revert-of-error"
    )


def test_precedence_correction_over_fix() -> None:
    assert (
        classify_commit_intent("Fix wrong return-type in docstring")
        == "correction"
    )


def test_handles_multiline_message() -> None:
    msg = (
        "feat: add helper\n"
        "\n"
        "This also fixes a regression in the cache lookup.\n"
    )
    assert classify_commit_intent(msg) == "fix"


def test_word_boundary_no_substring_match() -> None:
    # 'prefix' contains 'fix' but should not match.
    assert classify_commit_intent("Add prefix to URL paths") is None
    # 'transparent' contains nothing matching.
    assert classify_commit_intent("Add transparent option to logger") is None
