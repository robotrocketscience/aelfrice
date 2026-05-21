"""Linked-issue extraction for <recent-work> sub-block (#887)."""
from __future__ import annotations

from aelfrice.hook import _extract_linked_issues


def test_branch_issue_slug() -> None:
    assert _extract_linked_issues("feat/issue-887-recent-work", []) == ["#887"]


def test_branch_issues_slug_plural() -> None:
    assert _extract_linked_issues("fix/issues/42-thing", []) == ["#42"]


def test_subject_hash_ref() -> None:
    assert _extract_linked_issues(None, ["feat: thing (#150)"]) == ["#150"]


def test_dedupes_and_sorts() -> None:
    out = _extract_linked_issues(
        "feat/issue-887",
        ["feat: A (#150)", "feat: B (#42)", "fix: C (#887)"],
    )
    assert out == ["#42", "#150", "#887"]


def test_no_refs_returns_empty() -> None:
    assert _extract_linked_issues("main", ["docs: update README"]) == []


def test_none_branch_handled() -> None:
    assert _extract_linked_issues(None, ["feat: x (#1)"]) == ["#1"]


def test_empty_subjects_handled() -> None:
    assert _extract_linked_issues("feat/issue-42-x", []) == ["#42"]


def test_cap_at_max() -> None:
    subjects = [f"feat: thing (#{n})" for n in range(1, 30)]
    out = _extract_linked_issues(None, subjects)
    assert len(out) == 16
    assert out[0] == "#1"
    assert out[-1] == "#16"


def test_ignores_non_digit_hash_strings() -> None:
    # `#abc123` is not an issue ref; SHA-ish trailing strings excluded.
    out = _extract_linked_issues(
        "feat/something",
        ["feat: pin #abc123 hash", "fix: real (#5)"],
    )
    assert out == ["#5"]
