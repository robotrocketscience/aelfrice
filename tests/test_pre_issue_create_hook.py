"""Tests for pre_issue_create_hook — tokenizer, scorer, and run_guard.

All fixtures are entirely synthetic.  No real issue titles from the
implementation session appear here.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from aelfrice.pre_issue_create_hook import (
    BLOCK_THRESHOLD,
    jaccard,
    run_guard,
    score_candidate,
    tokenize_title,
    _is_gh_issue_create,
    _extract_title,
    _extract_body_file,
    _safe_read_body_file,
)


# ---------------------------------------------------------------------------
# tokenize_title
# ---------------------------------------------------------------------------


class TestTokenizeTitle:
    def test_basic_lowercase_and_split(self) -> None:
        assert "widget" in tokenize_title("Add widget factory")
        assert "factory" in tokenize_title("Add widget factory")

    def test_strips_conventional_commit_prefix_feat(self) -> None:
        tokens = tokenize_title("feat(ui): add dark mode toggle")
        assert "feat" not in tokens
        assert "ui" not in tokens  # scope is inside prefix, stripped
        assert "dark" in tokens
        assert "mode" in tokens
        assert "toggle" in tokens

    def test_strips_conventional_commit_prefix_fix(self) -> None:
        tokens = tokenize_title("fix: broken nav-bar dropdown menu")
        assert "fix" not in tokens
        assert "broken" in tokens
        assert "nav" in tokens
        assert "bar" in tokens
        assert "dropdown" in tokens

    def test_strips_fix_with_scope(self) -> None:
        tokens = tokenize_title("fix(retrieval): BM25 query expansion crash")
        assert "fix" not in tokens
        assert "retrieval" not in tokens
        assert "bm25" in tokens

    def test_drops_stop_words(self) -> None:
        tokens = tokenize_title("a widget and the factory of doom")
        assert "a" not in tokens
        assert "and" not in tokens
        assert "the" not in tokens
        assert "of" not in tokens
        assert "widget" in tokens
        assert "factory" in tokens
        assert "doom" in tokens

    def test_drops_single_char_tokens(self) -> None:
        tokens = tokenize_title("x y z long-word here")
        assert "x" not in tokens
        assert "y" not in tokens
        assert "z" not in tokens
        assert "long" in tokens

    def test_no_prefix_plain_title(self) -> None:
        tokens = tokenize_title("Cache eviction policy redesign")
        assert "cache" in tokens
        assert "eviction" in tokens
        assert "policy" in tokens
        assert "redesign" in tokens

    def test_empty_title(self) -> None:
        assert tokenize_title("") == set()

    def test_prefix_only_title(self) -> None:
        # After stripping the prefix nothing remains
        tokens = tokenize_title("fix:")
        assert tokens == set()

    def test_breaking_exclamation_prefix(self) -> None:
        # feat!: breaking change syntax
        tokens = tokenize_title("feat!: drop Python 3.9 support")
        assert "feat" not in tokens
        assert "drop" in tokens


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets(self) -> None:
        a = {"foo", "bar", "baz"}
        assert jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        assert jaccard({"alpha", "beta"}, {"gamma", "delta"}) == pytest.approx(0.0)

    def test_empty_both(self) -> None:
        assert jaccard(set(), set()) == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        assert jaccard({"x"}, set()) == pytest.approx(0.0)

    def test_half_overlap(self) -> None:
        a = {"a", "b", "c", "d"}
        b = {"c", "d", "e", "f"}
        # intersection={c,d}=2, union={a,b,c,d,e,f}=6
        assert jaccard(a, b) == pytest.approx(2 / 6)

    def test_subset(self) -> None:
        a = {"x", "y"}
        b = {"x", "y", "z"}
        # 2/3
        assert jaccard(a, b) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_verbatim_match(self) -> None:
        title = "webhook retry logic"
        score = score_candidate(tokenize_title(title), title)
        assert score == pytest.approx(1.0)

    def test_near_match(self) -> None:
        query_tokens = tokenize_title("webhook retry logic")
        score = score_candidate(query_tokens, "feat: add webhook retry logic")
        # After stripping feat prefix: candidate tokens = {add, webhook, retry, logic}
        # query tokens = {webhook, retry, logic}; intersection=3, union=4 → 0.75
        assert score == pytest.approx(0.75)

    def test_zero_overlap(self) -> None:
        query_tokens = tokenize_title("cache eviction policy")
        score = score_candidate(query_tokens, "network timeout")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        query_tokens = tokenize_title("rate limit backoff")
        score = score_candidate(query_tokens, "exponential backoff support")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsGhIssueCreate:
    def test_plain(self) -> None:
        assert _is_gh_issue_create("gh issue create --title foo")

    def test_leading_whitespace(self) -> None:
        assert _is_gh_issue_create("  gh issue create --title x")

    def test_not_gh(self) -> None:
        assert not _is_gh_issue_create("git commit -m foo")

    def test_gh_pr_create(self) -> None:
        assert not _is_gh_issue_create("gh pr create --title foo")

    def test_gh_issue_list(self) -> None:
        assert not _is_gh_issue_create("gh issue list")

    def test_env_prefix(self) -> None:
        # KEY=val gh issue create ...
        assert _is_gh_issue_create("GH_TOKEN=abc gh issue create --title x")


class TestExtractTitle:
    def test_long_flag(self) -> None:
        assert _extract_title("gh issue create --title 'My Bug'") == "My Bug"

    def test_short_flag(self) -> None:
        assert _extract_title("gh issue create -t 'Widget crash'") == "Widget crash"

    def test_eq_form(self) -> None:
        assert _extract_title("gh issue create --title=EqForm") == "EqForm"

    def test_no_title(self) -> None:
        assert _extract_title("gh issue create --body foo") == ""


class TestExtractBodyFile:
    def test_long_flag(self) -> None:
        assert _extract_body_file("gh issue create --body-file /tmp/body.md") == "/tmp/body.md"

    def test_short_flag(self) -> None:
        assert _extract_body_file("gh issue create -F /tmp/b.md") == "/tmp/b.md"

    def test_eq_form(self) -> None:
        assert _extract_body_file("gh issue create --body-file=/tmp/b.md") == "/tmp/b.md"

    def test_no_body_file(self) -> None:
        assert _extract_body_file("gh issue create --title x") == ""


class TestSafeReadBodyFile:
    def test_real_file(self, tmp_path: pytest.TempdirFactory) -> None:
        p = tmp_path / "body.md"
        p.write_text("hello world")
        assert _safe_read_body_file(str(p)) == "hello world"

    def test_nonexistent(self) -> None:
        assert _safe_read_body_file("/nonexistent/path/body.md") == ""

    def test_empty_string(self) -> None:
        assert _safe_read_body_file("") == ""

    def test_claude_dir_refused(self) -> None:
        # Path under ~/.claude/ must be refused even if the file exists
        from pathlib import Path
        claude_path = Path.home() / ".claude" / "settings.json"
        # We do not require the file to exist; the safety check fires first.
        result = _safe_read_body_file(str(claude_path))
        assert result == ""


# ---------------------------------------------------------------------------
# run_guard — with mocked runners
# ---------------------------------------------------------------------------


def _make_stdin(command: str) -> dict[str, Any]:
    return {
        "tool_name": "Bash",
        "tool_input": {"command": command, "description": ""},
    }


def _gh_empty(_: list[str]) -> str:
    return "[]"


def _git_empty(_: list[str]) -> str:
    return ""


def _gh_with_dup(number: int, title: str, state: str = "CLOSED") -> Any:
    def runner(_: list[str]) -> str:
        return json.dumps([{
            "number": number,
            "title": title,
            "state": state,
            "stateReason": "completed",
            "closedAt": "2024-01-01T00:00:00Z",
        }])
    return runner


def _git_with_commit(msg: str) -> Any:
    def runner(_: list[str]) -> str:
        return f"abc1234 {msg}\n"
    return runner


class TestRunGuard:

    def test_non_bash_tool_passes(self) -> None:
        payload: dict[str, Any] = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/foo/bar"},
        }
        import io
        assert run_guard(payload, gh_runner=_gh_empty, git_runner=_git_empty,
                         stderr_out=io.StringIO()) == 0

    def test_non_gh_issue_create_passes(self) -> None:
        import io
        payload = _make_stdin("git commit -m 'fix: something'")
        assert run_guard(payload, gh_runner=_gh_empty, git_runner=_git_empty,
                         stderr_out=io.StringIO()) == 0

    def test_gh_pr_create_passes(self) -> None:
        import io
        payload = _make_stdin("gh pr create --title 'some PR'")
        assert run_guard(payload, gh_runner=_gh_empty, git_runner=_git_empty,
                         stderr_out=io.StringIO()) == 0

    def test_no_title_passes(self) -> None:
        import io
        payload = _make_stdin("gh issue create --body-file /tmp/b.md")
        assert run_guard(payload, gh_runner=_gh_empty, git_runner=_git_empty,
                         stderr_out=io.StringIO()) == 0

    def test_novel_title_passes(self) -> None:
        """A title with no overlapping candidates → exit 0."""
        import io
        payload = _make_stdin(
            "gh issue create --title 'widget renderer memory leak'"
        )
        assert run_guard(
            payload,
            gh_runner=_gh_with_dup(99, "completely unrelated network timeout"),
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        ) == 0

    def test_duplicate_blocks(self) -> None:
        """A title very similar to an existing issue → exit 2."""
        import io
        dup_title = "feat: widget renderer memory leak on resize"
        proposed = "widget renderer memory leak on resize"
        payload = _make_stdin(
            f"gh issue create --title '{proposed}'"
        )
        err = io.StringIO()
        result = run_guard(
            payload,
            gh_runner=_gh_with_dup(42, dup_title, state="CLOSED"),
            git_runner=_git_empty,
            stderr_out=err,
        )
        assert result == 2
        assert "BLOCK" in err.getvalue()

    def test_git_log_duplicate_blocks(self) -> None:
        """A title matching a git commit subject → exit 2."""
        import io
        commit_msg = "fix: webhook retry logic on transient network failure"
        proposed = "webhook retry logic on transient network failure"
        payload = _make_stdin(
            f"gh issue create --title '{proposed}'"
        )
        err = io.StringIO()
        result = run_guard(
            payload,
            gh_runner=_gh_empty,
            git_runner=_git_with_commit(commit_msg),
            stderr_out=err,
        )
        assert result == 2

    def test_allow_dup_issue_env_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ALLOW_DUP_ISSUE=1 bypasses the guard entirely."""
        import io
        monkeypatch.setenv("ALLOW_DUP_ISSUE", "1")
        dup_title = "feat: widget renderer memory leak on resize"
        proposed = "widget renderer memory leak on resize"
        payload = _make_stdin(f"gh issue create --title '{proposed}'")
        result = run_guard(
            payload,
            gh_runner=_gh_with_dup(42, dup_title),
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        )
        assert result == 0

    def test_no_pre_issue_guard_env_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AELFRICE_NO_PRE_ISSUE_GUARD=1 bypasses the guard."""
        import io
        monkeypatch.setenv("AELFRICE_NO_PRE_ISSUE_GUARD", "1")
        dup_title = "feat: widget renderer memory leak on resize"
        proposed = "widget renderer memory leak on resize"
        payload = _make_stdin(f"gh issue create --title '{proposed}'")
        result = run_guard(
            payload,
            gh_runner=_gh_with_dup(42, dup_title),
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        )
        assert result == 0

    def test_false_positive_guard_distinct_domains(self) -> None:
        """Titles sharing only the cc-prefix + one common word do NOT block.

        fix: nav bar dropdown  vs  fix: navigation menu close
        Meaningful overlapping tokens: just 'nav' / 'navigation' — different.
        Neither 'fix' nor common stop-words count.  Jaccard should be < 0.5.
        """
        import io
        existing = "fix: nav bar dropdown behaviour"
        proposed = "fix: navigation menu close animation"
        payload = _make_stdin(f"gh issue create --title '{proposed}'")
        result = run_guard(
            payload,
            gh_runner=_gh_with_dup(10, existing),
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        )
        assert result == 0

    def test_body_file_from_claude_dir_body_treated_as_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A --body-file under ~/.claude/ is refused; guard still runs on title."""
        import io
        from pathlib import Path
        fake_claude_path = str(Path.home() / ".claude" / "fake-body.md")
        # Title alone has no overlap with empty candidates → PASS
        payload = _make_stdin(
            f"gh issue create --title 'database migration helper' "
            f"--body-file {fake_claude_path}"
        )
        result = run_guard(
            payload,
            gh_runner=_gh_empty,
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        )
        assert result == 0  # no dup candidates → PASS regardless of body path

    def test_empty_candidates_passes(self) -> None:
        """When both runners return nothing, always PASS."""
        import io
        payload = _make_stdin("gh issue create --title 'some novel feature'")
        assert run_guard(
            payload,
            gh_runner=_gh_empty,
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        ) == 0

    def test_gh_runner_exception_passes(self) -> None:
        """A failing gh runner is silently swallowed; guard returns PASS."""
        import io

        def _bad_gh(_: list[str]) -> str:
            raise RuntimeError("gh not found")

        payload = _make_stdin("gh issue create --title 'cache invalidation bug'")
        assert run_guard(
            payload,
            gh_runner=_bad_gh,
            git_runner=_git_empty,
            stderr_out=io.StringIO(),
        ) == 0

    def test_block_message_contains_candidate_ref(self) -> None:
        """The BLOCK message includes the matching candidate's number."""
        import io
        dup_title = "feat: document upload preview widget"
        proposed = "document upload preview widget"
        payload = _make_stdin(f"gh issue create --title '{proposed}'")
        err = io.StringIO()
        run_guard(
            payload,
            gh_runner=_gh_with_dup(777, dup_title),
            git_runner=_git_empty,
            stderr_out=err,
        )
        assert "#777" in err.getvalue()
