"""Tests for the post-compact working-state projector (#587).

Each test stands up a real (tiny) git repo in a tmp_path so the
projector exercises its actual subprocess pipeline. We only mock when
the unit under test is the pure transformation half.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import RecentTurn
from aelfrice.working_state import (
    WorkingState,
    _earliest_session_ts,
    _project_user_prompts,
    project_working_state,
)


def _git(repo: Path, *args: str) -> None:
    """Run a git command in `repo`. Fail loudly on non-zero exit."""
    subprocess.run(  # noqa: S603 -- list-form
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
            timeout=30,
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Throwaway git repo with a single commit on `main`.

    Default branch is forced to `main` so the projector's branch
    assertion is portable across hosts whose `init.defaultBranch` is
    something else (`master`, `trunk`, …).
    """
    _git(tmp_path, "init", "-q", "--initial-branch=main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test Author")
    _git(tmp_path, "commit", "--allow-empty", "-m", "initial commit")
    return tmp_path


# --- _project_user_prompts (pure) -----------------------------------------


class TestProjectUserPrompts:
    def test_takes_only_user_role(self) -> None:
        turns = [
            RecentTurn(role="user", text="hello"),
            RecentTurn(role="assistant", text="hi back"),
            RecentTurn(role="user", text="follow up"),
        ]
        assert _project_user_prompts(turns, max_prompts=10) == ["hello", "follow up"]

    def test_caps_at_max(self) -> None:
        turns = [RecentTurn(role="user", text=f"q{i}") for i in range(10)]
        assert _project_user_prompts(turns, max_prompts=3) == ["q7", "q8", "q9"]

    def test_zero_max_returns_empty(self) -> None:
        turns = [RecentTurn(role="user", text="q")]
        assert _project_user_prompts(turns, max_prompts=0) == []

    def test_no_user_turns(self) -> None:
        turns = [RecentTurn(role="assistant", text="solo")]
        assert _project_user_prompts(turns, max_prompts=5) == []


# --- _earliest_session_ts (pure) ------------------------------------------


class TestEarliestSessionTs:
    def test_picks_min_ts_within_session(self) -> None:
        turns = [
            RecentTurn(role="user", text="a", session_id="s1", ts="2026-05-10T08:00:00Z"),
            RecentTurn(role="assistant", text="b", session_id="s1", ts="2026-05-10T08:00:01Z"),
            RecentTurn(role="user", text="c", session_id="s2", ts="2026-05-10T09:00:00Z"),
            RecentTurn(role="assistant", text="d", session_id="s2", ts="2026-05-10T09:00:05Z"),
        ]
        assert _earliest_session_ts(turns) == "2026-05-10T09:00:00Z"

    def test_no_session_id_returns_none(self) -> None:
        turns = [RecentTurn(role="user", text="x", ts="2026-05-10T00:00:00Z")]
        assert _earliest_session_ts(turns) is None

    def test_no_ts_returns_none(self) -> None:
        turns = [RecentTurn(role="user", text="x", session_id="s1")]
        assert _earliest_session_ts(turns) is None

    def test_empty_returns_none(self) -> None:
        assert _earliest_session_ts([]) is None

    def test_partial_ts_skips_unts_turns(self) -> None:
        # Latest turn has session_id and at least one matching turn has ts → use that ts.
        turns = [
            RecentTurn(role="user", text="a", session_id="s1", ts="2026-05-10T08:00:00Z"),
            RecentTurn(role="user", text="b", session_id="s1"),  # ts=None
        ]
        assert _earliest_session_ts(turns) == "2026-05-10T08:00:00Z"


# --- project_working_state (subprocess) -----------------------------------


class TestProjectWorkingState:
    def test_clean_repo_emits_branch_and_log_only(self, tmp_repo: Path) -> None:
        ws = project_working_state(tmp_repo, recent_turns=[])
        assert ws.branch == "main"
        assert ws.status_porcelain == []
        assert len(ws.recent_log) == 1
        assert "initial commit" in ws.recent_log[0]
        assert ws.recent_user_prompts == []
        assert ws.session_commits == []

    def test_dirty_repo_surfaces_status(self, tmp_repo: Path) -> None:
        (tmp_repo / "newfile.txt").write_text("x")
        ws = project_working_state(tmp_repo, recent_turns=[])
        # Untracked files show as "?? newfile.txt"
        assert any("newfile.txt" in line for line in ws.status_porcelain)

    def test_status_capped_at_max_lines(self, tmp_repo: Path) -> None:
        for i in range(20):
            (tmp_repo / f"f{i}.txt").write_text("x")
        ws = project_working_state(tmp_repo, recent_turns=[], max_status_lines=5)
        assert len(ws.status_porcelain) == 5

    @pytest.mark.timeout(30)
    def test_recent_log_capped(self, tmp_repo: Path) -> None:
        for i in range(5):
            _git(tmp_repo, "commit", "--allow-empty", "-m", f"commit-{i}")
        ws = project_working_state(tmp_repo, recent_turns=[], recent_commit_count=3)
        assert len(ws.recent_log) == 3
        # Most-recent first
        assert "commit-4" in ws.recent_log[0]

    def test_user_prompts_drawn_from_turns(self, tmp_repo: Path) -> None:
        turns = [
            RecentTurn(role="user", text="first prompt"),
            RecentTurn(role="assistant", text="reply"),
            RecentTurn(role="user", text="second prompt"),
        ]
        ws = project_working_state(tmp_repo, recent_turns=turns, max_user_prompts=10)
        assert ws.recent_user_prompts == ["first prompt", "second prompt"]

    @pytest.mark.timeout(30)
    def test_session_commits_bounded_by_session_ts(self, tmp_repo: Path) -> None:
        # The initial commit lands first; sleep so the next commit is at
        # least 2s later (git --since has 1s granularity). Take the ts of
        # the in-session commit as session start so it lands inside the
        # window and the initial commit doesn't.
        import time
        time.sleep(2)
        _git(tmp_repo, "commit", "--allow-empty", "-m", "in-session")
        out = subprocess.run(  # noqa: S603
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(tmp_repo), check=True, capture_output=True, text=True,
        timeout=30).stdout.strip()
        earliest_ts = out
        turns = [
            RecentTurn(role="user", text="q", session_id="s1", ts=earliest_ts),
        ]
        ws = project_working_state(tmp_repo, recent_turns=turns)
        assert any("in-session" in line for line in ws.session_commits)
        assert all("initial commit" not in line for line in ws.session_commits)

    def test_session_commits_empty_without_ts(self, tmp_repo: Path) -> None:
        turns = [RecentTurn(role="user", text="q", session_id="s1")]  # ts=None
        ws = project_working_state(tmp_repo, recent_turns=turns)
        assert ws.session_commits == []


# --- non-git directory ----------------------------------------------------


class TestNonGitCwd:
    def test_non_git_returns_empty(self, tmp_path: Path) -> None:
        ws = project_working_state(tmp_path, recent_turns=[])
        assert ws.is_empty()


# --- WorkingState.is_empty ------------------------------------------------


class TestIsEmpty:
    def test_default_construction_is_empty(self) -> None:
        assert WorkingState().is_empty() is True

    def test_branch_only_is_not_empty(self) -> None:
        assert WorkingState(branch="main").is_empty() is False

    def test_log_only_is_not_empty(self) -> None:
        assert WorkingState(recent_log=["abc Hello"]).is_empty() is False


# --- Latency budget (#587 AC#3) ---------------------------------------


class TestLatency:
    @pytest.mark.timeout(60)
    def test_clean_repo_run_finishes_without_a_git_timeout(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC#3 sets ≤50ms p95 in production. This test catches a subprocess
        hang, so it watches for the hang and reads no clock (#1853).

        A hung call runs into the module's timeout and raises TimeoutExpired on
        any host, so a run with no expiry had no hung call. That timeout is the
        only bound here, so a slow host can't widen it. It runs once, so a git
        call that takes over 1.5s on a busy host fails it too.
        """
        real_run = subprocess.run
        expired: list[str] = []

        def watching_run(cmd: list[str], **kwargs: object) -> object:
            try:
                return real_run(cmd, **kwargs)  # type: ignore[call-overload]
            except subprocess.TimeoutExpired:
                expired.append(" ".join(cmd))
                raise

        monkeypatch.setattr(subprocess, "run", watching_run)
        ws = project_working_state(tmp_repo, recent_turns=[])
        assert expired == []
        assert ws.branch == "main"

    def test_every_git_call_carries_the_timeout(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing timeout costs nothing until git hangs, so no clock sees it.
        Read the argument instead, with and without a session in the turns. The
        literal states the module's hard 1.5s, so raising it fails here too. The
        process counts are upper bounds, so merging two calls passes and adding
        one fails.
        """
        real_run = subprocess.run
        subcommands = {"rev-parse", "status", "log"}
        seen: list[tuple[str, object]] = []

        def recording_run(cmd: list[str], **kwargs: object) -> object:
            subcommand = next((a for a in cmd[1:] if a in subcommands), " ".join(cmd))
            seen.append((subcommand, kwargs.get("timeout")))
            return real_run(cmd, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(subprocess, "run", recording_run)
        session = [
            RecentTurn(role="user", text="q", session_id="s1", ts="2026-01-01T00:00:00Z"),
        ]
        for turns, max_processes in (([], 3), (session, 4)):
            seen.clear()
            project_working_state(tmp_repo, recent_turns=turns)
            assert seen
            assert len(seen) <= max_processes
            for subcommand, timeout in seen:
                assert subcommand in subcommands
                assert timeout == 1.5

    def test_projector_own_time_with_git_stubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every git call stubbed, what's left is the projector's own time.

        Stubbed runs stayed at or below 0.033ms at every load measured, so 50ms
        holds on a slow host and a sleep or a slow loop in the projector fails
        it (#1853). The fastest of five runs is judged, so a slowdown on some
        runs passes. Delay inside the git process stays uncaught.
        """
        import time

        def stub(cmd: list[str], **kwargs: object) -> object:
            return subprocess.CompletedProcess(cmd, 0, "main\n", "")

        monkeypatch.setattr(subprocess, "run", stub)
        best_ms = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            project_working_state(tmp_path, recent_turns=[])
            best_ms = min(best_ms, (time.perf_counter() - t0) * 1000)
            if best_ms < 50:
                break
        assert best_ms < 50, f"fastest of 5 stubbed runs took {best_ms:.1f}ms; expected <50ms"
