"""Subprocess tests for `.githooks/pre-push` (#580).

The hook is a POSIX shell script that gets invoked by git itself with
`<remote-name> <remote-url>` as positional args and the ref-update lines
on stdin (one per ref being pushed). We build a tiny throwaway git
repo + bare "remote" in a tempdir and feed the hook the same stdin git
would, asserting on exit code and stderr.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

_HOOK = Path(__file__).parent.parent / ".githooks" / "pre-push"
_Z40 = "0" * 40


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return result.stdout.strip()


def _setup_repo(
    tmp_path: Path,
    base_age_seconds: int,
    extra_local_commits: int = 1,
) -> tuple[Path, Path]:
    """Build origin (bare) + local clone with a feature branch.

    The merge-base between the feature branch tip and origin/main has
    its committer-date set to ``now - base_age_seconds`` so the hook's
    age check has a deterministic input.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)

    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(origin), str(local)], check=True, capture_output=True)
    _git(local, "config", "user.email", "test@example.com")
    _git(local, "config", "user.name", "Test")
    _git(local, "config", "commit.gpgsign", "false")
    _git(local, "checkout", "-q", "-b", "main")

    # Seed commit on main with a fixed committer-date so its age is
    # deterministic. This becomes the merge-base for the feature branch.
    seed = local / "seed.txt"
    seed.write_text("seed\n")
    _git(local, "add", "seed.txt")
    backdate = f"@{int(__import__('time').time()) - base_age_seconds} +0000"
    _git(
        local,
        "commit",
        "-q",
        "--date", backdate,
        "-m", "feat: seed",
        env={"GIT_COMMITTER_DATE": backdate},
    )
    _git(local, "push", "-q", "origin", "main")

    # Branch off main and add fresh commits (committer-date = now).
    _git(local, "checkout", "-q", "-b", "feature/x")
    for i in range(extra_local_commits):
        f = local / f"f{i}.txt"
        f.write_text(f"x{i}\n")
        _git(local, "add", f.name)
        _git(local, "commit", "-q", "-m", f"feat: x{i}")

    return local, origin


def _stdin_for(local: Path, branch: str) -> str:
    sha = _git(local, "rev-parse", branch)
    return f"refs/heads/{branch} {sha} refs/heads/{branch} {_Z40}\n"


def _run_hook(
    local: Path,
    stdin: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_HOOK), "origin", str(local.parent / "origin.git")],
        cwd=local,
        input=stdin,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


# --- positive cases ---------------------------------------------------------


def test_fresh_branch_under_threshold_passes(tmp_path: Path) -> None:
    """Merge-base age 1 hour < 4h default → push allowed."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=3600)
    res = _run_hook(local, _stdin_for(local, "feature/x"))
    assert res.returncode == 0, res.stderr


def test_main_push_skipped(tmp_path: Path) -> None:
    """Push to main itself: hook skips even if main is stale."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=99 * 3600)
    sha = _git(local, "rev-parse", "main")
    stdin = f"refs/heads/main {sha} refs/heads/main {_Z40}\n"
    res = _run_hook(local, stdin)
    assert res.returncode == 0, res.stderr


def test_branch_deletion_skipped(tmp_path: Path) -> None:
    """local sha = z40 → branch deletion → skip."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=99 * 3600)
    stdin = f"refs/heads/feature/x {_Z40} refs/heads/feature/x {_Z40}\n"
    res = _run_hook(local, stdin)
    assert res.returncode == 0, res.stderr


# --- negative cases ---------------------------------------------------------


def test_stale_branch_aborts_with_message(tmp_path: Path) -> None:
    """Merge-base 5 hours old > 4h default → abort with rebase instructions."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=5 * 3600)
    res = _run_hook(local, _stdin_for(local, "feature/x"))
    assert res.returncode != 0, res.stdout + res.stderr
    assert "pre-push aborted" in res.stderr
    assert "git rebase" in res.stderr
    assert "ALLOW_STALE_BRANCH_PUSH=1" in res.stderr


def test_override_env_var_bypasses(tmp_path: Path) -> None:
    """ALLOW_STALE_BRANCH_PUSH=1 lets a stale push through, with stderr warning."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=10 * 3600)
    res = _run_hook(
        local,
        _stdin_for(local, "feature/x"),
        env={"ALLOW_STALE_BRANCH_PUSH": "1"},
    )
    assert res.returncode == 0, res.stderr
    assert "OVERRIDE" in res.stderr


# --- threshold knobs --------------------------------------------------------


def test_env_var_threshold_override(tmp_path: Path) -> None:
    """AELF_PRE_PUSH_FRESHNESS_HOURS=24 lets a 5h-stale branch through."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=5 * 3600)
    res = _run_hook(
        local,
        _stdin_for(local, "feature/x"),
        env={"AELF_PRE_PUSH_FRESHNESS_HOURS": "24"},
    )
    assert res.returncode == 0, res.stderr


def test_invalid_threshold_falls_back_to_default(tmp_path: Path) -> None:
    """Non-numeric threshold → warning + fallback to 4h default."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=2 * 3600)
    res = _run_hook(
        local,
        _stdin_for(local, "feature/x"),
        env={"AELF_PRE_PUSH_FRESHNESS_HOURS": "not-a-number"},
    )
    # 2h < 4h fallback → push allowed.
    assert res.returncode == 0, res.stderr
    assert "invalid" in res.stderr.lower()


def test_git_config_threshold_override(tmp_path: Path) -> None:
    """aelfrice.prepushFreshnessHours git config raises threshold."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=5 * 3600)
    _git(local, "config", "aelfrice.prepushFreshnessHours", "24")
    res = _run_hook(local, _stdin_for(local, "feature/x"))
    assert res.returncode == 0, res.stderr


def test_force_push_after_real_rebase_passes(tmp_path: Path) -> None:
    """After actually rebasing onto fresh main, the merge-base is fresh and push goes through."""
    local, _origin = _setup_repo(tmp_path, base_age_seconds=10 * 3600)
    # Land a fresh commit on main and fast-forward origin.
    _git(local, "checkout", "-q", "main")
    fresh = local / "fresh.txt"
    fresh.write_text("fresh\n")
    _git(local, "add", "fresh.txt")
    _git(local, "commit", "-q", "-m", "feat: fresh")
    _git(local, "push", "-q", "origin", "main")
    # Rebase the feature branch onto the now-fresh main.
    _git(local, "checkout", "-q", "feature/x")
    _git(local, "rebase", "main")
    res = _run_hook(local, _stdin_for(local, "feature/x"))
    assert res.returncode == 0, res.stderr
