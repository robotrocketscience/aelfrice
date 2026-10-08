"""`run_bench_gate.sh` says which corpus revision it reads (#1735)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_bench_gate.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def _env(root: Path, ceiling: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["AELFRICE_CORPUS_ROOT"] = str(root)
    env["GIT_CEILING_DIRECTORIES"] = str(ceiling)
    return env


def _run(
    root: Path, ceiling: Path, *args: str, extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _env(root, ceiling) | (extra or {})
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *(args or ("-k", "probe"))],
        env=env, capture_output=True, text=True, check=True, timeout=60,
    )


@pytest.fixture
def corpus_checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "lab"
    root = repo / "tests" / "corpus" / "v2_0"
    root.mkdir(parents=True)
    (root / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "corpus")
    return root


@pytest.mark.timeout(60)
def test_a_root_inside_a_checkout_reports_branch_and_commit(
    corpus_checkout: Path, tmp_path: Path,
) -> None:
    repo = corpus_checkout.parent.parent.parent
    out = _run(corpus_checkout, tmp_path).stdout
    commit = _git(repo, "rev-parse", "--short=12", "HEAD")
    assert f"bench-gate corpus root: {corpus_checkout}\n" in out
    assert f"bench-gate corpus checkout: {repo.resolve()}\n" in out
    assert "bench-gate corpus branch: main\n" in out
    assert f"bench-gate corpus commit: {commit}\n" in out
    assert "bench-gate corpus uncommitted changes: no\n" in out
    assert "dry run, would run: uv run pytest tests/bench_gate/ -v -m bench_gated -k probe\n" in out


@pytest.mark.timeout(60)
def test_another_branch_or_uncommitted_rows_warn(
    corpus_checkout: Path, tmp_path: Path,
) -> None:
    repo = corpus_checkout.parent.parent.parent
    assert "warning" not in _run(corpus_checkout, tmp_path).stderr
    _git(repo, "switch", "-q", "-c", "campaign/x")
    branch_run = _run(corpus_checkout, tmp_path)
    assert "bench-gate corpus branch: campaign/x\n" in branch_run.stdout
    assert "must read the corpus at a clean main" in branch_run.stderr
    _git(repo, "switch", "-q", "main")
    (corpus_checkout / "rows.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    dirty_run = _run(corpus_checkout, tmp_path)
    assert "bench-gate corpus uncommitted changes: yes\n" in dirty_run.stdout
    assert "must read the corpus at a clean main" in dirty_run.stderr


@pytest.mark.timeout(60)
def test_a_root_outside_any_checkout_says_so(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    out = _run(root, tmp_path).stdout
    assert f"bench-gate corpus root: {root}\n" in out
    assert "bench-gate corpus checkout: none (the root is not inside a git checkout)\n" in out
    assert "bench-gate corpus branch" not in out


@pytest.mark.timeout(60)
def test_a_root_outside_any_checkout_warns(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    assert "must read the corpus at a clean main" in _run(root, tmp_path).stderr


@pytest.mark.timeout(60)
def test_a_checkout_with_no_commits_warns(tmp_path: Path) -> None:
    repo = tmp_path / "lab"
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    run = _run(repo, tmp_path)
    assert "bench-gate corpus commit: (no commits)\n" in run.stdout
    assert "must read the corpus at a clean main" in run.stderr


@pytest.mark.timeout(60)
def test_an_ignored_corpus_file_counts_as_uncommitted(
    corpus_checkout: Path, tmp_path: Path,
) -> None:
    repo = corpus_checkout.parent.parent.parent
    (repo / ".git" / "info" / "exclude").write_text("*.extra\n", encoding="utf-8")
    (corpus_checkout / "rows.extra").write_text("{}\n", encoding="utf-8")
    run = _run(corpus_checkout, tmp_path)
    assert "bench-gate corpus uncommitted changes: yes\n" in run.stdout
    assert "must read the corpus at a clean main" in run.stderr


@pytest.mark.timeout(60)
def test_changes_outside_the_corpus_root_are_not_counted(
    corpus_checkout: Path, tmp_path: Path,
) -> None:
    repo = corpus_checkout.parent.parent.parent
    (repo / "elsewhere.txt").write_text("x\n", encoding="utf-8")
    run = _run(corpus_checkout, tmp_path)
    assert "bench-gate corpus uncommitted changes: no\n" in run.stdout
    assert "warning" not in run.stderr


@pytest.mark.timeout(60)
def test_an_inherited_git_dir_does_not_change_the_report(
    corpus_checkout: Path, tmp_path: Path,
) -> None:
    repo = corpus_checkout.parent.parent.parent
    _git(repo, "switch", "-q", "-c", "side")
    other = tmp_path / "other"
    _git(tmp_path, "init", "-q", "-b", "main", str(other))
    run = _run(corpus_checkout, tmp_path, extra={
        "GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other),
    })
    assert f"bench-gate corpus checkout: {repo.resolve()}\n" in run.stdout
    assert "bench-gate corpus branch: side\n" in run.stdout


@pytest.mark.timeout(60)
def test_the_dry_run_line_quotes_arguments(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    out = _run(root, tmp_path, "-k", "a b").stdout
    assert "would run: uv run pytest tests/bench_gate/ -v -m bench_gated -k a\\ b\n" in out
