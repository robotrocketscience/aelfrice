"""Deterministic tests for `aelfrice.project_warm`.

Hard rules from issue #137 / the GSD playbook:

- No real `~/.aelfrice` writes — every test redirects `aelfrice_home`
  via `monkeypatch` or the `WarmConfig.aelfrice_home` field.
- No network. No probabilistic assertions. Each test ≤2s.
- Real git operations are allowed in `tmp_path`-scoped throwaway repos
  because `git init` is fast and deterministic; we don't mock the
  binary.

Coverage matrix per the issue body:

  * git repo                      → resolves to repo root
  * git worktree                  → resolves to the worktree, not main
  * non-git dir                   → returns None / UNKNOWN_PROJECT
  * denied path                   → DENIED_PATH
  * debounce window               → DEBOUNCED on the second call
  * unknown project               → UNKNOWN_PROJECT
  * config.json deny override     → custom deny applies
  * sentinel write                → file appears with the right ts
  * CLI silent no-op              → exit 0 + empty stdout

Plus a few near-tests for the helpers that aren't directly observable
through the public API (project-id determinism, deny matcher).
"""
from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.project_warm import (
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_DENY_GLOBS,
    ProjectRef,
    WarmConfig,
    WarmResult,
    load_config,
    resolve_project_root,
    warm_path,
    warm_project,
)


# --- fixtures ---------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    """Run `git <args>` in `cwd` with deterministic identity + no signing."""
    subprocess.run(
        [
            "git",
            "-c", "user.email=test@example.invalid",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=str(cwd),
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    """Make `repo` a git work-tree with one commit."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], cwd=repo)
    (repo / "README").write_text("seed", encoding="utf-8")
    _git(["add", "."], cwd=repo)
    _git(["commit", "-q", "-m", "seed"], cwd=repo)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `~/.aelfrice/` to a tmp_path so tests never write to $HOME.

    Returns the path that `Path.home() / ".aelfrice"` resolves to under
    the redirect — the CLI test needs this exact mapping because
    `aelf project-warm` builds `WarmConfig` via `load_config()` which
    reads `Path.home() / ".aelfrice"`. Setting HOME alone is enough on
    macOS / Linux where `Path.home()` consults `$HOME`.
    """
    fake_user_home = tmp_path / "fake_user_home"
    fake_user_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_user_home))
    home = fake_user_home / ".aelfrice"
    home.mkdir()
    return home


# --- resolve_project_root ---------------------------------------------------


def test_resolve_project_root_git_repo(tmp_path: Path, fake_home: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)

    ref = resolve_project_root(sub, aelfrice_home=fake_home)

    assert ref is not None
    assert ref.root == repo.resolve()
    assert len(ref.id) == 12


def test_resolve_project_root_git_worktree(
    tmp_path: Path, fake_home: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "feat/x", str(wt)], cwd=repo)

    ref = resolve_project_root(wt, aelfrice_home=fake_home)

    # ProjectRef.root is the worktree working directory — not the main
    # checkout — so _warm_store can os.chdir to the right place.
    assert ref is not None
    assert ref.root == wt.resolve()
    assert ref.root != repo.resolve()


def test_resolve_project_root_worktrees_share_id(
    tmp_path: Path, fake_home: Path,
) -> None:
    """Two worktrees of the same repo share a single ProjectRef.id.

    The sentinel is keyed by git-common-dir, so both worktrees hit the
    same debounce sentinel, matching the v1.1.0 design that worktrees of
    one repo share a single belief store.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-q", "-b", "feat/x", str(wt)], cwd=repo)

    ref_main = resolve_project_root(repo, aelfrice_home=fake_home)
    ref_wt = resolve_project_root(wt, aelfrice_home=fake_home)

    assert ref_main is not None
    assert ref_wt is not None
    # Roots differ — each worktree is a separate working directory.
    assert ref_main.root != ref_wt.root
    # IDs must be identical — both map to the same git-common-dir.
    assert ref_main.id == ref_wt.id


def test_resolve_project_root_non_git_dir(
    tmp_path: Path, fake_home: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    ref = resolve_project_root(plain, aelfrice_home=fake_home)

    assert ref is None


def test_resolve_project_root_ancestor_with_layout(
    tmp_path: Path, fake_home: Path,
) -> None:
    """Non-git dir that has its `.aelfrice/projects/<id>/` provisioned."""
    project = tmp_path / "notebook"
    (project / "deep" / "nested").mkdir(parents=True)
    # Provision the project layout the way `aelf onboard` would.
    from aelfrice.project_warm import _project_id  # pyright: ignore[reportPrivateUsage]
    pid = _project_id(project.resolve())
    (fake_home / "projects" / pid).mkdir(parents=True)

    ref = resolve_project_root(project / "deep" / "nested", aelfrice_home=fake_home)

    assert ref is not None
    assert ref.root == project.resolve()
    assert ref.id == pid


def test_resolve_project_root_returns_none_for_missing_path(
    tmp_path: Path, fake_home: Path,
) -> None:
    missing = tmp_path / "does-not-exist"

    ref = resolve_project_root(missing, aelfrice_home=fake_home)

    # Path doesn't exist → not a git repo, not under an opted-in
    # ancestor → None. Hook callers downgrade this to a silent no-op.
    assert ref is None


# --- warm_project: core paths ----------------------------------------------


def _ref_for(root: Path) -> ProjectRef:
    from aelfrice.project_warm import _project_id  # pyright: ignore[reportPrivateUsage]
    return ProjectRef(root=root.resolve(), id=_project_id(root.resolve()))


def test_warm_project_writes_sentinel(
    tmp_path: Path, fake_home: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    ref = _ref_for(repo)
    cfg = WarmConfig(deny_globs=(), aelfrice_home=fake_home)

    result = warm_project(ref, config=cfg, now=1_700_000_000.0)

    assert result is WarmResult.WARMED
    sentinel = fake_home / "projects" / ref.id / ".last_warm"
    assert sentinel.is_file()
    assert float(sentinel.read_text(encoding="utf-8").strip()) == 1_700_000_000.0


def test_warm_project_debounces_within_window(
    tmp_path: Path, fake_home: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    ref = _ref_for(repo)
    cfg = WarmConfig(deny_globs=(), aelfrice_home=fake_home)

    # First call seeds the sentinel.
    first = warm_project(ref, config=cfg, now=1_700_000_000.0)
    # Second call inside the 60s window must be a no-op.
    second = warm_project(
        ref, config=cfg, debounce_seconds=60, now=1_700_000_030.0,
    )
    # Third call past the window warms again.
    third = warm_project(
        ref, config=cfg, debounce_seconds=60, now=1_700_000_061.0,
    )

    assert first is WarmResult.WARMED
    assert second is WarmResult.DEBOUNCED
    assert third is WarmResult.WARMED


def test_warm_project_denied_path(
    tmp_path: Path, fake_home: Path,
) -> None:
    # Use a deny-glob the test root actually matches so we don't depend
    # on the host having /tmp/* layouts under tmp_path (it varies).
    repo = tmp_path / "scratch"
    _init_repo(repo)
    ref = _ref_for(repo)
    cfg = WarmConfig(
        deny_globs=(str(tmp_path) + "/**",),
        aelfrice_home=fake_home,
    )

    result = warm_project(ref, config=cfg, now=1_700_000_000.0)

    assert result is WarmResult.DENIED_PATH
    sentinel = fake_home / "projects" / ref.id / ".last_warm"
    assert not sentinel.exists()  # denied path must not seed sentinel


def test_warm_project_skips_when_db_does_not_exist(
    tmp_path: Path, fake_home: Path,
) -> None:
    """No DB yet (first onboard hasn't run) is still a successful warm.

    The warm step is best-effort. The sentinel still updates so the
    debounce window applies the next time a real DB is present.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    ref = _ref_for(repo)
    cfg = WarmConfig(deny_globs=(), aelfrice_home=fake_home)

    result = warm_project(ref, config=cfg, now=1_700_000_000.0)

    assert result is WarmResult.WARMED
    db = repo / ".git" / "aelfrice" / "memory.db"
    assert not db.exists()


def test_warm_project_warms_real_store(
    tmp_path: Path, fake_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a populated DB has its locked-beliefs path exercised."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    db_dir = repo / ".git" / "aelfrice"
    db_dir.mkdir(parents=True)
    db = db_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    # Seed one belief so count_beliefs / list_locked_beliefs touch real
    # pages. Using the public store API keeps the test resilient.
    from aelfrice.models import (
        BELIEF_FACTUAL,
        LOCK_USER,
        Belief,
    )
    from aelfrice.store import MemoryStore

    seeded = MemoryStore(str(db))
    try:
        seeded.insert_belief(
            Belief(
                id="warm-test-0001",
                content="warming smoke belief",
                content_hash="warm-test-0001-hash",
                alpha=9.0,
                beta=0.5,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_USER,
                locked_at="2026-04-27T00:00:00Z",
                demotion_pressure=0,
                created_at="2026-04-27T00:00:00Z",
                last_retrieved_at=None,
                origin="user_stated",
            ),
        )
    finally:
        seeded.close()

    monkeypatch.delenv("AELFRICE_DB", raising=False)
    ref = _ref_for(repo)
    cfg = WarmConfig(deny_globs=(), aelfrice_home=fake_home)

    result = warm_project(ref, config=cfg, now=1_700_000_000.0)

    assert result is WarmResult.WARMED


# --- warm_path: end-to-end via the convenience wrapper ---------------------


def test_warm_path_unknown_project(
    tmp_path: Path, fake_home: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    cfg = WarmConfig(deny_globs=(), aelfrice_home=fake_home)

    result = warm_path(plain, config=cfg, now=1_700_000_000.0)

    assert result is WarmResult.UNKNOWN_PROJECT


# --- config -----------------------------------------------------------------


def test_load_config_defaults_when_no_file(fake_home: Path) -> None:
    cfg = load_config(aelfrice_home=fake_home)
    assert cfg.deny_globs == DEFAULT_DENY_GLOBS
    assert cfg.aelfrice_home == fake_home


def test_load_config_overrides_deny_globs(fake_home: Path) -> None:
    (fake_home / "config.json").write_text(
        json.dumps({"project_warm": {"deny_globs": ["/srv/**", "~/scratch/**"]}}),
        encoding="utf-8",
    )

    cfg = load_config(aelfrice_home=fake_home)

    assert cfg.deny_globs == ("/srv/**", "~/scratch/**")


def test_load_config_ignores_malformed_globs(fake_home: Path) -> None:
    (fake_home / "config.json").write_text(
        json.dumps({"project_warm": {"deny_globs": ["/ok/**", 42]}}),
        encoding="utf-8",
    )

    cfg = load_config(aelfrice_home=fake_home)

    # One non-string in the list discards the whole override.
    assert cfg.deny_globs == DEFAULT_DENY_GLOBS


def test_load_config_falls_back_on_invalid_json(fake_home: Path) -> None:
    (fake_home / "config.json").write_text("{not json", encoding="utf-8")

    cfg = load_config(aelfrice_home=fake_home)

    assert cfg.deny_globs == DEFAULT_DENY_GLOBS


# --- CLI surface ------------------------------------------------------------


def test_cli_project_warm_silent_for_unknown_path(
    tmp_path: Path, fake_home: Path,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    buf = io.StringIO()

    code = cli_main(["project-warm", str(plain)], out=buf)

    assert code == 0
    assert buf.getvalue() == ""


def test_cli_project_warm_silent_for_denied_path(
    tmp_path: Path, fake_home: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    # Configure deny so this tmp_path is blocked.
    (fake_home / "config.json").write_text(
        json.dumps({"project_warm": {"deny_globs": [str(tmp_path) + "/**"]}}),
        encoding="utf-8",
    )
    buf = io.StringIO()

    code = cli_main(["project-warm", str(repo)], out=buf)

    assert code == 0
    assert buf.getvalue() == ""
    sentinel_dir = fake_home / "projects"
    # No project layout should have been created — denied paths are
    # invisible to the warm pipeline.
    if sentinel_dir.exists():
        assert list(sentinel_dir.iterdir()) == []


def test_cli_project_warm_warms_and_debounces(
    tmp_path: Path, fake_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    # Disable deny globs so tmp_path-rooted repos can warm.
    (fake_home / "config.json").write_text(
        json.dumps({"project_warm": {"deny_globs": []}}),
        encoding="utf-8",
    )
    # Pin time so the second call lands inside the debounce window.
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(time, "time", lambda: fixed_now)

    buf1 = io.StringIO()
    code1 = cli_main(["project-warm", str(repo)], out=buf1)
    buf2 = io.StringIO()
    code2 = cli_main(["project-warm", str(repo)], out=buf2)

    assert code1 == 0
    assert code2 == 0
    # The CLI surface stays silent on both paths — observable evidence
    # is the sentinel file alone.
    assert buf1.getvalue() == ""
    assert buf2.getvalue() == ""
    from aelfrice.project_warm import _project_id  # pyright: ignore[reportPrivateUsage]
    # The sentinel id is keyed off git-common-dir (repo/.git for a
    # non-worktree repo), not the repo root itself.
    pid = _project_id((repo / ".git").resolve())
    sentinel = fake_home / "projects" / pid / ".last_warm"
    assert sentinel.is_file()
    assert float(sentinel.read_text(encoding="utf-8").strip()) == fixed_now


def test_cli_project_warm_default_debounce_constant() -> None:
    """Issue #137 pins the default to 60 seconds — guard against drift."""
    assert DEFAULT_DEBOUNCE_SECONDS == 60
