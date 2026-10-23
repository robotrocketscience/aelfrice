"""#1796: a `[store]` table in `.aelfrice.toml` selects a store per folder.

Covers the issue's acceptance items 1-4, 6 and 7. Item 5, the hook
payload's working directory, belongs to a later change.

Every repository, marker, store and home directory is planted under
`tmp_path`. `AELFRICE_DB` and `AELFRICE_TRANSCRIPTS_DIR`, which the suite
pins, are deleted per test so the marker and git rules are reachable.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aelfrice import cli, db_paths, transcript_logger
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.db_paths import (
    STORE_RULE_ENV,
    STORE_RULE_GIT,
    STORE_RULE_GLOBAL,
    STORE_RULE_MARKER,
    StoreMarkerError,
    resolve_store,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# Most tests run `git init`, `git commit`, or `git worktree add`, which
# outlast the suite's default budget under load (#1307).
pytestmark = pytest.mark.timeout(60)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, timeout=30,
    )


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=repo)
    return repo


def _commit_all(repo: Path) -> None:
    _git("add", "-A", cwd=repo)
    _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", "seed",
        cwd=repo,
    )


def _marker(folder: Path, value: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / CONFIG_FILENAME
    config.write_text(f'[store]\npath = "{value}"\n', encoding="utf-8")
    return config


def _common_dir(repo: Path) -> Path:
    return (repo / ".git").resolve()


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid} for the store marker tests",
        content_hash=f"{bid}-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-10-22T00:00:00Z",
        last_retrieved_at=None,
    )


def _write(ids: list[str]) -> None:
    store = db_paths._open_store()
    try:
        for bid in ids:
            store.insert_belief(_belief(bid))
    finally:
        store.close()


def _context_of(path: Path, bid: str) -> str:
    store = MemoryStore(str(path))
    try:
        got = store.get_belief(bid)
    finally:
        store.close()
    assert got is not None
    return got.project_context


def _status() -> str:
    buf = io.StringIO()
    assert cli._cmd_stats(argparse.Namespace(), buf) == 0
    return buf.getvalue()


def _status_value(out: str, label: str) -> str:
    for line in out.splitlines():
        if line.startswith(f"{label}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no {label!r} line in:\n{out}")


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Reach the marker, git and global rules, all inside `tmp_path`."""
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.delenv("AELFRICE_TRANSCRIPTS_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()
    return tmp_path


@pytest.fixture()
def monorepo(env: Path) -> Path:
    """A repository with two project folders, each with its own marker."""
    repo = _init_repo(env / "monorepo")
    _marker(repo / "project-a", ".aelfrice/store")
    _marker(repo / "project-b", ".aelfrice/store")
    return repo


# --- acceptance 1, 2: one store per marker folder; git and env rules ------


def test_two_marker_folders_report_two_stores_and_counts(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_a = monorepo / "project-a"
    project_b = monorepo / "project-b"
    (project_a / "src").mkdir()

    monkeypatch.chdir(project_a / "src")
    _write(["a1"])
    out_a = _status()
    monkeypatch.chdir(project_b)
    _write(["b1", "b2"])
    out_b = _status()

    assert Path(_status_value(out_a, "store")) == (
        project_a.resolve() / ".aelfrice" / "store" / "memory.db"
    )
    assert Path(_status_value(out_b, "store")) == (
        project_b.resolve() / ".aelfrice" / "store" / "memory.db"
    )
    assert _status_value(out_a, "beliefs") == "1"
    assert _status_value(out_b, "beliefs") == "2"
    assert not (_common_dir(monorepo) / "aelfrice" / "memory.db").exists()


def test_the_repository_root_without_a_marker_uses_the_git_store(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(monorepo)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_GIT
    assert selection.path == _common_dir(monorepo) / "aelfrice" / "memory.db"
    assert selection.marker is None


def test_aelfrice_db_outranks_a_marker(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pinned = tmp_path / "pinned" / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(pinned))
    monkeypatch.chdir(monorepo / "project-a")
    selection = resolve_store()
    assert selection.rule == STORE_RULE_ENV
    assert selection.path == pinned
    assert db_paths.db_path() == pinned


def test_outside_git_with_no_marker_uses_the_global_store(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = env / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_GLOBAL
    assert selection.path == env / "home" / ".aelfrice" / "memory.db"


def test_a_marker_outside_git_still_selects_its_store(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = env / "notes"
    config = _marker(folder, "store")
    monkeypatch.chdir(folder)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.marker == config.resolve()
    assert selection.path == folder.resolve() / "store" / "memory.db"


# --- acceptance 3, 4: walk bound and nesting -------------------------------


def test_a_marker_above_the_git_root_is_ignored(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = env / "outer"
    _marker(outer, "outer-store")
    repo = _init_repo(outer / "repo")
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    selection = resolve_store()
    assert selection.rule == STORE_RULE_GIT
    assert selection.path == _common_dir(repo) / "aelfrice" / "memory.db"


def test_a_marker_at_the_git_root_applies(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinguishing case for the test above: the bound is inclusive."""
    repo = _init_repo(env / "repo")
    _marker(repo, "root-store")
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.path == repo.resolve() / "root-store" / "memory.db"


def test_the_nearest_of_nested_markers_wins(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(env / "repo")
    outer = repo / "project"
    inner = outer / "inner"
    _marker(outer, "outer-store")
    _marker(inner, "inner-store")
    (inner / "deep").mkdir()
    (outer / "other").mkdir()

    monkeypatch.chdir(inner / "deep")
    assert resolve_store().path == inner.resolve() / "inner-store" / "memory.db"
    monkeypatch.chdir(outer / "other")
    assert resolve_store().path == outer.resolve() / "outer-store" / "memory.db"


@pytest.mark.parametrize(
    "inner_text",
    [
        "[retrieval]\nposterior_weight = 0.5\n",
        "[retrieval\nx = \n",
    ],
    ids=["no-store-table", "unparseable"],
)
def test_a_nearer_config_without_a_store_table_keeps_the_outer_marker(
    env: Path, monkeypatch: pytest.MonkeyPatch, inner_text: str,
) -> None:
    """An unrelated or broken config in a subfolder can't move that
    subfolder's writes from the marker store to the shared store."""
    repo = _init_repo(env / "repo")
    outer = repo / "project"
    config = _marker(outer, "outer-store")
    inner = outer / "inner"
    inner.mkdir()
    (inner / CONFIG_FILENAME).write_text(inner_text, encoding="utf-8")
    monkeypatch.chdir(inner)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.marker == config.resolve()
    assert selection.path == outer.resolve() / "outer-store" / "memory.db"


def test_a_nearer_config_hides_no_broken_outer_marker(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken root marker stays an error below a nearer config."""
    repo = _init_repo(env / "repo")
    (repo / CONFIG_FILENAME).write_text("[store]\npath = 3\n", encoding="utf-8")
    sub = repo / "sub"
    sub.mkdir()
    (sub / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    monkeypatch.chdir(sub)
    with pytest.raises(StoreMarkerError):
        resolve_store()


def test_the_marker_walk_stops_at_the_git_root(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing over a config without `[store]` never climbs past the
    work-tree root, even when a config sits at the root."""
    outer = env / "outer"
    _marker(outer, "outer-store")
    repo = _init_repo(outer / "repo")
    (repo / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    sub = repo / "sub"
    sub.mkdir()
    (sub / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    monkeypatch.chdir(sub)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_GIT
    assert selection.path == _common_dir(repo) / "aelfrice" / "memory.db"


def test_outside_git_only_the_nearest_config_counts(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no work-tree root to bound it, the walk does not climb, so a
    marker in an ancestor folder can't capture a nested folder."""
    outer = env / "notes"
    _marker(outer, "outer-store")
    inner = outer / "inner"
    inner.mkdir()
    (inner / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    monkeypatch.chdir(inner)
    assert resolve_store().rule == STORE_RULE_GLOBAL


# --- path forms ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("stores/a", "stores/a/memory.db"),
        ("stores/a/", "stores/a/memory.db"),
        ("../sibling", "../sibling/memory.db"),
    ],
)
def test_a_path_names_the_directory_that_holds_the_store(
    env: Path, monkeypatch: pytest.MonkeyPatch, value: str, expected: str,
) -> None:
    folder = env / "repo" / "proj"
    _init_repo(env / "repo")
    _marker(folder, value)
    monkeypatch.chdir(folder)
    assert resolve_store().path == folder.resolve() / expected


@pytest.mark.parametrize("value", ["stores/a.db", "~/b.db", "x.db/"])
def test_a_path_naming_a_db_file_is_an_error(
    env: Path, monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    """Each store owns its folder, so two stores never share the identity
    file and the turn log kept beside them."""
    folder = env / "repo" / "proj"
    _init_repo(env / "repo")
    _marker(folder, value)
    monkeypatch.chdir(folder)
    with pytest.raises(StoreMarkerError, match="names a file"):
        resolve_store()


def test_absolute_and_home_paths(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(env / "repo")
    absolute = env / "elsewhere" / "abs"
    _marker(repo / "a", str(absolute))
    _marker(repo / "b", "~/stores/b")

    monkeypatch.chdir(repo / "a")
    assert resolve_store().path == absolute / "memory.db"
    monkeypatch.chdir(repo / "b")
    assert resolve_store().path == env / "home" / "stores" / "b" / "memory.db"


def test_a_relative_store_is_per_worktree(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(env / "repo")
    _marker(repo / "project-a", ".aelfrice/store")
    _commit_all(repo)
    linked = env / "linked"
    _git("worktree", "add", "-q", "-b", "wt", str(linked), cwd=repo)

    monkeypatch.chdir(repo / "project-a")
    main_store = resolve_store().path
    monkeypatch.chdir(linked / "project-a")
    linked_store = resolve_store().path

    assert main_store == repo.resolve() / "project-a" / ".aelfrice" / "store" / "memory.db"
    assert linked_store == (
        linked.resolve() / "project-a" / ".aelfrice" / "store" / "memory.db"
    )


def test_a_git_store_is_shared_by_every_worktree(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(env / "repo")
    _marker(repo / "project-a", "git:project-a")
    _commit_all(repo)
    linked = env / "linked"
    _git("worktree", "add", "-q", "-b", "wt", str(linked), cwd=repo)
    expected = (
        _common_dir(repo) / "aelfrice" / "stores" / "project-a" / "memory.db"
    )

    monkeypatch.chdir(repo / "project-a")
    assert resolve_store().path == expected
    _write(["from-main"])
    monkeypatch.chdir(linked / "project-a")
    assert resolve_store().path == expected
    _write(["from-linked"])

    store = MemoryStore(str(expected))
    try:
        assert store.get_belief("from-main") is not None
        assert store.get_belief("from-linked") is not None
    finally:
        store.close()


@pytest.mark.parametrize(
    "value",
    ["git:", "git:../x", "git:a/b", "git:a\\\\b", "git:..", "git:.hidden",
     "git: a", "git:a:b"],
)
def test_a_bad_git_store_name_is_an_error_not_a_fallback(
    env: Path, monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    repo = _init_repo(env / "repo")
    _marker(repo / "proj", value)
    monkeypatch.chdir(repo / "proj")
    with pytest.raises(StoreMarkerError, match="git:"):
        db_paths.db_path()


def test_a_git_store_outside_a_repository_is_an_error(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _marker(env / "plain", "git:x")
    monkeypatch.chdir(env / "plain")
    with pytest.raises(StoreMarkerError, match="needs a git repository"):
        db_paths.db_path()


@pytest.mark.parametrize(
    "body",
    [
        "[store]\n",
        '[store]\npath = ""\n',
        '[store]\npath = "   "\n',
        "[store]\npath = 3\n",
        'store = "x"\n',
        "[store]\npath = .aelfrice/store\n",
    ],
    ids=["no-path", "empty", "blank", "not-a-string", "not-a-table",
         "unparseable"],
)
def test_a_marker_with_no_usable_path_is_an_error(
    env: Path, monkeypatch: pytest.MonkeyPatch, body: str,
) -> None:
    repo = _init_repo(env / "repo")
    folder = repo / "proj"
    folder.mkdir()
    (folder / CONFIG_FILENAME).write_text(body, encoding="utf-8")
    monkeypatch.chdir(folder)
    with pytest.raises(StoreMarkerError, match=CONFIG_FILENAME):
        db_paths.db_path()


def test_an_unparseable_config_without_a_store_header_is_no_marker(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken file that never held a marker resolves as it always did."""
    repo = _init_repo(env / "repo")
    (repo / CONFIG_FILENAME).write_text("[retrieval\nx = \n", encoding="utf-8")
    monkeypatch.chdir(repo)
    assert resolve_store().rule == STORE_RULE_GIT


def test_aelf_reports_a_bad_marker_and_creates_no_store(
    env: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    repo = _init_repo(env / "repo")
    config = _marker(repo / "proj", "git:../escape")
    monkeypatch.chdir(repo / "proj")
    assert cli.main(["status"], out=io.StringIO()) == 2
    err = capsys.readouterr().err
    assert str(config.resolve()) in err
    assert not (_common_dir(repo) / "aelfrice").exists()


def test_db_path_takes_a_start_directory(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch, env: Path,
) -> None:
    """The seam a hook uses to resolve for the agent's directory."""
    elsewhere = env / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    project_a = monorepo / "project-a"
    assert db_paths.db_path(project_a) == (
        project_a.resolve() / ".aelfrice" / "store" / "memory.db"
    )
    assert db_paths.db_path(monorepo) == (
        _common_dir(monorepo) / "aelfrice" / "memory.db"
    )
    assert resolve_store().rule == STORE_RULE_GLOBAL


# --- acceptance 6: identity -------------------------------------------------


def test_a_marker_store_gets_a_sidecar_named_for_its_folder(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_a = monorepo / "project-a"
    monkeypatch.chdir(project_a)
    _write(["a1"])
    store_path = resolve_store().path
    expected = db_paths._identity_from_marker_dir(project_a)

    assert expected.startswith("project-a-")
    sidecar = store_path.parent / "identity"
    assert db_paths._first_sidecar_entry(sidecar) == expected
    assert _context_of(store_path, "a1") == expected
    assert db_paths.repo_identity() == expected


def test_two_marker_folders_get_two_identities(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    idents: list[str] = []
    for name in ("project-a", "project-b"):
        monkeypatch.chdir(monorepo / name)
        idents.append(db_paths.store_identity(resolve_store()))
    assert idents[0] != idents[1]


def test_an_existing_sidecar_wins_over_the_folder(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(monorepo / "project-a")
    store_dir = resolve_store().path.parent
    store_dir.mkdir(parents=True)
    (store_dir / "identity").write_text("seeded-elsewhere\n", encoding="utf-8")
    _write(["a1"])
    assert _context_of(store_dir / "memory.db", "a1") == "seeded-elsewhere"


def test_a_copied_store_keeps_its_identity(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch, env: Path,
) -> None:
    monkeypatch.chdir(monorepo / "project-a")
    _write(["before"])
    original = resolve_store().path
    before = _context_of(original, "before")

    copy_dir = env / "backup" / "store-copy"
    shutil.copytree(original.parent, copy_dir)
    monkeypatch.setenv("AELFRICE_DB", str(copy_dir / "memory.db"))
    _write(["after"])

    assert before != ""
    assert _context_of(copy_dir / "memory.db", "after") == before
    assert db_paths.repo_identity_from_db_path(copy_dir / "memory.db") == before


def test_a_read_only_open_seeds_no_sidecar(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(monorepo / "project-a")
    db_paths.open_store_for_read().close()
    assert not (resolve_store().path.parent / "identity").exists()


@pytest.fixture()
def shared(env: Path) -> tuple[Path, Path]:
    """A repository and a linked worktree, both marked `git:shared`."""
    repo = _init_repo(env / "repo")
    _marker(repo, "git:shared")
    _commit_all(repo)
    linked = env / "linked"
    _git("worktree", "add", "-q", "-b", "wt", str(linked), cwd=repo)
    return repo, linked


@pytest.mark.parametrize("first", ["main", "linked"])
def test_a_git_store_identity_is_the_same_from_every_worktree(
    shared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, first: str,
) -> None:
    repo, linked = shared
    order = [repo, linked] if first == "main" else [linked, repo]
    expected = db_paths._identity_from_git_store(_common_dir(repo), "shared")
    for i, where in enumerate(order):
        monkeypatch.chdir(where)
        db_paths.clear_git_common_dir_cache()
        _write([f"b{i}"])
    store_path = resolve_store().path
    assert expected.startswith("repo-shared-")
    assert _context_of(store_path, "b0") == expected
    assert _context_of(store_path, "b1") == expected
    sidecar = (store_path.parent / "identity").read_text(encoding="utf-8")
    entries = [
        line.strip() for line in sidecar.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # One entry: a second worktree adds no alias line.
    assert entries == [expected]


def test_a_git_store_keeps_its_identity_when_the_sidecar_is_lost(
    shared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, linked = shared
    monkeypatch.chdir(repo)
    _write(["before"])
    store_path = resolve_store().path
    (store_path.parent / "identity").unlink()
    monkeypatch.chdir(linked)
    db_paths.clear_git_common_dir_cache()
    _write(["after"])
    assert _context_of(store_path, "after") == _context_of(store_path, "before")


def test_a_marker_naming_the_repo_store_keeps_the_repo_identity(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker that points back at the shared store must not seed that
    store's sidecar with the marker folder's identity."""
    repo = _init_repo(env / "repo")
    repo_store = _common_dir(repo) / "aelfrice" / "memory.db"
    _marker(repo / "project", str(repo_store.parent))
    monkeypatch.chdir(repo / "project")
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.owned_store
    _write(["via-marker"])
    expected = db_paths._identity_from_git_common_dir(_common_dir(repo))
    assert _context_of(repo_store, "via-marker") == expected
    monkeypatch.chdir(repo)
    db_paths.clear_git_common_dir_cache()
    assert db_paths.repo_identity() == expected


# --- acceptance 7: visibility ----------------------------------------------


def test_status_names_each_rule(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch, env: Path,
) -> None:
    project_a = monorepo / "project-a"
    monkeypatch.chdir(project_a)
    marker = project_a.resolve() / CONFIG_FILENAME
    assert _status_value(_status(), "store rule") == f"marker ({marker})"

    monkeypatch.chdir(monorepo)
    assert _status_value(_status(), "store rule") == "git"

    plain = env / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert _status_value(_status(), "store rule") == "global"

    monkeypatch.setenv("AELFRICE_DB", str(env / "pinned.db"))
    out = _status()
    assert _status_value(out, "store rule") == "env (AELFRICE_DB)"
    assert _status_value(out, "store") == str(env / "pinned.db")


def test_doctor_names_the_rule_and_the_marker(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_a = monorepo / "project-a"
    monkeypatch.chdir(project_a)
    buf = io.StringIO()
    cli._print_doctor_store_check(buf)
    marker = project_a.resolve() / CONFIG_FILENAME
    assert f"store rule: marker ({marker})" in buf.getvalue().splitlines()


# --- stores beside the marker store ----------------------------------------


def test_the_turn_log_follows_the_marker_store(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turns logged in one folder are ingested into that folder's store."""
    monkeypatch.chdir(monorepo / "project-a")
    assert transcript_logger.transcripts_dir() == (
        resolve_store().path.parent / "transcripts"
    )
    monkeypatch.chdir(monorepo)
    assert transcript_logger.transcripts_dir() == (
        _common_dir(monorepo) / "aelfrice" / "transcripts"
    )


def test_no_marker_resolution_is_unchanged(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository with a config file but no `[store]` table resolves to
    the store it resolved to before markers existed, with the identity it
    had then."""
    repo = _init_repo(env / "repo")
    (repo / CONFIG_FILENAME).write_text(
        "[retrieval]\nposterior_weight = 0.5\n", encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    path = db_paths.db_path()
    assert path == _common_dir(repo) / "aelfrice" / "memory.db"
    assert db_paths.store_identity(resolve_store()) == (
        db_paths._identity_from_git_common_dir(_common_dir(repo))
    )
    assert os.path.basename(path.parent) == "aelfrice"


# --- review fixes: HOME repos, encodings, owned stores --------------------


def test_a_repository_rooted_at_home_still_passes_over_a_nearer_config(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `$HOME` is itself a work tree, a folder under it is inside
    git, so a nearer config without `[store]` can't hide its marker."""
    home = env / "home"
    _init_repo(home)
    notes = home / "notes"
    _marker(notes, ".aelfrice/store")
    sub = notes / "sub"
    sub.mkdir()
    (sub / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    monkeypatch.chdir(sub)
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.path == notes.resolve() / ".aelfrice" / "store" / "memory.db"


@pytest.mark.parametrize(
    "text",
    [
        '﻿[store]\npath = "s"\n',
        'store.path = "s"\n[retrieval\n',
        'store = { path = "s"\n',
        "[store.extra]\nx = \n",
    ],
    ids=["bom", "dotted", "inline", "subtable"],
)
def test_a_broken_marker_in_any_spelling_is_an_error(
    env: Path, monkeypatch: pytest.MonkeyPatch, text: str,
) -> None:
    repo = _init_repo(env / "repo")
    folder = repo / "proj"
    folder.mkdir()
    (folder / CONFIG_FILENAME).write_text(text, encoding="utf-8")
    monkeypatch.chdir(folder)
    with pytest.raises(StoreMarkerError) as raised:
        resolve_store()
    if text.startswith("﻿"):
        assert "byte order mark" in str(raised.value)


def test_a_marker_naming_the_global_store_seeds_no_identity(
    env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global store carries no identity, whoever opens it."""
    global_dir = env / "home" / ".aelfrice"
    _marker(env / "scratch", str(global_dir))
    monkeypatch.chdir(env / "scratch")
    selection = resolve_store()
    assert selection.rule == STORE_RULE_MARKER
    assert selection.owned_store
    _write(["via-marker"])
    assert not (global_dir / "identity").exists()
    assert _context_of(global_dir / "memory.db", "via-marker") == ""

    # Even a sidecar planted there doesn't label global-store rows.
    (global_dir / "identity").write_text("planted\n", encoding="utf-8")
    plain = env / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert resolve_store().rule == STORE_RULE_GLOBAL
    _write(["no-marker"])
    assert _context_of(global_dir / "memory.db", "no-marker") == ""


# --- project-warm ------------------------------------------------------------


def test_project_warm_opens_the_marker_store(
    monorepo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import project_warm
    from aelfrice import store as store_mod

    folder = monorepo / "project-a"
    monkeypatch.chdir(folder)
    _write(["a1"])
    marker_store = resolve_store().path
    opened: list[str] = []
    real = store_mod.MemoryStore

    def spy(path: str) -> MemoryStore:
        opened.append(path)
        return real(path)

    monkeypatch.setattr(store_mod, "MemoryStore", spy)
    monkeypatch.chdir(monorepo)
    project_warm._warm_store(monorepo, folder)
    assert opened == [str(marker_store)]

    (folder / CONFIG_FILENAME).write_text("[store]\npath = 3\n", encoding="utf-8")
    opened.clear()
    project_warm._warm_store(monorepo, folder)
    assert opened == []


# --- commit ingest -----------------------------------------------------------

_TRIPLE_MSG = "the spec is derived from the prior memo"


def _commit_staged(repo: Path, message: str) -> str:
    _git(
        "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", message,
        cwd=repo,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def _stage(repo: Path, rel: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{rel}\n", encoding="utf-8")
    _git("add", rel, cwd=repo)


def _commit_sessions(path: Path) -> int:
    if not path.exists():
        return 0
    store = MemoryStore(str(path))
    try:
        row = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) FROM sessions WHERE model = 'commit-ingest'",
        ).fetchone()
    finally:
        store.close()
    return int(row[0])


@pytest.fixture()
def marked_repo(env: Path) -> Path:
    repo = _init_repo(env / "repo")
    _marker(repo / "project-a", "git:project-a")
    _marker(repo / "project-b", "git:project-b")
    _commit_all(repo)
    return repo


def _named_store(repo: Path, name: str) -> Path:
    return _common_dir(repo) / "aelfrice" / "stores" / name / "memory.db"


def _repo_store(repo: Path) -> Path:
    return _common_dir(repo) / "aelfrice" / "memory.db"


def test_a_commit_in_a_marker_folder_lands_in_that_store(
    marked_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import hook_commit_ingest as hk

    monkeypatch.chdir(marked_repo)
    _stage(marked_repo, "project-a/notes.txt")
    rev = _commit_staged(marked_repo, _TRIPLE_MSG)
    hk.git_hook_ingest(rev, str(marked_repo))
    assert _commit_sessions(_named_store(marked_repo, "project-a")) == 1
    assert _commit_sessions(_named_store(marked_repo, "project-b")) == 0
    assert not _repo_store(marked_repo).exists()


def test_a_commit_spanning_stores_lands_in_each(
    marked_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import hook_commit_ingest as hk

    monkeypatch.chdir(marked_repo)
    _stage(marked_repo, "project-b/b.txt")
    _stage(marked_repo, "top.txt")
    rev = _commit_staged(marked_repo, _TRIPLE_MSG)
    hk.git_hook_ingest(rev, str(marked_repo))
    assert _commit_sessions(_named_store(marked_repo, "project-b")) == 1
    assert _commit_sessions(_repo_store(marked_repo)) == 1
    assert _commit_sessions(_named_store(marked_repo, "project-a")) == 0


def test_a_broken_marker_skips_the_commit(
    marked_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import hook_commit_ingest as hk

    monkeypatch.chdir(marked_repo)
    (marked_repo / "project-a" / CONFIG_FILENAME).write_text(
        "[store]\npath = 3\n", encoding="utf-8",
    )
    _stage(marked_repo, "project-a/notes.txt")
    rev = _commit_staged(marked_repo, _TRIPLE_MSG)
    hk.git_hook_ingest(rev, str(marked_repo))
    assert not _repo_store(marked_repo).exists()
    assert _commit_sessions(_named_store(marked_repo, "project-a")) == 0


def test_a_deleted_folder_resolves_from_its_nearest_existing_parent(
    marked_repo: Path, env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import hook_commit_ingest as hk

    monkeypatch.chdir(marked_repo)
    _stage(marked_repo, "project-a/gone/x.txt")
    _commit_staged(marked_repo, "seed")
    _git("rm", "-q", "-r", "project-a/gone", cwd=marked_repo)
    rev = _commit_staged(marked_repo, _TRIPLE_MSG)
    hk.git_hook_ingest(rev, str(marked_repo))
    assert _commit_sessions(_named_store(marked_repo, "project-a")) == 1
    assert not (env / "home" / ".aelfrice" / "memory.db").exists()


# --- hook entry points with a broken marker -----------------------------------


_HOOKS = [
    "aelf-hook", "aelf-transcript-logger", "aelf-pre-compact-hook",
    "aelf-commit-ingest", "aelf-search-tool-hook", "aelf-agent-context-hook",
    "aelf-session-start-hook", "aelf-stop-hook", "aelf-pre-issue-hook",
    "aelf-claude-memory-mirror",
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("hook", _HOOKS)
def test_a_hook_with_a_broken_marker_exits_0_and_writes_no_store(
    env: Path, hook: str,
) -> None:
    import json
    import sys

    script = Path(sys.executable).parent / hook
    if not script.exists():
        pytest.skip(f"{hook} is not installed next to the interpreter")
    repo = _init_repo(env / "repo")
    folder = repo / "proj"
    folder.mkdir()
    (folder / CONFIG_FILENAME).write_text("[store]\npath = 3\n", encoding="utf-8")
    payload = {
        "hook_event_name": "UserPromptSubmit", "session_id": "s",
        "cwd": str(folder), "prompt": "probe the store marker",
        "source": "startup", "tool_name": "Bash",
        "tool_input": {"command": "git commit -m x"}, "tool_response": {},
    }
    child_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("AELFRICE_", "AELF_", "GIT_"))
    }
    child_env["HOME"] = str(env / "home")
    child_env["USERPROFILE"] = str(env / "home")
    result = subprocess.run(
        [str(script)], input=json.dumps(payload), capture_output=True,
        text=True, cwd=folder, env=child_env, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert not _repo_store(repo).exists()
    assert not (env / "home" / ".aelfrice" / "memory.db").exists()
