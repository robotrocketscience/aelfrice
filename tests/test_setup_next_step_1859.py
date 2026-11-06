"""`aelf setup` names the onboard command for its host (#1859).

Released 5.3.0 printed `aelf onboard .` on the claude host, the terminal
command with the regex classifier, printed it as "this project" from a
folder outside any repository, and printed nothing on Codex.

Every test goes through `main`, so the host routing and both call sites are
in the path. `AELFRICE_DB` stays pinned except where a test is about what
the resolver chooses: the "outside a repository" line has to hold with the
variable set, because `resolve_store()` then reports the `env` rule inside
and outside a repository alike.
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import subprocess
from pathlib import Path

import pytest

from aelfrice.cli import (
    _cmd_setup_codex,
    _print_setup_next_step,
    build_parser,
    main,
)
from aelfrice.db_paths import GIT_STATE_OUTSIDE, _open_store, git_folder_state
from aelfrice.models import Belief

IN_AGENT = "`/aelf:onboard .` in your agent"
IN_CODEX = "`$aelf-onboard .` in Codex"
IN_TERMINAL = "`aelf onboard .`"

PROJECT_LINE = "next step: this project's belief store is empty. Run {} to populate it."
OUTSIDE_LINE = (
    "next step: this folder isn't in a git repository. "
    "Open a project first, then run {}."
)
NO_WORK_TREE_LINE = (
    "next step: this folder isn't in a git work tree. "
    "Open a project first, then run {}."
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway home, Codex home, and pinned store, with the claude
    slash folder and the #1053 opt-out file pointed below it."""
    import aelfrice.auto_install as auto_install
    import aelfrice.cli as cli_mod
    import aelfrice.setup as setup_mod
    from aelfrice.lifecycle import MigrationResult

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "store" / "memory.db"))
    monkeypatch.setattr(
        setup_mod, "SLASH_COMMANDS_DIR_DEFAULT", home / ".claude" / "commands" / "aelf"
    )
    monkeypatch.setattr(
        auto_install, "OPT_OUT_PATH", home / ".aelfrice" / "opt-out-hooks.json"
    )
    monkeypatch.setattr(
        cli_mod,
        "maybe_migrate_to_uv",
        lambda: MigrationResult(False, False, "already on uv tool"),
    )
    return tmp_path


def _folder(root: Path, *, repo: bool, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = root / "work"
    folder.mkdir()
    if repo:
        subprocess.run(["git", "init", "-q"], cwd=folder, check=True, timeout=30)
    monkeypatch.chdir(folder)
    return folder


def _seed() -> None:
    store = _open_store()
    try:
        store.insert_belief(
            Belief(
                id="seed1859",
                content="the store already holds a belief",
                content_hash="seed1859",
                alpha=1.0,
                beta=1.0,
                type="factual",
                lock_level="none",
                locked_at=None,
                created_at="2026-01-01T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    finally:
        store.close()


def _next_steps(*argv: str) -> tuple[int, list[str]]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    lines = buf.getvalue().splitlines()
    return code, [line for line in lines if line.startswith("next step:")]


def _setup(host: str, *extra: str) -> tuple[int, list[str]]:
    if host == "claude":
        return _next_steps("setup", "--scope", "project", *extra)
    return _next_steps("setup", "--host", host, *extra)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("host", "onboard"), [("claude", IN_AGENT), ("codex", IN_CODEX)]
)
def test_repo_with_an_empty_store_names_the_host_command(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str, onboard: str
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    code, lines = _setup(host)
    assert code == 0
    assert lines == [PROJECT_LINE.format(onboard)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("host", "onboard"), [("claude", IN_AGENT), ("codex", IN_CODEX)]
)
def test_outside_a_repo_says_to_open_a_project_first(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str, onboard: str
) -> None:
    _folder(env, repo=False, monkeypatch=monkeypatch)
    code, lines = _setup(host)
    assert code == 0
    assert lines == [OUTSIDE_LINE.format(onboard)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("host", ["claude", "codex"])
@pytest.mark.parametrize("repo", [True, False])
def test_a_store_with_beliefs_prints_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str, repo: bool
) -> None:
    _folder(env, repo=repo, monkeypatch=monkeypatch)
    _seed()
    code, lines = _setup(host)
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
def test_codex_setup_still_creates_no_store_file(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    code, lines = _setup("codex")
    assert code == 0
    assert len(lines) == 1
    assert not (env / "store" / "memory.db").exists()


def _store_files(store_dir: Path) -> dict[str, bytes]:
    """The store's bytes, with an absent write-ahead log read as empty.

    A read-only handle on a WAL store leaves an empty `-wal` and a `-shm`
    behind, because only a writable connection removes them on close. An
    empty log holds no data, so the comparison treats it as absent, and
    `-shm` is a lock file that holds none either.
    """
    wal = store_dir / "memory.db-wal"
    return {
        "db": (store_dir / "memory.db").read_bytes(),
        "wal": wal.read_bytes() if wal.exists() else b"",
    }


def _edit_store(env: Path, *statements: str) -> None:
    conn = sqlite3.connect(env / "store" / "memory.db")
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "statements",
    [
        pytest.param(("DROP TABLE exploration_events",), id="old-schema"),
        pytest.param(
            (
                "UPDATE beliefs SET lock_level = 'user', "
                "locked_at = '2020-01-01T00:00:00Z', "
                "lock_expires_at = '2020-01-02T00:00:00Z'",
            ),
            id="expired-lock",
        ),
    ],
)
def test_codex_setup_leaves_a_store_with_beliefs_byte_identical(
    env: Path, monkeypatch: pytest.MonkeyPatch, statements: tuple[str, ...]
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    _seed()
    _edit_store(env, *statements)
    before = _store_files(env / "store")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []
    assert _store_files(env / "store") == before


@pytest.mark.timeout(60)
def test_codex_setup_counts_an_old_empty_store_without_migrating_it(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    _open_store().close()
    _edit_store(env, "DROP TABLE exploration_events")
    before = _store_files(env / "store")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_CODEX)]
    assert _store_files(env / "store") == before


@pytest.mark.timeout(60)
@pytest.mark.parametrize("kind", ["zero-byte", "another-app"])
def test_codex_setup_leaves_a_file_it_cannot_count_byte_identical(
    env: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    db = env / "store" / "memory.db"
    db.parent.mkdir()
    db.write_bytes(b"")
    if kind == "another-app":
        _edit_store(env, "CREATE TABLE mine (a)")
    before = _store_files(env / "store")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []
    assert _store_files(env / "store") == before


@pytest.mark.timeout(60)
@pytest.mark.parametrize("host", ["claude", "codex"])
def test_a_corrupt_store_exits_0_with_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    db = env / "store" / "memory.db"
    db.parent.mkdir()
    db.write_bytes(b"this is not a database file, " * 300)
    code, lines = _setup(host)
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("host", ["claude", "codex"])
def test_a_store_path_under_a_file_exits_0_with_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """`aelf onboard .` can't create this store, so calling it empty and
    sending the user there ends in a traceback."""
    _folder(env, repo=True, monkeypatch=monkeypatch)
    (env / "blocker").write_text("a file, not a folder", encoding="utf-8")
    monkeypatch.setenv("AELFRICE_DB", str(env / "blocker" / "memory.db"))
    code, lines = _setup(host)
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("host", ["claude", "codex"])
def test_a_broken_marker_exits_0_with_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    folder = _folder(env, repo=False, monkeypatch=monkeypatch)
    (folder / ".aelfrice.toml").write_text("[store]\npath = 5\n", encoding="utf-8")
    monkeypatch.delenv("AELFRICE_DB")
    code, lines = _setup(host)
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
def test_the_line_itself_writes_nothing_to_a_healthy_store(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claude path opens the store for its backfill, so this reads the
    helper alone: whatever setup writes, the line adds nothing to it."""
    _folder(env, repo=True, monkeypatch=monkeypatch)
    _seed()
    _edit_store(
        env,
        "UPDATE beliefs SET lock_level = 'user', "
        "locked_at = '2020-01-01T00:00:00Z', "
        "lock_expires_at = '2020-01-02T00:00:00Z'",
    )
    before = _store_files(env / "store")
    buf = io.StringIO()
    _print_setup_next_step(buf, IN_AGENT)
    assert buf.getvalue() == ""
    assert _store_files(env / "store") == before


@pytest.mark.timeout(60)
def test_codex_without_the_skill_names_the_terminal_command(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    code, lines = _setup("codex", "--no-codex-skills")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_TERMINAL)]


@pytest.mark.timeout(60)
def test_codex_without_the_skill_outside_a_repo(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=False, monkeypatch=monkeypatch)
    code, lines = _setup("codex", "--no-codex-skills")
    assert code == 0
    assert lines == [OUTSIDE_LINE.format(IN_TERMINAL)]


@pytest.mark.timeout(60)
def test_codex_skill_installed_earlier_is_still_named(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    assert _setup("codex")[0] == 0
    code, lines = _setup("codex", "--no-codex-skills")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_CODEX)]


@pytest.mark.timeout(60)
def test_codex_skill_held_by_someone_else_is_not_named(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    skill = env / "home" / ".agents" / "skills" / "aelf-onboard"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: aelf-onboard\ndescription: mine\n---\nmine\n", encoding="utf-8"
    )
    code, lines = _setup("codex")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_TERMINAL)]


@pytest.mark.timeout(60)
def test_codex_skill_that_is_not_utf8_is_not_named(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    skill = env / "home" / ".agents" / "skills" / "aelf-onboard"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_bytes(b"\xff\xfe\x00 not utf8 \x80\x81")
    code, lines = _setup("codex", "--no-codex-skills")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_TERMINAL)]


@pytest.mark.timeout(60)
def test_codex_skills_folder_without_permission_is_not_named(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`is_dir()` raises this on Python 3.12 and returns False on 3.14, so
    the stub gives every version the 3.12 behavior."""
    _folder(env, repo=True, monkeypatch=monkeypatch)

    def refuse(self: Path) -> bool:
        if self.name == "aelf-onboard":
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self)

    real_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", refuse)
    code, lines = _setup("codex", "--no-codex-skills")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_TERMINAL)]


@pytest.mark.timeout(60)
def test_codex_skill_is_looked_for_where_setup_put_it(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    buf = io.StringIO()
    code = _cmd_setup_codex(
        argparse.Namespace(host="codex", force=False, codex_skills=True),
        buf,
        hooks_path=env / "codex-hooks.json",
        skills_dest=env / "skills",
    )
    assert code == 0
    assert PROJECT_LINE.format(IN_CODEX) in buf.getvalue().splitlines()


@pytest.mark.timeout(60)
def test_a_marked_folder_outside_git_is_a_project(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _folder(env, repo=False, monkeypatch=monkeypatch)
    (folder / ".aelfrice.toml").write_text('[store]\npath = "local-store"\n', encoding="utf-8")
    monkeypatch.delenv("AELFRICE_DB")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_CODEX)]


@pytest.mark.timeout(60)
def test_the_shared_store_outside_git_says_to_open_a_project(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aelfrice.db_paths as db_paths

    _folder(env, repo=False, monkeypatch=monkeypatch)
    monkeypatch.delenv("AELFRICE_DB")
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", env / "global")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == [OUTSIDE_LINE.format(IN_CODEX)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("host", "onboard"), [("claude", IN_AGENT), ("codex", IN_CODEX)]
)
def test_a_bare_repository_is_not_called_a_project(
    env: Path, monkeypatch: pytest.MonkeyPatch, host: str, onboard: str
) -> None:
    bare = env / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, timeout=30)
    monkeypatch.chdir(bare)
    code, lines = _setup(host)
    assert code == 0
    assert lines == [NO_WORK_TREE_LINE.format(onboard)]


@pytest.mark.timeout(60)
def test_the_inside_of_a_git_folder_is_not_called_a_project(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _folder(env, repo=True, monkeypatch=monkeypatch)
    monkeypatch.chdir(folder / ".git")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == [NO_WORK_TREE_LINE.format(IN_CODEX)]


@pytest.mark.timeout(60)
def test_a_repository_git_refuses_is_not_called_outside_one(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _folder(env, repo=True, monkeypatch=monkeypatch)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    refused = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=folder, capture_output=True, text=True, timeout=30, check=False,
    )
    if refused.returncode == 0:
        pytest.skip("this git has no ownership check")
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
def test_a_bad_git_dir_is_no_answer_about_the_folder(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=False, monkeypatch=monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(env / "no-such-git-dir"))
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(subprocess.TimeoutExpired(["git"], 5), id="timeout"),
        pytest.param(FileNotFoundError("git"), id="no-git"),
    ],
)
def test_git_giving_no_answer_prints_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    real_run = subprocess.run

    def run(argv: list[str], *a: object, **k: object) -> object:
        if "--is-inside-work-tree" in argv:
            raise failure
        return real_run(argv, *a, **k)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess, "run", run)
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []


@pytest.mark.timeout(60)
def test_git_is_asked_in_the_c_locale(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "outside" answer is read from git's message, which translates."""
    seen: dict[str, object] = {}
    real_run = subprocess.run

    def run(argv: list[str], *a: object, **k: object) -> object:
        seen.update(k)
        return real_run(argv, *a, **k)  # type: ignore[call-overload]

    folder = _folder(env, repo=False, monkeypatch=monkeypatch)
    monkeypatch.setattr(subprocess, "run", run)
    assert git_folder_state(folder) == GIT_STATE_OUTSIDE
    env_used = seen["env"]
    assert isinstance(env_used, dict)
    assert env_used["LC_ALL"] == "C"


@pytest.mark.timeout(60)
def test_project_root_in_a_repo_is_described_from_outside_one(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=False, monkeypatch=monkeypatch)
    root = env / "root"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=30)
    code, lines = _setup("claude", "--project-root", str(root))
    assert code == 0
    assert lines == [PROJECT_LINE.format(IN_AGENT)]


@pytest.mark.timeout(60)
def test_project_root_outside_a_repo_is_described_from_inside_one(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(env, repo=True, monkeypatch=monkeypatch)
    root = env / "root"
    root.mkdir()
    code, lines = _setup("claude", "--project-root", str(root))
    assert code == 0
    assert lines == [OUTSIDE_LINE.format(IN_AGENT)]


@pytest.mark.timeout(60)
def test_project_root_store_is_the_one_counted(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the pin, each folder has its own store: the root's holds a
    belief and the current folder's is empty, so no line prints."""
    import aelfrice.db_paths as db_paths

    root = env / "root"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=30)
    monkeypatch.delenv("AELFRICE_DB")
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", env / "global")
    monkeypatch.chdir(root)
    _seed()
    _folder(env, repo=False, monkeypatch=monkeypatch)
    code, lines = _setup("claude", "--project-root", str(root))
    assert code == 0
    assert lines == []


def test_another_host_is_refused_before_setup_runs() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["setup", "--host", "gemini"])
    assert exc.value.code == 2
