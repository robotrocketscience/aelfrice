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
import subprocess
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_setup_codex, build_parser, main
from aelfrice.db_paths import _open_store
from aelfrice.models import Belief

IN_AGENT = "`/aelf:onboard .` in your agent"
IN_CODEX = "`$aelf-onboard .` in Codex"
IN_TERMINAL = "`aelf onboard .`"

PROJECT_LINE = "next step: this project's belief store is empty. Run {} to populate it."
OUTSIDE_LINE = (
    "next step: this folder isn't in a git repository. "
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


@pytest.mark.timeout(60)
def test_a_store_that_cannot_be_read_prints_no_next_step(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aelfrice.cli as cli_mod

    _folder(env, repo=True, monkeypatch=monkeypatch)
    _seed()

    def boom(*_a: object, **_k: object) -> object:
        raise OSError("store is unreadable")

    monkeypatch.setattr(cli_mod, "open_store_for_read", boom)
    code, lines = _setup("codex")
    assert code == 0
    assert lines == []


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


def test_another_host_is_refused_before_setup_runs() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["setup", "--host", "gemini"])
    assert exc.value.code == 2
