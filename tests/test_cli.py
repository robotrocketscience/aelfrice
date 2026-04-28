"""cli.main: per-command in-process invocation tests.

Each test sets `AELFRICE_DB` to a tmp_path-scoped DB so the CLI runs
hermetically against a fresh on-disk store. `main(argv=...)` is called
directly with `out` redirected to an io.StringIO so output is
capturable without subprocess overhead. Atomic short tests, one
property each, per the deterministic-atomic-short policy.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from aelfrice.cli import DEFAULT_DB_DIR, DEFAULT_DB_FILENAME, db_path, main
from aelfrice.models import LOCK_USER
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every CLI test gets its own throwaway DB at <tmp>/aelf.db."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


# --- db_path resolution -------------------------------------------------


def test_db_path_honors_env_override(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "elsewhere.db"
    monkeypatch.setenv("AELFRICE_DB", str(target))
    assert db_path() == target


def test_db_path_resolves_to_git_common_dir_when_in_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a git work-tree, the DB lives at <git-common-dir>/aelfrice/memory.db."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.chdir(repo)
    resolved = db_path()
    assert resolved == (repo / ".git" / "aelfrice" / DEFAULT_DB_FILENAME).resolve()


def test_db_path_worktrees_share_one_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two worktrees of one repo resolve to the same DB path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    # need at least one commit before adding a worktree
    (repo / "README").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false",
         "commit", "-q", "-m", "seed"],
        cwd=repo, check=True,
    )
    wt = tmp_path / "worktree-feature"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "feature", str(wt)],
        cwd=repo, check=True,
    )
    monkeypatch.delenv("AELFRICE_DB", raising=False)

    monkeypatch.chdir(repo)
    main_db = db_path()
    monkeypatch.chdir(wt)
    wt_db = db_path()
    assert main_db == wt_db


def test_db_path_falls_back_to_home_outside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-git directory falls back to ~/.aelfrice/memory.db."""
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    non_git = tmp_path / "no-repo"
    non_git.mkdir()
    monkeypatch.chdir(non_git)
    assert db_path() == DEFAULT_DB_DIR / DEFAULT_DB_FILENAME


def test_db_path_env_override_wins_inside_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """$AELFRICE_DB beats git-common-dir resolution."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    target = tmp_path / "explicit.db"
    monkeypatch.setenv("AELFRICE_DB", str(target))
    monkeypatch.chdir(repo)
    assert db_path() == target


# --- onboard ------------------------------------------------------------


def test_onboard_against_empty_dir_exits_zero(tmp_path: Path) -> None:
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    code, _ = _run("onboard", str(repo))
    assert code == 0


def test_onboard_inserts_beliefs_in_db(tmp_path: Path,
                                        isolated_db: Path) -> None:
    repo = tmp_path / "small_repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    code, out = _run("onboard", str(repo))
    assert code == 0
    assert "added" in out
    s = MemoryStore(str(isolated_db))
    try:
        assert s.count_beliefs() >= 1
    finally:
        s.close()


# --- lock + locked + demote ---------------------------------------------


def test_lock_inserts_locked_belief(isolated_db: Path) -> None:
    code, out = _run("lock", "we always sign commits with ssh")
    assert code == 0
    assert "locked:" in out
    s = MemoryStore(str(isolated_db))
    try:
        assert s.count_locked() == 1
    finally:
        s.close()


def test_locked_lists_inserted_lock(isolated_db: Path) -> None:
    _run("lock", "we always sign commits with ssh")
    code, out = _run("locked")
    assert code == 0
    assert "we always sign commits with ssh" in out


def test_locked_pressured_filters_to_pressured_only(isolated_db: Path) -> None:
    """An unpressured lock is hidden by --pressured, then a pressured one shows."""
    _run("lock", "we always sign commits with ssh")
    code, out = _run("locked", "--pressured")
    assert code == 0
    assert "no pressured locks" in out

    # Manually pressure the lock so the next call sees it.
    s = MemoryStore(str(isolated_db))
    try:
        for b in s.list_locked_beliefs():
            b.demotion_pressure = 2
            s.update_belief(b)
    finally:
        s.close()
    code, out = _run("locked", "--pressured")
    assert code == 0
    assert "pressure=2" in out


def test_demote_removes_user_lock(isolated_db: Path) -> None:
    _run("lock", "we always sign commits with ssh")
    s = MemoryStore(str(isolated_db))
    try:
        bid = s.list_locked_beliefs()[0].id
    finally:
        s.close()
    code, out = _run("demote", bid)
    assert code == 0
    assert "demoted" in out
    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
        assert b is not None
        assert b.lock_level != LOCK_USER
    finally:
        s.close()


def test_demote_unknown_id_exits_nonzero(isolated_db: Path) -> None:
    code, _ = _run("demote", "nonexistent")
    assert code != 0


def test_demote_already_unlocked_exits_zero_with_message(
    isolated_db: Path,
) -> None:
    """Demoting a belief that is already unlocked is a no-op success."""
    _run("lock", "the source of truth is the manifest")
    s = MemoryStore(str(isolated_db))
    try:
        bid = s.list_locked_beliefs()[0].id
    finally:
        s.close()
    _run("demote", bid)
    code, out = _run("demote", bid)
    assert code == 0
    assert "not locked" in out


# --- validate (v1.2) ----------------------------------------------------


def _seed_agent_inferred(db: Path, content: str) -> str:
    """Insert one agent_inferred belief and return its id."""
    from aelfrice.models import (
        BELIEF_FACTUAL,
        LOCK_NONE,
        ORIGIN_AGENT_INFERRED,
        Belief,
    )

    s = MemoryStore(str(db))
    bid = "abc123def456"
    try:
        s.insert_belief(Belief(
            id=bid, content=content, content_hash="h",
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE, locked_at=None,
            demotion_pressure=0,
            created_at="2026-04-26T00:00:00Z",
            last_retrieved_at=None,
            origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()
    return bid


def test_validate_promotes_agent_inferred_to_user_validated(
    isolated_db: Path,
) -> None:
    bid = _seed_agent_inferred(isolated_db, "Python is a programming language")
    code, out = _run("validate", bid)
    assert code == 0
    assert "validated" in out
    assert "agent_inferred -> user_validated" in out
    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
        assert b is not None
        from aelfrice.models import ORIGIN_USER_VALIDATED
        assert b.origin == ORIGIN_USER_VALIDATED
    finally:
        s.close()


def test_validate_unknown_id_exits_nonzero(isolated_db: Path) -> None:
    code, _ = _run("validate", "ghost")
    assert code == 1


def test_validate_locked_belief_exits_nonzero(isolated_db: Path) -> None:
    _run("lock", "we always sign commits with ssh")
    s = MemoryStore(str(isolated_db))
    try:
        bid = s.list_locked_beliefs()[0].id
    finally:
        s.close()
    code, _ = _run("validate", bid)
    assert code == 1


def test_validate_idempotent_already_validated(isolated_db: Path) -> None:
    bid = _seed_agent_inferred(isolated_db, "x")
    _run("validate", bid)
    code, out = _run("validate", bid)
    assert code == 0
    assert "already validated" in out


def test_demote_devalidates_user_validated_belief(isolated_db: Path) -> None:
    bid = _seed_agent_inferred(isolated_db, "x")
    _run("validate", bid)
    code, out = _run("demote", bid)
    assert code == 0
    assert "devalidated" in out
    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
        assert b is not None
        from aelfrice.models import ORIGIN_AGENT_INFERRED
        assert b.origin == ORIGIN_AGENT_INFERRED
    finally:
        s.close()


# --- feedback -----------------------------------------------------------


def test_feedback_used_increments_alpha(isolated_db: Path) -> None:
    _run("lock", "the source of truth is the manifest")
    s = MemoryStore(str(isolated_db))
    try:
        bid = s.list_locked_beliefs()[0].id
        pre = s.get_belief(bid)
        assert pre is not None
        pre_alpha = pre.alpha
    finally:
        s.close()
    code, _ = _run("feedback", bid, "used")
    assert code == 0
    s = MemoryStore(str(isolated_db))
    try:
        post = s.get_belief(bid)
        assert post is not None
        assert post.alpha > pre_alpha
    finally:
        s.close()


def test_feedback_invalid_signal_raises_systemexit(isolated_db: Path) -> None:
    """argparse rejects values outside `choices=` by raising SystemExit
    before the handler runs."""
    _run("lock", "anything")
    with pytest.raises(SystemExit):
        _run("feedback", "deadbeef", "ambiguous")


def test_feedback_unknown_belief_exits_nonzero(isolated_db: Path) -> None:
    code, _ = _run("feedback", "nonexistent", "used")
    assert code != 0


# --- search -------------------------------------------------------------


def test_search_finds_locked_belief(isolated_db: Path) -> None:
    _run("lock", "we always sign commits with ssh")
    code, out = _run("search", "ssh")
    assert code == 0
    assert "[locked]" in out


def test_search_no_match_says_no_results(isolated_db: Path) -> None:
    """No locks + no FTS match -> 'no results'. Locks always L0 so this
    test runs on a fresh empty store."""
    code, out = _run("search", "xenomorph12345")
    assert code == 0
    assert "no results" in out


def test_search_empty_store_distinguishes_empty_from_no_match(
    isolated_db: Path,
) -> None:
    """Issue #116: empty-store hit must point at `aelf onboard`."""
    code, out = _run("search", "anything")
    assert code == 0
    assert "store is empty" in out
    assert "aelf onboard" in out


def test_search_populated_store_says_query_not_indexed(
    isolated_db: Path,
) -> None:
    """Issue #116: with beliefs but no match, message names the count."""
    _run("lock", "we always sign commits with ssh")
    code, out = _run("search", "xenomorph12345")
    assert code == 0
    # Either the substring "store has" appears, or hits found one (false
    # positive guard); we only assert the empty-store branch did NOT fire.
    assert "store is empty" not in out


def test_search_with_dot_in_query_does_not_crash(isolated_db: Path) -> None:
    """Regression for the FTS5 escape bug surfaced at v0.5.0."""
    _run("lock", "the project ships at v0.5 with the regex fallback")
    code, out = _run("search", "v0.5")
    assert code == 0
    assert "regex" in out


# --- stats --------------------------------------------------------------


def test_stats_on_empty_db_shows_zeros(isolated_db: Path) -> None:
    code, out = _run("stats")
    assert code == 0
    assert "beliefs:" in out
    assert "0" in out


def test_stats_after_lock_shows_one_belief_one_locked(isolated_db: Path) -> None:
    _run("lock", "we always sign commits with ssh")
    code, out = _run("stats")
    assert code == 0
    # 'beliefs:           1' and 'locked:            1'
    lines = {line.split(":")[0].strip(): line.split(":")[1].strip()
             for line in out.strip().split("\n") if ":" in line}
    assert lines.get("beliefs") == "1"
    assert lines.get("locked") == "1"


# --- health (v1.1.0 auditor) -------------------------------------------


def test_health_on_empty_db_passes_audit(isolated_db: Path) -> None:
    """Empty store: no findings can fire. Exit 0."""
    code, out = _run("health")
    assert code == 0
    assert "audit:" in out
    assert "[ok  ]" in out  # at least one ok line


def test_health_emits_metrics_section(isolated_db: Path) -> None:
    code, out = _run("health")
    assert code == 0
    assert "metrics:" in out
    assert "beliefs" in out


def test_health_points_at_aelf_regime_for_classifier(isolated_db: Path) -> None:
    code, out = _run("health")
    assert code == 0
    assert "aelf regime" in out


def test_status_aliases_health(isolated_db: Path) -> None:
    """`aelf status` produces the same output as `aelf health`."""
    code_h, out_h = _run("health")
    code_s, out_s = _run("status")
    assert code_h == code_s == 0
    assert out_h == out_s


def test_health_exits_nonzero_when_audit_fails(isolated_db: Path) -> None:
    """Forcing an FTS5 drift makes `aelf health` exit 1."""
    _run("lock", "rule for the failing case")
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute("DELETE FROM beliefs_fts")  # type: ignore[attr-defined]
        s._conn.commit()  # type: ignore[attr-defined]
    finally:
        s.close()
    code, out = _run("health")
    assert code == 1
    assert "FAIL" in out


# --- regime (v1.0 classifier preserved) --------------------------------


def test_regime_on_empty_db_reports_insufficient_data(isolated_db: Path) -> None:
    code, out = _run("regime")
    assert code == 0
    assert "insufficient_data" in out


def test_regime_output_contains_brain_mode_label(isolated_db: Path) -> None:
    code, out = _run("regime")
    assert code == 0
    assert "brain mode" in out


# --- General CLI behavior ----------------------------------------------


def test_unknown_subcommand_exits_nonzero(isolated_db: Path) -> None:
    """argparse exits with code 2 on unknown subcommands; SystemExit raised."""
    with pytest.raises(SystemExit):
        _run("definitely-not-a-command")


def test_no_subcommand_exits_nonzero() -> None:
    """argparse `required=True` rejects empty subcommand path."""
    with pytest.raises(SystemExit):
        _run()


# --- --version ----------------------------------------------------------


def test_version_flag_prints_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`aelf --version` prints `aelf <__version__>` to stdout and
    exits 0. argparse's version action raises SystemExit with code 0
    after writing; we catch and inspect captured stdout."""
    from aelfrice import __version__ as version

    with pytest.raises(SystemExit) as excinfo:
        main(argv=["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert f"aelf {version}" in captured.out


def test_version_flag_short_circuits_required_subcommand() -> None:
    """`--version` must short-circuit the required-subcommand check
    (otherwise argparse exits 2 demanding a subcommand)."""
    with pytest.raises(SystemExit) as excinfo:
        main(argv=["--version"])
    assert excinfo.value.code == 0
