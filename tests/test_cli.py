"""cli.main: per-command in-process invocation tests.

Each test sets `AELFRICE_DB` to a tmp_path-scoped DB so the CLI runs
hermetically against a fresh on-disk store. `main(argv=...)` is called
directly with `out` redirected to an io.StringIO so output is
capturable without subprocess overhead. Atomic short tests, one
property each, per the deterministic-atomic-short policy.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import db_path, main
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


def test_db_path_falls_back_to_default_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    assert db_path().name == "memory.db"


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


# --- health -------------------------------------------------------------


def test_health_on_empty_db_reports_insufficient_data(isolated_db: Path) -> None:
    code, out = _run("health")
    assert code == 0
    assert "insufficient_data" in out


def test_health_output_contains_brain_mode_label(isolated_db: Path) -> None:
    code, out = _run("health")
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
