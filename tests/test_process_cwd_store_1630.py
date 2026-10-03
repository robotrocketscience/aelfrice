"""One UserPromptSubmit turn resolves every store consumer from the process cwd (#1630).

The hook payload carries a `cwd`, and the turn threads it as
`payload_cwd` into config and project-context lookups. Every consumer
of the store resolves through `db_path()` instead, which reads the hook
PROCESS cwd: the turn's shared handle (`ups_store`), the executed
`/aelf:` command (#1626), the command-outcome row (#1622), and the
session-state files that `/aelf:scope-out` keys on.

The decision recorded in #1630 is to keep the process cwd. The host
runs a hook in the session's current directory, and the payload `cwd`
follows it, so the two agree; a measured 2,026 matched turns showed no
genuine divergence. Moving some consumers to the payload cwd and not
others splits one turn across two stores, and a typed
`/aelf:scope-out` then fails because its session state stays behind.

These tests pin that decision. Each arm sets the process cwd to repo A
and the payload `cwd` to a different repo B, runs the real hook entry
point, and reads both repositories back. Every arm fails if its one
consumer moves to the payload cwd.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice.hook import user_prompt_submit
from aelfrice.hook_audit import command_outcomes_path_for_db
from aelfrice.session_exclusions import (
    exclusions_path,
    load_exclusions,
    read_current_session_id,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(120)

SESSION = "s1630"
LOCK_STATEMENT = "widgetprompt alpha marker statement."
SEEDED_STATEMENT = "widgetprompt beta marker statement."
SCOPED_STATEMENT = "widgetprompt gammaword marker statement."
RETRIEVAL_PROMPT = "what is the widgetprompt marker statement"


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(path)], check=True, timeout=30,
        capture_output=True,
    )
    return path.resolve()


def _state_dir_of(repo: Path) -> Path:
    return repo / ".git" / "aelfrice"


def _store_of(repo: Path) -> Path:
    return _state_dir_of(repo) / "memory.db"


def _locked(db: Path) -> list[str]:
    if not db.exists():
        return []
    store = MemoryStore(str(db))
    try:
        return [b.content for b in store.list_locked_beliefs()]
    finally:
        store.close()


def _seed_lock(
    db: Path, statement: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock `statement` into `db`, then restore the unset `AELFRICE_DB`."""
    from aelfrice import cli

    db.parent.mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as mp:
        mp.setenv("AELFRICE_DB", str(db))
        assert cli.main(["lock", statement], out=io.StringIO()) == 0
    assert statement in _locked(db)


def _run_hook(prompt: str, cwd: Path) -> tuple[int, str, str]:
    payload = {"prompt": prompt, "session_id": SESSION, "cwd": str(cwd)}
    out, err = io.StringIO(), io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=err,
    )
    return rc, out.getvalue(), err.getvalue()


def _exclusions_in(state_dir: Path) -> list[str]:
    """The exclusions the hook would apply for `SESSION` from `state_dir`.

    Resolved through the session id the state file names, the way the
    hook and `aelf scope-out` resolve it, not by reading the JSON: an
    exclusion stored under the wrong session reads back as empty.
    """
    sid = read_current_session_id(state_dir)
    if sid != SESSION:
        return []
    return load_exclusions(exclusions_path(state_dir), sid)


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Repo A is the process cwd; repo B is the payload cwd.

    `AELFRICE_DB` is cleared so the git-dir branch of resolution is
    live; conftest restores the suite-wide pin after the test. The
    home-dir store points into `tmp_path` too, so a regression that
    falls through to the non-git branch cannot reach a real store.
    """
    from aelfrice import db_paths

    a = _git_repo(tmp_path / "repo_a")
    b = _git_repo(tmp_path / "repo_b")
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", tmp_path / "home_store")
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "1")
    monkeypatch.chdir(a)
    return a, b


def test_the_turn_reads_the_process_cwd_store(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ups_store` reads A: a lock seeded only in A reaches the injection."""
    a, b = repos
    _seed_lock(_store_of(a), SEEDED_STATEMENT, monkeypatch)

    rc, out, err = _run_hook(RETRIEVAL_PROMPT, cwd=b)

    assert rc == 0
    assert SEEDED_STATEMENT in out, (
        f"the turn did not read the process-cwd store; stderr {err!r}"
    )
    assert not _store_of(b).exists(), "the turn opened a store in the payload cwd"


def test_typed_lock_lands_in_the_process_cwd_store(
    repos: tuple[Path, Path],
) -> None:
    a, b = repos

    rc, out, err = _run_hook(f"/aelf:lock {LOCK_STATEMENT}", cwd=b)

    assert rc == 0
    assert "<aelfrice-command-executed>" in out, out
    assert LOCK_STATEMENT in _locked(_store_of(a)), (
        f"the lock did not land in the process-cwd store; stderr {err!r}"
    )
    assert not _store_of(b).exists(), "the lock created a payload-cwd store"


def test_the_outcome_row_sits_beside_the_process_cwd_store(
    repos: tuple[Path, Path],
) -> None:
    """A refused lock is recorded beside the store the turn uses."""
    a, b = repos

    _run_hook("/aelf:lock -widgetprompt", cwd=b)

    assert command_outcomes_path_for_db(_store_of(a)).exists()
    assert not command_outcomes_path_for_db(_store_of(b)).exists()


def test_the_session_state_sits_beside_the_process_cwd_store(
    repos: tuple[Path, Path],
) -> None:
    a, b = repos

    _run_hook(RETRIEVAL_PROMPT, cwd=b)

    assert read_current_session_id(_state_dir_of(a)) == SESSION
    assert not _state_dir_of(b).exists(), (
        "session state was written under the payload cwd"
    )


def test_typed_scope_out_takes_effect_and_holds_on_the_next_turn(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression guard for a split turn.

    `aelf scope-out` resolves its session from the session-state file.
    If the executor resolved a different store than the one the state
    file sits beside, the command finds no active session and fails.
    The control turn proves the statement is injected before the
    exclusion, so its absence afterwards is the exclusion's doing.
    """
    a, b = repos
    _seed_lock(_store_of(a), SCOPED_STATEMENT, monkeypatch)

    _rc, before, _err = _run_hook(RETRIEVAL_PROMPT, cwd=b)
    assert SCOPED_STATEMENT in before, "control: statement not injected"

    _rc, out, err = _run_hook("/aelf:scope-out gammaword", cwd=b)
    assert "<aelfrice-command-executed>" in out, (
        f"scope-out did not take effect; out {out!r} stderr {err!r}"
    )
    assert _exclusions_in(_state_dir_of(a)) == ["gammaword"]

    _rc, after, _err = _run_hook(RETRIEVAL_PROMPT, cwd=b)
    assert SCOPED_STATEMENT not in after, "the exclusion was not honoured"
    assert not _state_dir_of(b).exists()


def test_one_turn_touches_nothing_under_the_payload_cwd(
    repos: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every in-turn consumer together: A gets all of it, B gets nothing."""
    a, b = repos
    _seed_lock(_store_of(a), SEEDED_STATEMENT, monkeypatch)

    rc, out, _err = _run_hook(f"/aelf:lock {LOCK_STATEMENT}", cwd=b)

    assert rc == 0
    assert "<aelfrice-command-executed>" in out, out
    assert {LOCK_STATEMENT, SEEDED_STATEMENT} <= set(_locked(_store_of(a)))
    assert command_outcomes_path_for_db(_store_of(a)).exists()
    assert read_current_session_id(_state_dir_of(a)) == SESSION
    assert not _state_dir_of(b).exists(), sorted(
        p.name for p in _state_dir_of(b).iterdir()
    )


def test_an_explicit_aelfrice_db_wins_for_every_consumer(
    repos: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AELFRICE_DB` overrides both cwds for every consumer in the turn."""
    a, b = repos
    explicit = tmp_path / "explicit" / "memory.db"
    _seed_lock(explicit, SCOPED_STATEMENT, monkeypatch)
    monkeypatch.setenv("AELFRICE_DB", str(explicit))

    _rc, before, _err = _run_hook(RETRIEVAL_PROMPT, cwd=b)
    assert SCOPED_STATEMENT in before, "the turn did not read the explicit store"

    _rc, out, _err = _run_hook(f"/aelf:lock {LOCK_STATEMENT}", cwd=b)
    assert "<aelfrice-command-executed>" in out, out
    assert LOCK_STATEMENT in _locked(explicit)

    _run_hook("/aelf:lock -widgetprompt", cwd=b)
    assert command_outcomes_path_for_db(explicit).exists()

    _rc, out, err = _run_hook("/aelf:scope-out gammaword", cwd=b)
    assert "<aelfrice-command-executed>" in out, (out, err)
    assert read_current_session_id(explicit.parent) == SESSION
    assert _exclusions_in(explicit.parent) == ["gammaword"]

    _rc, after, _err = _run_hook(RETRIEVAL_PROMPT, cwd=b)
    assert SCOPED_STATEMENT not in after

    assert not _state_dir_of(a).exists()
    assert not _state_dir_of(b).exists()
