"""Integration tests for `aelf reconcile-claude-memory` (#1089).

All fixtures are synthetic. Path.home and the reconcile sentinel dir are
sandboxed under tmp_path so nothing touches the runner's real ~/.claude or
~/.aelfrice trees.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.claude_memory import derive_memory_dir
from aelfrice.cli import main
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_MIRROR_CLAUDE_MEMORY", raising=False)
    # Sandbox Path.home so both derive_memory_dir and the reconcile
    # sentinel resolve under tmp_path rather than the real ~/.claude / ~.
    monkeypatch.setattr("aelfrice.claude_memory.Path.home", lambda: tmp_path)
    return db


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _fact(mtype: str, body: str) -> str:
    return f"---\nname: x\nmetadata:\n  type: {mtype}\n---\n\n{body}\n"


def test_reconcile_no_memory_dir_is_clean_noop(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    code, out = _run("reconcile-claude-memory", "--project", str(project))
    assert code == 0
    assert "reconcile-claude-memory:" in out


def test_reconcile_force_ingests_fact_files(
    tmp_path: Path, sandbox: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    memory_dir = derive_memory_dir(project)  # resolves under sandboxed home
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "one.md").write_text(
        _fact("user", "the curated fact to ingest"), encoding="utf-8",
    )

    code, out = _run(
        "reconcile-claude-memory", "--project", str(project), "--force",
    )
    assert code == 0
    assert "reconciled 1/1" in out

    s = MemoryStore(str(sandbox))
    try:
        assert s.count_beliefs() == 1
    finally:
        s.close()


def test_reconcile_without_force_respects_sentinel(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    # First run consents + writes the sentinel; second run short-circuits.
    _run("reconcile-claude-memory", "--project", str(project))
    code, out = _run("reconcile-claude-memory", "--project", str(project))
    assert code == 0
    assert "already reconciled" in out


def test_setup_helper_announces_when_memory_dir_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the project has a claude-memory dir, the `aelf setup` helper
    reconciles it and announces the consent event on stderr."""
    from aelfrice.cli import _maybe_reconcile_claude_memory_at_setup

    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    memory_dir = derive_memory_dir(project)  # under sandboxed home
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "one.md").write_text(
        _fact("user", "the curated fact"), encoding="utf-8",
    )

    _maybe_reconcile_claude_memory_at_setup()
    err = capsys.readouterr().err
    assert "claude-memory" in err
    assert "mirror is now ON" in err


def test_setup_helper_defers_when_no_memory_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No claude-memory dir -> the helper defers silently (no consent, no
    sentinel), so a fresh host that hasn't adopted the tool is untouched."""
    from aelfrice.cli import _maybe_reconcile_claude_memory_at_setup

    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)  # derive_memory_dir(project) does not exist

    _maybe_reconcile_claude_memory_at_setup()
    assert capsys.readouterr().err == ""
