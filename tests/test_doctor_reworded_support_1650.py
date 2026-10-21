"""#1650: `aelf doctor` counts logged reworded-support rows when the log is on.

The section names the flag and the number of `belief_support_evidence`
rows with applied=0. It appears only when `[ingest] log_reworded_support`
or its env var is on, and it reads the store read-only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aelfrice.doctor import diagnose, diagnose_reworded_support_log, format_report
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    SUPPORT_EVIDENCE_KIND_REWORDED,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.supports_log import ENV_LOG_REWORDED_SUPPORT

_HEADER = "reworded-support log (`[ingest] log_reworded_support`): on"


@pytest.fixture(autouse=True)
def _flag_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_LOG_REWORDED_SUPPORT, raising=False)


def _seed(tmp_path: Path) -> Path:
    """A store with two logged rows and one applied row."""
    path = tmp_path / "doctor-support.db"
    store = MemoryStore(str(path))
    try:
        for bid in ("u1", "u2", "u3", "p1"):
            store.insert_belief(Belief(
                id=bid, content=f"The build step {bid} runs under uv.",
                content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
                type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
                created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
            ))
        for bid, applied in (("u1", False), ("u2", False), ("u3", True)):
            store.record_support_evidence(
                bid, "p1", kind=SUPPORT_EVIDENCE_KIND_REWORDED,
                target_sentence_index=0, supporter_session_id="s2",
                supporter_ingest_log_id=None,
                matched_span="The build step runs under uv.", applied=applied,
            )
    finally:
        store.close()
    return path


def _project(tmp_path: Path, *, on: bool) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".aelfrice.toml").write_text(
        f"[ingest]\nlog_reworded_support = {'true' if on else 'false'}\n")
    return root


def test_the_count_is_the_unapplied_rows_when_the_log_is_on(
    tmp_path: Path,
) -> None:
    store = _seed(tmp_path)
    assert diagnose_reworded_support_log(
        str(store), _project(tmp_path, on=True)) == 2


def test_nothing_is_reported_when_the_log_is_off(tmp_path: Path) -> None:
    store = _seed(tmp_path)
    root = _project(tmp_path, on=False)
    assert diagnose_reworded_support_log(str(store), root) is None
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=root, store_path=str(store),
    )
    assert report.reworded_support_logged is None
    assert _HEADER not in format_report(report)


def test_the_env_var_turns_the_section_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed(tmp_path)
    monkeypatch.setenv(ENV_LOG_REWORDED_SUPPORT, "1")
    assert diagnose_reworded_support_log(
        str(store), _project(tmp_path, on=False)) == 2


def test_the_report_renders_the_count(tmp_path: Path) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=_project(tmp_path, on=True),
        store_path=str(_seed(tmp_path)),
    )
    assert report.reworded_support_logged == 2
    text = format_report(report)
    assert _HEADER in text
    assert "  2 reworded restatement(s) logged, none applied" in text


def test_the_report_renders_the_count_with_settings_scanned(
    tmp_path: Path,
) -> None:
    user = tmp_path / "user-settings.json"
    user.write_text("{}", encoding="utf-8")
    report = diagnose(
        user_settings=user, project_root=_project(tmp_path, on=True),
        store_path=str(_seed(tmp_path)),
    )
    assert report.scopes_scanned
    assert "2 reworded restatement(s) logged" in format_report(report)


def test_a_store_without_the_table_counts_zero(tmp_path: Path) -> None:
    store = _seed(tmp_path)
    conn = sqlite3.connect(store)
    try:
        conn.execute("DROP TABLE belief_support_evidence")
        conn.commit()
    finally:
        conn.close()
    assert diagnose_reworded_support_log(
        str(store), _project(tmp_path, on=True)) == 0
    # Read as found: a writable open would have created the table.
    conn = sqlite3.connect(store)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name = 'belief_support_evidence'").fetchone()[0] == 0
    finally:
        conn.close()


def test_an_unreadable_store_is_not_reported(tmp_path: Path) -> None:
    root = _project(tmp_path, on=True)
    assert diagnose_reworded_support_log(":memory:", root) is None
    assert diagnose_reworded_support_log(str(tmp_path / "nope.db"), root) is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    assert diagnose_reworded_support_log(str(empty), root) is None


def test_the_store_is_not_written(tmp_path: Path) -> None:
    store = _seed(tmp_path)
    before = store.read_bytes()
    diagnose_reworded_support_log(str(store), _project(tmp_path, on=True))
    assert store.read_bytes() == before


def test_the_report_renders_a_zero_count(tmp_path: Path) -> None:
    # A store with no logged row: the section still shows, so a user who
    # turned the log on can see it is running.
    path = tmp_path / "doctor-empty.db"
    MemoryStore(str(path)).close()
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=_project(tmp_path, on=True), store_path=str(path),
    )
    assert report.reworded_support_logged == 0
    text = format_report(report)
    assert _HEADER in text
    assert "  0 reworded restatement(s) logged, none applied" in text
