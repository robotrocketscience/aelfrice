"""Tests for the shared claude-memory ingest core (#1089).

`ingest_memory_text` is the per-file logic lifted out of the #985 mirror
hook so the reconcile sweep and the hook share one frontmatter ->
origin/prior mapping. These tests pin that mapping at the function level.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.claude_memory import (
    ENV_MIRROR_CLAUDE_MEMORY,
    is_mirror_enabled,
    mirror_opted_out,
    reconcile_sentinel_path,
)
from aelfrice.claude_memory_reconcile import (
    ingest_memory_text,
    maybe_reconcile_claude_memory,
    reconcile_claude_memory,
)
from aelfrice.models import ORIGIN_AGENT_INFERRED, ORIGIN_USER_VALIDATED
from aelfrice.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(":memory:")


def _file(mtype: str, body: str = "The build tool is standardised.") -> str:
    return f"---\nname: x\nmetadata:\n  type: {mtype}\n---\n\n{body}\n"


def _write_fact(memory_dir: Path, name: str, mtype: str, body: str) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / f"{name}.md").write_text(_file(mtype, body), encoding="utf-8")


def test_user_type_maps_to_user_validated() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("user"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_USER_VALIDATED
    finally:
        s.close()


def test_feedback_type_maps_to_user_validated() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("feedback"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_USER_VALIDATED
    finally:
        s.close()


def test_project_type_maps_to_agent_inferred() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("project"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_AGENT_INFERRED
    finally:
        s.close()


def test_no_frontmatter_returns_none_and_writes_nothing() -> None:
    s = _store()
    try:
        assert ingest_memory_text(s, "just a body, no fence") is None
        assert s.count_beliefs() == 0
    finally:
        s.close()


def test_empty_body_returns_none() -> None:
    s = _store()
    try:
        assert ingest_memory_text(s, "---\nname: x\n---\n\n   \n") is None
    finally:
        s.close()


def test_reingest_is_idempotent_corroborates_not_duplicates() -> None:
    s = _store()
    try:
        text = _file("user")
        first = ingest_memory_text(s, text)
        second = ingest_memory_text(s, text)
        assert first == second  # content-derived id
        assert s.count_beliefs() == 1
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Full-set reconcile sweep
# ---------------------------------------------------------------------------


def test_reconcile_ingests_all_fact_files_and_skips_index(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    _write_fact(memory_dir, "one", "user", "the first curated fact")
    _write_fact(memory_dir, "two", "project", "the second curated fact")
    # The MEMORY.md index is a pointer list, not a fact — must be skipped.
    (memory_dir / "MEMORY.md").write_text("- [one](one.md)\n", encoding="utf-8")

    s = _store()
    try:
        res = reconcile_claude_memory(s, memory_dir)
        assert res.ran is True
        assert res.n_files == 2  # index not counted
        assert res.n_ingested == 2
        assert s.count_beliefs() == 2
    finally:
        s.close()


def test_reconcile_missing_dir_is_noop_not_error(tmp_path: Path) -> None:
    s = _store()
    try:
        res = reconcile_claude_memory(s, tmp_path / "does-not-exist")
        assert res.ran is True
        assert res.n_files == 0
        assert s.count_beliefs() == 0
    finally:
        s.close()


def test_reconcile_is_idempotent_across_runs(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _write_fact(memory_dir, "one", "user", "the first curated fact")
    s = _store()
    try:
        reconcile_claude_memory(s, memory_dir)
        reconcile_claude_memory(s, memory_dir)
        assert s.count_beliefs() == 1  # corroborated, not duplicated
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Sentinel-gated maybe_reconcile (the consent event)
# ---------------------------------------------------------------------------


def test_maybe_reconcile_runs_then_writes_sentinel(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _write_fact(memory_dir, "one", "user", "the first curated fact")
    sentinel = tmp_path / "sentinel"
    s = _store()
    try:
        res = maybe_reconcile_claude_memory(
            s, project_path=tmp_path, sentinel_path=sentinel, opted_out=False,
        )
        assert res.ran is True
        # project_path derives the memory dir via the upstream encoding, not
        # tmp_path/memory, so nothing is ingested here — the point is the
        # sentinel is written so the second call short-circuits.
        assert sentinel.exists()
    finally:
        s.close()


def test_maybe_reconcile_second_call_short_circuits(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("already\n", encoding="utf-8")
    s = _store()
    try:
        res = maybe_reconcile_claude_memory(
            s, project_path=tmp_path, sentinel_path=sentinel, opted_out=False,
        )
        assert res.ran is False
        assert "already reconciled" in res.reason
    finally:
        s.close()


def test_maybe_reconcile_opted_out_defers_without_sentinel(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel"
    s = _store()
    try:
        res = maybe_reconcile_claude_memory(
            s, project_path=tmp_path, sentinel_path=sentinel, opted_out=True,
        )
        assert res.ran is False
        assert not sentinel.exists()  # re-arms after opt-out removed
        assert "opted out" in res.reason
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Consent sentinel flips is_mirror_enabled on (default-on-post-consent)
# ---------------------------------------------------------------------------


def test_consent_sentinel_flips_mirror_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_MIRROR_CLAUDE_MEMORY, raising=False)
    monkeypatch.chdir(tmp_path)  # isolate from any real .aelfrice.toml
    db = tmp_path / "memory.db"
    # AELFRICE_DB sandboxes db_path() -> the db-adjacent sentinel lands
    # under tmp, not beside a real store.
    monkeypatch.setenv("AELFRICE_DB", str(db))

    # No sentinel yet -> mirror stays at its opt-in default (off).
    assert is_mirror_enabled() is False

    sentinel = reconcile_sentinel_path(db)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("consented\n", encoding="utf-8")

    # Sentinel present -> mirror flips on.
    assert is_mirror_enabled() is True


def test_explicit_opt_out_beats_consent_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sentinel = reconcile_sentinel_path(db)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("consented\n", encoding="utf-8")

    # Explicit env opt-out wins over the consent sentinel.
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "0")
    assert is_mirror_enabled() is False
    assert mirror_opted_out() is True


def test_mirror_opted_out_reflects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "0")
    assert mirror_opted_out() is True
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "1")
    assert mirror_opted_out() is False
