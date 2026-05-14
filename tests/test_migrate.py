"""Tests for `aelfrice.migrate` — legacy global DB to per-project store.

The migrate path is read-only against the source DB and write-only
against the target. Tests cover: dry-run vs apply, project filter vs
--all, edge handling, idempotence, refusal when source==target.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.migrate import (
    default_legacy_db_path,
    migrate,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _b(bid: str, content: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h-{bid}",
        alpha=9.0 if locked else 1.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-27T00:00:00Z" if locked else None,
        created_at="2026-04-27T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    p = tmp_path / "myproj"
    p.mkdir()
    return p


@pytest.fixture
def legacy_db(tmp_path: Path, project_root: Path) -> Path:
    """A legacy DB seeded with two beliefs:
    - one mentions the project_root absolute path → matches the filter
    - one is unrelated to the project → does not match
    """
    db = tmp_path / "legacy.db"
    s = MemoryStore(str(db))
    try:
        proj_str = str(project_root.absolute())
        s.insert_belief(_b("aaaa", f"project rule lives at {proj_str}/RULES",
                           locked=True))
        s.insert_belief(_b("bbbb", "unrelated belief from a different project"))
    finally:
        s.close()
    return db


# --- default_legacy_db_path ----------------------------------------


def test_default_legacy_db_path_under_home() -> None:
    p = default_legacy_db_path()
    assert p.name == "memory.db"
    assert p.parent.name == ".aelfrice"
    assert p.parent.parent == Path.home()


# --- guard rails ----------------------------------------------------


def test_migrate_refuses_when_source_equals_target(
    tmp_path: Path, project_root: Path,
) -> None:
    same = tmp_path / "same.db"
    s = MemoryStore(str(same))
    s.close()
    with pytest.raises(ValueError, match="same DB"):
        migrate(
            legacy_path=same,
            target_path=same,
            project_root=project_root,
            apply=True,
            copy_all=False,
        )


def test_migrate_raises_when_source_missing(
    tmp_path: Path, project_root: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        migrate(
            legacy_path=tmp_path / "ghost.db",
            target_path=tmp_path / "tgt.db",
            project_root=project_root,
            apply=False,
            copy_all=False,
        )


# --- dry-run --------------------------------------------------------


def test_dry_run_does_not_create_target(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    target = tmp_path / "tgt.db"
    report = migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=False,
        copy_all=False,
    )
    assert not report.applied
    assert not target.exists()


def test_dry_run_filter_matches_only_project_belief(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    report = migrate(
        legacy_path=legacy_db,
        target_path=tmp_path / "tgt.db",
        project_root=project_root,
        apply=False,
        copy_all=False,
    )
    c = report.counts
    assert c.legacy_beliefs == 2
    assert c.matched_beliefs == 1
    assert c.inserted_beliefs == 1


def test_dry_run_all_matches_every_legacy_belief(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    report = migrate(
        legacy_path=legacy_db,
        target_path=tmp_path / "tgt.db",
        project_root=project_root,
        apply=False,
        copy_all=True,
    )
    assert report.counts.matched_beliefs == 2


# --- apply ---------------------------------------------------------


def test_apply_writes_only_filtered_beliefs(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    target = tmp_path / "tgt.db"
    report = migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=False,
    )
    assert report.applied
    assert report.counts.inserted_beliefs == 1
    s = MemoryStore(str(target))
    try:
        assert s.count_beliefs() == 1
        b = s.get_belief("aaaa")
        assert b is not None
        assert b.lock_level == LOCK_USER
    finally:
        s.close()


def test_apply_preserves_origin_from_legacy_row(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    """Migrating from a v1.2+ legacy DB preserves the origin column.
    The seeded locked belief is backfilled to 'user_stated' by the
    v1.2 catch-up migration (lock_level=user → user_stated); we assert
    the migrate path passes that value through to the target store
    rather than dropping it. Regression for #224."""
    target = tmp_path / "tgt.db"
    migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=True,
    )
    s = MemoryStore(str(target))
    try:
        b = s.get_belief("aaaa")
        assert b is not None
        assert b.origin == "user_stated"
        b2 = s.get_belief("bbbb")
        assert b2 is not None
        assert b2.origin == "unknown"
    finally:
        s.close()


def test_apply_all_writes_every_legacy_belief(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    target = tmp_path / "tgt.db"
    report = migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=True,
    )
    assert report.counts.inserted_beliefs == 2
    s = MemoryStore(str(target))
    try:
        assert s.count_beliefs() == 2
    finally:
        s.close()


def test_apply_is_idempotent(
    tmp_path: Path, project_root: Path, legacy_db: Path,
) -> None:
    target = tmp_path / "tgt.db"
    first = migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=False,
    )
    second = migrate(
        legacy_path=legacy_db,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=False,
    )
    assert first.counts.inserted_beliefs == 1
    assert second.counts.inserted_beliefs == 0
    assert second.counts.skipped_existing_beliefs == 1


# --- edge handling -------------------------------------------------


def test_apply_copies_edges_when_both_endpoints_match(
    tmp_path: Path, project_root: Path,
) -> None:
    legacy = tmp_path / "legacy.db"
    proj_str = str(project_root.absolute())
    s = MemoryStore(str(legacy))
    try:
        s.insert_belief(_b("aaaa", f"project rule {proj_str}/A"))
        s.insert_belief(_b("bbbb", f"project rule {proj_str}/B"))
        s.insert_edge(Edge(src="aaaa", dst="bbbb", type=EDGE_SUPPORTS,
                            weight=1.0))
    finally:
        s.close()
    target = tmp_path / "tgt.db"
    report = migrate(
        legacy_path=legacy,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=False,
    )
    assert report.counts.inserted_edges == 1
    assert report.counts.skipped_orphan_edges == 0


def test_apply_skips_edges_with_unfiltered_endpoint(
    tmp_path: Path, project_root: Path,
) -> None:
    """Edge from project belief to unrelated belief is NOT copied."""
    legacy = tmp_path / "legacy.db"
    proj_str = str(project_root.absolute())
    s = MemoryStore(str(legacy))
    try:
        s.insert_belief(_b("aaaa", f"project rule {proj_str}/A"))
        s.insert_belief(_b("bbbb", "unrelated to project"))
        s.insert_edge(Edge(src="aaaa", dst="bbbb", type=EDGE_SUPPORTS,
                            weight=1.0))
    finally:
        s.close()
    target = tmp_path / "tgt.db"
    report = migrate(
        legacy_path=legacy,
        target_path=target,
        project_root=project_root,
        apply=True,
        copy_all=False,
    )
    assert report.counts.inserted_edges == 0
    assert report.counts.skipped_orphan_edges == 1
