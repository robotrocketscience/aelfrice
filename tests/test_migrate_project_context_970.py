"""Migrate must preserve project_context provenance (#970).

Before #970, `aelf migrate` dropped project_context, collapsing every
migrated row to '' so two repos' beliefs became mutually visible in the
merged store. These tests pin: already-stamped rows are carried over,
source-'' project rows are stamped with the *source* repo identity, and
locked/global rows stay ''.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice import db_paths
from aelfrice.migrate import migrate
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    BELIEF_SCOPE_PROJECT,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


def _b(
    bid: str,
    *,
    project_context: str = "",
    scope: str = BELIEF_SCOPE_PROJECT,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"{bid}-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-06-18T00:00:00Z",
        last_retrieved_at=None,
        scope=scope,
        project_context=project_context,
    )


def _seed_source(path: Path, beliefs: list[Belief]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    src = MemoryStore(str(path))  # no default → values land verbatim
    for b in beliefs:
        src.insert_belief(b)
    src.close()


def _migrate(src: Path, tgt: Path) -> None:
    migrate(
        legacy_path=src,
        target_path=tgt,
        project_root=Path("/"),
        apply=True,
        copy_all=True,
    )


def _pc(path: Path, bid: str) -> str:
    store = MemoryStore(str(path))
    try:
        b = store.get_belief(bid)
        assert b is not None
        return b.project_context
    finally:
        store.close()


def test_migrate_preserves_already_stamped_context(tmp_path: Path) -> None:
    src = tmp_path / "src" / "aelfrice" / "memory.db"
    tgt = tmp_path / "tgt" / "memory.db"  # non-layout target → no re-stamp
    _seed_source(src, [_b("b1", project_context="prior-ctx")])
    _migrate(src, tgt)
    assert _pc(tgt, "b1") == "prior-ctx"


def test_migrate_stamps_source_identity_on_empty_project_rows(tmp_path: Path) -> None:
    src = tmp_path / "src" / "aelfrice" / "memory.db"  # repo-layout source
    tgt = tmp_path / "tgt" / "memory.db"
    _seed_source(src, [_b("b1")])  # project_context=''
    expected = db_paths.repo_identity_from_db_path(src)
    assert expected != ""
    _migrate(src, tgt)
    assert _pc(tgt, "b1") == expected


def test_migrate_leaves_locked_and_global_unstamped(tmp_path: Path) -> None:
    src = tmp_path / "src" / "aelfrice" / "memory.db"
    tgt = tmp_path / "tgt" / "memory.db"
    _seed_source(src, [
        _b("locked", lock_level=LOCK_USER),
        _b("global", scope=BELIEF_SCOPE_GLOBAL),
    ])
    _migrate(src, tgt)
    assert _pc(tgt, "locked") == ""
    assert _pc(tgt, "global") == ""


def test_migrate_non_repo_source_leaves_empty(tmp_path: Path) -> None:
    src = tmp_path / "memory.db"  # not under 'aelfrice' → source_identity ''
    tgt = tmp_path / "tgt" / "memory.db"
    _seed_source(src, [_b("b1")])
    assert db_paths.repo_identity_from_db_path(src) == ""
    _migrate(src, tgt)
    assert _pc(tgt, "b1") == ""
