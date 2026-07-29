"""Artifact-complete uninstall (#1173).

`uninstall` used to delete exactly one file, `memory.db`. Everything else
the package writes beside it survived, including four artifacts holding
verbatim belief content. Worse, `--archive` encrypted
`db_path.read_bytes()` without checkpointing the write-ahead log, so a
store held open by a live process archived a valid-but-stale main file
while the real content stayed in plaintext in `memory.db-wal`.

The load-bearing tests here are:

* `test_archive_of_a_live_store_recovers_every_belief` -- the data-loss
  regression. Fails on pre-#1173 code with 0 beliefs recovered.
* `test_purge_leaves_no_plaintext_belief_content` -- the privacy
  regression, asserted on file *content* rather than on a filename list,
  so a newly-added artifact type fails it without anyone remembering to
  update a fixture.
* `test_purge_of_an_unowned_store_dir_spares_generic_siblings` -- the
  guard that keeps the fix from deleting `$HOME/transcripts/` when a user
  points `AELFRICE_DB` at `~/memory.db`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice import lifecycle
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

_SECRET = "vector embeddings are prohibited in this project"


def _seed_store(db: Path, n: int = 12) -> MemoryStore:
    """Insert `n` beliefs and return the still-open store.

    Returning the store OPEN is deliberate: that is the shape of a live
    hook process, and it is the only state in which the WAL holds
    un-checkpointed content. Closing it here would mask the bug.
    """
    store = MemoryStore(str(db))
    for i in range(n):
        store.insert_belief(
            Belief(
                id=f"B{i:015x}",
                content=f"{_SECRET} ({i})",
                content_hash=f"h{i:015x}",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-07-29T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    return store


def _seed_siblings(store_dir: Path, db: Path) -> None:
    """Write one of every artifact the package creates beside the store."""
    (store_dir / "hook_audit.jsonl").write_text(
        '{"rendered_block": "' + _SECRET + '"}\n', encoding="utf-8",
    )
    (store_dir / "hook_audit.jsonl.1").write_text(
        '{"rendered_block": "' + _SECRET + '"}\n', encoding="utf-8",
    )
    (store_dir / "feed.jsonl").write_text(
        '{"content": "' + _SECRET + '"}\n', encoding="utf-8",
    )
    (store_dir / "memory.db.bm25f").write_bytes(
        b"bm25f tokens: " + _SECRET.encode("utf-8"),
    )
    (store_dir / "memory.db.bak-20260629").write_bytes(db.read_bytes())
    (store_dir / "memory.db.pre-clamp-2026-05-11.bak").write_bytes(
        db.read_bytes(),
    )
    for sub, name in (
        ("transcripts", "turns.jsonl"),
        ("rebuild_logs", "r1.json"),
        ("telemetry", "user_prompt_submit.jsonl"),
    ):
        (store_dir / sub).mkdir(exist_ok=True)
        (store_dir / sub / name).write_text(
            '{"text": "' + _SECRET + '"}\n', encoding="utf-8",
        )


@pytest.fixture
def live_store(tmp_path: Path) -> tuple[Path, Path, MemoryStore]:
    """An `aelfrice`-named store dir, seeded, with the connection open."""
    store_dir = tmp_path / "aelfrice"
    store_dir.mkdir()
    db = store_dir / "memory.db"
    store = _seed_store(db)
    _seed_siblings(store_dir, db)
    return store_dir, db, store


def _files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _plaintext_leaks(root: Path) -> list[str]:
    """Relative paths of surviving files containing belief content."""
    needle = _SECRET.encode("utf-8")
    return [
        p.relative_to(root).as_posix()
        for p in _files_under(root)
        if needle in p.read_bytes()
    ]


# --- The data-loss regression -------------------------------------------


def test_archive_of_a_live_store_recovers_every_belief(
    live_store: tuple[Path, Path, MemoryStore], tmp_path: Path,
) -> None:
    """#1173: the archive must contain the WAL's committed content.

    Pre-fix this recovered 0 of 12 beliefs: `memory.db` was a nearly
    empty shell and every belief lived in the un-checkpointed WAL, which
    `read_bytes()` never saw. The user was told "original deleted" and
    handed an archive of an empty database.
    """
    _store_dir, db, store = live_store
    expected = len(store.list_belief_ids())
    assert expected == 12

    archive = tmp_path / "backup.enc"
    result = lifecycle.uninstall(
        db, archive_path=archive, archive_password="hunter2",
    )
    assert result.mode == "archived"

    recovered = tmp_path / "recovered.db"
    recovered.write_bytes(lifecycle.decrypt_archive(archive, "hunter2"))
    assert len(MemoryStore(str(recovered)).list_belief_ids()) == expected


def test_checkpoint_wal_folds_the_log_into_the_main_db(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """The mechanism the archive fix rests on, asserted directly."""
    _store_dir, db, _store = live_store
    wal = db.parent / "memory.db-wal"
    assert wal.exists() and wal.stat().st_size > 0, "expected a live WAL"

    assert lifecycle.checkpoint_wal(db) is True
    assert wal.stat().st_size == 0, "TRUNCATE checkpoint should empty the WAL"
    assert _SECRET.encode("utf-8") in db.read_bytes()


def test_checkpoint_wal_is_false_for_a_missing_db(tmp_path: Path) -> None:
    assert lifecycle.checkpoint_wal(tmp_path / "nope.db") is False


def test_checkpoint_wal_survives_a_non_sqlite_file(tmp_path: Path) -> None:
    """Best-effort: a corrupt store must not block an uninstall."""
    junk = tmp_path / "memory.db"
    junk.write_bytes(b"not a database")
    assert lifecycle.checkpoint_wal(junk) is False


# --- The privacy regression ---------------------------------------------


def test_purge_leaves_no_plaintext_belief_content(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """AC4: nothing the package created may survive a --purge.

    Asserted on content, not on an expected-filename list, so a future
    artifact type that leaks belief text fails this without needing
    anyone to update a fixture.
    """
    store_dir, db, _store = live_store
    assert _plaintext_leaks(store_dir), "fixture should start leaky"

    result = lifecycle.uninstall(db, purge=True)

    assert result.mode == "purged"
    assert _plaintext_leaks(store_dir) == []
    assert _files_under(store_dir) == []
    assert result.orphaned == ()


def test_purge_removes_the_wal_where_the_content_actually_lives(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """The single worst pre-fix survivor gets its own assertion.

    `--purge` deleted a 4 KiB `memory.db` and left a 3.5 MB
    `memory.db-wal` holding every belief in plaintext.
    """
    store_dir, db, _store = live_store
    assert (store_dir / "memory.db-wal").exists()

    lifecycle.uninstall(db, purge=True)

    assert not (store_dir / "memory.db-wal").exists()
    assert not (store_dir / "memory.db-shm").exists()


def test_archive_removes_the_derived_plaintext_artifacts(
    live_store: tuple[Path, Path, MemoryStore], tmp_path: Path,
) -> None:
    """An encrypted archive beside plaintext copies is not a guarantee."""
    store_dir, db, _store = live_store
    archive = tmp_path / "backup.enc"

    lifecycle.uninstall(
        db, archive_path=archive, archive_password="hunter2",
    )

    assert archive.exists()
    assert _plaintext_leaks(store_dir) == []


def test_backup_databases_are_removed(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """Both backup naming schemes seen in the wild, via one glob."""
    store_dir, db, _store = live_store
    lifecycle.uninstall(db, purge=True)
    assert list(store_dir.glob("*.bak*")) == []
    assert list(store_dir.glob("*.bm25f")) == []


# --- The guard against deleting things that are not ours ----------------


@pytest.mark.parametrize("dirname", ["aelfrice", ".aelfrice"])
def test_recognised_store_dirs_are_owned(tmp_path: Path, dirname: str) -> None:
    assert lifecycle.store_dir_is_owned(tmp_path / dirname / "memory.db")


def test_an_arbitrary_dir_is_not_owned(tmp_path: Path) -> None:
    assert not lifecycle.store_dir_is_owned(tmp_path / "memory.db")


def test_purge_of_an_unowned_store_dir_spares_generic_siblings(
    tmp_path: Path,
) -> None:
    """`AELFRICE_DB=~/memory.db` must not cost the user `~/transcripts/`.

    The db-anchored artifacts are still removed -- their names are
    prefixed by the store filename, so they cannot belong to anyone
    else. The generically-named ones are reported instead.
    """
    db = tmp_path / "memory.db"          # tmp_path is not named aelfrice
    store = _seed_store(db)
    _seed_siblings(tmp_path, db)
    store.close()
    precious = tmp_path / "transcripts" / "turns.jsonl"

    result = lifecycle.uninstall(db, purge=True)

    assert not db.exists()
    assert not (tmp_path / "memory.db-wal").exists()
    assert not (tmp_path / "memory.db.bm25f").exists()
    assert precious.exists(), "generic sibling outside a store dir was deleted"
    orphan_names = {p.name for p in result.orphaned}
    assert "transcripts" in orphan_names
    assert "hook_audit.jsonl" in orphan_names
    assert "feed.jsonl" in orphan_names
    # Reported, never silently skipped -- silence here is the same defect
    # one level down.
    assert result.orphaned != ()


def test_archive_destination_inside_the_store_dir_is_not_deleted(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """Writing the archive next to the store must not destroy it."""
    store_dir, db, _store = live_store
    archive = store_dir / "backup.enc"

    result = lifecycle.uninstall(
        db, archive_path=archive, archive_password="hunter2",
    )

    assert archive.exists()
    assert archive not in result.removed
    recovered = lifecycle.decrypt_archive(archive, "hunter2")
    assert _SECRET.encode("utf-8") in recovered


# --- Enumeration contract ------------------------------------------------


def test_artifact_paths_puts_the_db_first(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """Deterministic order, DB first: a partially-failed removal must
    never leave the store as the only survivor."""
    _store_dir, db, _store = live_store
    owned, _orphaned = lifecycle.artifact_paths(db)
    assert owned[0] == db


def test_artifact_paths_is_deterministic(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    _store_dir, db, _store = live_store
    assert lifecycle.artifact_paths(db) == lifecycle.artifact_paths(db)


def test_artifact_paths_reports_nothing_for_a_missing_store(
    tmp_path: Path,
) -> None:
    owned, orphaned = lifecycle.artifact_paths(tmp_path / "gone.db")
    assert owned == ()
    assert orphaned == ()


def test_artifact_paths_excludes_the_named_path(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    _store_dir, db, _store = live_store
    sidecar = db.parent / "memory.db.bm25f"
    owned, _ = lifecycle.artifact_paths(db, exclude=sidecar)
    assert sidecar not in owned


def test_sibling_filenames_match_their_owning_modules() -> None:
    """The removal set is single-sourced with the write set.

    `lifecycle` spells these as literals rather than imports, because
    `hook` and `session_ring` sit above it in the import graph. This test
    is what keeps the two in step: renaming a filename at its source
    without updating the removal set fails here, instead of silently
    orphaning that file on every future uninstall.
    """
    from aelfrice import claude_memory, feed_log, hook, hook_audit, session_ring

    expected = {
        hook_audit.AUDIT_FILENAME,
        hook_audit.AUDIT_FILENAME + hook_audit.AUDIT_ROTATED_SUFFIX,
        feed_log.FEED_FILENAME,
        session_ring.SESSION_RING_FILENAME,
        session_ring.SESSION_RING_LOCK_FILENAME,
        hook.SESSION_STATE_FILENAME,
        hook._RECAP_LAST_TS_FILENAME,
        claude_memory._RECONCILE_SENTINEL_NAME,
    }
    assert expected <= set(lifecycle._SIBLING_FILENAMES), (
        "an artifact filename changed at its source but not in the "
        "uninstall removal set: "
        f"{sorted(expected - set(lifecycle._SIBLING_FILENAMES))}"
    )


def test_sibling_dirnames_match_their_owning_modules() -> None:
    """Same contract for the directories."""
    from aelfrice import context_rebuilder, transcript_logger

    expected = {
        transcript_logger.TRANSCRIPTS_SUBDIR,
        context_rebuilder.REBUILD_LOG_DIRNAME,
    }
    assert expected <= set(lifecycle._SIBLING_DIRNAMES), (
        "an artifact directory changed at its source but not in the "
        "uninstall removal set: "
        f"{sorted(expected - set(lifecycle._SIBLING_DIRNAMES))}"
    )


def test_bm25f_sidecar_suffix_is_covered_by_the_db_glob() -> None:
    """The BM25F index is matched by the `<dbname>.*` glob, not a literal."""
    from aelfrice import bm25

    assert bm25._SIDECAR_SUFFIX.startswith(".")


def test_session_state_files_are_removed(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """Per-session hook state must not outlive the store it indexes.

    `session_injected_ids.json` holds belief ids; leaving it behind after
    a purge means a reinstall starts with a ring pointing at rows that no
    longer exist.
    """
    store_dir, db, _store = live_store
    from aelfrice import claude_memory, hook, session_ring

    names = (
        session_ring.SESSION_RING_FILENAME,
        session_ring.SESSION_RING_LOCK_FILENAME,
        hook.SESSION_STATE_FILENAME,
        hook._RECAP_LAST_TS_FILENAME,
        claude_memory._RECONCILE_SENTINEL_NAME,
    )
    for name in names:
        (store_dir / name).write_text("{}", encoding="utf-8")

    lifecycle.uninstall(db, purge=True)

    for name in names:
        assert not (store_dir / name).exists(), f"{name} survived --purge"


def test_purge_is_idempotent(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """A second purge is a no-op, not an error."""
    _store_dir, db, _store = live_store
    first = lifecycle.uninstall(db, purge=True)
    second = lifecycle.uninstall(db, purge=True)
    assert first.removed
    assert second.mode == "purged"
    assert second.removed == ()


def test_keep_db_touches_nothing(
    live_store: tuple[Path, Path, MemoryStore],
) -> None:
    """--keep-db must remain a pure read: no artifact disposition."""
    store_dir, db, _store = live_store
    before = _files_under(store_dir)

    result = lifecycle.uninstall(db, keep_db=True)

    assert result.mode == "kept"
    assert result.removed == ()
    assert result.orphaned == ()
    assert _files_under(store_dir) == before


def test_symlinked_artifact_dir_is_unlinked_not_followed(
    tmp_path: Path,
) -> None:
    """A symlink named `transcripts` must lose the link, not the target.

    Defensive: `_remove_artifact` checks `is_symlink()` before `rmtree`,
    so a symlink planted in the store dir cannot be used to delete an
    arbitrary tree.
    """
    store_dir = tmp_path / "aelfrice"
    store_dir.mkdir()
    db = store_dir / "memory.db"
    _seed_store(db).close()
    outside = tmp_path / "important"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
    (store_dir / "transcripts").symlink_to(outside, target_is_directory=True)

    lifecycle.uninstall(db, purge=True)

    assert not (store_dir / "transcripts").exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == (
        "do not delete"
    )
