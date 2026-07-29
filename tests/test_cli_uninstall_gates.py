"""CLI-level gates for `aelf uninstall` (#1173).

Before #1173 the --purge gate printed one path and one size while the
command went on to delete only that one file. It now deletes the whole
artifact set, so the gate has to enumerate it -- a "type PURGE to
confirm" prompt is only meaningful if the user can see what they are
confirming.

Every test here sandboxes the install-state sentinels. `_cmd_uninstall`
clears them, and pointing them at the real `~/.aelfrice/` would break the
developer's own install as a side effect of running the suite.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pytest

from aelfrice import auto_install, cli
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

_SECRET = "no vector embeddings"


@pytest.fixture
def sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """A seeded store plus redirected install-state sentinels."""
    store_dir = tmp_path / "aelfrice"
    store_dir.mkdir()
    db = store_dir / "memory.db"
    store = MemoryStore(str(db))
    store.insert_belief(
        Belief(
            id="B" + "0" * 15, content=_SECRET, content_hash="h" + "0" * 15,
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, created_at="2026-07-29T00:00:00Z",
            last_retrieved_at=None,
        )
    )
    # Left OPEN deliberately: that is the live-hook shape, and it is the
    # only state in which the WAL sidecars exist to be enumerated.
    (store_dir / "hook_audit.jsonl").write_text(
        '{"rendered_block": "' + _SECRET + '"}\n', encoding="utf-8",
    )
    (store_dir / "memory.db.bm25f").write_bytes(b"sidecar")
    (store_dir / "transcripts").mkdir()
    (store_dir / "transcripts" / "turns.jsonl").write_text(
        "{}\n", encoding="utf-8",
    )

    fake_home = tmp_path / "home" / ".aelfrice"
    fake_home.mkdir(parents=True)
    stamp = fake_home / "installed-manifest-version"
    stamp.write_text("4.2.0", encoding="utf-8")
    uv_sentinel = fake_home / "migrated-to-uv"
    uv_sentinel.touch()
    opt_outs = fake_home / "opt-out-hooks.json"
    opt_outs.write_text('{"hooks": ["Stop"]}', encoding="utf-8")

    monkeypatch.setattr(auto_install, "STAMP_PATH", stamp)
    monkeypatch.setattr(cli, "_MIGRATED_TO_UV_SENTINEL", uv_sentinel)
    monkeypatch.setattr(cli, "db_path", lambda: db)
    # Real one would unlink the developer's ~/.cache/aelfrice cache.
    monkeypatch.setattr(cli, "_clear_update_cache", lambda: None)
    # Password acquisition is covered by _read_password's own tests; these
    # tests are about the gates, and the real one reaches for a terminal.
    monkeypatch.setattr(cli, "_read_password", lambda _a: "hunter2")
    return {
        "store_dir": store_dir, "db": db, "stamp": stamp,
        "uv_sentinel": uv_sentinel, "opt_outs": opt_outs, "store": store,
    }


def _args(**over: object) -> argparse.Namespace:
    base: dict[str, Any] = {
        "keep_db": False, "purge": False, "archive": None, "yes": False,
        "keep_hook": True, "settings_path": None, "password_stdin": False,
        "host": "claude",
    }
    base.update(over)
    return argparse.Namespace(**base)


def _run(args: argparse.Namespace) -> tuple[int, str]:
    out = io.StringIO()
    code = cli._cmd_uninstall(args, out)
    return code, out.getvalue()


# --- The gate must disclose what it is about to destroy ------------------


def test_purge_gate_lists_every_artifact(sandbox: dict[str, Any]) -> None:
    """AC: the manifest is auditable at the confirmation prompt."""
    code, text = _run(_args(purge=True, yes=True))

    assert code == 0
    assert "will permanently delete 6 artifacts" in text
    for name in (
        "memory.db", "memory.db-wal", "memory.db-shm", "memory.db.bm25f",
        "hook_audit.jsonl", "transcripts",
    ):
        assert name in text, f"{name} missing from the purge manifest"


def test_purge_gate_declines_without_confirmation(
    sandbox: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declined gate must leave every artifact in place."""
    monkeypatch.setattr("builtins.input", lambda *_a: "nope")
    before = sorted(p.name for p in sandbox["store_dir"].iterdir())

    code, text = _run(_args(purge=True))

    assert code == 1
    assert "aborted" in text
    assert sorted(p.name for p in sandbox["store_dir"].iterdir()) == before


def test_purge_reports_the_deleted_count(sandbox: dict[str, Any]) -> None:
    code, text = _run(_args(purge=True, yes=True))

    assert code == 0
    assert "deleted 6 artifacts" in text
    assert list(sandbox["store_dir"].iterdir()) == []


def test_archive_gate_says_the_extras_are_deleted_not_encrypted(
    sandbox: dict[str, Any], tmp_path: Path,
) -> None:
    """The disclosure that distinguishes 'encrypted' from 'destroyed'.

    `--archive` encrypts the belief DB only. Users must be told, before
    they type a password, that the remaining artifacts are being deleted
    rather than folded into the archive.
    """
    archive = tmp_path / "backup.enc"

    code, text = _run(_args(
        archive=str(archive), yes=True, password_stdin=False,
    ))

    assert code == 0
    assert "DELETED, not " in text and "encrypted" in text
    assert "hook_audit.jsonl" in text
    assert archive.exists()


def test_archive_gate_can_be_declined(
    sandbox: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining leaves the store intact and writes no archive."""
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    archive = tmp_path / "backup.enc"

    code, text = _run(_args(archive=str(archive)))

    assert code == 1
    assert "aborted" in text
    assert not archive.exists()
    assert sandbox["db"].exists()


def test_keep_db_leaves_everything(sandbox: dict[str, Any]) -> None:
    code, text = _run(_args(keep_db=True))

    assert code == 0
    assert "DB preserved" in text
    assert sandbox["db"].exists()
    assert (sandbox["store_dir"] / "hook_audit.jsonl").exists()


# --- Install-state sentinels ---------------------------------------------


def test_uninstall_clears_the_stamps_so_a_reinstall_re_merges(
    sandbox: dict[str, Any],
) -> None:
    """`maybe_install_manifest` short-circuits on an unchanged version.

    Leaving the stamp meant a same-version reinstall restored none of the
    hooks uninstall had removed: the product looked installed and was
    inert.
    """
    assert sandbox["stamp"].exists()

    code, _text = _run(_args(purge=True, yes=True))

    assert code == 0
    assert not sandbox["stamp"].exists()
    assert not sandbox["uv_sentinel"].exists()


def test_uninstall_preserves_hook_opt_outs(sandbox: dict[str, Any]) -> None:
    """Opt-outs record a user decision; a reinstall should honour it."""
    _run(_args(purge=True, yes=True))

    assert sandbox["opt_outs"].exists()
    assert "Stop" in sandbox["opt_outs"].read_text(encoding="utf-8")


def test_stamp_clearing_survives_absent_sentinels(
    sandbox: dict[str, Any],
) -> None:
    """Idempotent: a second uninstall must not raise on missing stamps."""
    _run(_args(purge=True, yes=True))
    code, _text = _run(_args(purge=True, yes=True))
    assert code == 0


# --- Hook removal --------------------------------------------------------


def test_uninstall_removes_the_rebuilder_hook(
    sandbox: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aelf unsetup` leaves the PreCompact entry unless --rebuilder.

    uninstall means "remove all of it", so it must opt in on the user's
    behalf. Otherwise the entry keeps pointing at a binary the closing
    message tells the user to pip uninstall, and every later compaction
    spawns a missing command.
    """
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(
        cli, "_cmd_unsetup", lambda a, _o: (seen.append(a), 0)[1],
    )

    code, _text = _run(_args(purge=True, yes=True, keep_hook=False))

    assert code == 0
    assert len(seen) == 1
    assert getattr(seen[0], "rebuilder", False) is True


def test_keep_hook_skips_unsetup_entirely(
    sandbox: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(
        cli, "_cmd_unsetup", lambda a, _o: (seen.append(a), 0)[1],
    )

    _run(_args(purge=True, yes=True, keep_hook=True))

    assert seen == []


# --- Mode selection is unchanged -----------------------------------------


def test_no_disposition_flag_is_an_error(sandbox: dict[str, Any]) -> None:
    code, _text = _run(_args())
    assert code == 2
    assert sandbox["db"].exists()


def test_conflicting_disposition_flags_are_an_error(
    sandbox: dict[str, Any],
) -> None:
    code, _text = _run(_args(purge=True, keep_db=True))
    assert code == 2
    assert sandbox["db"].exists()


# --- Size formatting ------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0 bytes"),
        (1, "1 bytes"),
        (1023, "1,023 bytes"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1024 ** 2, "1.0 MiB"),
        (283742208, "270.6 MiB"),
        (1024 ** 3, "1.0 GiB"),
    ],
)
def test_format_size(n: int, expected: str) -> None:
    assert cli._format_size(n) == expected


# --- Size accounting ------------------------------------------------------


def test_artifact_size_sums_a_directory_tree(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "transcripts"
    artifact_dir.mkdir()
    (artifact_dir / "a.jsonl").write_bytes(b"x" * 100)
    (artifact_dir / "nested").mkdir()
    (artifact_dir / "nested" / "b.jsonl").write_bytes(b"x" * 200)
    loose = tmp_path / "feed.jsonl"
    loose.write_bytes(b"x" * 50)

    assert cli._artifact_size(artifact_dir) == 300
    assert cli._artifact_total_bytes([artifact_dir, loose]) == 350


def test_artifact_size_does_not_follow_a_symlinked_dir(
    tmp_path: Path,
) -> None:
    """The manifest must not bill bytes the purge will not free.

    `_remove_artifact` unlinks the link, leaving the target intact, so
    walking the target here would report a size the deletion never
    recovers.
    """
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "big.bin").write_bytes(b"x" * 10_000)
    link = tmp_path / "transcripts"
    link.symlink_to(target, target_is_directory=True)

    assert cli._artifact_size(link) == link.lstat().st_size
    assert cli._artifact_size(link) < 10_000


def test_artifact_size_skips_symlinks_inside_a_directory(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 10_000)
    artifact_dir = tmp_path / "rebuild_logs"
    artifact_dir.mkdir()
    (artifact_dir / "real.json").write_bytes(b"x" * 10)
    (artifact_dir / "link.bin").symlink_to(outside / "big.bin")

    assert cli._artifact_size(artifact_dir) == 10


def test_artifact_size_is_zero_when_stat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stat failure must degrade to 0, never abort the gate."""
    target = tmp_path / "memory.db"
    target.write_bytes(b"x" * 10)
    real_stat = Path.stat

    def boom(self: Path, *a: object, **k: object) -> object:
        if self == target:
            raise OSError("permission denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", boom)
    assert cli._artifact_size(target) == 0


def test_artifact_size_of_a_missing_path_is_zero(tmp_path: Path) -> None:
    assert cli._artifact_size(tmp_path / "gone") == 0


# --- The orphaned-artifact warning ---------------------------------------


def test_purge_warns_about_artifacts_it_will_not_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AELFRICE_DB=~/memory.db` must produce a visible warning, not silence.

    The generically-named artifacts cannot be attributed to aelfrice
    outside a store directory, so they are left alone -- but saying
    nothing about them would be the #1173 defect one level down.
    """
    store_dir = tmp_path / "Downloads"      # deliberately not aelfrice
    store_dir.mkdir()
    db = store_dir / "memory.db"
    store = MemoryStore(str(db))
    store.insert_belief(
        Belief(
            id="B" + "0" * 15, content=_SECRET, content_hash="h" + "0" * 15,
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, created_at="2026-07-29T00:00:00Z",
            last_retrieved_at=None,
        )
    )
    store.close()
    (store_dir / "hook_audit.jsonl").write_text("{}\n", encoding="utf-8")
    (store_dir / "feed.jsonl").write_text("{}\n", encoding="utf-8")
    (store_dir / "transcripts").mkdir()
    (store_dir / "transcripts" / "mine.jsonl").write_text(
        "not aelfrice's\n", encoding="utf-8",
    )

    fake_dotdir = tmp_path / "home" / ".aelfrice"
    fake_dotdir.mkdir(parents=True)
    monkeypatch.setattr(
        auto_install, "STAMP_PATH", fake_dotdir / "installed-manifest-version",
    )
    monkeypatch.setattr(
        cli, "_MIGRATED_TO_UV_SENTINEL", fake_dotdir / "migrated-to-uv",
    )
    monkeypatch.setattr(cli, "db_path", lambda: db)
    monkeypatch.setattr(cli, "_clear_update_cache", lambda: None)

    code, text = _run(_args(purge=True, yes=True))

    assert code == 0
    assert "is not an aelfrice store directory" in text
    assert "will NOT be touched" in text
    for name in ("hook_audit.jsonl", "feed.jsonl", "transcripts"):
        assert name in text, f"{name} missing from the orphan warning"
    # Warned about, and genuinely left alone.
    assert (store_dir / "transcripts" / "mine.jsonl").exists()
    assert (store_dir / "hook_audit.jsonl").exists()
    # The db-anchored artifacts are still removed.
    assert not db.exists()
