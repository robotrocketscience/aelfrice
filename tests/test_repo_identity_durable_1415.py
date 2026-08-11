"""One physical repo, one identity, however the host spells its path (#1415).

The identity was a BLAKE2b digest of the absolute git-common-dir. Native
Windows and WSL address the same directory as
`C:\\tmp\\aelfrice_identity_repro\\.git` and
`/mnt/c/tmp/aelfrice_identity_repro/.git`, so they minted two identities for
one store — while reading and writing the same `memory.db`. Provenance for a
single repository split in two depending on which host a command ran from.

The fix seeds a sidecar in the directory the store already shares, so the
first host's answer becomes canonical. The seed is the *path-derived* value
rather than a fresh UUID, which is what makes the change migration-free on
the host that writes it: the identity there does not move, so rows already
stamped with it stay reachable.

The two host spellings are simulated by calling the resolver with different
path strings against one shared directory — the mechanism under test is
"which value does the sidecar hand back", and that is host-independent. The
genuinely native leg needs both a Windows host and a WSL guest over one NTFS
directory, and is left unchecked on the issue.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice import db_paths
from aelfrice.db_paths import (
    _durable_identity,
    _identity_from_git_common_dir,
    repo_identity_from_db_path,
)
from aelfrice.migrate import migrate
from aelfrice.store import MemoryStore


def _store_dir(root: Path) -> Path:
    """The `<git-common-dir>/aelfrice/` directory the store lives in."""
    d = root / ".git" / "aelfrice"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_two_spellings_of_one_repo_agree(tmp_path: Path) -> None:
    """The headline defect. On main these two disagree.

    The two values are the pair observed on #1415 — what the shipped
    derivation returns for the two host spellings of one repository, not
    two spellings of a path (this resolver is handed identities).
    """
    shared = _store_dir(tmp_path / "repo")
    windows_identity = "aelfrice_identity_repro-b0068862"
    wsl_identity = "aelfrice_identity_repro-74ec8541"

    first = _durable_identity(shared, windows_identity, create=True)
    second = _durable_identity(shared, wsl_identity, create=True)

    assert first == second == windows_identity


def test_the_creating_host_keeps_the_identity_it_already_had(
    tmp_path: Path,
) -> None:
    """Migration-free by construction, which is AC4.

    Whatever this host computed before the change is what it computes after,
    so rows stamped with the old path-derived value remain reachable without
    an alias table or a backfill.
    """
    shared = _store_dir(tmp_path / "repo")
    path_derived = _identity_from_git_common_dir(tmp_path / "repo" / ".git")

    assert _durable_identity(shared, path_derived, create=True) == path_derived
    # ... and still, on every subsequent open.
    assert _durable_identity(shared, path_derived, create=True) == path_derived


def test_the_other_spelling_is_recorded_as_an_alias(tmp_path: Path) -> None:
    """Resolving the split silently would lose the evidence of it.

    An alias line is what lets a later alias-aware filter, or a human
    reading the file, see that two spellings addressed this store.
    """
    shared = _store_dir(tmp_path / "repo")
    _durable_identity(shared, "repo-aaaaaaaa", create=True)
    _durable_identity(shared, "repo-bbbbbbbb", create=True)

    body = (shared / "identity").read_text(encoding="utf-8")
    entries = [
        ln.strip() for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert entries == ["repo-aaaaaaaa", "repo-bbbbbbbb"]


def test_an_alias_is_written_once_not_per_open(tmp_path: Path) -> None:
    """This runs on the store-open path, so a per-open append would grow
    the file without bound."""
    shared = _store_dir(tmp_path / "repo")
    _durable_identity(shared, "repo-aaaaaaaa", create=True)
    for _ in range(5):
        _durable_identity(shared, "repo-bbbbbbbb", create=True)

    body = (shared / "identity").read_text(encoding="utf-8")
    assert body.count("repo-bbbbbbbb") == 1


def test_two_clones_with_the_same_basename_stay_distinct(
    tmp_path: Path,
) -> None:
    """AC2. Each clone has its own git-common-dir, so its own sidecar."""
    one = _store_dir(tmp_path / "a" / "proj")
    two = _store_dir(tmp_path / "b" / "proj")
    id_one = _durable_identity(
        one, _identity_from_git_common_dir(tmp_path / "a" / "proj" / ".git"),
        create=True,
    )
    id_two = _durable_identity(
        two, _identity_from_git_common_dir(tmp_path / "b" / "proj" / ".git"),
        create=True,
    )

    assert id_one != id_two
    assert id_one.startswith("proj-") and id_two.startswith("proj-")


def test_worktrees_sharing_a_common_dir_share_an_identity(
    tmp_path: Path,
) -> None:
    """AC3. Worktrees resolve to one git-common-dir, hence one sidecar.

    Both worktrees and both call sites land on the same file, so the first
    answer written there is the answer every one of them gets — including
    a worktree opened from the host that spells the path the other way.
    """
    shared = _store_dir(tmp_path / "repo")
    common = _identity_from_git_common_dir(tmp_path / "repo" / ".git")
    # Whichever host's spelling this stands for, it is not `common`.
    other_host_spelling = "repo-00000001"

    assert _durable_identity(shared, common, create=True) == common
    assert _durable_identity(shared, other_host_spelling, create=True) == common
    assert repo_identity_from_db_path(shared / "memory.db") == common


def test_repo_identity_from_db_path_uses_the_sidecar(tmp_path: Path) -> None:
    """The store-open path, not just the helper.

    `_open_store` passes `repo_identity_from_db_path(p)` as
    `project_context_default`, so this is the call that decides what new
    rows are stamped with.
    """
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("canonical-deadbeef\n", encoding="utf-8")

    assert repo_identity_from_db_path(shared / "memory.db") == "canonical-deadbeef"


def test_repo_identity_reads_the_sidecar_not_just_its_own_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the fix, and the documented export surface.

    `repo_identity()` is what a user evaluates to set
    ``AELFRICE_PROJECT_CONTEXT``. Left deriving from the path while
    `_open_store` stamped rows from the sidecar, it would disagree with
    the store about the same repository — #1415 again, one layer up. The
    inequality below is what pins that: with a sidecar the resolver
    ignores, the exported value and the stamped value part company.
    """
    git_dir = tmp_path / "repo" / ".git"
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("otherhost-0abcdef1\n", encoding="utf-8")
    monkeypatch.setattr(db_paths, "_git_common_dir", lambda: git_dir)

    assert db_paths.repo_identity() == "otherhost-0abcdef1"
    assert db_paths.repo_identity() != _identity_from_git_common_dir(git_dir)
    # ... and this host's own spelling is recorded beside it, once.
    body = (shared / "identity").read_text(encoding="utf-8")
    entries = [
        ln.strip() for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert entries == [
        "otherhost-0abcdef1", _identity_from_git_common_dir(git_dir),
    ]


def test_resolving_an_identity_does_not_write_into_that_repo(
    tmp_path: Path,
) -> None:
    """Reading another repo's identity must not touch its `.git`.

    `migrate` resolves the SOURCE store's identity to stamp provenance on
    the rows it copies, and opens that store ``mode=ro`` on purpose. A
    resolver that seeded a sidecar for whatever path it was handed made
    `aelf migrate --from <other-repo>/.git/aelfrice/memory.db` write into
    the other repository — in the default DRY RUN, which promises to write
    nothing at all.
    """
    src_dir = _store_dir(tmp_path / "other-repo")
    src = src_dir / "memory.db"
    MemoryStore(str(src)).close()
    before = sorted(p.name for p in src_dir.iterdir())

    migrate(
        legacy_path=src,
        target_path=tmp_path / "mine" / "memory.db",
        project_root=tmp_path,
        apply=False,
        copy_all=True,
    )

    assert not (src_dir / "identity").exists()
    # SQLite's own `-wal`/`-shm` companions are the only thing a `mode=ro`
    # attach is allowed to leave behind; aelfrice adds nothing.
    added = {p.name for p in src_dir.iterdir()} - set(before)
    assert added <= {"memory.db-shm", "memory.db-wal"}


def test_a_comment_only_sidecar_is_not_an_identity(tmp_path: Path) -> None:
    """A file containing only its own header must not read as canonical."""
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("# header only\n\n", encoding="utf-8")

    assert _durable_identity(shared, "repo-cafebabe", create=True) == "repo-cafebabe"


def test_a_zero_byte_sidecar_self_heals(tmp_path: Path) -> None:
    """The state a kill between file creation and the flushed value left.

    A zero-byte `identity` reads as "no entry", so before the re-seed
    every host fell back to its own path-derived value — the #1415 split,
    silently back and permanent, with nothing to repair it.
    """
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_bytes(b"")

    first = _durable_identity(shared, "hostA-11111111", create=True)
    second = _durable_identity(shared, "hostB-22222222", create=True)

    assert first == second == "hostA-11111111"


def test_a_seed_that_cannot_be_placed_leaves_no_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed lands whole or not at all.

    The value is written to a temporary file and moved into place, so a
    failure anywhere before the move leaves nothing for a later open to
    read. Creating `identity` first and writing into it could not promise
    that — which is how the zero-byte sidecar above came to exist.
    """
    shared = _store_dir(tmp_path / "repo")

    def _die(*_a: object, **_k: object) -> None:
        raise OSError("interrupted before the move")

    monkeypatch.setattr(db_paths.os, "link", _die)
    monkeypatch.setattr(db_paths.os, "replace", _die)

    assert _durable_identity(shared, "repo-cafe1234", create=True) == "repo-cafe1234"
    assert not (shared / "identity").exists()
    assert list(shared.iterdir()) == []  # and no temporary file stranded


def test_an_alias_is_not_spliced_onto_an_unterminated_line(
    tmp_path: Path,
) -> None:
    """A canonical line missing its newline must not absorb the alias.

    Appending in place produced `canonical-deadbeefhostB-22222222`, and
    the next open returned that concatenation as the repo identity.
    """
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("canonical-deadbeef", encoding="utf-8")

    assert _durable_identity(shared, "hostB-22222222", create=True) == (
        "canonical-deadbeef"
    )
    assert _durable_identity(shared, "hostB-22222222", create=True) == (
        "canonical-deadbeef"
    )
    body = (shared / "identity").read_text(encoding="utf-8")
    entries = [
        ln.strip() for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert entries == ["canonical-deadbeef", "hostB-22222222"]


def test_an_unwritable_store_dir_falls_back_to_the_path_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft. A read-only checkout must still open its store.

    Degrading to the pre-#1415 behaviour is exactly as correct as it was
    before, and is strictly better than refusing to open. The seam is the
    temporary file: on a read-only filesystem that is where the seed
    first fails.
    """
    shared = _store_dir(tmp_path / "repo")

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(db_paths, "NamedTemporaryFile", _boom)

    assert _durable_identity(shared, "repo-fallback", create=True) == "repo-fallback"
    assert not (shared / "identity").exists()


def test_the_identity_format_is_unchanged(tmp_path: Path) -> None:
    """`models.py` documents the `<basename>-<hash>` shape, and users
    export the value as AELFRICE_PROJECT_CONTEXT. A durable identity that
    changed the format would break both."""
    shared = _store_dir(tmp_path / "repo")
    got = _durable_identity(
        shared, _identity_from_git_common_dir(tmp_path / "repo" / ".git"),
        create=True,
    )

    basename, _, digest = got.rpartition("-")
    assert basename == "repo"
    assert len(digest) == 8 and int(digest, 16) >= 0
