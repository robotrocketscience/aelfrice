"""One physical repo, one identity, however the host spells its path (#1415).

The identity was a BLAKE2b digest of the absolute git-common-dir. Native
Windows and WSL address the same directory as `C:\\repo\\.git` and
`/mnt/c/repo/.git`, so they minted two identities for one store — while
reading and writing the same `memory.db`. Provenance for a single repository
split in two depending on which host a command ran from.

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

from aelfrice.db_paths import (
    _durable_identity,
    _identity_from_git_common_dir,
    repo_identity_from_db_path,
)


def _store_dir(root: Path) -> Path:
    """The `<git-common-dir>/aelfrice/` directory the store lives in."""
    d = root / ".git" / "aelfrice"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_two_spellings_of_one_repo_agree(tmp_path: Path) -> None:
    """The headline defect. On main these two disagree."""
    shared = _store_dir(tmp_path / "repo")
    windows_spelling = "repo-b0068862"
    wsl_spelling = "repo-74ec8541"

    first = _durable_identity(shared, windows_spelling)
    second = _durable_identity(shared, wsl_spelling)

    assert first == second == windows_spelling


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

    assert _durable_identity(shared, path_derived) == path_derived
    # ... and still, on every subsequent open.
    assert _durable_identity(shared, path_derived) == path_derived


def test_the_other_spelling_is_recorded_as_an_alias(tmp_path: Path) -> None:
    """Resolving the split silently would lose the evidence of it.

    An alias line is what lets a later alias-aware filter, or a human
    reading the file, see that two spellings addressed this store.
    """
    shared = _store_dir(tmp_path / "repo")
    _durable_identity(shared, "repo-aaaaaaaa")
    _durable_identity(shared, "repo-bbbbbbbb")

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
    _durable_identity(shared, "repo-aaaaaaaa")
    for _ in range(5):
        _durable_identity(shared, "repo-bbbbbbbb")

    body = (shared / "identity").read_text(encoding="utf-8")
    assert body.count("repo-bbbbbbbb") == 1


def test_two_clones_with_the_same_basename_stay_distinct(
    tmp_path: Path,
) -> None:
    """AC2. Each clone has its own git-common-dir, so its own sidecar."""
    one = _store_dir(tmp_path / "a" / "proj")
    two = _store_dir(tmp_path / "b" / "proj")
    id_one = _durable_identity(
        one, _identity_from_git_common_dir(tmp_path / "a" / "proj" / ".git")
    )
    id_two = _durable_identity(
        two, _identity_from_git_common_dir(tmp_path / "b" / "proj" / ".git")
    )

    assert id_one != id_two
    assert id_one.startswith("proj-") and id_two.startswith("proj-")


def test_worktrees_sharing_a_common_dir_share_an_identity(
    tmp_path: Path,
) -> None:
    """AC3. Worktrees resolve to one git-common-dir, hence one sidecar."""
    shared = _store_dir(tmp_path / "repo")
    common = _identity_from_git_common_dir(tmp_path / "repo" / ".git")

    assert _durable_identity(shared, common) == _durable_identity(shared, common)


def test_repo_identity_from_db_path_uses_the_sidecar(tmp_path: Path) -> None:
    """The store-open path, not just the helper.

    `_open_store` passes `repo_identity_from_db_path(p)` as
    `project_context_default`, so this is the call that decides what new
    rows are stamped with.
    """
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("canonical-deadbeef\n", encoding="utf-8")

    assert repo_identity_from_db_path(shared / "memory.db") == "canonical-deadbeef"


def test_a_comment_only_sidecar_is_not_an_identity(tmp_path: Path) -> None:
    """A file containing only its own header must not read as canonical."""
    shared = _store_dir(tmp_path / "repo")
    (shared / "identity").write_text("# header only\n\n", encoding="utf-8")

    assert _durable_identity(shared, "repo-cafebabe") == "repo-cafebabe"


def test_an_unwritable_store_dir_falls_back_to_the_path_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft. A read-only checkout must still open its store.

    Degrading to the pre-#1415 behaviour is exactly as correct as it was
    before, and is strictly better than refusing to open.
    """
    shared = _store_dir(tmp_path / "repo")

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "open", _boom)
    monkeypatch.setattr(Path, "read_text", _boom)

    assert _durable_identity(shared, "repo-fallback") == "repo-fallback"


def test_the_identity_format_is_unchanged(tmp_path: Path) -> None:
    """`models.py` documents the `<basename>-<hash>` shape, and users
    export the value as AELFRICE_PROJECT_CONTEXT. A durable identity that
    changed the format would break both."""
    shared = _store_dir(tmp_path / "repo")
    got = _durable_identity(
        shared, _identity_from_git_common_dir(tmp_path / "repo" / ".git")
    )

    basename, _, digest = got.rpartition("-")
    assert basename == "repo"
    assert len(digest) == 8 and int(digest, 16) >= 0
