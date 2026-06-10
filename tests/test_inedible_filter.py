"""INEDIBLE filename marker — files containing the marker are
unconditionally skipped by every aelfrice ingest path.

Privacy/security primitive. The marker is case-sensitive
(`INEDIBLE` only) and matches anywhere in the basename.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.inedible import INEDIBLE_MARKER, is_inedible, is_inedible_path
from aelfrice.ingest import ingest_jsonl, ingest_jsonl_dir
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore


# --- Predicate ---------------------------------------------------------


def test_marker_constant_is_uppercase() -> None:
    assert INEDIBLE_MARKER == "INEDIBLE"


def test_is_inedible_matches_full_basename() -> None:
    assert is_inedible("/path/INEDIBLE.md")


def test_is_inedible_matches_prefix() -> None:
    assert is_inedible("/path/INEDIBLE_secrets.txt")


def test_is_inedible_matches_suffix() -> None:
    assert is_inedible("/path/notes_INEDIBLE.txt")


def test_is_inedible_matches_inside() -> None:
    assert is_inedible("/path/partINEDIBLEpart.py")


def test_is_inedible_does_not_match_lowercase() -> None:
    assert not is_inedible("/path/inedible.md")


def test_is_inedible_does_not_match_titlecase() -> None:
    assert not is_inedible("/path/Inedible.md")


def test_is_inedible_only_inspects_basename() -> None:
    """The `is_inedible` primitive inspects the basename only — a
    directory ancestor named INEDIBLE does NOT propagate through it.
    Directory-scoped exclusion is `is_inedible_path`'s job (see
    patterns, deferred."""
    assert not is_inedible("/path/INEDIBLE/file.md")


# --- is_inedible_path (directory-aware) --------------------------------


def test_is_inedible_path_matches_basename() -> None:
    assert is_inedible_path("/path/INEDIBLE_secrets.txt")


def test_is_inedible_path_accepts_path_or_string(tmp_path: Path) -> None:
    """is_inedible_path accepts Path or str for both `path` and `root`,
    mirroring test_is_inedible_accepts_path_or_string and keeping the
    polymorphic API contract explicit (`root` is keyword-only)."""
    root = tmp_path / "root"
    nested = root / "INEDIBLE_drafts"
    nested.mkdir(parents=True)
    f = nested / "notes.jsonl"
    assert is_inedible_path(f, root=root)
    assert is_inedible_path(str(f), root=root)
    assert is_inedible_path(f, root=str(root))
    assert is_inedible_path(str(f), root=str(root))
    # root=None path also accepts both types
    assert is_inedible_path("/a/INEDIBLE/b.jsonl")
    assert is_inedible_path(Path("/a/INEDIBLE/b.jsonl"))


def test_is_inedible_path_propagates_from_ancestor_dir() -> None:
    """Unlike is_inedible, the path-aware helper excludes a file beneath
    an INEDIBLE-named directory (#958)."""
    assert is_inedible_path("/path/INEDIBLE_drafts/notes.jsonl")
    assert is_inedible_path("/a/b/INEDIBLE/c/d/session.jsonl")


def test_is_inedible_path_clean_path_is_not_inedible() -> None:
    assert not is_inedible_path("/path/projects/myapp/session.jsonl")


def test_is_inedible_path_root_bounds_ancestor_check() -> None:
    """With `root` set, the root's own name (and anything above it) is
    not inspected — matching scan_repo, which pushes its walk root
    unconditionally. Components strictly between root and the file are."""
    root = Path("/home/u/INEDIBLE_x/proj")
    # root itself carries the marker, but it is not inspected:
    assert not is_inedible_path(root / "a/session.jsonl", root=root)
    # a directory between root and the file IS inspected:
    assert is_inedible_path(root / "INEDIBLE_sub/session.jsonl", root=root)


def test_is_inedible_path_root_unrelated_falls_back_to_full_walk() -> None:
    """If the path is not under `root`, the helper walks all ancestors
    rather than silently passing."""
    assert is_inedible_path("/other/INEDIBLE_dir/x.jsonl", root=Path("/home/u"))


def test_is_inedible_accepts_path_or_string() -> None:
    assert is_inedible(Path("INEDIBLE.md"))
    assert is_inedible("INEDIBLE.md")


# --- Scanner integration -----------------------------------------------


def test_scanner_skips_inedible_doc_file(tmp_path: Path) -> None:
    (tmp_path / "ok.md").write_text(
        "We always sign commits with ssh.\n"
        "Stick to atomic commits in this project.\n"
    )
    (tmp_path / "INEDIBLE_secrets.md").write_text(
        "We always sign commits with ssh.\n"
        "Secret password: hunter2.\n"
    )
    s = MemoryStore(":memory:")
    try:
        result = scan_repo(s, tmp_path)
    finally:
        s.close()
    assert result.inserted >= 0
    # No belief content from the INEDIBLE file should survive.
    s2 = MemoryStore(":memory:")
    try:
        scan_repo(s2, tmp_path)
        rows = s2._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s2.close()
    contents = "\n".join(row["content"] for row in rows)
    assert "hunter2" not in contents


def test_scanner_skips_inedible_python_file(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(
        '"""This module follows the project conventions for naming."""\n'
    )
    (tmp_path / "INEDIBLE_creds.py").write_text(
        '"""Hardcoded API key for the production database."""\n'
    )
    s = MemoryStore(":memory:")
    try:
        scan_repo(s, tmp_path)
        rows = s._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s.close()
    contents = "\n".join(row["content"] for row in rows)
    assert "Hardcoded API key" not in contents


def test_scanner_skips_inedible_directory(tmp_path: Path) -> None:
    """A directory whose basename contains INEDIBLE is not descended."""
    secret_dir = tmp_path / "INEDIBLE_drafts"
    secret_dir.mkdir()
    (secret_dir / "draft.md").write_text(
        "We always sign commits with ssh.\nSecret leaked content here.\n"
    )
    (tmp_path / "ok.md").write_text(
        "We always sign commits with ssh.\nProject convention applies.\n"
    )
    s = MemoryStore(":memory:")
    try:
        scan_repo(s, tmp_path)
        rows = s._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s.close()
    contents = "\n".join(row["content"] for row in rows)
    assert "leaked content" not in contents


# --- ingest_jsonl single-file ------------------------------------------


def test_ingest_jsonl_skips_inedible_single_file(tmp_path: Path) -> None:
    """A single-file ingest of an INEDIBLE-named JSONL is a no-op."""
    p = tmp_path / "INEDIBLE_session.jsonl"
    p.write_text(
        '{"role": "user", "text": "this should not land", '
        '"session_id": "s1", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    s = MemoryStore(":memory:")
    try:
        result = ingest_jsonl(s, p)
    finally:
        s.close()
    assert result.lines_read == 0
    assert result.turns_ingested == 0
    assert result.beliefs_inserted == 0


def test_ingest_jsonl_processes_safe_filename(tmp_path: Path) -> None:
    """Sanity: a non-INEDIBLE file in the same dir works normally."""
    p = tmp_path / "ok_session.jsonl"
    p.write_text(
        '{"role": "user", "text": "this should land cleanly", '
        '"session_id": "s1", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    s = MemoryStore(":memory:")
    try:
        result = ingest_jsonl(s, p)
    finally:
        s.close()
    assert result.lines_read == 1


# --- ingest_jsonl_dir batch --------------------------------------------


def test_ingest_jsonl_dir_skips_inedible_and_counts(tmp_path: Path) -> None:
    """The batch path skips INEDIBLE files and reports them in
    `files_skipped_inedible`."""
    (tmp_path / "ok1.jsonl").write_text(
        '{"role": "user", "text": "this project always uses uv for environment management", '
        '"session_id": "s1", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    (tmp_path / "INEDIBLE_secret.jsonl").write_text(
        '{"role": "user", "text": "the production database password is secret content", '
        '"session_id": "s2", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    (tmp_path / "ok2.jsonl").write_text(
        '{"role": "user", "text": "we prefer atomic commits over batched commits", '
        '"session_id": "s3", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    s = MemoryStore(":memory:")
    try:
        result = ingest_jsonl_dir(s, tmp_path)
        rows = s._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s.close()
    assert result.files_skipped_inedible == 1
    assert result.files_walked == 3
    contents = "\n".join(row["content"] for row in rows)
    assert "secret content" not in contents
    assert "uv for environment management" in contents
    assert "atomic commits" in contents


def test_ingest_jsonl_single_file_skips_inedible_ancestor_dir(
    tmp_path: Path,
) -> None:
    """A single-file ingest of a JSONL beneath an INEDIBLE-named
    directory is a no-op, even though the file's own name is clean
    (#958)."""
    nested = tmp_path / "INEDIBLE_old" / "projects"
    nested.mkdir(parents=True)
    p = nested / "session.jsonl"
    p.write_text(
        '{"role": "user", "text": "the production database password is secret content", '
        '"session_id": "s1", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    s = MemoryStore(":memory:")
    try:
        result = ingest_jsonl(s, p)
        rows = s._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s.close()
    assert result.lines_read == 0
    assert rows == []


def test_ingest_jsonl_dir_skips_files_under_inedible_dir(
    tmp_path: Path,
) -> None:
    """The batch path prunes files nested under an INEDIBLE-named
    directory, matching scan_repo's directory pruning (#958)."""
    (tmp_path / "ok.jsonl").write_text(
        '{"role": "user", "text": "this project always uses uv for environment management", '
        '"session_id": "s1", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    secret_dir = tmp_path / "INEDIBLE_drafts"
    secret_dir.mkdir()
    (secret_dir / "clean_name.jsonl").write_text(
        '{"role": "user", "text": "the production database password is secret content", '
        '"session_id": "s2", "ts": "2026-01-01T00:00:00Z"}\n'
    )
    s = MemoryStore(":memory:")
    try:
        result = ingest_jsonl_dir(s, tmp_path)
        rows = s._conn.execute(  # type: ignore[attr-defined]
            "SELECT content FROM beliefs"
        ).fetchall()
    finally:
        s.close()
    # Both files are walked; the nested one is skipped as inedible.
    assert result.files_walked == 2
    assert result.files_skipped_inedible == 1
    contents = "\n".join(row["content"] for row in rows)
    assert "secret content" not in contents
    assert "uv for environment management" in contents
