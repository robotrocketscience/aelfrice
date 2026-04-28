"""INEDIBLE filename marker — files containing the marker are
unconditionally skipped by every aelfrice ingest path.

Privacy/security primitive. The marker is case-sensitive
(`INEDIBLE` only) and matches anywhere in the basename.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.inedible import INEDIBLE_MARKER, is_inedible
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
    """A directory ancestor named INEDIBLE does NOT propagate to
    files; users want directory-scoped exclusion via gitignore-style
    patterns, deferred."""
    assert not is_inedible("/path/INEDIBLE/file.md")


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
