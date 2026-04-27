"""extract_filesystem: walks doc files and emits paragraph-shaped candidates.

Tests use pytest's tmp_path fixture for hermetic file-system isolation.
Every test rebuilds its own tree, runs the extractor, and asserts one
property — split per the deterministic-atomic-short policy.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.scanner import SentenceCandidate, extract_filesystem


# --- Empty / missing inputs ---------------------------------------------


def test_missing_path_returns_empty_list(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist"
    assert extract_filesystem(bogus) == []


def test_file_path_instead_of_directory_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "README.md"
    f.write_text("a long paragraph of words here", encoding="utf-8")
    assert extract_filesystem(f) == []


def test_empty_directory_returns_empty(tmp_path: Path) -> None:
    assert extract_filesystem(tmp_path) == []


# --- Single-file extraction ---------------------------------------------


def test_single_paragraph_yields_one_candidate(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    candidates = extract_filesystem(tmp_path)
    assert len(candidates) == 1


def test_two_paragraphs_yield_two_candidates(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5\n\n"
        "the polymorphic onboard handshake lands at v0.6",
        encoding="utf-8",
    )
    candidates = extract_filesystem(tmp_path)
    assert len(candidates) == 2


def test_short_paragraph_is_dropped(tmp_path: Path) -> None:
    """Below 24 chars -> dropped to avoid trivial-bullet pollution."""
    (tmp_path / "README.md").write_text(
        "ok\n\nthe project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    texts = [c.text for c in extract_filesystem(tmp_path)]
    assert "ok" not in texts
    assert any("regex fallback" in t for t in texts)


def test_candidate_text_is_stripped(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "   the project ships the regex fallback at v0.5   ",
        encoding="utf-8",
    )
    c = extract_filesystem(tmp_path)[0]
    assert c.text == "the project ships the regex fallback at v0.5"


def test_candidate_source_format_doc_path_index(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    c = extract_filesystem(tmp_path)[0]
    assert c.source == "doc:README.md:p0"


# --- File-extension filter ----------------------------------------------


def test_md_files_are_picked_up(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert len(extract_filesystem(tmp_path)) == 1


def test_rst_files_are_picked_up(tmp_path: Path) -> None:
    (tmp_path / "doc.rst").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert len(extract_filesystem(tmp_path)) == 1


def test_txt_files_are_picked_up(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert len(extract_filesystem(tmp_path)) == 1


def test_py_files_are_not_picked_up_by_filesystem_extractor(
    tmp_path: Path,
) -> None:
    """Python files are handled by the AST extractor, not this one."""
    (tmp_path / "module.py").write_text(
        "x = 1  # the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


def test_unknown_extension_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text(
        '{"comment": "the project ships only the regex fallback at v0.5"}',
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


# --- Skip-dirs filter ----------------------------------------------------


def test_dot_git_directory_is_skipped(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


def test_node_modules_directory_is_skipped(tmp_path: Path) -> None:
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


def test_venv_directory_is_skipped(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


def test_pycache_directory_is_skipped(tmp_path: Path) -> None:
    pc = tmp_path / "__pycache__"
    pc.mkdir()
    (pc / "stuff.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert extract_filesystem(tmp_path) == []


# --- Recursion + ordering ------------------------------------------------


def test_nested_directories_are_walked(tmp_path: Path) -> None:
    sub = tmp_path / "docs" / "guide"
    sub.mkdir(parents=True)
    (sub / "deep.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    assert len(extract_filesystem(tmp_path)) == 1


def test_nested_path_in_source_is_relative_posix(tmp_path: Path) -> None:
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "deep.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    c = extract_filesystem(tmp_path)[0]
    assert c.source == "doc:docs/deep.md:p0"


def test_files_returned_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text(
        "a paragraph long enough to qualify here",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        "another paragraph long enough to qualify",
        encoding="utf-8",
    )
    sources = [c.source for c in extract_filesystem(tmp_path)]
    assert sources == ["doc:a.md:p0", "doc:b.md:p0"]


def test_repeated_extraction_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "a paragraph long enough to qualify here\n\n"
        "another paragraph long enough to qualify",
        encoding="utf-8",
    )
    one = extract_filesystem(tmp_path)
    two = extract_filesystem(tmp_path)
    assert [c.source for c in one] == [c.source for c in two]
    assert [c.text for c in one] == [c.text for c in two]


# --- Result types --------------------------------------------------------


def test_each_candidate_is_typed(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "a paragraph long enough to qualify here",
        encoding="utf-8",
    )
    candidates = extract_filesystem(tmp_path)
    assert all(isinstance(c, SentenceCandidate) for c in candidates)
