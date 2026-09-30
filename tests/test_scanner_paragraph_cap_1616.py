"""#1616: a doc paragraph longer than 4,000 characters is not a candidate.

A `.txt` data file with no blank lines is one paragraph. Before the cap,
one such file became a 3.3 MB candidate (54 MB in a real repo) and went
whole into the onboard classifier prompt. The cap was ruled on
2026-09-25: 4,000 characters, drop rather than truncate.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.classification import start_onboard_session
from aelfrice.scanner import (
    _MAX_PARAGRAPH_CHARS,  # pyright: ignore[reportPrivateUsage]
    extract_filesystem,
)
from aelfrice.store import MemoryStore


def _para(n: int, fill: str = "a") -> str:
    """A paragraph of exactly n characters, with no blank line inside."""
    head = "prose paragraph "
    return head + fill * (n - len(head))


def test_the_cap_is_the_ruled_value() -> None:
    assert _MAX_PARAGRAPH_CHARS == 4_000


def test_a_paragraph_at_the_cap_is_kept_and_one_past_it_is_dropped(
    tmp_path: Path,
) -> None:
    (tmp_path / "at.md").write_text(_para(4_000), encoding="utf-8")
    (tmp_path / "over.md").write_text(_para(4_001), encoding="utf-8")
    sources = {c.source: len(c.text) for c in extract_filesystem(tmp_path)}
    assert sources == {"doc:at.md:p0": 4_000}


def test_the_issue_reproduction_yields_no_oversized_candidate(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vectors.txt").write_text(
        " ".join(f"0x{i:08x}" for i in range(300_000)), encoding="utf-8",
    )
    (docs / "README.md").write_text(
        "a short readme paragraph that is real prose about the project",
        encoding="utf-8",
    )
    candidates = extract_filesystem(tmp_path)
    assert [c.source for c in candidates] == ["doc:docs/README.md:p0"]
    assert max(len(c.text) for c in candidates) <= _MAX_PARAGRAPH_CHARS


def test_dropping_a_paragraph_does_not_renumber_the_ones_after_it(
    tmp_path: Path,
) -> None:
    """The index is part of the belief id, so it must not shift."""
    first = "first paragraph, a real sentence about the design"
    last = "last paragraph, another real sentence about the design"
    (tmp_path / "doc.md").write_text(
        f"{first}\n\n{_para(10_000)}\n\n{last}", encoding="utf-8",
    )
    got = {c.source: c.text for c in extract_filesystem(tmp_path)}
    assert got == {"doc:doc.md:p0": first, "doc:doc.md:p2": last}


def test_the_onboard_handshake_emits_no_oversized_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "data.txt").write_text(_para(50_000), encoding="utf-8")
    (repo / "NOTES.md").write_text(
        "a note paragraph with enough words to be a candidate belief",
        encoding="utf-8",
    )
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
    finally:
        store.close()
    lengths = [len(s.text) for s in result.sentences]
    assert lengths, "the NOTES.md paragraph must still be emitted"
    assert max(lengths) <= _MAX_PARAGRAPH_CHARS
