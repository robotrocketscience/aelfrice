"""Regression: scan_repo drops noise candidates before classification.

End-to-end coverage of the v1.0.1 noise_filter integration. A
synthetic doc tree is scanned; the result reports `skipped_noise`
greater than zero, and no belief in the resulting store carries
content that matches one of the four noise categories.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore


pytestmark = pytest.mark.regression


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_scan_repo_skips_heading_only_paragraphs(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        # Heading-only paragraph (>24 chars; would survive _split_paragraphs)
        "# Project README Documentation\n## Architecture overview reference\n\n"
        # Real prose paragraph (note: `posterior` is a unique, unrelated token)
        "The system stores beliefs in a SQLite database with a small "
        "feedback loop driving posterior updates."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.skipped_noise >= 1
    # Heading text should not have produced a belief; prose belief should.
    assert s.search_beliefs("Documentation", limit=10) == []
    assert len(s.search_beliefs("posterior", limit=10)) >= 1


def test_scan_repo_skips_checklist_only_paragraphs(tmp_path: Path) -> None:
    _write(
        tmp_path / "TODO.md",
        # Checklist-only paragraph (>24 chars total across lines)
        "- [ ] Wire the noise filter into scan_repo\n"
        "- [x] Write unit tests for noise_filter\n"
        "- [ ] Document in LIMITATIONS\n\n"
        "The actual implementation introduces a new module with one "
        "predicate per noise category and a top-level dispatcher."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.skipped_noise >= 1
    # Checklist tokens should not have produced beliefs.
    assert s.search_beliefs("Wire", limit=10) == []
    # Real prose belief survives.
    assert len(s.search_beliefs("dispatcher", limit=10)) >= 1


def test_scan_repo_skips_license_boilerplate(tmp_path: Path) -> None:
    _write(
        tmp_path / "LICENSE.md",
        "MIT License\n\n"
        "Copyright (c) 2026 ExampleCorp. All rights reserved.\n\n"
        "Permission is hereby granted, free of charge, to any person "
        "obtaining a copy of this software and associated documentation "
        "files, to deal in the Software without restriction."
    )
    _write(
        tmp_path / "README.md",
        "The project is licensed under MIT and ships with the standard "
        "permissive grant; consumers may redistribute freely."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.skipped_noise >= 2
    # License-clause tokens absent from store. (`hereby` and
    # `ExampleCorp` are unique to the LICENSE file; the README uses
    # `grant` casually so we don't search for that — stemmer would
    # match.)
    assert s.search_beliefs("hereby", limit=10) == []
    assert s.search_beliefs("ExampleCorp", limit=10) == []


def test_scan_repo_skips_three_word_fragments(tmp_path: Path) -> None:
    """Three-word fragments are NOT caught by _MIN_PARAGRAPH_CHARS = 24
    if the words are long. Confirm the noise filter catches them."""
    _write(
        tmp_path / "labels.md",
        "INSTRUCTIONS_FOR_LATER:_BEGIN_AT_TOP\n\n"
        "The actual content explains how this module integrates with "
        "the scanner orchestrator and updates the ScanResult counters."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.skipped_noise >= 1
    # Fragment token absent; real-prose token present.
    assert s.search_beliefs("INSTRUCTIONS_FOR_LATER", limit=10) == []
    assert len(s.search_beliefs("orchestrator", limit=10)) >= 1


def test_scan_repo_keeps_real_prose(tmp_path: Path) -> None:
    """Negative case: a doc tree with no noise should produce zero
    skipped_noise and full inserts."""
    _write(
        tmp_path / "ARCH.md",
        "The retrieval pipeline uses BM25 over an FTS5 virtual table.\n"
        "L0 locked beliefs are auto-loaded and never trimmed.\n\n"
        "Feedback events update the Beta-Bernoulli posterior on the "
        "addressed belief and write one row to feedback_history."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.skipped_noise == 0
    assert result.inserted >= 2


def test_scan_result_total_accounts_for_noise(tmp_path: Path) -> None:
    """Invariant: inserted + skipped_existing + skipped_non_persisting
    + skipped_noise == total_candidates."""
    _write(
        tmp_path / "mixed.md",
        "## Architecture overview reference here\n\n"
        "- [ ] One short task line goes here\n"
        "- [x] Two short task line goes here\n\n"
        "The mixed document contains both noise and real prose so the "
        "counter invariants can be checked."
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    total_resolved = (
        result.inserted
        + result.skipped_existing
        + result.skipped_non_persisting
        + result.skipped_noise
    )
    assert total_resolved == result.total_candidates


def test_scan_repo_idempotent_with_noise(tmp_path: Path) -> None:
    """Re-running scan_repo on a tree with noise produces the same
    skipped_noise count and the same insert count (existing beliefs
    move to skipped_existing)."""
    _write(
        tmp_path / "doc.md",
        "## Heading only paragraph\n\n"
        "Real prose belief that should persist on the first scan and "
        "be skipped as existing on the second scan."
    )
    s = MemoryStore(":memory:")
    first = scan_repo(s, tmp_path)
    second = scan_repo(s, tmp_path)
    assert first.skipped_noise == second.skipped_noise
    assert second.inserted == 0
    assert second.skipped_existing >= first.inserted
