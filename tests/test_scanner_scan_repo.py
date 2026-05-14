"""scan_repo: orchestrator combining the 3 extractors + classification + store.

Atomic-short tests use tmp_path for hermetic file-system isolation and
:memory: SQLite for the store. Each test asserts one property; helpers
build the tree and run the orchestrator.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aelfrice.models import LOCK_NONE
from aelfrice.scanner import ScanResult, scan_repo
from aelfrice.store import MemoryStore

_GIT_AVAILABLE = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not _GIT_AVAILABLE, reason="git binary not on PATH")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )


# --- Empty / missing inputs ---------------------------------------------


def test_missing_root_yields_no_inserts(tmp_path: Path) -> None:
    s = MemoryStore(":memory:")
    bogus = tmp_path / "nope"
    result = scan_repo(s, bogus)
    assert result.inserted == 0


def test_empty_directory_yields_no_inserts(tmp_path: Path) -> None:
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted == 0


def test_empty_directory_yields_no_candidates(tmp_path: Path) -> None:
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.total_candidates == 0


# --- Single-source extraction --------------------------------------------


def test_single_md_paragraph_inserts_one_belief(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted == 1


def test_single_md_paragraph_belief_persists_lock_none(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    beliefs = s.search_beliefs("regex")
    assert len(beliefs) == 1
    assert beliefs[0].lock_level == LOCK_NONE


def test_single_md_paragraph_belief_has_deterministic_id(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s1 = MemoryStore(":memory:")
    s2 = MemoryStore(":memory:")
    scan_repo(s1, tmp_path, now="2026-04-26T00:00:00Z")
    scan_repo(s2, tmp_path, now="2026-04-26T00:00:00Z")
    ids1 = {b.id for b in s1.search_beliefs("regex")}
    ids2 = {b.id for b in s2.search_beliefs("regex")}
    assert ids1 == ids2


def test_single_md_paragraph_uses_provided_now_timestamp(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path, now="2026-04-26T12:34:56Z")
    beliefs = s.search_beliefs("regex")
    assert beliefs[0].created_at == "2026-04-26T12:34:56Z"


# --- Idempotence --------------------------------------------------------


def test_repeat_scan_does_not_duplicate(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    second = scan_repo(s, tmp_path)
    assert second.inserted == 0


def test_repeat_scan_counts_skipped_existing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    second = scan_repo(s, tmp_path)
    assert second.skipped_existing >= 1


def test_repeat_scan_total_belief_count_matches_first_inserted(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5\n\n"
        "another paragraph long enough to qualify",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    first = scan_repo(s, tmp_path)
    scan_repo(s, tmp_path)
    # Beliefs in the store after two runs == beliefs after one.
    all_after = s.search_beliefs("paragraph")
    # cannot search all without query; use specific query that hits both.
    # Use a more comprehensive search:
    p1 = s.search_beliefs("regex")
    p2 = s.search_beliefs("paragraph")
    total_unique_ids = {b.id for b in p1} | {b.id for b in p2}
    assert len(total_unique_ids) == first.inserted
    # Also: re-scan does not silently dedupe wrong thing; same set holds.
    p1b = s.search_beliefs("regex")
    p2b = s.search_beliefs("paragraph")
    total_after_b = {b.id for b in p1b} | {b.id for b in p2b}
    assert total_after_b == total_unique_ids
    # tidy unused
    _ = all_after


# --- Multi-source --------------------------------------------------------


@needs_git
def test_md_plus_git_plus_py_inserts_three_or_more(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    (tmp_path / "module.py").write_text(
        '"""the module ships the regex fallback only at v0.5"""\n',
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "feat: ship regex fallback for v0.5")
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted >= 3


# --- Non-persisting filter ----------------------------------------------


def test_question_in_doc_does_not_persist(tmp_path: Path) -> None:
    """Question-form sentences from any source -> persist=False."""
    (tmp_path / "FAQ.md").write_text(
        "what does the project ship at v0.5?",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted == 0
    assert result.skipped_non_persisting >= 1


def test_persist_filter_does_not_block_other_paragraphs(tmp_path: Path) -> None:
    (tmp_path / "FAQ.md").write_text(
        "what does the project ship at v0.5?\n\n"
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted == 1
    assert result.skipped_non_persisting >= 1


# --- Result type --------------------------------------------------------


def test_result_is_scan_result_typed(tmp_path: Path) -> None:
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert isinstance(result, ScanResult)


def test_result_total_candidates_consistent(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "one paragraph long enough to qualify here",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert (
        result.inserted
        + result.skipped_existing
        + result.skipped_non_persisting
        == result.total_candidates
    )


# --- Belief properties --------------------------------------------------


def test_inserted_belief_alpha_below_user_prior(tmp_path: Path) -> None:
    """Scanner-extracted beliefs use the deflated alpha (non-user source)."""
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    b = s.search_beliefs("regex")[0]
    # Factual user-prior alpha is 3.0; deflation factor 0.2 -> 0.6 (above 0.5 floor).
    assert b.alpha < 3.0


# --- session_id propagation (#192) -------------------------------------


def test_scan_repo_tags_beliefs_with_synthetic_session_id(tmp_path: Path) -> None:
    """Every belief inserted by scan_repo carries a non-null session_id."""
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    rows = s.search_beliefs("regex")
    assert rows, "scan_repo did not insert any belief from the fixture"
    for b in rows:
        assert b.session_id is not None and b.session_id != ""


def test_scan_repo_explicit_session_id_propagates(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path, session_id="explicit-test-sid")
    rows = s.search_beliefs("regex")
    assert rows
    for b in rows:
        assert b.session_id == "explicit-test-sid"


def test_scan_repo_synthetic_id_stable_for_same_root_and_now(
    tmp_path: Path,
) -> None:
    """Two scans of the same tree at the same `now` produce the same sid."""
    (tmp_path / "README.md").write_text(
        "the project ships only the regex fallback at v0.5",
        encoding="utf-8",
    )
    s1 = MemoryStore(":memory:")
    s2 = MemoryStore(":memory:")
    scan_repo(s1, tmp_path, now="2026-05-02T00:00:00+00:00")
    scan_repo(s2, tmp_path, now="2026-05-02T00:00:00+00:00")
    sid1 = s1.search_beliefs("regex")[0].session_id
    sid2 = s2.search_beliefs("regex")[0].session_id
    assert sid1 is not None and sid1 == sid2
