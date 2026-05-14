"""End-to-end regression: scan a fixture repo and find what was scanned.

Cumulative integration scenario added at v0.5.0. Exercises scanner
(filesystem + git + AST extractors + scan_repo orchestrator) +
classification (regex fallback) + store (CRUD + FTS5) + retrieval
(L0 locked + L1 BM25) in one realistic flow:

  populate fixture repo
  → scan_repo
  → assert beliefs landed
  → lock one of them
  → assert retrieve() returns it in L0 above any L1 match
  → re-scan
  → assert no duplicates inserted

Each atomic test rebuilds a fresh fixture and runs an isolated property
check, per the deterministic-atomic-short policy.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aelfrice.models import LOCK_USER
from aelfrice.retrieval import retrieve
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore

_GIT_AVAILABLE = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not _GIT_AVAILABLE, reason="git binary not on PATH")

pytestmark = pytest.mark.regression


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


def _seed_fixture(root: Path, with_git: bool = True) -> None:
    """Mixed-source fixture with stable content across the three extractors."""
    (root / "README.md").write_text(
        "the project ships only the regex fallback at v0.5\n\n"
        "the polymorphic onboard handshake lands at v0.6 with the mcp server",
        encoding="utf-8",
    )
    (root / "module.py").write_text(
        '"""the module exposes the onboarding flow for the scanner pipeline"""\n'
        '\n'
        'def greet():\n'
        '    """says hello to the world from a documented helper"""\n'
        '    return "hi"\n',
        encoding="utf-8",
    )
    if with_git:
        _git(root, "init", "-q")
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "feat: ship the regex fallback for the v0.5 milestone")


# --- Onboarding lands beliefs ------------------------------------------


def test_scan_inserts_at_least_one_belief(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    result = scan_repo(s, tmp_path)
    assert result.inserted >= 1


def test_doc_paragraph_lands_in_store(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = s.search_beliefs("regex")
    assert len(hits) >= 1


def test_module_docstring_lands_in_store(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = s.search_beliefs("scanner")
    assert any("module exposes" in h.content for h in hits)


@needs_git
def test_git_commit_subject_lands_in_store(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=True)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = s.search_beliefs("milestone")
    assert any("v0.5 milestone" in h.content for h in hits)


# --- Retrieval finds onboarded beliefs ---------------------------------


def test_retrieve_finds_doc_belief_via_l1(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = retrieve(s, query="regex", token_budget=10_000)
    contents = " ".join(h.content for h in hits)
    assert "regex fallback" in contents


def test_retrieve_query_with_no_match_returns_empty_when_no_locks(
    tmp_path: Path,
) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = retrieve(s, query="nonexistentterm12345", token_budget=10_000)
    assert hits == []


# --- Locking a scanned belief moves it to L0 ---------------------------


def test_locked_scanned_belief_appears_first(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    docs = s.search_beliefs("regex")
    assert len(docs) >= 1
    target = docs[0]
    target.lock_level = LOCK_USER
    target.locked_at = "2026-04-26T00:00:00Z"
    s.update_belief(target)
    hits = retrieve(s, query="regex", token_budget=10_000)
    assert hits[0].id == target.id
    assert hits[0].lock_level == LOCK_USER


# --- Idempotence ------------------------------------------------------


def test_rescan_inserts_zero_new_beliefs(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    second = scan_repo(s, tmp_path)
    assert second.inserted == 0


def test_rescan_total_belief_set_unchanged(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, with_git=False)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    ids_one = {b.id for b in s.search_beliefs("regex")} | {
        b.id for b in s.search_beliefs("scanner")
    }
    scan_repo(s, tmp_path)
    ids_two = {b.id for b in s.search_beliefs("regex")} | {
        b.id for b in s.search_beliefs("scanner")
    }
    assert ids_one == ids_two


# --- Distinct sources produce distinct beliefs ---------------------------


@needs_git
def test_doc_and_git_with_overlapping_text_remain_distinct(
    tmp_path: Path,
) -> None:
    """A doc paragraph and a git commit subject sharing 'regex fallback'
    text get stored as two distinct beliefs because the source-keyed
    id derivation distinguishes them."""
    _seed_fixture(tmp_path, with_git=True)
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    hits = s.search_beliefs("regex")
    sources_distinct = len({h.id for h in hits})
    assert sources_distinct >= 2


# --- Non-persisting filter is honored end-to-end -----------------------


def test_question_in_doc_does_not_land(tmp_path: Path) -> None:
    (tmp_path / "FAQ.md").write_text(
        "what is the project shipping at the next milestone?",
        encoding="utf-8",
    )
    s = MemoryStore(":memory:")
    scan_repo(s, tmp_path)
    # Use a plain alphanumeric query to avoid FTS5 syntax characters.
    # The FTS5-escape bug surfaced by version-string queries like "v0.5"
    # is tracked separately; the persist-filter check stands either way.
    hits = s.search_beliefs("milestone")
    assert hits == []
