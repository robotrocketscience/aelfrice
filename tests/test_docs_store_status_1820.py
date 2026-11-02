"""User-doc claims about the store and `aelf status` (#1820).

The #1820 docs sweep left statements that disagreed with the 5.3.0 code:

  1. LIMITATIONS.md said two projects in one repository share a store
     unless each project folder holds a `[store]` table. One table is
     enough: the folder with it resolves to its own store, and the
     other keeps the repository's store.

Each test checks the code fact first, then the page. The page checks
match terms, not whole sentences.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from aelfrice import db_paths
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.db_paths import STORE_RULE_GIT, STORE_RULE_MARKER, resolve_store

REPO = Path(__file__).resolve().parents[1]
USER_DOCS = REPO / "docs" / "user"

pytestmark = pytest.mark.timeout(60)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert heading in text, f"{path.name} lost its {heading!r} section"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Reach the marker and git rules with every path under `tmp_path`."""
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("AELFRICE_USER_DB", str(home / "user" / "memory.db"))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()
    return tmp_path


def test_one_store_table_separates_two_projects_in_a_repository(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One `[store]` table separates the projects; the page must not
    ask for a table in every project folder."""
    repo = sandbox / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"], cwd=repo, check=True, capture_output=True,
        timeout=30,
    )
    marked = repo / "a"
    marked.mkdir()
    (marked / CONFIG_FILENAME).write_text(
        '[store]\npath = ".aelfrice/store"\n', encoding="utf-8",
    )
    unmarked = repo / "b"
    unmarked.mkdir()

    monkeypatch.chdir(marked)
    sel_a = resolve_store()
    monkeypatch.chdir(unmarked)
    sel_b = resolve_store()
    assert sel_a.rule == STORE_RULE_MARKER
    assert sel_b.rule == STORE_RULE_GIT
    assert sel_a.path != sel_b.path

    section = _collapse(
        _section(USER_DOCS / "LIMITATIONS.md", "### Multi-project query"),
    )
    assert "each project folder holds" not in section, section
    assert "unless one of them holds a `[store]` table" in section, section
