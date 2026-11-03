"""User-doc claims about the store and `aelf status` (#1820).

The #1820 docs sweep left statements that disagreed with the 5.3.0 code:

  1. LIMITATIONS.md said two projects in one repository share a store
     unless each project folder holds a `[store]` table. One table is
     enough: the folder with it resolves to its own store, and the
     other keeps the repository's store. A project nested below the
     marked folder shares the marked store unless it holds a
     `[store]` table of its own.
  2. QUICKSTART.md's `aelf status` sample showed an `aelfrice <version>`
     first line and no `store:`, `store rule:` or `phantoms:` line.
  3. The `/aelf:status` description listed counts only, though the
     command prints the store and its rule first.

Each test checks the code fact first, then the page. The page checks
match terms, not whole sentences.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
from pathlib import Path

import pytest

from aelfrice import cli, db_paths
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.db_paths import STORE_RULE_GIT, STORE_RULE_MARKER, resolve_store

REPO = Path(__file__).resolve().parents[1]
LIMITATIONS = REPO / "docs" / "user" / "LIMITATIONS.md"
QUICKSTART = REPO / "docs" / "user" / "QUICKSTART.md"
STATUS_PROMPT = "$ aelf status"
SLASH_STATUS = REPO / "src" / "aelfrice" / "slash_commands" / "status.md"

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
    nested = marked / "nested"
    nested.mkdir()
    (nested / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    own = marked / "own"
    own.mkdir()
    (own / CONFIG_FILENAME).write_text(
        '[store]\npath = ".aelfrice/store"\n', encoding="utf-8",
    )

    monkeypatch.chdir(marked)
    sel_a = resolve_store()
    monkeypatch.chdir(unmarked)
    sel_b = resolve_store()
    monkeypatch.chdir(nested)
    sel_nested = resolve_store()
    monkeypatch.chdir(own)
    sel_own = resolve_store()
    assert sel_a.rule == STORE_RULE_MARKER
    assert sel_b.rule == STORE_RULE_GIT
    assert sel_a.path != sel_b.path
    assert sel_nested.path == sel_a.path
    assert sel_own.rule == STORE_RULE_MARKER
    assert sel_own.path not in (sel_a.path, sel_b.path)

    section = _collapse(
        _section(LIMITATIONS, "### Multi-project query"),
    )
    assert "each project folder holds" not in section, section
    assert "unless one of them holds a `[store]` table" in section, section
    assert "nested below it" in section, section
    assert "table of its own" in section, section


def _labels(lines: list[str]) -> list[str]:
    return [ln.split(":", 1)[0] for ln in lines]


def _quickstart_status_sample() -> list[str]:
    text = QUICKSTART.read_text(encoding="utf-8")
    assert text.count(STATUS_PROMPT + "\n") == 1, "expected one aelf status sample"
    body = text.split(STATUS_PROMPT + "\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


def test_quickstart_status_sample_has_the_labels_status_prints(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sample's labels, in order, are the ones `aelf status` prints."""
    monkeypatch.setenv("AELFRICE_DB", str(sandbox / "store" / "memory.db"))
    buf = io.StringIO()
    assert cli._cmd_stats(argparse.Namespace(), buf) == 0
    printed = _labels(buf.getvalue().splitlines())
    assert "store" in printed and "store rule" in printed, printed

    assert _labels(_quickstart_status_sample()) == printed


def test_status_slash_description_names_the_store_and_rule(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aelf status` prints the store and its rule, so the slash
    command's description has to name both."""
    monkeypatch.setenv("AELFRICE_DB", str(sandbox / "store" / "memory.db"))
    buf = io.StringIO()
    assert cli._cmd_stats(argparse.Namespace(), buf) == 0
    printed = _labels(buf.getvalue().splitlines())
    assert printed[1:3] == ["store", "store rule"], printed

    text = SLASH_STATUS.read_text(encoding="utf-8")
    found = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert found is not None, "status.md lost its description"
    description = found.group(1).lower()
    assert "store" in description, description
    assert "rule" in description, description
