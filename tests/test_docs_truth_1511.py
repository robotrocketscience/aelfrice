"""Documentation claims that a machine can check, pinned to the tree (#1511).

#1509 converted the user-facing documents to ASD-STE100 under a "change no
facts" scope. Reading them against the source surfaced seven statements that
disagreed with the code. Two of those are not prose problems at all — they are
counts and a platform promise, and both drift by construction the moment
someone edits a workflow or adds a module.

So they are checked here rather than typed there:

  1. **The CI platform claim.** INSTALL.md and LIMITATIONS.md said the full
     suite runs on macOS and Linux on every pull request. No workflow has ever
     used a macOS runner. The operator ruling of 2026-08-19 was to describe
     reality rather than buy the CI leg, so the documents now say Linux plus a
     Windows smoke job. If a macOS runner is ever added, this test goes red and
     the sentence gets revisited — which is the point.

  2. **The ARCHITECTURE module and file counts.** The page named 31 modules
     against 117 `.py` files. The table had 32 rows and the tree held 127
     files. A hand-maintained count is wrong the day after it is written.

Deliberately not asserting prose wording. A text match breaks on rephrasing and
proves nothing about whether the claim is true. What is checkable is the tree
the prose describes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
ARCHITECTURE = REPO / "docs" / "concepts" / "ARCHITECTURE.md"
SRC = REPO / "src" / "aelfrice"


def _runner_labels() -> set[str]:
    """Every `runs-on:` label declared across the workflow directory."""
    labels: set[str] = set()
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        for line in wf.read_text(encoding="utf-8").splitlines():
            m = re.search(r"^\s*runs-on:\s*(.+?)\s*$", line)
            if m:
                labels.add(m.group(1).strip().strip("\"'"))
    return labels


def test_no_workflow_uses_a_macos_runner() -> None:
    """The documents say no workflow tests macOS. Hold them to it.

    If this goes red, a macOS leg was added and
    `docs/user/INSTALL.md` plus `docs/user/LIMITATIONS.md § Compatibility`
    must be corrected in the same change.
    """
    labels = _runner_labels()
    assert labels, "found no runs-on: labels at all — the scan is vacuous"
    macos = {lbl for lbl in labels if "macos" in lbl.lower()}
    assert not macos, (
        f"a macOS runner is now declared ({sorted(macos)}), but "
        "docs/user/INSTALL.md and docs/user/LIMITATIONS.md say no workflow "
        "tests macOS. Update the documents in this change."
    )


def test_the_linux_and_windows_legs_the_documents_name_still_exist() -> None:
    """The replacement sentence names two legs. Neither may vanish silently."""
    labels = _runner_labels()
    assert any("ubuntu" in lbl for lbl in labels), (
        "the documents promise a Linux full-suite leg; no ubuntu runner found"
    )
    assert any("windows" in lbl for lbl in labels), (
        "the documents promise a Windows smoke job; no windows runner found"
    )


def _module_table_rows() -> int:
    """Data rows in the `| Module | Responsibility |` table."""
    rows = 0
    in_table = False
    for line in ARCHITECTURE.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Module ") and "Responsibility" in line:
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if re.match(r"^\|\s*-{2,}", line):  # the header separator
                continue
            rows += 1
    return rows


def _py_file_count() -> int:
    return len([p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts])


def test_architecture_counts_match_the_tree() -> None:
    """The sentence's two numbers are the table's rows and the tree's files.

    The page says the table is a curated subset holding N modules against M
    `.py` files. Both numbers are derived here, so neither can drift.
    """
    text = ARCHITECTURE.read_text(encoding="utf-8")
    m = re.search(
        r"(\d+)\s*\n?\s*modules against the (\d+) `\.py` files", text
    )
    assert m, (
        "could not find the module/file count sentence in ARCHITECTURE.md — "
        "if it was reworded, update this test with it"
    )
    stated_modules, stated_files = int(m.group(1)), int(m.group(2))
    assert stated_modules == _module_table_rows(), (
        f"ARCHITECTURE.md says {stated_modules} modules; the table has "
        f"{_module_table_rows()} rows"
    )
    assert stated_files == _py_file_count(), (
        f"ARCHITECTURE.md says {stated_files} `.py` files; the tree holds "
        f"{_py_file_count()}"
    )


def test_exploration_ships_and_is_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PHILOSOPHY said exploration would ship "in a future version". It ships.

    The paragraph now says the slot exists and is off by default. Both halves
    are pinned: the module must be importable, and the resolver must return
    False with no environment override and no TOML file.
    """
    from aelfrice import exploration, retrieval

    assert exploration.DEFAULT_EXPLORATION_SLOTS >= 1, (
        "exploration module no longer declares a slot count — PHILOSOPHY "
        "claims the mechanism ships"
    )
    monkeypatch.delenv("AELFRICE_EXPLORATION", raising=False)
    assert retrieval.is_exploration_enabled(start=tmp_path) is False, (
        "exploration resolved default-ON; PHILOSOPHY says it is off by "
        "default. Update the paragraph in the same change as the flip."
    )
