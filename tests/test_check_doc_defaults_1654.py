"""#1654: `scripts/check_doc_defaults.py` holds documented defaults to the resolvers.

The repo run is the gate. The other tests copy the files that carry claims
into a temporary root and damage one claim each: a flipped, deleted, or
unlisted marker, and a flipped or reworded CONFIG.md line or heading. Each
must fail, so the guard can't pass by losing track of a claim.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_doc_defaults.py"
_FILES = (
    "src/aelfrice/ingest.py",
    "src/aelfrice/retrieval.py",
    "src/aelfrice/temporal_spine.py",
    "docs/user/CONFIG.md",
)


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(root)], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60,
    )


def _copy(tmp_path: Path) -> Path:
    for rel in _FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / rel, tmp_path / rel)
    return tmp_path


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


@pytest.mark.timeout(90)
def test_the_repo_docs_match_their_resolvers() -> None:
    result = _run(_REPO)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.timeout(90)
def test_an_undamaged_copy_passes(tmp_path: Path) -> None:
    result = _run(_copy(tmp_path))
    assert result.returncode == 0, result.stdout


_DAMAGE = {
    "flipped marker": ("src/aelfrice/retrieval.py",
                       "default-ON (is_entity_persist_demote_enabled)",
                       "default-OFF (is_entity_persist_demote_enabled)"),
    "deleted marker": ("src/aelfrice/ingest.py",
                       "Default-OFF (is_auto_relationship_detection_enabled)",
                       "Default-OFF"),
    "deleted backtick marker": ("src/aelfrice/temporal_spine.py",
                                "Default-ON (``is_temporal_spine_write_enabled``)",
                                "Default-ON"),
    "unlisted marker": ("src/aelfrice/retrieval.py",
                        "default-ON (is_entity_persist_demote_enabled).",
                        "default-ON (is_entity_persist_demote_enabled).\n"
                        "# default-OFF (is_bfs_enabled)."),
    "flipped config line": ("docs/user/CONFIG.md",
                            "Boolean, default `false` since #1162.",
                            "Boolean, default `true` since #1162."),
    "reworded config line": ("docs/user/CONFIG.md",
                             "Boolean, default `false` since #1162.",
                             "Boolean. The default is `false` since #1162."),
    "reworded section heading": ("docs/user/CONFIG.md",
                                 "## `[retrieval]` (v1.3+)",
                                 "## The `[retrieval]` table (v1.3+)"),
}


@pytest.mark.timeout(90)
@pytest.mark.parametrize("damage", sorted(_DAMAGE))
def test_a_damaged_claim_fails(tmp_path: Path, damage: str) -> None:
    root = _copy(tmp_path)
    _edit(root, *_DAMAGE[damage])
    result = _run(root)
    assert result.returncode == 1, result.stdout
