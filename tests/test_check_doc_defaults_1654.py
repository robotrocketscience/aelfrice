"""#1654: `scripts/check_doc_defaults.py` holds documented defaults to the resolvers.

The repo run is the gate. The other tests copy the files that carry claims
into a temporary root and damage one claim each: a flipped, deleted, or
unlisted marker, and a flipped or reworded CONFIG.md line or heading. Each
must fail, so the guard can't pass by losing track of a claim.
"""
from __future__ import annotations

import os
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


#: name -> (file, old text, new text, the failure line the guard must print).
_DAMAGE = {
    "flipped marker": ("src/aelfrice/retrieval.py",
                       "default-ON (is_entity_persist_demote_enabled)",
                       "default-OFF (is_entity_persist_demote_enabled)",
                       "says default-OFF, retrieval:is_entity_persist_demote_enabled() returns True"),
    "deleted marker": ("src/aelfrice/ingest.py",
                       "Default-OFF (is_auto_relationship_detection_enabled)",
                       "Default-OFF",
                       "no default-ON/OFF marker for is_auto_relationship_detection_enabled"),
    "deleted backtick marker": ("src/aelfrice/temporal_spine.py",
                                "Default-ON (``is_temporal_spine_write_enabled``)",
                                "Default-ON",
                                "temporal_spine.py: no default-ON/OFF marker for is_temporal_spine_write_enabled"),
    "unlisted marker": ("src/aelfrice/retrieval.py",
                        "default-ON (is_entity_persist_demote_enabled).",
                        "default-ON (is_entity_persist_demote_enabled).\n"
                        "# default-OFF (is_bfs_enabled).",
                        "marker for is_bfs_enabled is not listed in MARKERS"),
    "flipped config line": ("docs/user/CONFIG.md",
                            "Boolean, default `false` since #1162.",
                            "Boolean, default `true` since #1162.",
                            "[retrieval] use_heat_kernel says true"),
    "reworded config line": ("docs/user/CONFIG.md",
                             "Boolean, default `false` since #1162.",
                             "Boolean. The default is `false` since #1162.",
                             "no `Boolean, default` line for [retrieval] use_heat_kernel"),
    "reworded section heading": ("docs/user/CONFIG.md",
                                 "## `[retrieval]` (v1.3+)",
                                 "## The `[retrieval]` table (v1.3+)",
                                 "no `Boolean, default` line for [retrieval] entity_index_enabled"),
}


@pytest.mark.timeout(90)
@pytest.mark.parametrize("damage", sorted(_DAMAGE))
def test_a_damaged_claim_fails(tmp_path: Path, damage: str) -> None:
    rel, old, new, expected = _DAMAGE[damage]
    root = _copy(tmp_path)
    _edit(root, rel, old, new)
    result = _run(root)
    # A traceback exits 1 too, so the exit code alone proves nothing.
    assert result.returncode == 1, result.stdout
    assert "FAIL:" in result.stdout and expected in result.stdout, result.stdout
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.timeout(90)
def test_ambient_env_and_config_cannot_reach_the_check(tmp_path: Path) -> None:
    """Overrides in the env, the cwd, and HOME would each flip a checked
    default; the guard must clear or sidestep all three."""
    toml = "[retrieval]\nuse_heat_kernel = true\n"
    (tmp_path / ".aelfrice.toml").write_text(toml, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".aelfrice.toml").write_text(toml, encoding="utf-8")
    env = {**os.environ, "HOME": str(home),
           "AELFRICE_HEAT_KERNEL": "1", "AELFRICE_ENTITY_INDEX": "0"}
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_REPO)], cwd=tmp_path, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 failed" in result.stdout
