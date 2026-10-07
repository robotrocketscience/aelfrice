"""#1654: `scripts/check_doc_defaults.py` holds documented defaults to the resolvers.

The repo run is the gate. The synthetic roots prove the guard can fail: a
marker or a CONFIG.md line that disagrees with its resolver exits 1, and a
root with nothing to check exits 1 rather than passing vacuously.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_doc_defaults.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(root)], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=60,
    )


def _root(tmp_path: Path, *, marker: str = "", config: str = "") -> Path:
    src = tmp_path / "src" / "aelfrice"
    src.mkdir(parents=True)
    (src / "flags.py").write_text(f"# {marker}\n", encoding="utf-8")
    docs = tmp_path / "docs" / "user"
    docs.mkdir(parents=True)
    (docs / "CONFIG.md").write_text(config, encoding="utf-8")
    return tmp_path


_HEAT = "## `[retrieval]` (v1.3+)\n\n### `use_heat_kernel`\n\nBoolean, default `{}` since #1162.\n"


@pytest.mark.timeout(90)
def test_the_repo_docs_match_their_resolvers() -> None:
    result = _run(_REPO)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.timeout(90)
def test_a_marker_that_disagrees_with_its_resolver_fails(tmp_path: Path) -> None:
    result = _run(_root(tmp_path, marker="Default-ON (is_heat_kernel_enabled)"))
    assert result.returncode == 1
    assert "says default-ON, is_heat_kernel_enabled() returns False" in result.stdout


@pytest.mark.timeout(90)
def test_a_config_line_that_disagrees_with_its_resolver_fails(tmp_path: Path) -> None:
    result = _run(_root(tmp_path, config=_HEAT.format("true")))
    assert result.returncode == 1
    assert "[retrieval] use_heat_kernel says true" in result.stdout


@pytest.mark.timeout(90)
def test_matching_claims_pass(tmp_path: Path) -> None:
    root = _root(tmp_path, marker="default-OFF (is_heat_kernel_enabled)",
                 config=_HEAT.format("false"))
    result = _run(root)
    assert result.returncode == 0, result.stdout
    assert "2 claims checked" in result.stdout


@pytest.mark.timeout(90)
def test_a_root_with_nothing_to_check_fails(tmp_path: Path) -> None:
    assert _run(_root(tmp_path)).returncode == 1
