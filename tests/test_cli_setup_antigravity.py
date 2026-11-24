"""`aelf setup|unsetup|doctor --host antigravity`.

HOME is sandboxed by tests/conftest.py, so `~/.gemini/config` resolves
inside the test's temporary home.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

from aelfrice import cli
from aelfrice.host_antigravity import OWNED_HOOK_KEY, agy_hooks_path, agy_skills_dir

pytestmark = pytest.mark.timeout(60)


@pytest.fixture(autouse=True)
def fresh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home per test: the conftest sandbox home is shared by the session."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(argv, out=out)
    return code, out.getvalue()


def test_setup_doctor_unsetup_round_trip() -> None:
    hooks = agy_hooks_path()
    hooks.parent.mkdir(parents=True)
    hooks.write_text(json.dumps({"mine": {"Stop": []}}))
    code, text = _run(["setup", "--host", "antigravity"])
    assert code == 0, text
    data = json.loads(hooks.read_text())
    assert set(data) == {"mine", OWNED_HOOK_KEY}
    assert (agy_skills_dir() / "aelf-search" / "SKILL.md").is_file()
    assert "not on antigravity:" in text
    code, text = _run(["unsetup", "--host", "antigravity"])
    assert code == 0, text
    assert json.loads(hooks.read_text()) == {"mine": {"Stop": []}}
    assert not (agy_skills_dir() / "aelf-search").exists()


def test_setup_no_flags_become_skip_roles() -> None:
    code, text = _run(["setup", "--host", "antigravity", "--no-commit-ingest", "--no-stop-hook"])
    assert code == 0, text
    entry = json.loads(agy_hooks_path().read_text())[OWNED_HOOK_KEY]
    assert "PostToolUse" not in entry
    assert entry["Stop"][0]["command"].endswith("--skip commit_ingest,stop")


def test_doctor_reports_missing_entry() -> None:
    out = io.StringIO()
    assert cli._cmd_doctor_antigravity(  # pyright: ignore[reportPrivateUsage]
        argparse.Namespace(), out,
    ) == 1
    assert "no aelfrice entry" in out.getvalue()


def test_doctor_host_antigravity_routes_to_the_agy_check() -> None:
    """`aelf doctor --host antigravity` checks the agy entry, not Claude's."""
    code, text = _run(["doctor", "--host", "antigravity"])
    assert code == 1, text
    assert "no aelfrice entry" in text
    agy_hooks_path().parent.mkdir(parents=True)
    code, text = _run(["setup", "--host", "antigravity"])
    assert code == 0, text
    code, text = _run(["doctor", "--host", "antigravity"])
    assert code == 0, text
    assert "no aelfrice entry" not in text
