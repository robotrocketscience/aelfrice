"""#1716: measure_block_ceiling's figures don't depend on the runner's HOME.

On a machine with a `~/.aelfrice.toml` and no project config, SessionStart
adds the #1652 ignored-config line, which put about 40 tokens on the
`session_start` figures there and none on CI. The producer now measures
under an empty HOME, so the published figures hold wherever it runs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRODUCER = REPO / "scripts" / "measure_block_ceiling.py"


def _published(key: str) -> int:
    m = re.search(rf"measure_block_ceiling\.py#{re.escape(key)} = (\d+)", PRODUCER.read_text())
    assert m, key
    return int(m.group(1))


@pytest.mark.timeout(180)
def test_a_home_config_does_not_move_the_session_start_figures(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".aelfrice.toml").write_text("[cadence]\nenabled = true\n")
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    out = subprocess.run(
        [sys.executable, str(PRODUCER), "--emit-figures"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=170,
    )
    assert out.returncode == 0, out.stderr[-500:]
    figures = json.loads(out.stdout.strip().splitlines()[-1])
    for key in ("ref_lock_30026_session_start_frozen", "ref_lock_30026_session_start_reference"):
        assert figures[key] == _published(key), key


def test_the_home_is_restored_after_measuring(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_block_ceiling", PRODUCER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_block_ceiling"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setenv("HOME", "/original/home")
    with mod._hermetic_home() as home:  # noqa: SLF001
        assert os.environ["HOME"] == str(home)
        assert not any(home.iterdir())
    assert os.environ["HOME"] == "/original/home"
