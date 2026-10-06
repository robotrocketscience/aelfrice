"""#1714: an exported git config or template cannot reach the suite.

The sandbox `HOME` hides `~/.gitconfig`, but `GIT_CONFIG_GLOBAL` and
`GIT_CONFIG_SYSTEM` name config files directly, `GIT_CONFIG_COUNT` and
`GIT_CONFIG_PARAMETERS` carry settings in the environment, and
`GIT_TEMPLATE_DIR` names the hooks that `git init` copies. Exporting
`GIT_CONFIG_GLOBAL` at a config with an external `core.hooksPath` failed 20
tests and ran the external hooks 596 times.

As in #1707, checking this process proves nothing, because CI exports none
of them. The guard runs a second pytest with all of them exported at an
external hooks directory, and the probe below has to pass inside that run.
The probe checks what git does as well as the variables, so a pin that names
the wrong file still fails it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Exported by the guard. Listed here rather than read from conftest, so a
#: pin dropped from conftest cannot also drop out of the guard.
EXPORTED = (
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_TEMPLATE_DIR",
)
#: Of those, the ones the sandbox must remove rather than repoint.
CLEARED = (
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_TEMPLATE_DIR",
)

_HERE = Path(__file__).resolve()


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )


@pytest.mark.timeout(60)
def test_git_reads_no_outside_config_or_template(tmp_path: Path) -> None:
    """Holds in every run; the guard below makes it hold under export."""
    present = [v for v in CLEARED if v in os.environ]
    assert present == [], f"git config variables leaked into the suite: {present}"
    # Checked directly: on a host with no system config file, git behaves
    # the same with or without this switch, so the listing can't catch it.
    assert os.environ.get("GIT_CONFIG_NOSYSTEM") == "1"
    home = Path(os.environ["HOME"]).resolve()
    listing = _git(["config", "--list", "--show-origin"], tmp_path)
    assert listing.returncode in (0, 1), listing.stderr
    # Every setting must come from a file in the sandbox home. Settings
    # from the environment show the origin `command line:`.
    outside = [
        line for line in listing.stdout.splitlines()
        if not line.startswith("file:")
        or not Path(line[5:].split("\t", 1)[0]).resolve().is_relative_to(home)
    ]
    assert outside == [], f"git read config from outside the sandbox: {outside}"
    repo = tmp_path / "repo"
    init = _git(["init", "-q", str(repo)], tmp_path)
    assert init.returncode == 0, init.stderr
    hooks = repo / ".git" / "hooks"
    live = sorted(
        p.name for p in hooks.iterdir() if not p.name.endswith(".sample")
    ) if hooks.is_dir() else []
    assert live == [], f"git init copied hooks from an outside template: {live}"


@pytest.mark.timeout(120)
def test_every_exported_config_variable_is_pinned(tmp_path: Path) -> None:
    hooks = tmp_path / "external-hooks"
    hooks.mkdir()
    config = tmp_path / "external.gitconfig"
    config.write_text(f"[core]\n\thooksPath = {hooks}\n", encoding="utf-8")
    template = tmp_path / "external-template"
    (template / "hooks").mkdir(parents=True)
    hook = template / "hooks" / "post-commit"
    hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hook.chmod(0o755)
    exported = {
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_SYSTEM": str(config),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_PARAMETERS": f"'core.hooksPath'='{hooks}'",
        "GIT_TEMPLATE_DIR": str(template),
    }
    assert set(exported) == set(EXPORTED)
    clean = {
        k: v for k, v in os.environ.items()
        if k not in EXPORTED and k != "GIT_CONFIG_NOSYSTEM"
    }
    clean["GIT_CONFIG_KEY_0"] = "core.hooksPath"
    clean["GIT_CONFIG_VALUE_0"] = str(hooks)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{_HERE}::test_git_reads_no_outside_config_or_template"],
        cwd=_HERE.parents[1], env={**clean, **exported}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
