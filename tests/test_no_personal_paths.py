"""Guards for `scripts/check_no_personal_paths.py` (#1617).

The gate exists because PR #1610 added `.gemini/settings.json` carrying
a developer's absolute home path eleven times to a public repo, and
`secrets-scan`, `pattern-scan` and `history-scan` all reported PASS.
These tests are what stop the detector quietly rotting back into that
state.

Each arm is written to fail under a mutation that removes the rule it
covers, rather than merely asserting the checker runs.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_no_personal_paths.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_no_personal_paths", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load()


# The exact string PR #1610 published, with the user segment changed so
# this file does not itself republish it.
LEAKED_SHAPE = '"command": "AELFRICE_HOST=gemini /home/jdoe/projects/aelfrice/.venv/bin/aelf-hook"'


@pytest.mark.timeout(30)
def test_the_pattern_matches_the_shape_that_actually_leaked() -> None:
    """The #1610 line must match.

    This is the whole point of the gate. If it stops matching, the
    detector has been mutated into a no-op and the hole is reopened.
    """
    assert chk.HOME_PATH_RE.search(LEAKED_SHAPE), (
        "the detector no longer matches the line class that PR #1610 "
        "published to a public repo"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        "/home/jdoe/projects/aelfrice",
        "/Users/jdoe/projects/aelfrice",
        "C:\\Users\\Jdoe\\AppData\\Local",
        "C:\\\\Users\\\\Jdoe\\\\AppData",  # source- or JSON-escaped form
        'File "/home/mmueller/x.py", line 4',
        "cwd=/Users/kbrown/code",
    ],
)
def test_real_home_paths_are_caught(text: str) -> None:
    assert chk.HOME_PATH_RE.search(text), f"missed a real home path: {text!r}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        "/home/runner/work/aelfrice/aelfrice",  # GitHub Actions
        "/Users/synthetic/aelfrice",  # documented fixture marker
        "C:\\Users\\ci\\aelf.exe",  # Windows test fixtures
        "/home/user/project",
        "/Users/you/project",
        "/home/$USER/x",
        "/home/${USER}/x",
        "/Users/<user>/x",
        "/Users/%USERNAME%/x",
        "~/.aelfrice/memory.db",
        "$HOME/.config",
        "relative/home/path/x",  # not a path root
        "some/Users/thing/x",
        "/usr/local/bin/aelf",
        "/home/",
        "/Users/",
    ],
)
def test_placeholders_and_runner_paths_do_not_fire(text: str) -> None:
    """A gate that cries wolf gets disabled, which is the same as absent."""
    m = chk.HOME_PATH_RE.search(text)
    assert m is None, f"false positive on {text!r}: matched {m.group(0)!r}"


@pytest.mark.timeout(60)
def test_the_tracked_tree_is_clean() -> None:
    """No tracked file may carry an absolute home path.

    Fails loudly with the offenders rather than a bare boolean, because
    the next person to trip this needs to know which line to redact.
    """
    findings = chk.scan_tracked()
    assert not findings, "personal paths in tracked files: " + "; ".join(
        f"{p}:{n} {hit!r}" for p, n, hit, _ in findings
    )


@pytest.mark.timeout(60)
def test_no_host_configuration_directory_is_tracked() -> None:
    """`.claude/`, `.gemini/`, `.codex/` must never be tracked.

    This is the vector half of #1617, independent of the path pattern:
    #1610 got in by adding a whole host dotdir, and `.gitignore` had no
    rule for one.
    """
    tracked = chk.forbidden_tracked()
    assert not tracked, f"host configuration directories tracked: {tracked}"


@pytest.mark.timeout(30)
def test_gitignore_covers_every_forbidden_directory() -> None:
    """The ignore rule and the checker must not drift apart.

    `forbidden_tracked` only catches a dotdir already staged. The
    `.gitignore` lines are what stop it being staged at all, so if one
    list grows the other has to.
    """
    # Pin the set itself first. Without this the loop below goes vacuous
    # under the one mutation that matters — deleting an entry from
    # FORBIDDEN_TRACKED_DIRS removes both the checker's rule AND the
    # assertion that would have caught its removal.
    assert set(chk.FORBIDDEN_TRACKED_DIRS) >= {".claude/", ".gemini/", ".codex/"}, (
        "a host configuration directory was dropped from the refused set; "
        f"got {chk.FORBIDDEN_TRACKED_DIRS}"
    )
    ignored = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    entries = {line.strip() for line in ignored if line.strip()}
    for directory in chk.FORBIDDEN_TRACKED_DIRS:
        assert directory in entries, (
            f"{directory} is refused by the checker but not in .gitignore, "
            "so nothing stops it being staged in the first place"
        )


@pytest.mark.timeout(60)
def test_the_script_self_test_passes() -> None:
    """`--self-test` is the fixture set shipped with the pattern."""
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--self-test"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, f"self-test failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.timeout(60)
def test_dry_run_reports_but_does_not_fail() -> None:
    """`--dry-run` must exit 0 even when it finds something.

    Pinned because the repo's reusable-bash rule requires both a
    `--dry-run` and a non-zero exit on failure, and a `--dry-run` that
    still exits non-zero makes the script unusable as a report.
    """
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, f"--dry-run must exit 0, got {r.returncode}"
