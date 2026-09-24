"""Guards for `scripts/check_no_personal_paths.py` (#1617).

The gate exists because PR #1610 added `.gemini/settings.json` carrying
a developer's absolute venv path twelve times to a public repo, and
`secrets-scan`, `pattern-scan` and `history-scan` all reported PASS.

**Most of these arms drive the SCRIPT, not the regex.** An earlier
revision tested only `HOME_PATH_RE` and was vacuous over the thing that
matters: nine mutations survived it, including `main()` returning 0
unconditionally and `scan_tracked()` returning nothing — either of which
turns the required CI step into a no-op while every test still passes.
A gate whose suite cannot detect the gate's removal is the #1610 failure
mode one layer up, so the end-to-end arms below run the real script
against a real dirty repository and assert on its exit code.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_no_personal_paths.py"

# Split so this file does not itself contain a matchable literal.
_LEAK = "/home/" + "jdoe" + "/projects/app/.venv/bin/aelf-hook"


def _load():
    spec = importlib.util.spec_from_file_location("check_no_personal_paths", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load()


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd or _REPO_ROOT,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, timeout=60,
    )


@pytest.fixture
def dirty_repo(tmp_path: Path) -> Path:
    """A git repo carrying BOTH failure classes the gate exists for."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "settings.json").write_text(
        '{"command": "' + _LEAK + '"}\n', encoding="utf-8"
    )
    dotdir = repo / ".gemini"
    dotdir.mkdir()
    (dotdir / "settings.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "dirty")
    return repo


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "c"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README.md").write_text(
        "Install to ~/.local/bin, or /home/user/bin on Linux.\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "clean")
    return repo


# --- end-to-end: the arms that make the suite non-vacuous ---------------


@pytest.mark.timeout(120)
def test_the_script_fails_on_a_repo_carrying_a_personal_path(dirty_repo: Path) -> None:
    """The whole point. Kills `main() -> 0` and `scan_tracked() -> []`."""
    r = _run(cwd=dirty_repo)
    assert r.returncode != 0, f"gate passed a dirty repo:\n{r.stdout}\n{r.stderr}"
    assert "personal path" in r.stdout, r.stdout
    assert "clean:" not in r.stdout, f"reported clean while dirty:\n{r.stdout}"


@pytest.mark.timeout(120)
def test_the_script_reports_the_tracked_host_dotdir(dirty_repo: Path) -> None:
    """Kills `forbidden_tracked() -> []` and dropping its report."""
    r = _run(cwd=dirty_repo)
    assert "host configuration directory must not be tracked" in r.stdout, r.stdout
    assert ".gemini" in r.stdout, r.stdout


@pytest.mark.timeout(120)
def test_the_script_passes_a_clean_repo(clean_repo: Path) -> None:
    """The other half: a gate that always fails is also a broken gate."""
    r = _run(cwd=clean_repo)
    assert r.returncode == 0, f"gate failed a clean repo:\n{r.stdout}\n{r.stderr}"
    assert "clean:" in r.stdout


@pytest.mark.timeout(120)
def test_dry_run_exits_zero_even_when_it_finds_something(dirty_repo: Path) -> None:
    """`--dry-run` reports; it never fails.

    Run against a DIRTY repo. The earlier version ran on the clean tree,
    where the branch it claims to cover was never reached — so ignoring
    the flag entirely passed it.
    """
    r = _run("--dry-run", cwd=dirty_repo)
    assert r.returncode == 0, f"--dry-run must exit 0, got {r.returncode}"
    assert "personal path" in r.stdout, "--dry-run must still report"


@pytest.mark.timeout(120)
def test_an_inline_marker_suppresses_one_line(tmp_path: Path) -> None:
    """The escape hatch works, and only for the line carrying it."""
    repo = tmp_path / "m"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text(f"{_LEAK}  # {chk.INLINE_ALLOW}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    assert _run(cwd=repo).returncode == 0, "inline marker did not suppress"

    (repo / "b.txt").write_text(f"{_LEAK}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "y")
    assert _run(cwd=repo).returncode != 0, "marker leaked to an unmarked line"


@pytest.mark.timeout(120)
def test_a_nested_host_dotdir_is_refused(tmp_path: Path) -> None:
    """`.gitignore` covers every depth, so the checker must too.

    An earlier `startswith('.claude/')` matched only the repo root,
    making the backstop weaker than the thing it backs up.
    """
    repo = tmp_path / "n"
    (repo / "sub" / "proj" / ".gemini").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "sub" / "proj" / ".gemini" / "settings.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "nested")
    r = _run(cwd=repo)
    assert r.returncode != 0, f"nested host dotdir passed:\n{r.stdout}"
    assert "must not be tracked" in r.stdout


@pytest.mark.timeout(120)
def test_a_non_utf8_file_cannot_hide_a_path(tmp_path: Path) -> None:
    """Skipping undecodable files would make encoding an evasion."""
    repo = tmp_path / "e"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "latin.txt").write_bytes(
        "café ".encode("latin-1") + _LEAK.encode("latin-1") + b"\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "latin")
    r = _run(cwd=repo)
    assert r.returncode != 0, f"non-UTF-8 file hid a path:\n{r.stdout}"


@pytest.mark.timeout(120)
def test_a_symlink_is_not_followed_out_of_the_repo(tmp_path: Path) -> None:
    """The result must depend on the commit, not on the host filesystem.

    Reading the working tree followed a tracked symlink off the repo and
    reported findings from a file that is not in the commit — both a
    false positive and a determinism violation (#605).
    """
    outside = tmp_path / "outside.txt"
    outside.write_text(f"{_LEAK}\n", encoding="utf-8")
    repo = tmp_path / "s"
    repo.mkdir()
    _git(repo, "init", "-q")
    os.symlink(outside, repo / "link.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "link")
    r = _run(cwd=repo)
    assert r.returncode == 0, (
        "the scan followed a symlink outside the repository:\n" + r.stdout
    )


@pytest.mark.timeout(120)
def test_range_mode_flags_an_added_line(tmp_path: Path) -> None:
    """Diff mode is what gates a PR; it needs its own end-to-end arm."""
    repo = tmp_path / "d"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("nothing here\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "a.txt").write_text(f"nothing here\n{_LEAK}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "leak")
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode != 0, f"range mode missed an added path:\n{r.stdout}"


@pytest.mark.timeout(120)
def test_a_removed_line_is_not_a_finding(tmp_path: Path) -> None:
    """Deleting a leak shrinks exposure; it must not block the fix."""
    repo = tmp_path / "rm"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text(f"{_LEAK}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "a.txt").write_text("redacted\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "redact")
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode == 0, f"removing a leak was treated as adding one:\n{r.stdout}"


@pytest.mark.timeout(120)
def test_an_unreadable_range_fails_closed(tmp_path: Path) -> None:
    """Never report clean for history the gate could not read."""
    r = _run("--range", "definitely-not-a-ref..HEAD")
    assert r.returncode != 0
    assert "clean:" not in r.stdout
    assert "could not read" in r.stdout, r.stdout


# --- the pattern itself --------------------------------------------------


@pytest.mark.timeout(30)
def test_the_pattern_matches_the_shape_that_actually_leaked() -> None:
    assert chk.HOME_PATH_RE.search(
        '"command": "AELFRICE_HOST=gemini ' + _LEAK + '"'
    ), "the detector no longer matches the line class PR #1610 published"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        '"venvPath": "/Users/jdoe"',  # bare root, no trailing separator
        '"path":"\\/Users\\/jdoe\\/proj"',  # JSON-escaped slashes
        "C:\\Users\\Jdoe\\AppData",
        "C:\\\\Users\\\\Jdoe\\\\AppData",
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
        "/home/runner/work/aelfrice/aelfrice",
        "/Users/synthetic/aelfrice",
        "C:\\Users\\ci\\aelf.exe",
        "/home/user/project",
        "/home/user.",  # end of a sentence, not an account named "user."
        "/Users/yourname/projects/aelfrice",
        "/home/node/app",
        "/home/ubuntu/deploy",
        "C:\\Users\\Public\\x",
        "/home/u/proj",  # single char: this tree's fixture convention
        "/home/$USER/x",
        "/Users/<user>/x",
        "~/.aelfrice/memory.db",
        "$HOME/.config",
        "relative/home/path/x",
        "/home/",
        "/usr/local/bin/aelf",
    ],
)
def test_placeholders_and_service_accounts_do_not_fire(text: str) -> None:
    """A gate that cries wolf gets disabled, which is the same as absent."""
    m = chk.HOME_PATH_RE.search(text)
    assert m is None, f"false positive on {text!r}: matched {m.group(0)!r}"


@pytest.mark.timeout(60)
def test_the_tracked_tree_is_clean() -> None:
    findings = chk.scan_tracked()
    assert not findings, "personal paths in tracked files: " + "; ".join(
        f"{p}:{n} {hit!r}" for p, n, hit, _ in findings
    )


@pytest.mark.timeout(60)
def test_no_host_configuration_directory_is_tracked() -> None:
    tracked = chk.forbidden_tracked()
    assert not tracked, f"host configuration directories tracked: {tracked}"


@pytest.mark.timeout(30)
def test_gitignore_covers_every_forbidden_directory() -> None:
    """The ignore rule and the checker must not drift apart."""
    # Pin the set first, or the loop goes vacuous under the one mutation
    # that matters: deleting an entry removes the rule AND the assertion
    # that would have caught its removal.
    assert set(chk.FORBIDDEN_TRACKED_DIRS) >= {".claude", ".gemini", ".codex"}, (
        f"a host configuration directory was dropped: {chk.FORBIDDEN_TRACKED_DIRS}"
    )
    entries = {
        ln.strip()
        for ln in (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    for directory in chk.FORBIDDEN_TRACKED_DIRS:
        assert f"{directory}/" in entries, (
            f"{directory}/ is refused by the checker but not in .gitignore, "
            "so nothing stops it being staged in the first place"
        )


@pytest.mark.timeout(60)
def test_the_script_self_test_passes() -> None:
    r = _run("--self-test")
    assert r.returncode == 0, f"self-test failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.timeout(60)
def test_self_test_fails_when_the_pattern_stops_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--self-test` must be able to FAIL, or it certifies nothing."""
    import re as _re

    monkeypatch.setattr(chk, "HOME_PATH_RE", _re.compile(r"(?!x)x"))
    assert chk.self_test() == 1, "self_test passed a pattern that matches nothing"
