"""Guards for `scripts/check_commit_identity.py` (#1617).

Two commits reached `main` carrying a contributor's real name and
corporate email as the git author. Commit metadata is copied into every
clone and fork, served by the unauthenticated REST API, and pinned by
`refs/pull/<n>/head`, so unlike file content it cannot be withdrawn.
Refusing it before the push is the only real control.

**Most of these arms run `offenders()` over a real range containing a
real offence.** An earlier revision tested only the address pattern and
was vacuous over the gate itself: deleting the author check entirely
passed every arm while flipping a known-bad commit from exit 1 to exit
0. A suite that cannot detect the gate's removal certifies nothing.

The pattern also fails CLOSED — a false rejection blocks every push — so
both shapes GitHub actually issues are pinned explicitly. The first
draft of the pattern rejected both.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_commit_identity.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_commit_identity", _SCRIPT)
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


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    import os

    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, **(env or {})},
    )


def _repo_with(tmp_path: Path, name: str, email: str, tag: str) -> Path:
    """A two-commit repo whose second commit has the given identity."""
    repo = tmp_path / tag
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(
        repo, "-c", "user.name=rrs",
        "-c", "user.email=1+rrs@users.noreply.github.com",
        "add", "-A",
    )
    _git(
        repo, "-c", "user.name=rrs",
        "-c", "user.email=1+rrs@users.noreply.github.com",
        "commit", "-qm", "base",
    )
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(
        repo, "-c", f"user.name={name}", "-c", f"user.email={email}",
        "commit", "-qm", "second",
        env={
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        },
    )
    return repo


# --- end-to-end: the arms that make the suite non-vacuous ---------------


@pytest.mark.timeout(120)
def test_a_routable_author_email_is_refused(tmp_path: Path) -> None:
    """The gate's entire purpose. Kills deleting the author check."""
    repo = _repo_with(tmp_path, "rrs", "someone@corp.example", "email")
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode != 0, f"a routable author email passed:\n{r.stdout}"
    assert "author email" in r.stdout, r.stdout
    assert "clean:" not in r.stdout


@pytest.mark.timeout(120)
def test_an_unpublished_author_name_is_refused(tmp_path: Path) -> None:
    """A real NAME with a compliant address must still fail.

    This was the half the gate originally missed. GitHub's no-reply
    forms anonymise the address and do nothing about `%an`, and the
    commits that motivated this check carried BOTH.
    """
    repo = _repo_with(
        tmp_path, "Jane Q. Doe", "1+rrs@users.noreply.github.com", "name"
    )
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode != 0, f"a real name with a noreply address passed:\n{r.stdout}"
    assert "name" in r.stdout, r.stdout


@pytest.mark.timeout(120)
def test_an_address_hidden_in_the_name_field_is_refused(tmp_path: Path) -> None:
    repo = _repo_with(
        tmp_path, "jane@example.org", "1+rrs@users.noreply.github.com", "inname"
    )
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode != 0, f"an address in the name field passed:\n{r.stdout}"


@pytest.mark.timeout(120)
def test_a_published_identity_passes(tmp_path: Path) -> None:
    """A gate that always fails is also a broken gate."""
    repo = _repo_with(
        tmp_path, "rrs", "276464689+robotrocketscience@users.noreply.github.com", "ok"
    )
    r = _run("--range", "HEAD~1..HEAD", cwd=repo)
    assert r.returncode == 0, f"a compliant identity was refused:\n{r.stdout}"
    assert "clean:" in r.stdout


@pytest.mark.timeout(120)
def test_dry_run_exits_zero_even_when_it_finds_something(tmp_path: Path) -> None:
    repo = _repo_with(tmp_path, "Jane Q. Doe", "jane@corp.example", "dry")
    r = _run("--range", "HEAD~1..HEAD", "--dry-run", cwd=repo)
    assert r.returncode == 0, f"--dry-run must exit 0, got {r.returncode}"
    assert "is not a published identity" in r.stdout, "--dry-run must still report"


@pytest.mark.timeout(60)
def test_offenders_reports_both_halves(tmp_path: Path) -> None:
    """A commit with a bad name AND a bad address reports both."""
    repo = _repo_with(tmp_path, "Jane Q. Doe", "jane@corp.example", "both")
    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        found = chk.offenders("HEAD~1..HEAD")
    finally:
        os.chdir(cwd)
    kinds = {which for *_, which in found}
    assert "author email" in kinds, kinds
    assert "author name" in kinds, kinds


@pytest.mark.timeout(60)
def test_an_unreadable_range_fails_closed() -> None:
    r = _run("--range", "definitely-not-a-ref..HEAD")
    assert r.returncode != 0
    assert "clean:" not in r.stdout
    assert "could not walk" in r.stdout, r.stdout


@pytest.mark.timeout(60)
def test_an_empty_range_is_clean_not_an_error() -> None:
    r = _run("--range", "HEAD..HEAD")
    assert r.returncode == 0, f"empty range must be clean:\n{r.stdout}\n{r.stderr}"


# --- the patterns themselves ---------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "email",
    [
        "276464689+robotrocketscience@users.noreply.github.com",
        "49699333+dependabot[bot]@users.noreply.github.com",
        "octocat@users.noreply.github.com",
        "noreply@github.com",
    ],
)
def test_the_noreply_forms_github_issues_are_accepted(email: str) -> None:
    """Fails closed, so a false rejection blocks every push.

    The `<id>+<login>` and `[bot]` shapes are the two the first draft of
    this pattern got wrong.
    """
    assert chk.email_is_allowed(email), f"legitimate noreply form rejected: {email}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "email",
    [
        "jane.doe@example.org",
        "someone@gmail.com",
        "dev@corp.io",
        # Anchoring: an address that merely CONTAINS the domain must fail.
        "attacker@users.noreply.github.com.evil.test",
        "prefix noreply@github.com",
        "",
    ],
)
def test_routable_addresses_are_refused(email: str) -> None:
    assert not chk.email_is_allowed(email), f"routable address accepted: {email}"


@pytest.mark.timeout(30)
def test_the_address_pattern_is_anchored_at_both_ends() -> None:
    assert not chk.NOREPLY_RE.match("x noreply@github.com")
    assert not chk.NOREPLY_RE.match("noreply@github.com.attacker.test")
    assert not chk.NOREPLY_RE.match("a@users.noreply.github.comX")


@pytest.mark.timeout(30)
@pytest.mark.parametrize("name", ["rrs", "dependabot[bot]", "GitHub"])
def test_published_names_are_accepted(name: str) -> None:
    assert chk.name_is_allowed(name)


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "name",
    ["Jane Doe", "J. Random Hacker", "jane@example.org", "Jane <jane@example.org>", ""],
)
def test_unpublished_names_are_refused(name: str) -> None:
    assert not chk.name_is_allowed(name)


@pytest.mark.timeout(60)
def test_the_script_self_test_passes() -> None:
    r = _run("--self-test")
    assert r.returncode == 0, f"self-test failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.timeout(60)
def test_self_test_fails_when_the_pattern_stops_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--self-test` must be able to FAIL, or it certifies nothing."""
    import re as _re

    monkeypatch.setattr(chk, "NOREPLY_RE", _re.compile(r".*"))
    assert chk.self_test() == 1, "self_test passed a pattern that accepts everything"
