"""`publish.yml` refuses a tag that main does not contain (audit CI-03, CI-04).

Before this, pushing any `vX.Y.Z` tag published to PyPI, whatever commit
it named and whatever version pyproject.toml carried. The first half of
this module runs `scripts/release_preflight.sh` against a throwaway git
repository; the second half reads `publish.yml` as text and pins where
the preflight runs and how every action is referenced.

The workflow is hand-parsed, like the other workflow tests: PyYAML is not
importable in CI (#1436).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(60)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "release_preflight.sh"
_PUBLISH = _REPO / ".github" / "workflows" / "publish.yml"

_PYPROJECT = (
    '[build-system]\n'
    'requires = ["hatchling"]\n'
    '\n'
    '[tool.other]\n'
    'version = "9.9.9"\n'
    '\n'
    '[project]\n'
    'name = "probe"\n'
    'version = "{version}"\n'
)


def _env(tmp_path: Path) -> dict[str, str]:
    """No inherited git variables, a sandboxed HOME, and no store paths.

    A sandboxed HOME also keeps the developer's global git config (and its
    commit signing) out of the throwaway repository.
    """
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "GITHUB_REF_NAME"
    }
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["AELFRICE_DB"] = str(home / "memory.db")
    env["AELFRICE_USER_DB"] = str(home / "user.db")
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    return env


def _git(repo: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo),
         "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        env=env, check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def _commit(repo: Path, env: dict[str, str], version: str, msg: str) -> None:
    (repo / "pyproject.toml").write_text(
        _PYPROJECT.format(version=version), encoding="utf-8",
    )
    _git(repo, env, "add", "pyproject.toml")
    _git(repo, env, "commit", "-q", "--allow-empty", "-m", msg)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """main at version 1.2.3, with origin/main pointing at it.

    `origin/main` is a plain ref written by hand; the script reads a ref
    and never fetches, so no remote is needed.
    """
    env = _env(tmp_path)
    path = tmp_path / "repo"
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(path)],
        env=env, check=True, capture_output=True, text=True, timeout=30,
    )
    _commit(path, env, "1.2.3", "release 1.2.3")
    _git(path, env, "update-ref", "refs/remotes/origin/main", "main")
    return path, env


def _run(
    repo: Path, env: dict[str, str], *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=30,
    )


def test_a_tag_on_main_with_the_matching_version_passes(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.3")
    result = _run(path, env, "v1.2.3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight PASS: tag commit is on origin/main" in result.stdout
    assert (
        "preflight PASS: pyproject.toml version 1.2.3 matches tag v1.2.3"
        in result.stdout
    )


def test_an_annotated_tag_resolves_to_its_commit(
    repo: tuple[Path, dict[str, str]],
) -> None:
    """The log names the commit, not the annotated tag's own object."""
    path, env = repo
    _git(path, env, "tag", "-a", "-m", "annotated", "v1.2.3")
    commit = _git(path, env, "rev-parse", "HEAD")
    assert _git(path, env, "rev-parse", "v1.2.3") != commit
    result = _run(path, env, "v1.2.3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"preflight tag commit: {commit}\n" in result.stdout


def test_the_tag_defaults_to_github_ref_name(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.3")
    result = _run(path, env | {"GITHUB_REF_NAME": "v1.2.3"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight PASS: tag v1.2.3 names a final release" in result.stdout


def test_a_tag_on_a_commit_main_does_not_contain_fails(
    repo: tuple[Path, dict[str, str]],
) -> None:
    """The version matches, so only the ancestry check can fail this."""
    path, env = repo
    _git(path, env, "switch", "-q", "-c", "side")
    _commit(path, env, "1.2.3", "unreviewed change")
    _git(path, env, "tag", "v1.2.3")
    result = _run(path, env, "v1.2.3")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "is not an ancestor of origin/main" in result.stderr
    assert "matches tag v1.2.3" in result.stdout


def test_a_version_that_does_not_match_the_tag_fails(
    repo: tuple[Path, dict[str, str]],
) -> None:
    """The commit is on main, so only the version check can fail this.

    The fixture's `[tool.other]` table carries `version = "9.9.9"` ahead of
    `[project]`, so a reader that took the first `version =` in the file
    would report 9.9.9 here instead of 1.2.3.
    """
    path, env = repo
    _git(path, env, "tag", "v1.2.4")
    result = _run(path, env, "v1.2.4")
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "pyproject.toml version 1.2.3 does not match tag v1.2.4"
        in result.stderr
    )
    assert "preflight PASS: tag commit is on origin/main" in result.stdout


def test_the_version_is_read_at_the_tag_not_from_the_working_tree(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.4")
    (path / "pyproject.toml").write_text(
        _PYPROJECT.format(version="1.2.4"), encoding="utf-8",
    )
    result = _run(path, env, "v1.2.4")
    assert result.returncode == 1, result.stdout + result.stderr


def test_the_version_is_read_at_the_tag_not_at_head(
    repo: tuple[Path, dict[str, str]],
) -> None:
    """The tag sits behind HEAD, and the two commits disagree on version.

    Every other fixture tags HEAD, so a script that read HEAD:pyproject.toml
    would pass them all. Here only the tagged commit carries 1.2.4.
    """
    path, env = repo
    _commit(path, env, "1.2.4", "release 1.2.4")
    _git(path, env, "tag", "v1.2.4")
    _commit(path, env, "1.2.3", "later commit at 1.2.3")
    _git(path, env, "update-ref", "refs/remotes/origin/main", "main")
    result = _run(path, env, "v1.2.4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "preflight PASS: pyproject.toml version 1.2.4 matches tag v1.2.4"
        in result.stdout
    )


def test_a_tag_that_is_not_a_final_release_fails(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.3rc1")
    result = _run(path, env, "v1.2.3rc1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "is not vMAJOR.MINOR.PATCH" in result.stderr


def test_a_main_ref_that_does_not_resolve_is_a_usage_error(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.3")
    result = _run(path, env, "--main-ref", "origin/absent", "v1.2.3")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "main ref origin/absent does not resolve" in result.stderr


def test_dry_run_reports_a_failure_and_exits_zero(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.4")
    result = _run(path, env, "--dry-run", "v1.2.4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "preflight FAIL (dry run, not fatal): pyproject.toml version 1.2.3 "
        "does not match tag v1.2.4" in result.stdout
    )
    assert "dry run: a real run would exit 1" in result.stdout
    assert "::error::" not in result.stdout


def test_dry_run_on_a_passing_tag_says_a_real_run_would_pass(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    _git(path, env, "tag", "v1.2.3")
    result = _run(path, env, "--dry-run", "v1.2.3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dry run: a real run would exit 0" in result.stdout
    assert "FAIL" not in result.stdout


def test_dry_run_still_refuses_a_tag_that_does_not_exist(
    repo: tuple[Path, dict[str, str]],
) -> None:
    path, env = repo
    result = _run(path, env, "--dry-run", "v7.7.7")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "tag v7.7.7 does not resolve" in result.stderr


# --- publish.yml, read as text --------------------------------------------

_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")


def _publish_text() -> str:
    return _PUBLISH.read_text(encoding="utf-8")


def _jobs(text: str) -> dict[str, list[str]]:
    """Each job's lines, keyed by job id (two-space keys under `jobs:`)."""
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for line in text.splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        match = re.match(r"^  (?P<id>[A-Za-z0-9_-]+):\s*$", line)
        if match:
            current = match.group("id")
            jobs[current] = []
            continue
        if current is not None:
            jobs[current].append(line)
    return jobs


def _steps(job_lines: list[str]) -> list[str]:
    """The job's steps, one string per `- ` item under `steps:`.

    Whole-line comments are dropped, so a step mentioned only in a comment
    does not count as a step.
    """
    steps: list[str] = []
    item_indent: int | None = None
    in_steps = False
    for line in job_lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if stripped.startswith("steps:"):
            in_steps = True
            continue
        if not in_steps:
            continue
        if item_indent is None and stripped.startswith("- "):
            item_indent = indent
        if item_indent is not None and indent < item_indent:
            break
        if indent == item_indent and stripped.startswith("- "):
            steps.append(line + "\n")
        elif steps:
            steps[-1] += line + "\n"
    return steps


def _index(steps: list[str], pattern: str) -> list[int]:
    return [i for i, s in enumerate(steps) if re.search(pattern, s)]


_BUILD_OR_PUBLISH = (
    r"setup-uv@|uv sync|pytest|uv build"
    r"|attest-build-provenance@|gh-action-pypi-publish@|gh release"
)


def test_the_parser_finds_the_publish_job_and_its_steps() -> None:
    jobs = _jobs(_publish_text())
    assert list(jobs) == ["publish"], list(jobs)
    steps = _steps(jobs["publish"])
    assert len(steps) >= 8, steps
    assert _index(steps, r"gh-action-pypi-publish@"), steps


def test_the_preflight_is_the_first_step_after_checkout() -> None:
    steps = _steps(_jobs(_publish_text())["publish"])
    checkout = _index(steps, r"uses:\s*actions/checkout@")
    preflight = _index(steps, r"scripts/release_preflight\.sh")
    assert len(checkout) == 1, checkout
    assert len(preflight) == 1, preflight
    assert preflight[0] == checkout[0] + 1, (checkout, preflight)
    later = _index(steps, _BUILD_OR_PUBLISH)
    assert later, "no build or publish step found; the parser has rotted"
    assert preflight[0] < min(later), (preflight, later)


def test_the_preflight_fetches_main_and_is_not_a_dry_run() -> None:
    steps = _steps(_jobs(_publish_text())["publish"])
    (step,) = [s for s in steps if "scripts/release_preflight.sh" in s]
    assert "--dry-run" not in step
    fetch = step.find("refs/heads/main:refs/remotes/origin/main")
    call = step.find("scripts/release_preflight.sh")
    assert 0 <= fetch < call, step
    assert "--main-ref origin/main" in step
    assert '"${GITHUB_REF_NAME}"' in step
    assert "${{" not in step, "pass the tag through the env, not an expression"


def test_every_action_in_publish_is_pinned_to_a_commit_sha() -> None:
    refs = [
        m for m in (
            _USES.match(line) for line in _publish_text().splitlines()
            if not line.lstrip().startswith("#")
        ) if m
    ]
    assert len(refs) >= 5, refs
    for match in refs:
        ref = match.group("ref")
        _, _, version = ref.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", version), (
            f"{ref} is not pinned to a 40-hex commit SHA"
        )
        assert re.match(r"\s+#\s*v\d+\.\d+\.\d+\s*$", match.group("rest")), (
            f"{ref} has no '# vX.Y.Z' comment naming the pinned tag"
        )


_RELEASING = _REPO / "docs" / "concepts" / "RELEASING.md"


def test_the_release_guide_lists_the_preflight_first() -> None:
    """RELEASING.md lists publish.yml's steps in order; the preflight leads.

    The guide's numbered list is what a release author reads to learn why
    a tag was refused, so it has to name the script and put it ahead of
    pytest, the way the workflow runs them.
    """
    text = _RELEASING.read_text(encoding="utf-8")
    anchor = text.index("Pushing a tag starts `.github/workflows/publish.yml`")
    block = text[anchor:].split("\n\n", 2)[1]
    items = re.findall(r"^(\d+)\. (.+)$", block, re.M)
    assert [int(n) for n, _ in items] == list(range(1, len(items) + 1)), items
    assert "scripts/release_preflight.sh" in items[0][1], items
    pytest_steps = [i for i, (_, s) in enumerate(items) if "pytest" in s]
    assert pytest_steps and pytest_steps[0] > 0, items


def test_publish_job_settings() -> None:
    """Timeout, no persisted token, a pinned Python and no restored cache."""
    for job_id, lines in _jobs(_publish_text()).items():
        body = "\n".join(lines)
        assert re.search(r"^    timeout-minutes:\s*\d+\s*$", body, re.M), (
            f"job {job_id} has no timeout-minutes"
        )
    text = _publish_text()
    assert re.search(r"^    timeout-minutes:\s*30\s*$", text, re.M)
    steps = _steps(_jobs(text)["publish"])
    for step in steps:
        if "actions/checkout@" in step:
            assert re.search(r"persist-credentials:\s*false\b", step), step
        if "astral-sh/setup-uv@" in step:
            assert re.search(r"enable-cache:\s*false\b", step), step
            assert re.search(r"python-version:\s*'3\.13'", step), step
