"""Guards for `scripts/check_commit_identity.py` (#1617).

Two commits reached `main` carrying a contributor's real name and
corporate email as the git author. Commit metadata is copied into every
clone and fork, served by the unauthenticated REST API, and pinned by
`refs/pull/<n>/head`, so unlike file content it cannot be withdrawn.
Refusing it before the push is the only real control, and these tests
are what keep that refusal honest.

The pattern fails *closed*: a bug that rejects a legitimate form blocks
every push. Both real-world shapes GitHub issues — the `<id>+<login>`
form and the `[bot]` form — are therefore pinned explicitly, because
the first draft of this pattern rejected both.
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
def test_the_noreply_forms_github_actually_issues_are_accepted(email: str) -> None:
    """Fails closed, so a false rejection blocks every push.

    The `<id>+<login>` and `[bot]` shapes are the two the first draft of
    this pattern got wrong.
    """
    assert chk.NOREPLY_RE.match(email), f"legitimate noreply form rejected: {email}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "email",
    [
        "jane.doe@example.org",
        "someone@gmail.com",
        "dev@corp.io",
        # Anchoring matters: a routable address that merely *contains* the
        # noreply domain must not pass.
        "attacker@users.noreply.github.com.evil.test",
        "prefix noreply@github.com",
        "",
    ],
)
def test_routable_addresses_are_refused(email: str) -> None:
    assert not chk.NOREPLY_RE.match(email), f"routable address accepted: {email}"


@pytest.mark.timeout(30)
def test_the_pattern_is_anchored_at_both_ends() -> None:
    """A substring match would let any address through with a suffix.

    Pinned separately from the table above because this is the property
    that makes the check meaningful rather than decorative.
    """
    assert not chk.NOREPLY_RE.match("x noreply@github.com")
    assert not chk.NOREPLY_RE.match("noreply@github.com.attacker.test")
    assert not chk.NOREPLY_RE.match("a@users.noreply.github.comX")


@pytest.mark.timeout(60)
def test_the_script_self_test_passes() -> None:
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--self-test"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, f"self-test failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.timeout(60)
def test_dry_run_exits_zero() -> None:
    """The reusable-script rule wants a report mode that never fails."""
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--range", "HEAD~1..HEAD", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, f"--dry-run must exit 0, got {r.returncode}"


@pytest.mark.timeout(60)
def test_a_range_with_no_commits_is_clean_not_an_error() -> None:
    """An empty range must pass, or the gate blocks a no-op push."""
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--range", "HEAD..HEAD"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, f"empty range must be clean:\n{r.stdout}\n{r.stderr}"
