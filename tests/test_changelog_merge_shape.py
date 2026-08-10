"""The claim behind `CHANGELOG/unreleased/`, run rather than asserted (#1475).

Thirteen of fourteen open PRs on the 2026-08-10 board inserted into
`CHANGELOG/v4.md` within lines 8-16. The proposal is that one file per
entry drops the conflict tax to zero rather than merely reducing it.
That is a claim about git's merge behaviour, and a comment asserting it
is worth nothing — so these tests drive real `git merge` in a scratch
repository and read the exit code.

The pair is the point. `test_two_entries_in_the_unreleased_block_conflict`
is the control: remove it and the other test proves only that some
merge somewhere succeeds, which was never in doubt.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SEED_CHANGELOG = "\n".join([
    "# Changelog — v4.x",
    "",
    "## [Unreleased]",
    "",
    "### Fixed",
    "",
    "- **An entry that already shipped ([#1](u)).** Prose.",
    "",
    "## [4.2.0] - 2026-07-21",
    "",
])

# Realistic width. Real entries run 2,000-4,500 characters on one line,
# which is why the resolution has no intra-line granularity.
BODY = "x" * 2000


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c", "user.email=test@example.invalid",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo, capture_output=True, text=True, check=False,
        timeout=30,
    )


def _seed(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    assert _git(repo, "init", "-q", "-b", "main", ".").returncode == 0
    changelog = repo / "CHANGELOG"
    (changelog / "unreleased").mkdir(parents=True)
    (changelog / "v4.md").write_text(SEED_CHANGELOG, encoding="utf-8")
    # git stores no empty tree, so the directory needs a tracked file or
    # both branches would have to create it — an add/add on one path.
    (changelog / "unreleased" / "README.md").write_text(
        "One file per unreleased entry.\n", encoding="utf-8"
    )
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-qm", "seed").returncode == 0


def _branch(repo: Path, name: str) -> None:
    assert _git(repo, "switch", "-q", "-c", name, "main").returncode == 0


def _commit(repo: Path, message: str) -> None:
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-qm", message).returncode == 0


@pytest.mark.timeout(60)
def test_two_entries_in_the_unreleased_block_conflict(tmp_path: Path) -> None:
    """The status quo: both branches insert at the same offset."""
    repo = tmp_path / "block"
    _seed(repo)
    path = repo / "CHANGELOG" / "v4.md"
    for tag, issue in (("a", 101), ("b", 102)):
        _branch(repo, f"block-{tag}")
        path.write_text(
            SEED_CHANGELOG.replace(
                "### Fixed\n\n",
                f"### Fixed\n\n- **PR {tag} ([#{issue}](u)).** {BODY}\n",
                1,
            ),
            encoding="utf-8",
        )
        _commit(repo, f"docs: PR {tag} entry in the block")

    assert _git(repo, "switch", "-q", "block-a").returncode == 0
    merged = _git(repo, "merge", "--no-edit", "block-b")

    assert merged.returncode != 0, merged.stdout
    assert "CONFLICT (content)" in merged.stdout
    assert "<<<<<<<" in path.read_text(encoding="utf-8")
    # Both sides are one 2 KB line: nothing inside them to merge, so the
    # resolution is by hand and a dropped side leaves no trace.
    assert _git(repo, "status", "--short").stdout.strip() == (
        "UU CHANGELOG/v4.md"
    )


@pytest.mark.timeout(60)
def test_two_entry_files_under_unreleased_merge_clean(tmp_path: Path) -> None:
    """The change: distinct paths, so add/add never collides."""
    repo = tmp_path / "files"
    _seed(repo)
    directory = repo / "CHANGELOG" / "unreleased"
    for tag, issue in (("a", 101), ("b", 102)):
        _branch(repo, f"file-{tag}")
        (directory / f"{issue}-pr-{tag}.md").write_text(
            f"### Fixed\n\n- **PR {tag} ([#{issue}](u)).** {BODY}\n",
            encoding="utf-8",
        )
        _commit(repo, f"docs: PR {tag} entry file")

    assert _git(repo, "switch", "-q", "file-a").returncode == 0
    merged = _git(repo, "merge", "--no-edit", "file-b")

    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert "CONFLICT" not in merged.stdout
    assert _git(repo, "status", "--short").stdout == ""
    assert sorted(p.name for p in directory.iterdir()) == [
        "101-pr-a.md", "102-pr-b.md", "README.md",
    ]
