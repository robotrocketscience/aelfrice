"""The "agreement is not quality" rule stays committed and cited (#1453 §1).

The rule exists because a published table was retracted in-tree for treating
agreement between two approximations as validation. The retraction commit is
the evidence, so a note that cites a SHA which no longer resolves is worse than
no note — it reads as sourced while pointing at nothing.

Two things are pinned, and deliberately only two:

  1. The note is where a person running a retrieval measurement would open it.
  2. The SHA it cites resolves in this repository.

Not pinned: the note's wording. A text match on prose breaks on rephrasing and
says nothing about whether the rule is still stated.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTE = REPO / "benchmarks" / "README.md"

# The retraction the rule is built on: "revert(bench): stop calling the
# recorded query the production population".
RETRACTION_SHA = "848dbf83aab6e2c5198b1bc5b0f70ebcc7f11ea9"


def test_the_rule_lives_next_to_the_harnesses() -> None:
    """A rule filed where nobody looks is not a landed rule."""
    assert NOTE.exists(), f"{NOTE} is gone; the rule lost its home"
    text = NOTE.read_text(encoding="utf-8")
    assert "Agreement is not quality" in text, (
        "benchmarks/README.md no longer states the agreement-is-not-quality "
        "rule (#1453 §1)"
    )
    assert RETRACTION_SHA[:8] in text, (
        f"the note no longer cites {RETRACTION_SHA[:8]}, the retraction that "
        "motivates it"
    )


@pytest.mark.timeout(60)
def test_the_cited_retraction_still_resolves() -> None:
    """A citation to an unreachable object is a broken citation.

    `git cat-file -t` rather than `rev-parse`: rev-parse happily echoes a
    well-formed hex string it cannot resolve under some configurations, which
    would make this pass for a SHA that is not in the repository.
    """
    proc = subprocess.run(
        ["git", "cat-file", "-t", RETRACTION_SHA],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=30,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "commit", (
        f"{RETRACTION_SHA} does not resolve to a commit in this repository "
        f"(git said: {proc.stdout.strip() or proc.stderr.strip()}). The note "
        "in benchmarks/README.md cites it as the retraction."
    )
