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


def _is_shallow() -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=30,
    )
    return proc.stdout.strip() == "true"


@pytest.mark.timeout(60)
def test_the_cited_retraction_still_resolves() -> None:
    """A citation to an unreachable object is a broken citation.

    `git cat-file -t` rather than `rev-parse`: rev-parse happily echoes a
    well-formed hex string it cannot resolve under some configurations, which
    would make this pass for a SHA that is not in the repository.

    Resolution is attempted FIRST, and a shallow clone only excuses a failure.
    Probing shallowness up front is wrong: this repository reports
    `--is-shallow-repository true` while still holding the object, so an
    up-front skip would forfeit the check on the very clones that can run it.

    So the three cases separate correctly:
      * resolves               -> pass, wherever it ran
      * missing, history full  -> fail (a genuinely broken citation)
      * missing, history cut   -> skip (CI uses `fetch-depth: 1`, and a commit
                                 from two weeks ago is legitimately absent —
                                 this test failed in CI for exactly that)

    **Stated plainly, because a skip is not a gate:** on a shallow clone a
    mistyped SHA is indistinguishable from a truncated history, so this test
    skips rather than catches it. This repository reports shallow, so that is
    the common case, not the exotic one. `test_the_cited_sha_is_well_formed`
    is the assertion that runs everywhere; it catches corruption and
    truncation but cannot catch a plausible typo. Verify a new citation on a
    full clone.
    """
    proc = subprocess.run(
        ["git", "cat-file", "-t", RETRACTION_SHA],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=30,
    )
    if proc.returncode == 0 and proc.stdout.strip() == "commit":
        return
    if _is_shallow():
        pytest.skip(
            "shallow clone: history does not reach the cited commit, so "
            "resolution cannot be checked here (runs on full clones)"
        )
    raise AssertionError(
        f"{RETRACTION_SHA} does not resolve to a commit in this repository "
        f"(git said: {proc.stdout.strip() or proc.stderr.strip()}). The note "
        "in benchmarks/README.md cites it as the retraction."
    )


def test_the_cited_sha_is_well_formed() -> None:
    """Runs everywhere, shallow clones included.

    Not a substitute for resolution — it cannot tell a real commit from a
    plausible typo — but it catches a truncated or corrupted citation with no
    environment dependency, so the shallow-clone skip above does not leave the
    citation completely unguarded in CI.
    """
    assert len(RETRACTION_SHA) == 40, "a full 40-character SHA is expected"
    assert all(c in "0123456789abcdef" for c in RETRACTION_SHA), (
        f"{RETRACTION_SHA} is not lowercase hexadecimal"
    )
