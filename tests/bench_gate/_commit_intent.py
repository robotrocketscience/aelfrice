"""Commit-intent classifier for #438 correction-detection eval Path A1.

Classifies a commit message into one of {"fix", "correction", "revert-of-error", None}
via a deterministic regex over a small keyword bank. Test-utility scope only;
not shipped library code (see docs/design/feature-correction-detection-eval.md § Path A).

Usage:
    from tests.bench_gate._commit_intent import classify_commit_intent

    intent = classify_commit_intent("Fix #1234: docstring claimed bool, returns list")
    # -> "fix"

The classifier is intentionally narrow. Ambiguous or non-corrective commits
return None so the eval composite verdict can short-circuit before consulting
the more expensive substrate paths.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

CommitIntent = Literal["fix", "correction", "revert-of-error"]

_REVERT_RE = re.compile(
    r"\b(revert|reverts|reverting|reverted|rollback|rolls?\s+back|rolled\s+back|"
    r"backout|backed\s+out|undo|undid)\b",
    re.IGNORECASE,
)

_CORRECTION_RE = re.compile(
    r"\b(correct|corrects|corrected|correction|corrections|"
    r"wrong|incorrect|inaccurate|misleading|misstated|"
    r"typo|typos|misspelled|misspelling)\b",
    re.IGNORECASE,
)

_FIX_RE = re.compile(
    r"\b(fix|fixes|fixed|fixing|"
    r"bug|bugfix|hotfix|patch|patched|"
    r"regression|regressions|broken)\b",
    re.IGNORECASE,
)


def classify_commit_intent(message: str) -> Optional[CommitIntent]:
    """Return the commit-intent label for a commit message, or None.

    Precedence is revert-of-error > correction > fix: a "revert wrong commit"
    message classifies as revert-of-error rather than correction, and a
    "fix wrong docstring" classifies as correction rather than fix. The
    precedence reflects the labelling rationale in the spec — explicit
    revert and correction language carries stronger evidence of "A was wrong"
    than the broader fix vocabulary.
    """
    if not message:
        return None

    if _REVERT_RE.search(message):
        return "revert-of-error"
    if _CORRECTION_RE.search(message):
        return "correction"
    if _FIX_RE.search(message):
        return "fix"
    return None
