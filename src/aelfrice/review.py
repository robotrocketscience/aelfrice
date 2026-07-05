"""Weekly review workflow: generate a checkbox file and apply verdicts (#936).

Pure-function module. No CLI argparse, no print calls. All side-effectful
operations are delegated to the store via thin named methods so the module
stays unit-testable with an in-memory store.

Public surface
--------------
select_candidates(store, *, limit=10) -> list[Belief]
render_review_file(candidates, *, now: datetime) -> str
parse_review_file(text: str) -> list[ParsedDecision]
apply_decisions(store, decisions, *, now: str) -> ApplyReport

Exceptions
----------
AmbiguousRowError   — raised by parse_review_file when a row has >1 checked box
MalformedRowError   — defense-in-depth in parse_review_file's belief-ID
                      check; unreachable via the current row regex, which
                      requires at least one non-whitespace ID character to
                      match at all (rows with no ID token are silently
                      ignored, not raised on)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from aelfrice.models import (
    LOCK_USER,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore

# ── Errors ──────────────────────────────────────────────────────────────────

class AmbiguousRowError(ValueError):
    """Raised when a review-file row has more than one checkbox checked.

    apply_decisions raises immediately on the first ambiguous row without
    applying any changes (fail-closed semantics).
    """
    def __init__(self, belief_id: str, row_text: str) -> None:
        self.belief_id = belief_id
        self.row_text = row_text
        super().__init__(
            f"ambiguous verdict on belief {belief_id!r}: "
            f"exactly one box must be checked per row.\n"
            f"  row: {row_text!r}"
        )


class MalformedRowError(ValueError):
    """Raised when a review-file row matches the checkbox pattern but
    the belief ID cannot be parsed (empty or missing).

    apply_decisions raises immediately on the first malformed row.
    """
    def __init__(self, row_text: str) -> None:
        self.row_text = row_text
        super().__init__(
            f"malformed review row — belief ID missing or empty.\n"
            f"  row: {row_text!r}"
        )


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParsedDecision:
    """A single parsed verdict from the review markdown file."""
    belief_id: str
    verdict: Literal["keep", "remove", "lock", "skip"]
    row_text: str


@dataclass
class ApplyReport:
    """Summary of what apply_decisions did."""
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    locked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Row format regex ─────────────────────────────────────────────────────────

# Matches a single review row of the form:
#   - [ ] keep   [ ] remove   [ ] lock   | <id> (…) — …
# Groups: keep_box, remove_box, lock_box, belief_id
_ROW_RE = re.compile(
    r"^\s*-\s+"
    r"\[(?P<keep_box>[xX ])\]\s+keep\s+"
    r"\[(?P<remove_box>[xX ])\]\s+remove\s+"
    r"\[(?P<lock_box>[xX ])\]\s+lock\s+"
    r"\|\s+(?P<belief_id>\S+)",
    re.MULTILINE,
)

_SNIPPET_LEN = 80
_HEADER = (
    "## aelfrice review — generated {date}\n"
    "For each belief: leave checkboxes empty to no-op; "
    "check exactly one verdict per row.\n"
)


# ── Public helpers ───────────────────────────────────────────────────────────

def select_candidates(store: MemoryStore, *, limit: int = 10) -> list[Belief]:
    """Return the next batch of beliefs to include in a review file.

    Delegates to MemoryStore.list_review_candidates which sorts by:
      1. last_confirmed_at ASC NULLS FIRST
      2. last_retrieved_at ASC NULLS FIRST
      3. created_at ASC
    and excludes soft-deleted and user-locked beliefs.
    """
    return store.list_review_candidates(limit=limit)


def _age_days(created_at: str, now: datetime) -> int:
    """Return floor(days since created_at). Negative is clamped to 0."""
    try:
        ts = datetime.fromisoformat(created_at.rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
    except (ValueError, AttributeError):
        return 0
    delta = now - ts
    return max(0, delta.days)


def _cold_days(belief: Belief, now: datetime) -> int:
    """Return floor(days since the most recent retrieval or confirmation).

    Falls back to (now - created_at) when both last_retrieved_at and
    last_confirmed_at are NULL, capturing beliefs that have never been
    explicitly retrieved or confirmed.
    """
    candidates: list[str] = []
    if belief.last_retrieved_at:
        candidates.append(belief.last_retrieved_at)
    if belief.last_confirmed_at:
        candidates.append(belief.last_confirmed_at)
    if not candidates:
        return _age_days(belief.created_at, now)
    most_recent = max(candidates)
    try:
        ts = datetime.fromisoformat(most_recent.rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
    except (ValueError, AttributeError):
        return _age_days(belief.created_at, now)
    delta = now - ts
    return max(0, delta.days)


def _snippet(content: str) -> str:
    """First _SNIPPET_LEN characters of content, truncated with '...'."""
    if len(content) <= _SNIPPET_LEN:
        return content
    return content[:_SNIPPET_LEN] + "..."


def render_review_file(candidates: list[Belief], *, now: datetime) -> str:
    """Render a markdown checkbox file for the given candidates.

    Each row is:
      - [ ] keep   [ ] remove   [ ] lock   | <id> (<age>d old, <cold>d cold) — <snippet>

    The file starts with a datestamped header. Empty candidate list
    produces a header-only file with no rows.
    """
    today = now.date().isoformat() if hasattr(now, "date") else str(date.today())
    lines: list[str] = [_HEADER.format(date=today), ""]
    for b in candidates:
        age = _age_days(b.created_at, now)
        cold = _cold_days(b, now)
        snip = _snippet(b.content)
        lines.append(
            f"- [ ] keep   [ ] remove   [ ] lock"
            f"   | {b.id} ({age}d old, {cold}d cold) — {snip}"
        )
    return "\n".join(lines) + "\n"


def parse_review_file(text: str) -> list[ParsedDecision]:
    """Parse a review markdown file into a list of decisions.

    Rules:
    - Rows with all three boxes empty → verdict='skip'
    - Rows with exactly one box checked → verdict is that box's label
    - Rows with two or more boxes checked → raise AmbiguousRowError
      immediately (fail-closed; no partial list returned)
    - Rows with a missing/empty belief ID → raise MalformedRowError

    Non-matching lines are silently ignored so the user can add notes.
    """
    decisions: list[ParsedDecision] = []
    for m in _ROW_RE.finditer(text):
        row_text = m.group(0).strip()
        belief_id = m.group("belief_id").strip()
        if not belief_id:
            raise MalformedRowError(row_text)

        checked: list[str] = []
        if m.group("keep_box").lower() == "x":
            checked.append("keep")
        if m.group("remove_box").lower() == "x":
            checked.append("remove")
        if m.group("lock_box").lower() == "x":
            checked.append("lock")

        if len(checked) >= 2:
            raise AmbiguousRowError(belief_id, row_text)

        verdict: Literal["keep", "remove", "lock", "skip"]
        if len(checked) == 1:
            verdict = checked[0]  # type: ignore[assignment]
        else:
            verdict = "skip"

        decisions.append(ParsedDecision(
            belief_id=belief_id,
            verdict=verdict,
            row_text=row_text,
        ))
    return decisions


def apply_decisions(
    store: MemoryStore,
    decisions: list[ParsedDecision],
    *,
    now: str,
) -> ApplyReport:
    """Dispatch verdicts from a parsed review file.

    Verdict semantics:
    - keep   → update_last_confirmed_at(belief_id, now)
    - remove → soft_delete_belief (existing primitive), followed by an
               explicit insert_feedback_event(source="review:remove") audit row
    - lock   → set lock_level=user, locked_at=now, origin=user_stated,
               write a 'review:lock' audit row
    - skip   → no-op (belief stays in the next review cycle)

    An unknown belief_id (belief not found in store) is recorded in
    ApplyReport.errors; the rest of the decisions continue to apply.
    This is the only error that is non-fatal. Ambiguous rows must have
    been caught by parse_review_file before calling here.

    `now` must be an ISO-8601 UTC string (e.g. '2026-06-04T10:00:00Z').
    """
    report = ApplyReport()

    for decision in decisions:
        bid = decision.belief_id
        verdict = decision.verdict

        if verdict == "skip":
            report.skipped.append(bid)
            continue

        belief = store.get_belief(bid)
        if belief is None:
            report.errors.append(f"{bid}: not found in store")
            continue

        if verdict == "keep":
            store.update_last_confirmed_at(bid, now)
            report.kept.append(bid)

        elif verdict == "remove":
            store.soft_delete_belief(bid, ts=now)
            # Write an audit row so the removal is traceable.
            store.insert_feedback_event(
                belief_id=bid,
                valence=0.0,
                source="review:remove",
                created_at=now,
            )
            report.removed.append(bid)

        elif verdict == "lock":
            # Promote the belief to user-locked tier by direct field
            # update (mirrors the re-lock path in cli._cmd_lock).
            belief.lock_level = LOCK_USER
            belief.locked_at = now
            belief.origin = ORIGIN_USER_STATED
            store.update_belief(belief)
            # Audit row tags the lock as review-originated.
            store.insert_feedback_event(
                belief_id=bid,
                valence=0.0,
                source="review:lock",
                created_at=now,
            )
            report.locked.append(bid)

    return report
