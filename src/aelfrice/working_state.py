"""Post-compact working-state projector (#587).

The PreCompact rebuilder's existing emission packs the recent-turn tail
plus retrieved beliefs — *cold-start* retrieval against the just-
summarized window. What that loses is the agent's *working-state delta*:
the branch it was on, the files it had touched, the commits it just
made. Those are the state-of-work bits the agent was holding in working
memory at compaction time, and they are exactly what gets summarized
away. Beliefs surface what the turns were *about*; this module surfaces
what they *touched*.

This module is the pure projector half. It computes a `WorkingState`
from:
  * the cwd at compaction time (a Path; the rebuilder already resolves
    it from `payload.cwd`);
  * the `RecentTurn` list the rebuilder is already going to emit.

No retrieval, no BM25, no belief extraction. Deterministic projections
of state-at-compaction-time:
  1. current branch + a bounded `git status --porcelain=v1` snapshot;
  2. the last few `git log` entries (HEAD walk);
  3. the last few user prompts (no extra I/O — drawn from recent_turns);
  4. commits authored since the latest session_id's first turn.

Tool-call signatures (issue #587 acceptance bullet 2 second half) are
intentionally deferred — the v1.2 turns.jsonl schema does not capture
them, and reading the harness internal transcript format is volatile
across harness versions. A follow-up issue tracks extending the JSONL
schema.

Latency: every git invocation has a hard 1.5s timeout and a
log-on-error/return-empty fallback. Subprocess errors never propagate.
The PreCompact hook contract is "never block, never raise".
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aelfrice.context_rebuilder import RecentTurn

DEFAULT_MAX_USER_PROMPTS: int = 5
"""Cap the recent-user-prompt projection. Five turns at MAX_TURN_TEXT_CHARS
each is ~2.5KB worst case — well under the rebuilder's token budget."""

DEFAULT_MAX_STATUS_LINES: int = 50
"""Cap on `git status --porcelain` lines surfaced. A repo with hundreds
of dirty files is already in trouble; truncating is fine."""

DEFAULT_RECENT_COMMITS: int = 3
"""HEAD-walk depth for the recent-commits projection. Three is enough to
show "what just landed" without dragging in pre-session history."""

DEFAULT_GIT_TIMEOUT_S: float = 1.5
"""Per-subprocess-call timeout. Multiple git calls fan out, each with
this cap; total p95 budget is ~50ms in a healthy repo, multiples of
that on cold filesystems."""


@dataclass(frozen=True)
class WorkingState:
    """Snapshot of state-of-work at PreCompact-fire time.

    Empty-by-design when `cwd` is not a git repo or every projection
    fails: the rebuilder calls `is_empty()` and omits the sub-block
    rather than emitting hollow XML.
    """
    branch: str | None = None
    status_porcelain: list[str] = field(default_factory=list)
    recent_log: list[str] = field(default_factory=list)
    recent_user_prompts: list[str] = field(default_factory=list)
    session_commits: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.branch is None
            and not self.status_porcelain
            and not self.recent_log
            and not self.recent_user_prompts
            and not self.session_commits
        )


def _run_git(
    cwd: Path,
    args: list[str],
    *,
    timeout: float = DEFAULT_GIT_TIMEOUT_S,
) -> str | None:
    """Invoke git with a hard timeout. Return stdout or None on failure.

    Anything that could surface noise to the user (FileNotFoundError when
    git is absent, subprocess timeouts, non-zero exits) is squashed to
    None — the rebuilder caller treats None as "this projection is
    unavailable" and moves on.
    """
    try:
        result = subprocess.run(  # noqa: S603 -- list-form, no shell expansion
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _project_branch(cwd: Path) -> str | None:
    out = _run_git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"])
    if out is None:
        return None
    branch = out.strip()
    return branch if branch else None


def _project_status(cwd: Path, max_lines: int) -> list[str]:
    out = _run_git(cwd, ["status", "--porcelain=v1"])
    if out is None:
        return []
    lines = [line for line in out.splitlines() if line.strip()]
    return lines[:max_lines]


def _project_recent_log(cwd: Path, n: int) -> list[str]:
    if n <= 0:
        return []
    out = _run_git(cwd, ["log", f"-{n}", "--format=%h %s"])
    if out is None:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _project_user_prompts(
    recent_turns: "list[RecentTurn]", max_prompts: int,
) -> list[str]:
    if max_prompts <= 0:
        return []
    user_texts = [t.text for t in recent_turns if t.role == "user"]
    return user_texts[-max_prompts:]


def _earliest_session_ts(recent_turns: "list[RecentTurn]") -> str | None:
    """Find the oldest `ts` of turns sharing the latest turn's session_id.

    The turns.jsonl schema (docs/transcript_ingest.md) carries `ts` per
    turn. `read_recent_turns_aelfrice` populates `RecentTurn.ts` when the
    line has it; legacy callers and the Claude-Code transcript adapter
    pass through with `ts=None`.
    """
    if not recent_turns:
        return None
    latest_sid = recent_turns[-1].session_id
    if latest_sid is None:
        return None
    candidates = [
        t.ts for t in recent_turns
        if t.session_id == latest_sid and t.ts
    ]
    if not candidates:
        return None
    # turns.jsonl ts is RFC3339 / ISO-8601, lexicographically sortable.
    return min(candidates)


def _project_session_commits(
    cwd: Path, since_ts: str | None, max_lines: int = 10,
) -> list[str]:
    if since_ts is None:
        return []
    out = _run_git(
        cwd,
        ["log", f"--since={since_ts}", f"-{max_lines}", "--format=%h %s"],
    )
    if out is None:
        return []
    return [line for line in out.splitlines() if line.strip()]


def project_working_state(
    cwd: Path,
    recent_turns: "list[RecentTurn]",
    *,
    max_user_prompts: int = DEFAULT_MAX_USER_PROMPTS,
    max_status_lines: int = DEFAULT_MAX_STATUS_LINES,
    recent_commit_count: int = DEFAULT_RECENT_COMMITS,
) -> WorkingState:
    """Project working-state from cwd + recent turns. Pure-ish.

    Pure on `recent_turns`; subprocess on `cwd`. Each git call is
    individually fault-tolerant — a failing `git status` does not
    suppress `git log`, and vice versa. A non-git `cwd` returns an
    all-empty `WorkingState` rather than raising.

    Caller (the PreCompact rebuilder) checks `result.is_empty()` and
    omits the `<working-state>` sub-block when nothing populated.
    """
    branch = _project_branch(cwd)
    status = _project_status(cwd, max_status_lines)
    log = _project_recent_log(cwd, recent_commit_count)
    user_prompts = _project_user_prompts(recent_turns, max_user_prompts)
    since_ts = _earliest_session_ts(recent_turns)
    session_commits = _project_session_commits(cwd, since_ts)
    return WorkingState(
        branch=branch,
        status_porcelain=status,
        recent_log=log,
        recent_user_prompts=user_prompts,
        session_commits=session_commits,
    )
