"""PreToolUse:Bash guard — duplicate-detection before `gh issue create`.

Fires when the host harness is about to execute a Bash command that starts with
`gh issue create`. Runs two parallel dup-detection sweeps and blocks the call
(exit 2) when any candidate scores above the Jaccard threshold.

Hook contract
-------------
* stdin  — PreToolUse hook JSON envelope:
    {"tool_name": "Bash", "tool_input": {"command": "...", "description": "..."}, ...}
* stdout — human-readable additionalContext block (on PASS, may be empty).
* stderr — block message + candidate list (on BLOCK, exit 2).
* Non-Bash tools or non-gh-issue-create commands → exit 0 (transparent pass).

Token-overlap scoring
---------------------
Title tokens are derived by:
  1. Stripping a leading conventional-commit prefix  (``feat(scope):``, ``fix:``…).
  2. Lowercasing.
  3. Splitting on non-alphanumeric runs.
  4. Dropping English stop-words.
Similarity is Jaccard on token sets.  Default block threshold: 0.5.

Rationale for 0.5: a pair of issues that share half their meaningful tokens
almost certainly describe the same concept; lower thresholds produced
false-positives in manual evaluation; higher thresholds missed near-verbatim
duplicates that differed only in conventional-commit prefix.

Env overrides
-------------
* ``ALLOW_DUP_ISSUE=1``            — always PASS (emergency bypass).
* ``AELFRICE_NO_PRE_ISSUE_GUARD=1``— always PASS (install-time opt-out).
Both are mutually honoured; the first that evaluates truthy wins.

Body-file safety
----------------
``--body-file <F>`` paths that resolve under ``~/.claude/`` are refused; the
body read is silently skipped (treated as empty string) so the guard still
runs on title tokens alone.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

BLOCK_THRESHOLD: float = 0.5
"""Jaccard score above which a candidate triggers a BLOCK (exit 2)."""

TOP_N_CANDIDATES: int = 5
"""Maximum number of candidates printed in the BLOCK message."""

TOP_K_QUERY_TOKENS: int = 3
"""Number of highest-value query tokens forwarded to gh/git searches."""

GH_RESULT_LIMIT: int = 20
"""Maximum candidates requested from `gh issue list`."""

GIT_LOG_LIMIT: int = 20
"""Maximum commit-message candidates requested from `git log`."""

_CLAUDE_DIR: Path = Path.home() / ".claude"

# ---------------------------------------------------------------------------
# Stop-words (tiny set — filters noise, not meaning)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or",
    "is", "are", "be", "at", "by", "as", "it", "its", "from", "into",
    "that", "this", "was", "were", "has", "have", "been", "not", "but",
    "if", "so", "do", "no", "up", "out", "via", "per", "vs",
})

# Matches a leading conventional-commit prefix such as:
#   feat(scope):   fix:   perf(foo/bar):   docs:
_CC_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^[a-z]+(?:\([^)]*\))?!?\s*:\s*", re.IGNORECASE
)

# Non-alphanumeric run splitter
_SPLIT_RE: re.Pattern[str] = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------


def tokenize_title(title: str) -> set[str]:
    """Return meaningful lowercase tokens from an issue or commit title.

    Drops the conventional-commit prefix, lowercases, splits on
    non-alphanumeric runs, drops stop-words, and filters single-char tokens.
    """
    stripped = _CC_PREFIX_RE.sub("", title, count=1)
    lowered = stripped.lower()
    raw = _SPLIT_RE.split(lowered)
    return {t for t in raw if len(t) > 1 and t not in _STOP_WORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets.  Returns 0.0 when both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def score_candidate(query_tokens: set[str], cand_title: str) -> float:
    """Jaccard score of *query_tokens* against the tokens of *cand_title*."""
    return jaccard(query_tokens, tokenize_title(cand_title))


# ---------------------------------------------------------------------------
# Command parsing helpers
# ---------------------------------------------------------------------------


def _is_gh_issue_create(command: str) -> bool:
    """Return True when *command* begins with `gh issue create` (ignoring leading ws)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    # Tolerate leading env-var assignments (KEY=VAL gh ...)
    while tokens and "=" in tokens[0]:
        tokens = tokens[1:]
    return (
        len(tokens) >= 3
        and tokens[0] == "gh"
        and tokens[1] == "issue"
        and tokens[2] == "create"
    )


def _extract_title(command: str) -> str:
    """Pull the --title value out of a gh issue create command string."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-t", "--title") and i + 1 < len(tokens):
            return tokens[i + 1]
        # --title=value form
        if tok.startswith("--title="):
            return tok[len("--title="):]
        i += 1
    return ""


def _extract_body_file(command: str) -> str:
    """Pull the --body-file path out of a gh issue create command string."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-F", "--body-file") and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith("--body-file="):
            return tok[len("--body-file="):]
        i += 1
    return ""


def _safe_read_body_file(path_str: str) -> str:
    """Read *path_str* if it is a regular file outside ``~/.claude/``.

    Returns empty string on any failure or if the path is under ~/.claude/.
    """
    if not path_str:
        return ""
    p = Path(path_str).expanduser()
    try:
        resolved = p.resolve()
    except (OSError, ValueError):
        return ""
    # Refuse paths that originate under ~/.claude/
    try:
        resolved.relative_to(_CLAUDE_DIR.resolve())
        return ""  # inside ~/.claude/ — refuse
    except ValueError:
        # relative_to raises ValueError when resolved is NOT under
        # _CLAUDE_DIR — that's the path we want to read. Fall through.
        pass
    try:
        if resolved.is_file():
            return resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Unreadable / permission-denied / vanished mid-read — treat as
        # empty body so the guard fails open rather than aborting.
        pass
    return ""


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _top_query_tokens(tokens: set[str], n: int = TOP_K_QUERY_TOKENS) -> list[str]:
    """Return up to *n* tokens, sorted for determinism."""
    return sorted(tokens)[:n]


def _build_gh_candidates(
    query_tokens: list[str],
    gh_runner: Callable[[list[str]], str],
) -> list[dict[str, object]]:
    """Run `gh issue list` and return parsed JSON rows."""
    if not query_tokens:
        return []
    search_str = " ".join(query_tokens)
    try:
        raw = gh_runner([
            "gh", "issue", "list",
            "--state", "all",
            "--search", search_str,
            "--limit", str(GH_RESULT_LIMIT),
            "--json", "number,title,state,stateReason,closedAt",
        ])
    except Exception:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError):
        # Malformed JSON from `gh` (network hiccup, auth prompt to stderr
        # leaking into stdout, etc.) — treat as no candidates and fall
        # through so the guard fails open.
        pass
    return []


def _build_git_candidates(
    query_tokens: list[str],
    git_runner: Callable[[list[str]], str],
) -> list[dict[str, object]]:
    """Run `git log --grep` and synthesise pseudo-issue rows."""
    if not query_tokens:
        return []
    grep_pattern = "|".join(re.escape(t) for t in query_tokens)
    try:
        raw = git_runner([
            "git", "log",
            f"--grep={grep_pattern}",
            "--extended-regexp",
            "--oneline",
            f"-{GIT_LOG_LIMIT}",
        ])
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        title = parts[1] if len(parts) > 1 else line
        rows.append({
            "number": None,
            "title": title,
            "state": "MERGED",
            "stateReason": "shipped",
            "closedAt": None,
        })
    return rows


def _score_and_rank(
    query_tokens: set[str],
    candidates: list[dict[str, object]],
) -> list[tuple[float, dict[str, object]]]:
    """Return (score, row) pairs sorted descending by score."""
    scored: list[tuple[float, dict[str, object]]] = []
    for cand in candidates:
        raw_title = cand.get("title") or ""
        if not isinstance(raw_title, str):
            raw_title = ""
        sc = score_candidate(query_tokens, raw_title)
        scored.append((sc, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _format_block_message(
    proposed_title: str,
    top: list[tuple[float, dict[str, object]]],
) -> str:
    lines = [
        "aelf-pre-issue-guard: BLOCK — proposed title looks like a duplicate.",
        f"  Proposed: {proposed_title!r}",
        "",
        "  Top matches:",
    ]
    for score, row in top:
        number = row.get("number")
        title = row.get("title") or "(no title)"
        state = row.get("state") or ""
        reason = row.get("stateReason") or ""
        ref = f"#{number}" if number is not None else "(git-log)"
        state_str = f"{state}/{reason}" if reason else state
        lines.append(f"    {ref:>7}  [{state_str}]  score={score:.2f}  {title!r}")
    lines += [
        "",
        "  To bypass: set ALLOW_DUP_ISSUE=1 in the host's environment",
        "  (an inline KEY=VAL prefix on the gh command does not reach this",
        "  hook), or pass --no-pre-issue-guard to `aelf setup` to disable",
        "  globally.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public guard entry point
# ---------------------------------------------------------------------------


def run_guard(
    stdin_json: dict[str, object],
    *,
    gh_runner: Callable[[list[str]], str] | None = None,
    git_runner: Callable[[list[str]], str] | None = None,
    stderr_out: object = None,
) -> int:
    """Evaluate *stdin_json* and return an exit code.

    Returns:
      0 — PASS (transparent; the host harness continues normally).
      2 — BLOCK (the host harness surfaces the stderr message and aborts the tool call).

    *gh_runner* and *git_runner* are injectable callables
    ``(argv: list[str]) -> stdout_str`` used during testing.  When None, real
    subprocess calls are used.
    """
    import sys as _sys  # local ref so callers can patch

    _stderr = stderr_out if stderr_out is not None else _sys.stderr

    # --- Env overrides -------------------------------------------------------
    if os.environ.get("ALLOW_DUP_ISSUE") == "1":
        return 0
    if os.environ.get("AELFRICE_NO_PRE_ISSUE_GUARD") == "1":
        return 0

    # --- Gate on tool_name ---------------------------------------------------
    if stdin_json.get("tool_name") != "Bash":
        return 0

    tool_input = stdin_json.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    command = tool_input.get("command")
    if not isinstance(command, str):
        return 0

    if not _is_gh_issue_create(command):
        return 0

    # --- Extract title -------------------------------------------------------
    title = _extract_title(command)
    if not title:
        # No --title provided; cannot score.  PASS silently.
        return 0

    # --- Optional body read (title-only scoring is fine without body) --------
    body_file = _extract_body_file(command)
    _safe_read_body_file(body_file)  # read but not currently used in scoring

    # --- Tokenize and build query -------------------------------------------
    query_tokens = tokenize_title(title)
    if not query_tokens:
        return 0

    top_tokens = _top_query_tokens(query_tokens)

    # --- Set up runners ------------------------------------------------------
    # subprocess.run on a list (no shell=True) is shell-injection-safe; the
    # argv comes from constants + tokens we built ourselves. nosec to silence
    # the static-analysis flag without changing the call shape.
    def _default_gh_runner(argv: list[str]) -> str:
        result = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, timeout=10,
        )
        return result.stdout

    def _default_git_runner(argv: list[str]) -> str:
        result = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, timeout=5,
        )
        return result.stdout

    _gh = gh_runner if gh_runner is not None else _default_gh_runner
    _git = git_runner if git_runner is not None else _default_git_runner

    # --- Fetch candidates in "parallel" (sequential, both always run) -------
    gh_candidates = _build_gh_candidates(top_tokens, _gh)
    git_candidates = _build_git_candidates(top_tokens, _git)
    all_candidates = gh_candidates + git_candidates

    if not all_candidates:
        return 0

    # --- Score and filter ---------------------------------------------------
    ranked = _score_and_rank(query_tokens, all_candidates)
    above_threshold = [(sc, row) for sc, row in ranked if sc >= BLOCK_THRESHOLD]

    if not above_threshold:
        return 0

    # --- BLOCK --------------------------------------------------------------
    top = above_threshold[:TOP_N_CANDIDATES]
    msg = _format_block_message(title, top)
    print(msg, file=_stderr)  # type: ignore[arg-type]
    return 2


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``aelf-pre-issue-hook`` console script."""
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)
    if not isinstance(payload, dict):
        sys.exit(0)
    rc = run_guard(payload)
    sys.exit(rc)


if __name__ == "__main__":
    main()
