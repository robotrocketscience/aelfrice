"""PostToolUse hook that turns each successful `git commit` into an
ingest event: parse the message, run the triple extractor, persist
beliefs and edges under a session derived from git context.

Closes the v1.0 limitation that the belief graph only grows on
explicit `aelf onboard` / `aelf lock` calls. Each commit
message becomes a typed-edge ingest, which is the first ingest path
that densely populates `Edge.anchor_text`, `Belief.session_id`, and
`DERIVED_FROM` edges in production data.

Hook contract (Claude Code PostToolUse):
- payload includes `tool_name`, `tool_input`, `tool_response`,
  `cwd`, plus the standard event fields. We act only when:
    * tool_name == "Bash"
    * tool_input.command starts with `git commit`
    * tool_response is not flagged as an error / interrupted
- All failure modes return exit 0 silently. The hook may NEVER
  cause a `git commit` to feel broken.

Latency budget per docs/design/commit_ingest_hook.md:
    median <= 30 ms, p95 <= 100 ms

Tactics:
- Lazy imports of triple_extractor and store (cold-start dominates).
- Skip empty / merge / amend-without-message commits up front.
- Cap the message body at 4 KB before extraction.
- One git subprocess at most: `git log -1 --format=%B <hash>` to
  fetch the just-committed message body. The branch and short
  hash come from the bracketed prefix `[branch hash]` Claude Code
  already captured in `tool_response.stdout` — no extra git calls.

Local-only: brain-graph writes never cross the git boundary or any
network boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from typing import IO, Final, cast

from aelfrice.stream_encoding import ensure_utf8_streams, read_payload_text

MESSAGE_BYTE_CAP: Final[int] = 4096
"""Truncate commit messages above this many bytes before extraction.
Long commit messages are rare; the cap bounds the worst case."""

GIT_LOG_TIMEOUT_S: Final[float] = 2.0
"""Per-call timeout for `git log` so a hung git binary cannot block
the hook past the latency budget."""

# Conservative pattern: optional leading whitespace, then `git`, then
# `commit` as the first sub-token. Catches `git commit -m ...`,
# `git  commit --amend`, `  git commit -F ...`. Does NOT match
# `git -c user.email=x commit ...` — rare; if observed, broaden later.
_GIT_COMMIT_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*git\s+commit\b"
)

# `git commit` prints `[branch shorthash[ ...]] subject` on success.
# Capture both groups to derive session_id and look up the full body.
_COMMIT_BRACKET_RE: Final[re.Pattern[str]] = re.compile(
    r"^\[([^\s\]]+)\s+(?:\(root-commit\)\s+)?([0-9a-f]{4,40})[\s\]]"
)


def _read_payload(
    stdin: IO[str],
    stderr: IO[str] | None = None,
) -> dict[str, object] | None:
    raw = read_payload_text(stdin, stderr)
    if raw is None or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast(dict[str, object], parsed)


def _is_successful_git_commit(payload: dict[str, object]) -> bool:
    if payload.get("tool_name") != "Bash":
        return False
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    cmd = cast(dict[str, object], tool_input).get("command")
    if not isinstance(cmd, str) or not _GIT_COMMIT_RE.match(cmd):
        return False
    tool_response = payload.get("tool_response")
    if isinstance(tool_response, dict):
        resp = cast(dict[str, object], tool_response)
        if resp.get("isError") is True or resp.get("interrupted") is True:
            return False
    return True


def _branch_and_hash_from_stdout(stdout: str) -> tuple[str, str] | None:
    for line in stdout.splitlines():
        m = _COMMIT_BRACKET_RE.match(line)
        if m:
            return m.group(1), m.group(2)
    return None


def _read_full_commit_message(commit_hash: str, cwd: str | None) -> str | None:
    """Run `git log -1 --format=%B <hash>` to fetch the full message
    body. Returns None on any failure — hook stays silent."""
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%B", commit_hash],
            capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace",
            timeout=GIT_LOG_TIMEOUT_S, cwd=cwd,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.rstrip("\n")


def _derive_session_id(branch: str, commit_hash: str) -> str:
    """Stable id from sha256(branch + ':' + commit_hash)[:16].

    Idempotent: two hook invocations on the same commit produce the
    same id. Cross-machine stable: the same commit on two clones
    produces the same id."""
    raw = f"{branch}:{commit_hash}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _truncate_for_extraction(message: str) -> str:
    encoded = message.encode("utf-8")
    if len(encoded) <= MESSAGE_BYTE_CAP:
        return message
    return encoded[:MESSAGE_BYTE_CAP].decode("utf-8", errors="ignore")


def _extract_commit_context(
    payload: dict[str, object], cwd: str | None,
) -> tuple[str, str, str] | None:
    """Pull (branch, commit_hash, full_message) for the commit just made.

    Returns None when the prefix line cannot be parsed (unusual
    commit output) or when git log refuses to read the hash.
    """
    tool_response = payload.get("tool_response")
    stdout = ""
    if isinstance(tool_response, dict):
        s = cast(dict[str, object], tool_response).get("stdout")
        if isinstance(s, str):
            stdout = s
    if not stdout:
        return None
    parsed = _branch_and_hash_from_stdout(stdout)
    if parsed is None:
        return None
    branch, short_hash = parsed
    body = _read_full_commit_message(short_hash, cwd)
    if body is None:
        return None
    return branch, short_hash, body


def _do_ingest(payload: dict[str, object]) -> None:
    """Core hook body. Returns silently on any non-budget failure.

    Lazy imports keep the cold-start path light: the hook does not
    pay for `aelfrice.store` / `aelfrice.triple_extractor` import
    cost on commits that aren't `git commit` Bash calls.
    """
    if not _is_successful_git_commit(payload):
        return
    cwd_obj = payload.get("cwd")
    cwd = cwd_obj if isinstance(cwd_obj, str) else None
    extracted = _extract_commit_context(payload, cwd)
    if extracted is None:
        return
    branch, commit_hash, body = extracted
    _ingest_message(
        body,
        session_id=_derive_session_id(branch, commit_hash),
        project_context=cwd or os.getcwd(),
    )


def _ingest_message(message: str, *, session_id: str, project_context: str) -> None:
    """Extract triples from one commit message and record them under
    `session_id`. Shared by the PostToolUse path and the git hook."""
    if not message.strip():
        return
    body = _truncate_for_extraction(message)

    # Lazy imports: cold-start cost is paid only when we actually ingest.
    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415
    from aelfrice.triple_extractor import (  # noqa: PLC0415
        extract_triples, ingest_triples,
    )

    # #1376: commit bodies are the register where single-token relation
    # verbs collide with plural nouns ("Two tests that …"), and this is a
    # write path — a fragment minted here is an irreversible belief. The
    # read path in `context_rebuilder` deliberately does not pass this.
    triples = extract_triples(body, constrain_collision_verbs=True)
    if not triples:
        return  # no relations => nothing to record

    p = db_path()
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(p))
    try:
        # Persist a session row tagged with the git context. Idempotent
        # on re-fire because ingest_triples skips duplicate edges and
        # complete_session updates completed_at without erroring on a
        # known id.
        try:
            existing = store.get_session(session_id)
        except Exception:  # pyright: ignore[reportBroadException]
            existing = None
        if existing is None:
            store._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "INSERT OR IGNORE INTO sessions "
                "(id, started_at, completed_at, model, project_context) "
                "VALUES (?, ?, NULL, ?, ?)",
                (
                    session_id,
                    _iso_now(),
                    "commit-ingest",
                    project_context,
                ),
            )
            store._conn.commit()  # pyright: ignore[reportPrivateUsage]
        ingest_triples(store, triples, session_id=session_id)
        store.complete_session(session_id)
    finally:
        store.close()


# --- git post-commit hook (#1698) ------------------------------------------
#
# The PostToolUse path above guesses which commit a Bash call made from the
# command text and stdout; in this project's sessions it saw about 2 of 399
# commits (#1683). Git runs `post-commit` itself, inside the repository, after
# every commit it makes, so the commit (`HEAD`), the repository, and the store
# (resolved from the working directory) are all exact. `aelf setup` installs a
# `post-commit` block that, synchronously, skips a rebase replay and resolves
# `HEAD`, then runs `aelf-commit-ingest --git-hook <hash>` in the background,
# so a commit never waits on the ingest. Commits that `git rebase` replays
# were ingested when first made (operator ruling, 2026-10-05). The rebase
# check also skips a commit made by hand while a rebase is paused.

GIT_HOOK_FLAG: Final[str] = "--git-hook"

GIT_HOOK_LOG_NAME: Final[str] = "commit-ingest.log"
"""Errors from the detached git-hook run go here, next to the store, because
the hook's own stderr is discarded."""


def _git_out(args: list[str], cwd: str | None) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace",
            timeout=GIT_LOG_TIMEOUT_S, cwd=cwd,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _read_commit(rev: str, cwd: str | None) -> tuple[str, str, str] | None:
    """`(first_parent, author_date, full_message)` for `rev`.
    `first_parent` is empty for a root commit."""
    out = _git_out(["log", "-1", "--format=%P%x00%aI%x00%B", rev, "--"], cwd)
    if out is None or out.count("\x00") < 2:
        return None
    parents, author_date, message = out.split("\x00", 2)
    first = parents.split()
    return (first[0] if first else ""), author_date.strip(), message.rstrip("\n")


def _commit_session_id(first_parent: str, author_date: str) -> str:
    """sha256('commit:' + first parent + NUL + author date)[:16] (#1698).

    Keyed on what `git commit --amend` keeps: the parent and, unless
    `--reset-author` or `--date` is given, the author date. Every version
    of an amended commit shares one session, and the store never lets a
    commit session corroborate a belief it created itself
    (`MemoryStore.insert_or_corroborate`), so an amend adds only its new
    phrases. Commits on one branch have different parents. Two commits
    on one parent in the same second share a session."""
    raw = f"commit:{first_parent}\x00{author_date}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _is_revert(message: str) -> bool:
    """A `git revert` message quotes the subject it undoes, so ingesting
    it would corroborate the claim being reverted."""
    return message.startswith('Revert "') and "This reverts commit " in message


def git_hook_ingest(rev: str, cwd: str | None = None) -> None:
    """Ingest commit `rev` in `cwd` (the repository root, when git runs
    the hook).

    The hook block resolves `rev` from `HEAD` and checks for a rebase
    before it starts this process in the background. Reading `HEAD` here
    instead lost commits made in quick succession, and a rebase was over
    before a check here could see it (#1698 review).
    """
    commit = _read_commit(rev, cwd)
    if commit is None:
        return
    first_parent, author_date, message = commit
    if _is_revert(message):
        return
    _ingest_message(
        message,
        session_id=_commit_session_id(first_parent, author_date),
        project_context=cwd or os.getcwd(),
    )


def _log_git_hook_failure() -> None:
    """Append the current traceback to the log next to the store."""
    try:
        from aelfrice.db_paths import db_path  # noqa: PLC0415

        p = db_path()
        if str(p) == ":memory:":
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.parent / GIT_HOOK_LOG_NAME, "a", encoding="utf-8") as fh:
            fh.write(f"--- {_iso_now()}\n")
            traceback.print_exc(file=fh)
    except Exception:  # pyright: ignore[reportBroadException]
        pass


def _iso_now() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415
    return datetime.now(timezone.utc).isoformat()


def main(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract).

    With `--git-hook`, runs as the git `post-commit` hook (#1698) and
    ingests the named commit; otherwise reads a PostToolUse payload.
    """
    args = sys.argv[1:] if argv is None else argv
    if GIT_HOOK_FLAG in args:
        i = args.index(GIT_HOOK_FLAG)
        rev = args[i + 1] if i + 1 < len(args) else "HEAD"
        try:
            git_hook_ingest(rev)
        except Exception:  # non-blocking: log, never raise
            _log_git_hook_failure()
        return 0
    sin = stdin if stdin is not None else sys.stdin
    serr = stderr if stderr is not None else sys.stderr
    ensure_utf8_streams((serr,))
    try:
        payload = _read_payload(sin, serr)
        if payload is None:
            return 0
        _do_ingest(payload)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
