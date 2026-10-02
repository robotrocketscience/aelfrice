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
    * tool_input.command mentions both `git` and `commit`
    * tool_response is not flagged as an error / interrupted
- All failure modes return exit 0 silently. The hook may NEVER
  cause a `git commit` to feel broken.

**Which commits a call made (#1698).** The command text is only a
cheap prefilter. The commits come from `HEAD`'s reflog: every entry
whose subject starts with `commit` (`commit:`, `commit (amend):`,
`commit (initial):`, `commit (merge):`, `commit (cherry-pick):`)
written in the last `REFLOG_WINDOW_S` seconds. The reflog is read in
the directory of a leading `cd <path> &&`, or else in the payload
`cwd`. Before #1698 the hook matched only a command that started with
`git commit` and read the hash from the `[branch hash]` line, which
`git commit -q` doesn't print; it ingested about 2 of 399 commits in
this project's sessions (#1683). Reading the reflog means a chained
command, a quiet commit, and a call that makes several commits all
work. A commit older than the window is never read. A call inside the
window that made no commit re-reads the recent ones, and that re-read
is skipped once every phrase is already logged for the commit's
session. Entries are timed by committer date, so a commit made with a
past `GIT_COMMITTER_DATE` is missed.

Latency budget per docs/design/commit_ingest_hook.md:
    median <= 30 ms, p95 <= 100 ms
A call that passes the prefilter pays one `git log -g`, about 57 ms on
a development machine, so it exceeds the median budget; about 2.7% of
Bash calls in this project's transcripts pass it (#1698).

Tactics:
- Lazy imports of triple_extractor and store (cold-start dominates).
- A Bash call that doesn't mention `git` and `commit` costs two
  linear regex searches.
- Cap the message body at 4 KB before extraction.
- One `git log -g` for the reflog, plus one `git log -1` per commit
  found in the window.

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
import time
import traceback
from typing import IO, TYPE_CHECKING, Final, cast

from aelfrice.stream_encoding import ensure_utf8_streams, read_payload_text

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore
    from aelfrice.triple_extractor import Triple

MESSAGE_BYTE_CAP: Final[int] = 4096
"""Truncate commit messages above this many bytes before extraction.
Long commit messages are rare; the cap bounds the worst case."""

GIT_LOG_TIMEOUT_S: Final[float] = 2.0
"""Per-call timeout for `git log` so a hung git binary cannot block
the hook past the latency budget."""

REFLOG_WINDOW_S: Final[int] = 120
"""A reflog `commit…` entry this recent counts as made by the call that
just finished (#1698). Long enough for a commit chained after a slow
step; an older commit is never read. A commit made in another terminal
inside the window is ingested too; it's a real commit in the repository
either way."""

REFLOG_DEPTH: Final[int] = 20
"""How many reflog entries to read. More commits than this in one call
are rare; the oldest beyond it are skipped."""

# #1698: the cheapest test that a Bash call might have committed. The
# reflog decides whether it did. Two separate searches, so the cost stays
# linear in the command's length.
_GIT_WORD: Final[re.Pattern[str]] = re.compile(r"\bgit\b")
_COMMIT_WORD: Final[re.Pattern[str]] = re.compile(r"\bcommit\b")

# A leading `cd <path>` step: `cd ../wt && git commit …` commits in that
# directory, not in the payload's cwd. About a third of commit calls in
# this project's transcripts start this way (#1698).
_LEADING_CD: Final[re.Pattern[str]] = re.compile(
    r"\A\s*cd\s+(\"[^\"]+\"|'[^']+'|[^\s;&|]+)\s*(?:&&|;)"
)

# `git log -g --date=unix` renders the reflog selector as `HEAD@{<unix>}`.
_REFLOG_TS_RE: Final[re.Pattern[str]] = re.compile(r"@\{(\d+)\}$")


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


def _mentions_git_commit(cmd: str) -> bool:
    return bool(_GIT_WORD.search(cmd)) and bool(_COMMIT_WORD.search(cmd))


def _repo_dir(cmd: str, cwd: str | None) -> str | None:
    """The directory the commit ran in: a leading `cd <path>` target,
    resolved against `cwd`, or `cwd` itself."""
    m = _LEADING_CD.match(cmd)
    if m is None:
        return cwd
    target = os.path.expanduser(m.group(1).strip("\"'"))
    if cwd is not None and not os.path.isabs(target):
        target = os.path.join(cwd, target)
    return target


def _may_have_committed(payload: dict[str, object]) -> bool:
    if payload.get("tool_name") != "Bash":
        return False
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    cmd = cast(dict[str, object], tool_input).get("command")
    if not isinstance(cmd, str) or not _mentions_git_commit(cmd):
        return False
    tool_response = payload.get("tool_response")
    if isinstance(tool_response, dict):
        resp = cast(dict[str, object], tool_response)
        if resp.get("isError") is True or resp.get("interrupted") is True:
            return False
    return True


def _git(args: list[str], cwd: str | None) -> str | None:
    """Run one read-only git command; None on any failure."""
    try:
        r = subprocess.run(
            ["git", *args],
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
    return r.stdout


def _now() -> float:
    """Clock seam for tests."""
    return time.time()


def _recent_commits(cwd: str | None) -> list[str]:
    """Hashes of the `commit…` reflog entries of the last
    `REFLOG_WINDOW_S` seconds, oldest first, each once."""
    out = _git(
        ["log", "-g", f"-n{REFLOG_DEPTH}", "--date=unix",
         "--format=%H%x00%gd%x00%gs", "HEAD"],
        cwd,
    )
    if out is None:
        return []
    cutoff = _now() - REFLOG_WINDOW_S
    found: list[str] = []
    for line in out.splitlines():
        parts = line.split("\x00")
        if len(parts) != 3:
            continue
        commit_hash, selector, subject = parts
        m = _REFLOG_TS_RE.search(selector)
        if m is None or int(m.group(1)) < cutoff:
            continue
        if not subject.startswith("commit"):
            continue
        if commit_hash not in found:
            found.append(commit_hash)
    found.reverse()
    return found


def _read_commit(commit_hash: str, cwd: str | None) -> tuple[str, str, str] | None:
    """`(first_parent, author_date, full_message)`; `first_parent` is
    empty for a root commit. None on any failure."""
    out = _git(["log", "-1", "--format=%P%x00%aI%x00%B", commit_hash], cwd)
    if out is None or out.count("\x00") < 2:
        return None
    parents, author_date, message = out.split("\x00", 2)
    first_parent = parents.split()[0] if parents.split() else ""
    return first_parent, author_date.strip(), message.rstrip("\n")


def _derive_session_id(first_parent: str, author_date: str) -> str:
    """Stable id from sha256('commit:' + first parent + NUL + author date)[:16].

    #1698: keyed on what an amend keeps. `git commit --amend` keeps the
    parent and, unless `--reset-author` or `--date` is given, the author
    date, so every version of an amended commit shares one session, and
    the store never lets a commit session corroborate a belief it
    created itself (`MemoryStore.insert_or_corroborate`). Branches cut
    from the same tip share a parent, so their commits stay distinct only
    if their author dates differ: two commits on one parent in the same
    second share a session. A rebase or cherry-pick changes the parent,
    so the rewritten commit gets a new session and corroborates its
    earlier version (operator ruling, 2026-10-02). Cross-machine stable."""
    raw = f"commit:{first_parent}\x00{author_date}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _truncate_for_extraction(message: str) -> str:
    encoded = message.encode("utf-8")
    if len(encoded) <= MESSAGE_BYTE_CAP:
        return message
    return encoded[:MESSAGE_BYTE_CAP].decode("utf-8", errors="ignore")


def _do_ingest(payload: dict[str, object]) -> None:
    """Core hook body. Returns silently on any non-budget failure.

    Lazy imports keep the cold-start path light: the hook does not
    pay for `aelfrice.store` / `aelfrice.triple_extractor` import
    cost unless a recent commit has triples to record.
    """
    if not _may_have_committed(payload):
        return
    cwd_obj = payload.get("cwd")
    cwd = cwd_obj if isinstance(cwd_obj, str) else None
    tool_input = cast(dict[str, object], payload["tool_input"])
    repo = _repo_dir(cast(str, tool_input["command"]), cwd)
    for commit_hash in _recent_commits(repo):
        _ingest_commit(commit_hash, repo)


def _already_logged(store: "MemoryStore", session_id: str, triples: list["Triple"]) -> bool:
    """True when every subject and object phrase of `triples` already has
    an `ingest_log` row under `session_id`: a re-read of a commit this
    session ingested (#1698). An edited amend adds a phrase, so it is not
    skipped."""
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_text FROM ingest_log WHERE session_id = ?", (session_id,),
    ).fetchall()
    logged = {str(r[0]) for r in rows}
    phrases = {p for t in triples for p in (t.subject, t.object) if p}
    return phrases <= logged


def _ingest_commit(commit_hash: str, cwd: str | None) -> None:
    commit = _read_commit(commit_hash, cwd)
    if commit is None:
        return
    first_parent, author_date, message = commit
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

    session_id = _derive_session_id(first_parent, author_date)
    store = MemoryStore(str(p))
    try:
        # Persist a session row tagged with the git context. A re-fire on
        # the same commit, an amend, or a retry after a crash reuses it:
        # ingest_triples skips duplicate edges, and the store records no
        # corroboration for a belief this session created (#1698).
        try:
            existing = store.get_session(session_id)
        except Exception:  # pyright: ignore[reportBroadException]
            existing = None
        if existing is not None and _already_logged(store, session_id, triples):
            return
        if existing is None:
            store._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "INSERT OR IGNORE INTO sessions "
                "(id, started_at, completed_at, model, project_context) "
                "VALUES (?, ?, NULL, ?, ?)",
                (
                    session_id,
                    _iso_now(),
                    "commit-ingest",
                    cwd or os.getcwd(),
                ),
            )
            store._conn.commit()  # pyright: ignore[reportPrivateUsage]
        ingest_triples(store, triples, session_id=session_id)
        store.complete_session(session_id)
    finally:
        store.close()


def _iso_now() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415
    return datetime.now(timezone.utc).isoformat()


def main(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract)."""
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
