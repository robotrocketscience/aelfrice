"""PostToolUse hook that turns each successful `git commit` into an
ingest event: parse the message, run the triple extractor, persist
beliefs and edges under a session derived from git context.

Closes the v1.0 limitation that the belief graph only grows on
explicit `aelf onboard` / `aelf remember` calls. Each commit
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


def _read_payload(stdin: IO[str]) -> dict[str, object] | None:
    raw = stdin.read()
    if not raw.strip():
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
            timeout=GIT_LOG_TIMEOUT_S, cwd=cwd,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
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
    if not body.strip():
        return
    body = _truncate_for_extraction(body)

    # Lazy imports: cold-start cost is paid only when we actually ingest.
    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415
    from aelfrice.triple_extractor import (  # noqa: PLC0415
        extract_triples, ingest_triples,
    )

    triples = extract_triples(body)
    if not triples:
        return  # no relations => nothing to record

    p = db_path()
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)

    session_id = _derive_session_id(branch, commit_hash)
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
    try:
        payload = _read_payload(sin)
        if payload is None:
            return 0
        _do_ingest(payload)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
