"""PreToolUse hook that runs `aelf search` against the per-project belief
store before a `Grep` or `Glob` tool call fires, and emits the results as
`additionalContext` so the agent sees them and can decide to skip / refine
the tool call or use the tool to fill in gaps.

Hook contract (Claude Code PreToolUse):
- payload includes `tool_name`, `tool_input`, `cwd`, plus the standard
  event fields. We act only when:
    * tool_name in {"Grep", "Glob"}
    * tool_input.pattern is a string with at least one extractable token
- All failure modes return exit 0 silently. The hook may NEVER cause a
  `Grep` or `Glob` to feel broken.

Latency budget per docs/search_tool_hook.md:
    median <= 50 ms, p95 <= 200 ms

Tactics:
- Lazy imports of retrieval / store (cold-start dominates).
- Skip on empty token sets up front.
- Cap query at 5 tokens; require 3+ chars per token.
- Lower retrieval budget (token_budget=600, l1_limit=10) than the
  user-facing default — this is auxiliary context.

Local-only: brain-graph reads never cross the git boundary or any
network boundary.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Final, cast

QUERY_TOKEN_LIMIT: Final[int] = 5
"""Maximum number of tokens joined into the FTS5 OR query.
Bounds query complexity; longer patterns rarely add useful signal."""

MIN_TOKEN_LEN: Final[int] = 3
"""Minimum token length to be considered a query term.
Filters single-letter regex anchors (\\b, \\d) and 2-char noise."""

INJECTED_TOKEN_BUDGET: Final[int] = 600
"""Token budget for retrieve() — half the user-facing default. Auxiliary
context should not crowd out the user's primary turn budget."""

INJECTED_L1_LIMIT: Final[int] = 10
"""L1 result cap. Lower than retrieval default to bound injection size."""

PER_LINE_CHAR_CAP: Final[int] = 200
"""Truncate each emitted belief line to this many chars. Keeps the
injection bounded even when individual beliefs have long content."""

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    rf"[A-Za-z][A-Za-z0-9_-]{{{MIN_TOKEN_LEN - 1},}}"
)


# --- v1.5.0 Bash matcher (#155) ------------------------------------------

# Halved budget vs. the v1.2.x Grep|Glob path. Bash extraction is one
# parse hop further from the agent's intent so the auxiliary-context
# allowance shrinks correspondingly. Spec § Token budget.
BASH_INJECTED_TOKEN_BUDGET: Final[int] = 300
BASH_INJECTED_L1_LIMIT: Final[int] = 5

# Per-turn fire cap. State is keyed by session_id and reset when a
# new session_id appears. Prevents pipeline storms (e.g. a `for` loop
# running `rg` ten times). Spec § Per-turn fire cap.
BASH_FIRE_CAP_PER_TURN: Final[int] = 3

# Truncated `cmd` attribute on the emitted block. Keeps the
# additionalContext payload bounded.
BASH_CMD_ATTR_CHAR_CAP: Final[int] = 80

# Per-command flag table. Each entry maps a flag to whether the flag
# consumes the next token as its value. The parser walks token-by-
# token and skips both the flag and (if applicable) its value.
#
# Long-form flags with `=value` are recognised by the parser
# regardless of this table — only space-separated `--flag value`
# pairs need the takes-arg signal.
_GREP_FLAGS_WITH_ARG: Final[frozenset[str]] = frozenset({
    "-A", "-B", "-C", "-e", "-f", "-m",
    "--after-context", "--before-context", "--context",
    "--regexp", "--file", "--max-count",
    "--include", "--exclude", "--exclude-dir",
    "-d", "--directories",
})
_RG_FLAGS_WITH_ARG: Final[frozenset[str]] = frozenset(_GREP_FLAGS_WITH_ARG | {
    "-t", "-T", "--type", "--type-not",
    "-g", "--glob", "--iglob",
    "--max-depth", "--maxdepth",
})
_FIND_FLAGS_WITH_ARG: Final[frozenset[str]] = frozenset({
    "-name", "-iname", "-path", "-ipath",
    "-type", "-mindepth", "-maxdepth",
    "-newer", "-mtime", "-atime", "-ctime",
    "-size", "-user", "-group", "-perm",
    "-exec", "-execdir", "-ok", "-okdir",
})
_FD_FLAGS_WITH_ARG: Final[frozenset[str]] = frozenset({
    "-t", "-T", "--type", "--type-not",
    "-e", "--extension",
    "-d", "--max-depth", "--min-depth",
    "-x", "--exec", "-X", "--exec-batch",
    "--changed-within", "--changed-before",
    "--owner",
})

# Tokens that abort the Bash parser immediately. Pipeline / shell-
# control / command-substitution / redirection. Conservative: any
# of these implies the actual search query is something the parser
# can't reliably lift, so we silent-skip rather than guess.
_BASH_ABORT_TOKENS: Final[frozenset[str]] = frozenset({
    "|", "||", "&&", "&", ";", ";;",
    ">", ">>", "<", "<<", "<<<", "<<-", "2>", "2>>", "&>", "&>>",
    "$(", "`", ")", "(",
    "for", "while", "until", "if", "case", "do", "done", "fi", "esac",
    "then", "else", "elif",
})

# Allowlisted commands grouped by their micro-parser. The map
# value is the flag-value table the parser consults.
_BASH_ALLOWLIST: Final[dict[str, frozenset[str]]] = {
    "grep":     _GREP_FLAGS_WITH_ARG,
    "egrep":    _GREP_FLAGS_WITH_ARG,
    "fgrep":    _GREP_FLAGS_WITH_ARG,
    "rg":       _RG_FLAGS_WITH_ARG,
    "ripgrep":  _RG_FLAGS_WITH_ARG,
    "ack":      _GREP_FLAGS_WITH_ARG,
    "find":     _FIND_FLAGS_WITH_ARG,
    "fd":       _FD_FLAGS_WITH_ARG,
    "fdfind":   _FD_FLAGS_WITH_ARG,
}

# Per-process per-turn fire counter. Keyed by session_id; on a new
# session_id, the counter resets. Not thread-safe; the hook process
# is short-lived and single-threaded by Claude Code's contract.
_BASH_FIRE_STATE: dict[str, int] = {}


def _is_abort_token(token: str) -> bool:
    """Return True if `token` matches an abort-list entry exactly OR
    contains one as a prefix / substring (covers naive whitespace
    splits of $(cmd) / $cmd / backtick-quoted / inline backticks).
    """
    if token in _BASH_ABORT_TOKENS:
        return True
    # Command substitution / process substitution / arithmetic
    # expansion all start with `$(` or backtick. Backslash is not
    # in the regular tokens. Substring `$(` covers both naked and
    # quoted forms.
    if "$(" in token:
        return True
    if "`" in token:
        return True
    if "<(" in token or ">(" in token:
        return True
    return False


def _strip_command_prefix(tokens: list[str]) -> list[str]:
    """Skip leading no-op prefix tokens until an allowlisted command
    is reached.

    Handles:
      - env-assignment prefixes:  `RUST_LOG=trace rg ...`
      - `nohup` / `time` / `command` wrappers:  `nohup rg ...`
      - leading `cd foo &&` is rejected by the abort-token list,
        not by this function.

    Returns the suffix starting at the allowlisted command, or
    `[]` if no allowlisted command is found.
    """
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _BASH_ABORT_TOKENS:
            return []
        # KEY=VALUE env assignments at the start of a command are
        # standard shell syntax. Skip them.
        if "=" in t and t.split("=", 1)[0].isidentifier():
            i += 1
            continue
        if t in ("nohup", "time", "command", "exec"):
            i += 1
            continue
        # Strip path prefix on a real command (`/usr/bin/grep` -> `grep`).
        basename = t.rsplit("/", 1)[-1]
        if basename in _BASH_ALLOWLIST:
            return [basename, *tokens[i + 1:]]
        return []
    return []


def _parse_grep_like(tokens: list[str], flags_with_arg: frozenset[str]) -> str | None:
    """Walk grep / rg / ack tokens; return the first positional as the query.

    Skips flag tokens (leading `-`) and their values when the table
    indicates the flag takes an argument. `--flag=value` form is
    treated as flag-only (the value is embedded). Returns `None` if
    no positional remains or any token is in the abort set.
    """
    i = 1  # skip the command name itself
    while i < len(tokens):
        t = tokens[i]
        if t in _BASH_ABORT_TOKENS:
            return None
        if t.startswith("-"):
            # Long-form `--flag=value`: consume only the flag.
            if "=" in t:
                i += 1
                continue
            if t in flags_with_arg:
                # Skip flag + its value. If we run off the end, that's
                # a malformed command; abort.
                if i + 1 >= len(tokens):
                    return None
                if tokens[i + 1] in _BASH_ABORT_TOKENS:
                    return None
                i += 2
                continue
            # Bare flag with no argument.
            i += 1
            continue
        # First non-flag positional is the query.
        return t
    return None


def _parse_find(tokens: list[str]) -> str | None:
    """Walk `find` tokens; return the `-name` / `-iname` value.

    Find's contract is stricter than grep's: only `-name` or
    `-iname` produces a query. `find . -type f` and similar
    name-less invocations silent-skip.
    """
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in _BASH_ABORT_TOKENS:
            return None
        if t in ("-name", "-iname"):
            if i + 1 >= len(tokens):
                return None
            value = tokens[i + 1]
            if value in _BASH_ABORT_TOKENS:
                return None
            return value
        if t.startswith("-") and t in _FIND_FLAGS_WITH_ARG:
            # Consume the flag's value.
            i += 2
            continue
        i += 1
    return None


def _parse_fd(tokens: list[str]) -> str | None:
    """Walk `fd` / `fdfind` tokens; return first non-flag positional."""
    return _parse_grep_like(tokens, _FD_FLAGS_WITH_ARG)


def _parse_bash_command(command: str) -> tuple[str, str] | None:
    """Lift `(query, command_name)` from a raw Bash `tool_input.command`.

    Returns `None` when the command is not allowlisted, the parser
    cannot find a query, or the command contains pipeline /
    command-substitution / shell-control tokens.

    Tokenisation is a naive whitespace split. We deliberately do
    NOT shell-evaluate the command. Quoted arguments with embedded
    whitespace will tokenise incorrectly here; the parser's
    failure mode is to return `None` rather than guess.
    """
    if not command or not command.strip():
        return None
    tokens = command.split()
    # Reject the entire command if any abort token (pipeline, shell
    # control, command substitution, redirection) appears anywhere.
    # The narrow parser cannot reliably lift a query out of a multi-
    # stage shell expression; silent-skip beats wrong-query.
    if any(_is_abort_token(t) for t in tokens):
        return None
    suffix = _strip_command_prefix(tokens)
    if not suffix:
        return None
    cmd = suffix[0]
    flags_table = _BASH_ALLOWLIST.get(cmd)
    if flags_table is None:
        return None
    if cmd in ("find",):
        query = _parse_find(suffix)
    elif cmd in ("fd", "fdfind"):
        query = _parse_fd(suffix)
    else:
        query = _parse_grep_like(suffix, flags_table)
    if query is None:
        return None
    # Strip wrapping quotes.
    cleaned = query.strip("'\"")
    if not cleaned:
        return None
    return cleaned, cmd


def _extract_bash_query(
    payload: dict[str, object],
) -> tuple[str, str, str] | None:
    """Return `(fts5_query, command_name, raw_cmd_truncated)` for a
    Bash payload, or `None` to silent-skip.

    Tokenises the lifted query string the same way the Grep|Glob
    path does (3-char minimum, FTS5 OR-join, 5-token cap), so the
    Bash matcher's downstream `retrieve()` call sees the same
    shape of query the v1.2.x matcher emits.
    """
    if payload.get("tool_name") != "Bash":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw_cmd = cast(dict[str, object], tool_input).get("command")
    if not isinstance(raw_cmd, str) or not raw_cmd.strip():
        return None
    parsed = _parse_bash_command(raw_cmd)
    if parsed is None:
        return None
    query_str, cmd_name = parsed
    tokens = _TOKEN_RE.findall(query_str)
    if not tokens:
        return None
    fts5_query = " OR ".join(tokens[:QUERY_TOKEN_LIMIT])
    truncated = raw_cmd.strip().replace("\n", " ")
    if len(truncated) > BASH_CMD_ATTR_CHAR_CAP:
        truncated = truncated[: BASH_CMD_ATTR_CHAR_CAP - 3] + "..."
    return fts5_query, cmd_name, truncated


def _bash_fire_cap_reached(session_id: str | None) -> bool:
    """Return True if `session_id` has hit BASH_FIRE_CAP_PER_TURN.

    A session_id of `None` (missing from payload) collapses to a
    sentinel key and shares the cap with other untagged calls in
    the same process. Conservative: caps still apply.
    """
    key = session_id or "<no-session>"
    count = _BASH_FIRE_STATE.get(key, 0)
    return count >= BASH_FIRE_CAP_PER_TURN


def _record_bash_fire(session_id: str | None) -> None:
    key = session_id or "<no-session>"
    _BASH_FIRE_STATE[key] = _BASH_FIRE_STATE.get(key, 0) + 1


def _reset_bash_fire_state() -> None:
    """Test-only helper. Not part of the public API."""
    _BASH_FIRE_STATE.clear()


# --- Telemetry (v1.5.0 #155 AC3 prerequisite) ----------------------------

TELEMETRY_RING_CAP: Final[int] = 1000
"""Maximum entries retained in the JSONL ring buffer."""

TELEMETRY_SUBPATH: Final[str] = "aelfrice/telemetry/search_tool_hook.jsonl"
"""Path fragment appended to the git-common-dir to get the telemetry file.
Lives next to the DB and inherits its gitignore boundary."""


def _telemetry_path_for_db(db_path: Path) -> Path:
    """Derive the telemetry file path from the DB path.

    The DB lives at `<git-common-dir>/aelfrice/memory.db` (or
    ~/.aelfrice/memory.db for non-git dirs). The telemetry file lives at
    `<git-common-dir>/aelfrice/telemetry/search_tool_hook.jsonl`,
    adjacent to the DB.
    """
    return db_path.parent / "telemetry" / "search_tool_hook.jsonl"


def _append_telemetry(
    telemetry_path: Path,
    *,
    session_id: str | None,
    command: str,
    query: str,
    latency_ms: float,
    injected_l1: int,
    injected_l0: int,
    stderr: IO[str] | None = None,
) -> None:
    """Append one telemetry record to the JSONL ring buffer. Fail-soft.

    Uses read-all → trim → rewrite-atomically (tempfile + os.replace).
    If the write fails for any reason (read-only, disk-full, missing
    parent), traces one line to stderr and continues.
    """
    record: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": session_id or "",
        "command": command,
        "query": query,
        "latency_ms": round(latency_ms, 3),
        "injected_l1": injected_l1,
        "injected_l0": injected_l0,
    }
    try:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        if telemetry_path.exists():
            lines = [
                ln for ln in telemetry_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        else:
            lines = []
        lines.append(json.dumps(record))
        # Evict oldest to hold the ring cap.
        if len(lines) > TELEMETRY_RING_CAP:
            lines = lines[-TELEMETRY_RING_CAP:]
        payload = "\n".join(lines) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=telemetry_path.name + ".", suffix=".tmp",
            dir=str(telemetry_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, telemetry_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
    except Exception as exc:  # fail-soft contract
        serr = stderr if stderr is not None else sys.stderr
        print(
            f"aelfrice: telemetry write failed (non-fatal): {exc}",
            file=serr,
        )


def read_telemetry(path: Path) -> list[dict[str, object]]:
    """Read the JSONL ring buffer at `path`. Returns [] when missing.

    Raises `ValueError` if the file exists but contains a line that
    is not valid JSON (real corruption). Lines that are valid JSON
    but not objects are silently skipped (defensive).
    """
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"telemetry file {path} line {i + 1} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            continue
        records.append(cast(dict[str, object], parsed))
    return records


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


def _is_search_tool_call(payload: dict[str, object]) -> bool:
    tool_name = payload.get("tool_name")
    return tool_name in ("Grep", "Glob")


def _extract_query(payload: dict[str, object]) -> str | None:
    """Lift the search query out of tool_input.pattern.

    Both Grep and Glob use the field name `pattern`. Strips regex/glob
    metacharacters by extracting only alphanumeric word tokens, then
    joins the first QUERY_TOKEN_LIMIT with FTS5 ` OR ` to form a query.
    Returns None when no usable tokens are present (pure-glob patterns,
    single-character regexes, etc.) — caller treats None as "skip".
    """
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw = cast(dict[str, object], tool_input).get("pattern")
    if not isinstance(raw, str) or not raw.strip():
        return None
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return None
    return " OR ".join(tokens[:QUERY_TOKEN_LIMIT])


def _format_results(
    query: str,
    beliefs: list[object],
    locked_ids: set[str],
    *,
    bash_source: tuple[str, str] | None = None,
) -> str:
    """Render retrieve() output as a flat text block for additionalContext.

    Format: "[L0] {id-prefix}: {content}" for locked, "[L1] ..." for
    BM25-ranked. One belief per line, truncated to PER_LINE_CHAR_CAP.

    When `bash_source` is set (v1.5.0 #155 Bash matcher), the
    emitted block carries `source="bash:<cmd>"` and `cmd="<truncated>"`
    attributes so the agent can tell which Bash invocation triggered
    the injection. Default `None` preserves the v1.2.x output shape.
    """
    if bash_source is not None:
        cmd_name, raw_cmd = bash_source
        attrs = (
            f'query="{query}" source="bash:{cmd_name}" '
            f'cmd="{raw_cmd}"'
        )
    else:
        attrs = f'query="{query}"'
    lines: list[str] = []
    for b in beliefs:
        bid = getattr(b, "id", "") or ""
        content = getattr(b, "content", "") or ""
        if not bid or not content:
            continue
        tier = "L0" if bid in locked_ids else "L1"
        prefix = bid[:16]
        line = f"[{tier}] {prefix}: {content}".replace("\n", " ")
        if len(line) > PER_LINE_CHAR_CAP:
            line = line[: PER_LINE_CHAR_CAP - 3] + "..."
        lines.append(line)
    if not lines:
        return (
            f'<aelfrice-search {attrs}>'
            f"no matching beliefs in store; the tool result will fill the gap"
            f"</aelfrice-search>"
        )
    body = "\n".join(lines)
    return (
        f'<aelfrice-search {attrs}>aelf search ran on this query before '
        f"the tool fires; results:\n{body}\n"
        f"If this answers the question, you may skip the tool call. Otherwise "
        f"use the tool to fill gaps.</aelfrice-search>"
    )


def _emit(stdout: IO[str], context: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
    }
    stdout.write(json.dumps(payload))


def _do_search(
    payload: dict[str, object],
    stdout: IO[str],
    stderr: IO[str] | None = None,
) -> None:
    """Core hook body. Returns silently on any non-budget failure.

    Lazy imports keep the cold-start path light: the hook does not pay
    for `aelfrice.store` / `aelfrice.retrieval` import cost on tool
    calls that aren't `Grep` / `Glob`, or on patterns that have no
    extractable tokens.
    """
    bash_source: tuple[str, str] | None = None
    session_id: str | None = None
    t0: float | None = None
    if _is_search_tool_call(payload):
        query = _extract_query(payload)
        if query is None:
            return
    elif payload.get("tool_name") == "Bash":
        # v1.5.0 #155 Bash matcher path. Per-turn fire cap applies
        # only to this lane; Grep|Glob fires once per direct tool
        # call and is not capped.
        t0 = time.perf_counter()
        session_obj = payload.get("session_id")
        session_id = session_obj if isinstance(session_obj, str) else None
        if _bash_fire_cap_reached(session_id):
            return
        bash_extracted = _extract_bash_query(payload)
        if bash_extracted is None:
            return
        query, cmd_name, raw_cmd = bash_extracted
        bash_source = (cmd_name, raw_cmd)
        _record_bash_fire(session_id)
    else:
        return

    cwd_obj = payload.get("cwd")
    cwd = cwd_obj if isinstance(cwd_obj, str) else None

    # Lazy imports: cold-start cost is paid only when we actually search.
    from aelfrice.cli import db_path  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    from aelfrice.retrieval import retrieve  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    p = db_path(cwd=cwd) if _db_path_accepts_cwd(db_path) else db_path()
    if str(p) != ":memory:" and not p.exists():
        # Empty / not-yet-onboarded store — explicit sentinel so the agent
        # learns the check ran.
        _emit(stdout, _format_results(
            query, [], set(), bash_source=bash_source,
        ))
        if bash_source is not None and t0 is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            tel_path = _telemetry_path_for_db(p)
            _append_telemetry(
                tel_path,
                session_id=session_id,
                command=bash_source[0],
                query=query,
                latency_ms=latency_ms,
                injected_l1=0,
                injected_l0=0,
                stderr=stderr,
            )
        return

    # Bash matcher gets a halved token budget + L1 limit (spec § Token budget).
    token_budget = (
        BASH_INJECTED_TOKEN_BUDGET if bash_source else INJECTED_TOKEN_BUDGET
    )
    l1_limit = (
        BASH_INJECTED_L1_LIMIT if bash_source else INJECTED_L1_LIMIT
    )
    store = MemoryStore(str(p))
    try:
        locked = store.list_locked_beliefs()
        locked_ids = {b.id for b in locked}
        beliefs = retrieve(
            store,
            query,
            token_budget=token_budget,
            l1_limit=l1_limit,
        )
    finally:
        store.close()

    _emit(stdout, _format_results(
        query, beliefs, locked_ids, bash_source=bash_source,
    ))

    # Write telemetry for the Bash branch only (AC3 prerequisite).
    if bash_source is not None and t0 is not None:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        n_l0 = sum(1 for b in beliefs if getattr(b, "id", "") in locked_ids)
        n_l1 = len(beliefs) - n_l0
        tel_path = _telemetry_path_for_db(p)
        _append_telemetry(
            tel_path,
            session_id=session_id,
            command=bash_source[0],
            query=query,
            latency_ms=latency_ms,
            injected_l1=n_l1,
            injected_l0=n_l0,
            stderr=stderr,
        )


def _db_path_accepts_cwd(db_path_fn: object) -> bool:
    """Best-effort detection: does aelfrice.cli.db_path() accept a cwd kw?

    v1.1.0 db_path() reads cwd from os.getcwd(); a later patch may add a
    cwd parameter for callers that need to scope to a specific worktree.
    The hook detects either signature without a hard dependency on the
    later API.
    """
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(db_path_fn)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return False
    return "cwd" in sig.parameters


def main(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract)."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        payload = _read_payload(sin)
        if payload is None:
            return 0
        _do_search(payload, sout, stderr=serr)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
