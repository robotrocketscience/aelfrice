"""Codex host target for setup / doctor / uninstall (#1052).

Writes, validates, and removes the aelfrice hook set in Codex's
``$CODEX_HOME/hooks.json`` (``~/.codex/hooks.json`` when that variable
is unset — see ``resolve_codex_home``, #1427). Design constraints
established in #1054/#1052 triage:

- **hooks.json only.** The documented ``{"hooks": {"<Event>": [...]}}``
  shape is the stable public surface. The per-hook trust ledger
  (``[hooks.state]`` tables in ``config.toml``, keyed positionally with
  a ``trusted_hash`` over a canonical TOML serialization) is explicitly
  marked for replacement in the Codex source — we never write it.
  Setup instead prints approval guidance: the user runs ``/hooks``
  inside a Codex session to trust the new entries. Until approved,
  Codex silently skips the hooks. The ``hooks`` feature itself is
  stable and on by default (it was the under-development ``codex_hooks``
  flag on Codex 0.11x–0.12x); doctor surfaces an explicit disable.
- **Merge-aware and idempotent.** Entries whose command basename is one
  of ours are replaced wholesale on every setup run; everything else in
  the file is preserved byte-for-byte at the JSON level. An unparseable
  hooks.json is never overwritten without ``force`` — a real-world
  Codex ``hooks.json`` has been observed holding truncated JSON, and
  clobbering user content on a parse error is worse than refusing.
- **Portable hook subset.** Host-agnostic hooks are installed:
  retrieval injection (UserPromptSubmit), the transcript logger
  (UserPromptSubmit / Stop / PreCompact / PostCompact), session-start
  baseline injection (SessionStart, all sources — ``compact`` included,
  which is the rebuild-at-compaction channel per #1054), the stop
  lock-prompt, and the ``Bash``-matcher tool hooks (#1055): Codex
  canonicalizes hook tool names to the compatible surface — shell
  commands report ``tool_name == "Bash"`` — so the memory-first shell
  search, pre-issue duplicate guard, and commit-ingest hooks match
  unchanged. The search-tool hook on ``SEARCH_TOOL_MATCHER`` (Grep,
  Glob, WebSearch, WebFetch) is excluded: Grep and Glob do not exist
  on Codex, where greps arrive via Bash and are covered by the Bash
  matcher. The web tool names are excluded with them. The
  host-specific memory mirror is excluded too.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import tomllib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from aelfrice.hook_payload import HOOK_PAYLOAD_CHAR_LIMIT
from aelfrice.launcher import (
    LAUNCHER_SUFFIXES,
    command_launcher_key,
    command_program_keys,
    command_tokens,
    launcher_key,
    owned_keys,
    program_exists,
    program_token,
)
from aelfrice.setup import (
    SettingsScope,
    resolve_commit_ingest_command,
    resolve_hook_command,
    resolve_pre_issue_guard_command,
    resolve_search_tool_bash_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
)

CODEX_HOME_ENV: Final[str] = "CODEX_HOME"
CODEX_DEFAULT_DIRNAME: Final[str] = ".codex"
CODEX_HOOKS_FILENAME: Final[str] = "hooks.json"
CODEX_CONFIG_FILENAME: Final[str] = "config.toml"


class CodexHomeError(RuntimeError):
    """`$CODEX_HOME` is set to something that cannot be a Codex home.

    #1427. Raised instead of falling back to `~/.codex`: an explicitly
    configured home that we silently ignore is exactly the failure this
    issue is about — setup reports success against a directory the
    running Codex never reads.
    """


def resolve_codex_home() -> Path:
    """The directory Codex reads its configuration from (#1427).

    Codex resolves its home from `$CODEX_HOME` when that variable holds
    a non-empty value and falls back to `~/.codex` otherwise; aelfrice
    bound `Path.home() / ".codex"` into a module-level constant at
    import time, so `aelf setup --host codex` wrote hooks the running
    Codex never loaded, and `doctor`/`unsetup` inspected and stripped
    the wrong directory.

    Late-bound on purpose. A module constant is wrong twice over: it
    cannot see `$CODEX_HOME` at all, and it freezes `$HOME` at import,
    so a test or a wrapper that changes the environment between two
    calls gets the stale answer (#1320 is the same smell on another
    path).

    `~` is expanded and the result is made absolute, so a relative
    `$CODEX_HOME` resolves against the process cwd exactly once, here,
    rather than differently at each use site.

    An explicitly configured value raises `CodexHomeError` rather than
    reverting to the conventional path when it does not exist, or exists
    and is not a directory. Both refusals mirror the running Codex,
    measured against codex-cli 0.145.0 on this machine::

        $ CODEX_HOME=/tmp/definitely-not-here codex mcp list
        Error: failed to load configuration
        Caused by:
            CODEX_HOME points to "/tmp/definitely-not-here", but that
            path does not exist

    with the identical shape for a non-directory, and an empty value
    falling back to the real `~/.codex`. The non-existent case matters
    most: Codex refuses to start there, so creating it and reporting
    success would leave `aelf setup --host codex` claiming a directory
    Codex will not read — the same failure #1427 was filed about, one
    typo away.
    """
    raw = os.environ.get(CODEX_HOME_ENV)
    if not raw:
        # Unset or empty: Codex's own fallback, and the pre-#1427
        # behaviour, byte for byte.
        return Path.home() / CODEX_DEFAULT_DIRNAME
    try:
        home = Path(raw).expanduser().absolute()
    except (OSError, ValueError) as exc:  # pragma: no cover - platform-dep
        raise CodexHomeError(
            f"${CODEX_HOME_ENV} is set to {raw!r}, which is not a usable "
            f"path ({exc}); fix or unset it"
        ) from exc
    if not home.exists():
        raise CodexHomeError(
            f"${CODEX_HOME_ENV} points to {home}, but that path does not "
            "exist; Codex refuses to start against it, so aelfrice will "
            "not create it. Create the directory, fix the value, or unset "
            "it to use the conventional ~/.codex"
        )
    if not home.is_dir():
        raise CodexHomeError(
            f"${CODEX_HOME_ENV} points to {home}, but that path is not a "
            "directory; fix or unset it"
        )
    return home


# SessionStart matcher covering every source, so the baseline block fires
# on fresh sessions and the rebuild block fires post-compaction (#1054:
# SessionStart(source=="compact") is Codex's only compaction-injection
# channel — PreCompact/PostCompact stdout is ignored by the harness).
# #1812: `fork` is the source of a forked thread, which gets the baseline
# too. Setup drops the group that holds its own handler and writes this
# one, so a rerun moves an earlier install to it.
_SESSION_START_MATCHER: Final[str] = "startup|resume|clear|compact|fork"
#: The SessionStart matcher that earlier releases wrote, which leaves out
#: `fork`. Doctor reports a group with it that holds an aelfrice handler as
#: a stale install, which a setup rerun fixes (#1812).
_STALE_SESSION_START_MATCHERS: Final[frozenset[str]] = frozenset(
    {"startup|resume|clear|compact"},
)

# Basenames owned by aelfrice; setup replaces exactly these on re-run and
# uninstall removes exactly these. Mirrors the manifest-ownership posture
# of auto_install (additive within owned basenames).
_OWNED_BASENAMES: Final[frozenset[str]] = frozenset({
    "aelf-hook",
    "aelf-transcript-logger",
    "aelf-session-start-hook",
    "aelf-stop-hook",
    "aelf-search-tool-hook",
    "aelf-pre-issue-hook",
    "aelf-commit-ingest",
})


def codex_hooks_path(codex_dir: Path | None = None) -> Path:
    cdir = codex_dir if codex_dir is not None else resolve_codex_home()
    return cdir / CODEX_HOOKS_FILENAME


def codex_config_path(codex_dir: Path | None = None) -> Path:
    cdir = codex_dir if codex_dir is not None else resolve_codex_home()
    return cdir / CODEX_CONFIG_FILENAME


CODEX_ADDITIONAL_CONTEXT_LIMIT: Final[int] = HOOK_PAYLOAD_CHAR_LIMIT
"""`additionalContextLimit` on the handlers whose stdout reaches the model.

Codex limits each model-visible hook output to about 2,500 tokens by
default and saves a longer one to a file, so the model sees only a preview
(https://developers.openai.com/codex/hooks, "Large hook output"). The
field is an approximate token threshold. The docs say: "For any command
hook that returns additionalContext, set additionalContextLimit on the
handler", and "For events that can't produce additional context, Codex
ignores additionalContextLimit and reports a configuration warning."
Three aelfrice handlers return `additionalContext`: retrieval
(UserPromptSubmit), the SessionStart baseline, and the search-tool hook
on PreToolUse, an event whose docs say "To add model-visible context
without blocking, return hookSpecificOutput.additionalContext". Codex's
hook discovery (codex-rs/hooks/src/engine/discovery.rs in openai/codex)
keeps the field on PreToolUse, PostToolUse, SessionStart, and
UserPromptSubmit, among others, and on an event that can't emit context
logs "ignoring additionalContextLimit ... this event cannot emit
additionalContext" and runs the handler without it.

The value is the bound aelfrice already enforces on every hook fire's
whole stdout, `HOOK_PAYLOAD_CHAR_LIMIT` characters. A token estimate of a
string charges at most one token per character for any estimator that
counts bytes per token at 4 or more, so a fire never exceeds this many
tokens and Codex never cuts aelfrice's block to a preview. The retrieval
token budget is not the bound: locks are never trimmed to it, several
lanes are packed against separate budgets, and
`AELFRICE_RETRIEVAL_TOKEN_BUDGET` can raise it.
"""

#: The argument setup appends to the SessionStart command so the hook knows
#: it runs under Codex. The `--host` spelling matches the CLI option.
CODEX_SESSION_START_ARGS: Final[str] = "--host codex"


def _handler(
    command: str,
    *,
    timeout: int | None = None,
    status_message: str | None = None,
    additional_context_limit: int | None = None,
) -> dict[str, object]:
    """One Codex command handler.

    `statusMessage` and `additionalContextLimit` are the Codex handler
    fields (https://developers.openai.com/codex/hooks). Each is written only
    when it has a value, so a handler without them is byte-identical to
    the one earlier releases wrote.
    """
    entry: dict[str, object] = {"type": "command", "command": command}
    if timeout is not None:
        entry["timeout"] = timeout
    if status_message is not None:
        entry["statusMessage"] = status_message
    if additional_context_limit is not None:
        entry["additionalContextLimit"] = additional_context_limit
    return entry


def desired_codex_hooks(
    scope: SettingsScope = "user",
    *,
    status_message: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """The aelfrice hook set in Codex hooks.json shape, keyed by event.

    Commands resolve to absolute paths via the same resolvers the Claude
    installers use, so both hosts pin identical executables.

    `status_message` goes on the handlers that `aelf setup --status-message`
    covers on the claude host: retrieval (UserPromptSubmit), the
    SessionStart baseline, and the stop hook. The transcript logger and the
    tool hooks carry none there either.
    """
    hook_cmd = resolve_hook_command(scope)
    logger_cmd = resolve_transcript_logger_command(scope)
    session_cmd = (
        f"{resolve_session_start_hook_command(scope)} "
        f"{CODEX_SESSION_START_ARGS}"
    )
    stop_cmd = resolve_stop_hook_command(scope)
    search_bash_cmd = resolve_search_tool_bash_command(scope)
    pre_issue_cmd = resolve_pre_issue_guard_command(scope)
    commit_cmd = resolve_commit_ingest_command(scope)
    return {
        "UserPromptSubmit": [
            {"hooks": [
                _handler(
                    hook_cmd,
                    status_message=status_message,
                    additional_context_limit=CODEX_ADDITIONAL_CONTEXT_LIMIT,
                ),
                _handler(logger_cmd),
            ]},
        ],
        "Stop": [
            {"hooks": [
                _handler(logger_cmd),
                _handler(stop_cmd, status_message=status_message),
            ]},
        ],
        "PreCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "PostCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "SessionStart": [
            {
                "matcher": _SESSION_START_MATCHER,
                "hooks": [_handler(
                    session_cmd,
                    status_message=status_message,
                    additional_context_limit=CODEX_ADDITIONAL_CONTEXT_LIMIT,
                )],
            },
        ],
        # #1055: Codex reports shell commands as tool_name "Bash", so the
        # Bash-matcher hooks are host-portable verbatim. The search-tool
        # matcher (Grep|Glob|WebSearch|WebFetch) is omitted — Grep and
        # Glob do not exist on Codex.
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    _handler(
                        search_bash_cmd,
                        additional_context_limit=CODEX_ADDITIONAL_CONTEXT_LIMIT,
                    ),
                    _handler(pre_issue_cmd),
                ],
            },
        ],
        "PostToolUse": [
            {"matcher": "Bash", "hooks": [_handler(commit_cmd)]},
        ],
    }


def _command_keys(
    handler: object, *, windows: bool | None = None,
) -> list[str]:
    """Every ownership key a handler's command could carry, [] on miss.

    Replaces the single-valued `_command_basename` (#1412) at both of its
    call sites. #1482: a spaced POSIX path has more than one reading, and
    that helper returned only the first — deciding anything from it is the
    defect, so nothing here keeps a single-key view of a command that was
    written by somebody else.

    #1412's platform gating is unchanged and lives in `launcher_key`: on
    Windows the keys are case-folded with the launcher suffix removed, so
    ``...\\Scripts\\aelf-hook.EXE`` compares equal to ``aelf-hook``; on
    POSIX they are plain basenames.
    """
    if not isinstance(handler, dict):
        return []
    hd = cast(dict[str, object], handler)
    cmd = hd.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return []
    return command_program_keys(cmd, windows=windows)


def _owned_command_key(cmd: str, *, windows: bool | None = None) -> str:
    """The owned name a command resolves to, '' when it is not ours.

    #1482: doctor compares the keys it *found* against the keys it
    *expected*, and both sides go through here so a spaced install compares
    per handler. Under the single-key derivation every handler of a spaced
    install keyed to the same path fragment, which collapsed all seven
    expected names into one and made the partial-removal check (#1430) blind
    for exactly the users this issue is about.
    """
    owned = owned_keys(_OWNED_BASENAMES, windows=windows)
    return next(
        (k for k in command_program_keys(cmd, windows=windows) if k in owned),
        "",
    )


def _handler_is_owned(handler: object, *, windows: bool | None = None) -> bool:
    """True iff a single handler is one of ours.

    #1482: every reading of the command is offered, but membership is still
    the closed `_OWNED_BASENAMES` set, so the wider candidate list cannot
    claim a foreign handler — ``/home/first last/bin/foreign-hook`` yields
    ``first`` and ``foreign-hook``, and neither is ours. That matters
    because this predicate is what `remove_codex_hooks` deletes by.
    """
    owned = owned_keys(_OWNED_BASENAMES, windows=windows)
    return any(k in owned for k in _command_keys(handler, windows=windows))


def _handlers_in(group: object) -> list[object]:
    """Every handler inside a matcher group, in file order."""
    if not isinstance(group, dict):
        return []
    handlers = cast(dict[str, object], group).get("hooks")
    if not isinstance(handlers, list):
        return []
    return list(cast(list[object], handlers))


def _owned_handlers_in(
    group: object, *, windows: bool | None = None,
) -> list[object]:
    """Every aelfrice handler inside a matcher group, in file order."""
    return [
        h for h in _handlers_in(group)
        if _handler_is_owned(h, windows=windows)
    ]


def _without_owned_handlers(
    group: object, *, windows: bool | None = None,
) -> object | None:
    """Drop our handlers from a group; None when nothing survives.

    #1412: ownership has to be decided per *handler*, not per group. The
    previous group-level rule left a mixed group — one aelfrice handler
    beside a foreign one — entirely untouched, which produced a data-loss
    asymmetry: setup did not recognise the aelfrice handler and appended a
    second one, then unsetup removed the group it had just created and left
    the original stranded inside the mixed group forever.

    Foreign handlers and every other key on the group (its ``matcher``, any
    field a future Codex adds) are preserved exactly; only the ``hooks``
    list is rewritten, and only when it actually changes.
    """
    if not isinstance(group, dict):
        return group
    gd = cast(dict[str, object], group)
    handlers = gd.get("hooks")
    if not isinstance(handlers, list):
        return group
    kept = [
        h for h in cast(list[object], handlers)
        if not _handler_is_owned(h, windows=windows)
    ]
    if len(kept) == len(cast(list[object], handlers)):
        return group
    if not kept:
        return None
    survivor = dict(gd)
    survivor["hooks"] = kept
    return survivor


def claude_host_has_aelfrice_hooks(
    settings_path: Path, *, windows: bool | None = None,
) -> bool:
    """True iff the Claude-host settings.json wires any aelfrice hook.

    Used by `aelf setup --host codex` (#1053) to distinguish a
    Codex-only machine (write the claude auto-install opt-out) from a
    dual-host one (leave auto-install alone). Shape-tolerant and
    fail-closed: a missing or unreadable settings file counts as "no
    hooks" — the worst case of a false negative is an opt-out the user
    can undo with one explicit `aelf setup`.

    #1412: this is the third consumer of the ownership-key derivation, and
    the one whose failure is silent. On Windows the old basename never
    started with ``aelf-``, so a dual-host machine read as Codex-only and
    the Claude auto-install opt-out was written over a live install.

    #1482: a POSIX install path containing a space failed the same way,
    keying as the fragment before the space; the probe now reads every
    candidate rather than the first.
    """
    if not settings_path.is_file():
        return False
    try:
        parsed = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    hooks = cast(dict[str, object], parsed).get("hooks")
    if not isinstance(hooks, dict):
        return False
    for groups in cast(dict[str, object], hooks).values():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            if not isinstance(group, dict):
                continue
            gd = cast(dict[str, object], group)
            for handler in cast(list[object], gd.get("hooks", []) or []):
                if any(
                    key.startswith("aelf-")
                    for key in _command_keys(handler, windows=windows)
                ):
                    return True
    return False


@dataclass
class CodexInstallResult:
    path: Path
    changed: bool
    installed_events: list[str] = field(default_factory=list[str])
    guidance: list[str] = field(default_factory=list[str])
    error: str | None = None


# --- hooks.json as a transaction (#1428) ---------------------------------
#
# `hooks.json` is shared configuration: an editor, another installer, or a
# second aelfrice process can write it between our read and our write.
# The pre-#1428 code did `Path.read_text()` ... `Path.write_text()` with
# nothing in between, so a concurrent update was replaced by our stale
# snapshot — silently, because the result is still valid JSON, and the
# entries lost are whatever hooks the other writer had just added.
#
# The shape here mirrors the settings transaction added for the other host
# in #1161, and inherits its two hard-won lessons:
#
# * **Do not collapse the read-modify-write.** Batching alone WIDENS the
#   window it claims to close. What makes it safe is the fingerprint: the
#   bytes are hashed at the read and re-hashed immediately before the
#   replace, so a change under us is detected rather than overwritten.
# * **Do not truncate in place.** `write_text` truncates first, so a short
#   write leaves a half-document where a complete one used to be. Every
#   commit goes to a same-directory temp file, is flushed and `fsync`ed,
#   and is then `os.replace`d — a crash at any point leaves the previous
#   complete file.
#
# The lock serialises aelfrice's own writers. The fingerprint catches a
# non-cooperating one. What remains — a foreign process replacing the file
# in the instant between the final fingerprint check and `os.replace` — is
# not closeable without a shared protocol Codex does not offer, and is
# documented rather than papered over.
_HOOKS_LOCK_TIMEOUT: Final[float] = 10.0

# Bounded, because the merge is convergent: a retry re-reads the newer
# document and re-applies the same owned groups to it, so retrying is
# strictly better than aborting. Bounded rather than unbounded so a
# pathologically busy file surfaces as an error instead of a hang.
_HOOKS_COMMIT_ATTEMPTS: Final[int] = 3


#: Stands in for "the document could not be parsed at all". A unique
#: object rather than None, because `null` parses to None and must stay
#: distinguishable from a syntax error (see `_plan_install`).
_UNPARSEABLE: Final[object] = object()

#: Stands in for "this key is not in the document". Same reason as
#: `_UNPARSEABLE`, one level down: `dict.get(k)` answers None both for a
#: missing key and for a key explicitly set to `null`, and only the first
#: is ours to fill in. Reusing None here is what let `{"hooks": null}`
#: and `{"hooks":{"UserPromptSubmit":null}}` be reshaped with no error
#: and no `--force` after the top-level case had been fixed.
_ABSENT: Final[object] = object()


def _json_type_name(value: object) -> str:
    """Name `value`'s type as JSON names it, for a message about JSON.

    `type(None).__name__` is `NoneType`, which is a Python detail leaking
    into a refusal the user is meant to act on by editing a `.json` file.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _fingerprint(path: Path) -> str:
    """Content hash of `path`; `""` when it does not exist.

    Hashed rather than stat-compared: mtime granularity is coarse enough
    to miss a fast rewrite, and a same-size edit defeats size alone.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def _read_hooks_snapshot(path: Path) -> tuple[str | None, str]:
    """Return `(text, fingerprint)` for one read of `path`.

    `text` is None when the file does not exist. Both values come from
    the *same* `read_bytes`, so nothing can slip between the content we
    merge and the fingerprint we later check it against.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, ""
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _atomic_replace_hooks(path: Path, text: str, expected: str) -> bool:
    """Replace `path` with `text` iff it still hashes to `expected`.

    Returns False when the file changed under us — the caller retries or
    reports. The temp file is created in the destination directory so the
    rename is a same-filesystem `os.replace`, and the destination's
    permission bits are carried over: `mkstemp` creates 0600, and a shared
    config file silently narrowing to owner-only is its own bug.
    """
    if _fingerprint(path) != expected:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp_path, path.stat().st_mode & 0o777)
        except OSError:
            # No destination yet (first install), or a platform that will
            # not report/apply the mode. The default 0600 is restrictive,
            # not permissive, so this degrades safely.
            pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            # Best-effort cleanup of the temp file on the failure path. If the
            # unlink itself fails we still want the ORIGINAL exception to
            # propagate — a stray temp file in the destination directory is a
            # smaller problem than swallowing the reason the commit failed,
            # and `mkstemp` names do not collide, so a leftover cannot
            # corrupt a later attempt.
            pass
        raise
    return True


def _plan_install(
    text: str | None, scope: SettingsScope, force: bool, hooks_path: Path,
    *, windows: bool | None = None, status_message: str | None = None,
) -> tuple[str | None, CodexInstallResult]:
    """Merge our hook set into one snapshot. Pure — touches no file.

    Returns `(serialized_or_None, result)`. `serialized` is None when
    nothing should be written, either because the document is already
    current or because `result.error` explains why we refuse.

    Foreign structure is preserved or refused, never normalised away: the
    pre-#1428 code substituted `{}` for a non-object `hooks` and `[]` for
    a touched event whose value was not a list, so
    `{"hooks":{"UserPromptSubmit":{"foreign":"keep"}}}` lost its object
    during an ordinary setup run with no error and no `--force`.

    That guarantee is total over JSON types at all three levels, which
    took three sentinels to say: `null` is a legal document, a legal
    value for `hooks`, and a legal value for an event, and Python spells
    all three `None` — the same `None` that `json.loads` never returns
    for a syntax error and that `dict.get` returns for a missing key.
    Fixing only the document level, as the first pass did, left
    `{"hooks": null}` and `{"hooks":{"UserPromptSubmit":null}}` silently
    reshaped while `{"hooks": 42}` was refused.
    """
    def refuse(msg: str) -> tuple[str | None, CodexInstallResult]:
        return None, CodexInstallResult(
            path=hooks_path, changed=False, error=msg,
        )

    existing: dict[str, object] = {}
    if text is not None:
        # `_UNPARSEABLE`, not None: `null` is a legal JSON document, so
        # reusing None as the "could not parse" sentinel made a file
        # holding `null` indistinguishable from a syntax error — and the
        # `parsed is not _UNPARSEABLE` guard below then skipped the
        # refusal and overwrote it with no error and no `--force`.
        parsed: object = _UNPARSEABLE
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            if not force:
                return refuse(
                    f"existing hooks.json is invalid JSON ({exc}); "
                    "re-run with --force to replace it"
                )
        if isinstance(parsed, dict):
            existing = cast(dict[str, object], parsed)
        elif parsed is not _UNPARSEABLE and not force:
            return refuse(
                "existing hooks.json is not a JSON object; "
                "re-run with --force to replace it"
            )

    # `_ABSENT`, not None, at both levels: `null` is a legal value for
    # any JSON key, so `.get(k)` cannot tell "no such key" (ours to fill
    # in) from "the user wrote null there" (foreign structure, and not
    # ours to reshape).
    hooks_obj = existing.get("hooks", _ABSENT)
    if hooks_obj is _ABSENT or force:
        hooks_map: dict[str, object] = (
            cast(dict[str, object], hooks_obj)
            if isinstance(hooks_obj, dict) else {}
        )
    elif isinstance(hooks_obj, dict):
        hooks_map = cast(dict[str, object], hooks_obj)
    else:
        return refuse(
            "existing hooks.json has a non-object `hooks` value "
            f"({_json_type_name(hooks_obj)}); it is not ours to reshape — "
            "fix it, or re-run with --force to replace it"
        )

    desired = desired_codex_hooks(scope, status_message=status_message)
    if not force:
        for event in desired:
            current = hooks_map.get(event, _ABSENT)
            if current is not _ABSENT and not isinstance(current, list):
                return refuse(
                    f"existing hooks.json has a non-list value for the "
                    f"`{event}` event ({_json_type_name(current)}); it is "
                    "not ours to reshape — fix it, or re-run with --force "
                    "to replace it"
                )

    before = json.dumps({"hooks": hooks_map}, sort_keys=True)
    for event, groups in desired.items():
        current = hooks_map.get(event)
        current_list = (
            cast(list[object], current) if isinstance(current, list) else []
        )
        kept = [
            survivor
            for survivor in (
                _without_owned_handlers(g, windows=windows)
                for g in current_list
            )
            if survivor is not None
        ]
        hooks_map[event] = kept + cast(list[object], groups)
    existing["hooks"] = hooks_map
    after = json.dumps({"hooks": hooks_map}, sort_keys=True)

    changed = before != after or text is None
    result = CodexInstallResult(
        path=hooks_path,
        changed=changed,
        installed_events=sorted(desired.keys()),
        guidance=[
            (
                "Codex runs a hook only after per-hook trust approval: open "
                + "a Codex session and run /hooks to approve the new entries "
                + "(automation that vets its own hook sources may pass "
                + "--dangerously-bypass-hook-trust instead)."
            ),
            (
                "The Codex `hooks` feature is stable and enabled by default; "
                + "no action is needed unless you disabled it "
                + "([features].hooks = false in config.toml)."
            ),
        ],
    )
    if not changed:
        return None, result
    return json.dumps(existing, indent=2) + "\n", result


def _plan_remove(
    text: str | None, hooks_path: Path, *, windows: bool | None = None,
) -> tuple[str | None, CodexInstallResult]:
    """Strip aelfrice-owned groups from one snapshot. Pure.

    A missing or unparseable file is reported, not modified — uninstall
    never destroys content it cannot positively identify as ours. An
    event whose value is not a list is left exactly as found for the same
    reason.
    """
    if text is None:
        return None, CodexInstallResult(path=hooks_path, changed=False)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, CodexInstallResult(
            path=hooks_path, changed=False,
            error=f"hooks.json is invalid JSON ({exc}); not modified",
        )
    if not isinstance(parsed, dict):
        return None, CodexInstallResult(
            path=hooks_path, changed=False,
            error="hooks.json is not a JSON object; not modified",
        )
    doc = cast(dict[str, object], parsed)
    hooks_obj = doc.get("hooks")
    if not isinstance(hooks_obj, dict):
        return None, CodexInstallResult(path=hooks_path, changed=False)
    hooks_map = cast(dict[str, object], hooks_obj)

    changed = False
    removed_events: list[str] = []
    for event in list(hooks_map.keys()):
        groups = hooks_map[event]
        if not isinstance(groups, list):
            continue
        group_list = cast(list[object], groups)
        kept = [
            survivor
            for survivor in (
                _without_owned_handlers(g, windows=windows) for g in group_list
            )
            if survivor is not None
        ]
        if kept != group_list:
            changed = True
            removed_events.append(event)
            if kept:
                hooks_map[event] = kept
            else:
                del hooks_map[event]
    result = CodexInstallResult(
        path=hooks_path, changed=changed,
        installed_events=sorted(removed_events),
    )
    if not changed:
        return None, result
    return json.dumps(doc, indent=2) + "\n", result


def _commit_hooks_transaction(
    hooks_path: Path,
    plan: Callable[[str | None], tuple[str | None, CodexInstallResult]],
) -> CodexInstallResult:
    """Run `plan` against `hooks_path` under lock, and commit atomically.

    One read-modify-write per attempt: read a snapshot with its
    fingerprint, plan against it, and replace the file only if it still
    hashes the same. A mismatch means a writer that does not take our
    lock got in; the plan is convergent, so we re-read and re-apply
    rather than clobber, up to `_HOOKS_COMMIT_ATTEMPTS`.

    A lock we cannot acquire within `_HOOKS_LOCK_TIMEOUT` is reported as
    `result.error`, not raised. `FileLockTimeout` would otherwise escape
    to the #1161 wrapper in `cli`, which names *settings.json* — the
    other host's file, and not the one under contention here.

    The sibling `hooks.json.lock` is created on demand — `_open_lock`
    makes the parent directory, so even a first install into a home that
    does not exist yet is serialised — and is deliberately **left in
    place by `unsetup`**, unlike the artifacts #1173 sweeps. Unlinking a
    lock file another process may be holding is how two writers end up
    holding two different inodes and no mutual exclusion at all; a
    zero-byte file is the cheaper residue. That residue is bounded to a
    home that already had a `hooks.json`: `remove_codex_hooks` refuses to
    enter this transaction at all when the file is absent, precisely
    because entering it would create the directory and the lock.
    """
    from aelfrice.session_ring import FileLockTimeout, exclusive_file_lock

    try:
        with exclusive_file_lock(hooks_path, timeout=_HOOKS_LOCK_TIMEOUT):
            # Nothing inside this block raises `FileLockTimeout`, so the
            # handler below can only be the acquisition failing.
            return _commit_under_lock(hooks_path, plan)
    except FileLockTimeout as exc:
        return CodexInstallResult(
            path=hooks_path, changed=False,
            error=(
                f"another aelfrice process is writing {hooks_path} "
                f"({exc}); nothing was changed. Re-run the command."
            ),
        )


def _commit_under_lock(
    hooks_path: Path,
    plan: Callable[[str | None], tuple[str | None, CodexInstallResult]],
) -> CodexInstallResult:
    """The bounded read-plan-commit loop. Caller holds the lock."""
    for _ in range(_HOOKS_COMMIT_ATTEMPTS):
        text, fingerprint = _read_hooks_snapshot(hooks_path)
        serialized, result = plan(text)
        if serialized is None:
            return result
        try:
            committed = _atomic_replace_hooks(
                hooks_path, serialized, fingerprint,
            )
        except OSError as exc:
            # A full disk, a revoked permission, an antivirus holding
            # the rename. The previous document is still complete on
            # disk; report rather than unwind a traceback through the
            # CLI, which is all the caller could do with it anyway.
            return CodexInstallResult(
                path=hooks_path, changed=False,
                error=f"could not write {hooks_path} ({exc}); "
                      "the existing file is unchanged",
            )
        if committed:
            return result
    return CodexInstallResult(
        path=hooks_path, changed=False,
        error=(
            f"{hooks_path} was modified by another process during each of "
            f"{_HOOKS_COMMIT_ATTEMPTS} attempts; nothing was written. "
            "Re-run the command."
        ),
    )


def install_codex_hooks(
    hooks_path: Path,
    *,
    scope: SettingsScope = "user",
    force: bool = False,
    windows: bool | None = None,
    status_message: str | None = None,
) -> CodexInstallResult:
    """Write the aelfrice hook set into ``hooks_path``, merge-aware.

    Serialised against other aelfrice writers and committed atomically;
    see the transaction notes above. Refuses (with ``error`` set) when
    the existing file is unparseable, or holds a foreign structure at a
    key we would have to reshape, and ``force`` is False.

    ``status_message`` becomes the Codex ``statusMessage`` field on the
    handlers `desired_codex_hooks` names. A re-run without it writes the
    handlers without the field, the way the claude host replaces its
    entries on every run.
    """
    return _commit_hooks_transaction(
        hooks_path,
        lambda text: _plan_install(
            text, scope, force, hooks_path,
            windows=windows, status_message=status_message,
        ),
    )


def remove_codex_hooks(
    hooks_path: Path, *, windows: bool | None = None,
) -> CodexInstallResult:
    """Remove aelfrice-owned matcher groups; drop emptied events.

    Same transaction as ``install_codex_hooks`` — an uninstall racing a
    setup is exactly as capable of dropping a foreign entry as two
    setups are.

    **No file, no transaction.** The absence check runs here, ahead of
    the lock, because taking the lock is itself a write: `_open_lock`
    mkdirs the parent and creates `hooks.json.lock`, so routing the
    "nothing to remove" case through the transaction made
    `aelf unsetup --host codex` — and `aelf uninstall --host codex`,
    which reaches the same handler — *create* the Codex home on a host
    that never had Codex, and leave behind a lock file this code
    deliberately never sweeps. An uninstall verb that creates the
    configuration directory it removes from is the #1173 defect class,
    and it contradicts what INSTALL.md promises about a home aelfrice
    will not create for you.

    Racing an install that creates the file immediately after this check
    is not a correctness problem: nothing of ours existed to strip at the
    instant we looked, and the plan is re-read under the lock in every
    path that does take it.
    """
    if not hooks_path.is_file():
        return CodexInstallResult(path=hooks_path, changed=False)
    return _commit_hooks_transaction(
        hooks_path,
        lambda text: _plan_remove(text, hooks_path, windows=windows),
    )


# --- Codex agent skills (the `$aelf-*` port of `/aelf:*`) ---------------
#
# Codex's analogue of an `/aelf:*` slash command is an *agent skill*: a
# directory holding a ``SKILL.md`` (name + description frontmatter, then
# natural-language instructions), discovered from the user scope
# ``~/.agents/skills/`` and invoked explicitly as ``$<name>`` or triggered
# implicitly when a task matches the description. Custom prompts
# (``~/.codex/prompts``) are the closer 1:1 to a slash file but are
# deprecated upstream in favour of skills, so we target skills.
#
# The source of truth is the SAME bundle the Claude installer ships
# (``src/aelfrice/slash_commands/*.md``). Each file is transformed on
# install — no second copy is maintained, so editing the slash file
# updates both hosts. The transform: rename ``aelf:foo`` -> ``aelf-foo``
# (colons are invalid in skill/dir names), reduce the frontmatter to the
# required ``name``/``description`` pair, and prepend a short adapter
# preamble that (a) defines ``$ARGUMENTS`` for a host with no positional
# substitution engine and (b) maps the host-specific ``Task`` fan-out
# tool onto Codex's equivalent mechanism (``Task`` is the only tool name
# the adapter maps). The ``<objective>``/``<process>`` body is carried
# over verbatim — except that the host-management commands (setup /
# doctor / uninstall / upgrade) additionally get a ``<host-adapter>``
# note steering their ``aelf`` invocations to the ``--host codex`` form
# (#1136), since the bare form targets another host's configuration.

# Codex USER-scope skill discovery root (the open agent-skills standard
# path, shared with other agents' skills — hence the marker-gated prune).
#
# #1427: resolved from the user home and NOT from `$CODEX_HOME`. The
# agent-skills location is a cross-agent standard path, so a custom Codex
# configuration home must not move it. Late-bound for the same reason
# `resolve_codex_home` is: an import-time `Path.home()` freezes `$HOME`.
def resolve_agents_skills_dir() -> Path:
    """The user-scope agent-skills root, `~/.agents/skills` (#1427)."""
    return Path.home() / ".agents" / "skills"


# Every generated SKILL.md carries this marker on its first body line. It
# is the prune safety key: uninstall / orphan-prune only ever removes an
# ``aelf-*`` skill directory whose SKILL.md contains this exact marker, so
# a user's hand-authored ``aelf-*`` skill is never destroyed.
_SKILL_MARKER: Final[str] = "AELFRICE-CODEX-SKILL"
_SKILL_PREFIX: Final[str] = "aelf-"
_SKILL_FILENAME: Final[str] = "SKILL.md"

# Host-management commands (#1136). Their bundled bodies instruct bare
# ``aelf setup`` / ``aelf doctor`` / ``aelf unsetup`` / ``aelf uninstall``
# runs, which on this host would install, scan, or tear down ANOTHER
# host's configuration (settings-file hooks, statusline, slash bundle)
# instead of the Codex home's ``hooks.json`` + the ``$aelf-*`` skills. Their
# generated skills carry an adapter note steering every such invocation
# to the ``--host codex`` form.
_HOST_MANAGEMENT_SKILLS: Final[frozenset[str]] = frozenset({
    "aelf-setup",
    "aelf-doctor",
    "aelf-uninstall",
    "aelf-upgrade",
})

# The bundled ``setup`` description names another host's artifacts
# (settings file + statusline snippet). Describe the codex-host effect
# instead, so implicit skill triggering matches what the command
# actually does here (#1136).
_SETUP_DESCRIPTION_OVERRIDE: Final[str] = (
    "Install the aelfrice hooks in the Codex home's hooks.json "
    "($CODEX_HOME, else ~/.codex) and the $aelf-* "
    "agent skills under ~/.agents/skills/ on this host."
)

_HOST_MANAGEMENT_NOTE: Final[str] = (
    "<host-adapter>\n"
    "IMPORTANT — on this host, every `aelf setup`, `aelf doctor`,\n"
    "`aelf unsetup`, or `aelf uninstall` invocation in the steps below\n"
    "MUST use the `--host codex` form (e.g. `aelf setup --host codex`,\n"
    "`aelf doctor --host codex`, `aelf unsetup --host codex`,\n"
    "`aelf uninstall <flags> --host codex`). The bare form\n"
    "targets another host's configuration — it would not touch this\n"
    "host's install and must not be run here.\n"
    "</host-adapter>"
)

# #1413. A generated skill must not name ``uv run aelf``. ``uv run``
# initializes and locks a cache before aelfrice starts, so in a sandbox
# whose cache is read-only the skill fails at the wrapper — measured,
# `uv run aelf --version` exits 2 where `aelf --version` exits 0. It also
# resolves a project environment, so an unactivated checkout holding a
# different aelfrice version silently shadows the ``uv tool`` install
# that generated the skill.
#
# Since #1857 the slash sources invoke ``aelf`` by name on every host, so
# the only text this rewrite still changes is the sentence in upgrade.md
# warning against substituting ``uv run aelf upgrade``. No absolute path
# is baked in (ratified design fork, 2026-08-06), so the generated skill
# stays machine-independent and resolves ``aelf`` through the invoking
# shell's PATH/PATHEXT like any other command.
_UV_RUN_PREFIX: Final[str] = "uv run aelf "
_DIRECT_PREFIX: Final[str] = "aelf "


def _direct_cli_invocations(text: str) -> str:
    """Rewrite ``uv run aelf <cmd>`` to ``aelf <cmd>`` (#1413).

    The one bundled occurrence (upgrade.md, #1857) is the literal prefix
    followed by a subcommand, so a literal replace is exact — there is no
    ``uv run aelfrice``, no line-broken form, and no occurrence where
    ``aelf`` is the final token. A test pins that property against the
    bundle so a future slash edit introducing another shape fails rather
    than shipping a half-rewritten skill.
    """
    return text.replace(_UV_RUN_PREFIX, _DIRECT_PREFIX)


def _parse_slash_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a slash-command file into (frontmatter, body).

    Frontmatter values in the bundle are single-line ``key: value`` pairs
    (the only multi-line key, ``allowed-tools``, is a list we discard), so
    a line-based parse is exact and dependency-free. Returns the scalar
    keys we consume (``name``, ``description``, ``argument-hint``) and the
    body text after the closing delimiter (verbatim, stripped of a single
    leading newline).
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    # lines[0] == "---"; find the closing delimiter.
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    front: dict[str, str] = {}
    for raw in lines[1:close]:
        if raw[:1] in (" ", "\t") or ":" not in raw:
            # Indented list item (allowed-tools entries) or blank — skip.
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        if key in ("name", "description", "argument-hint"):
            front[key] = val.strip()
    body = "\n".join(lines[close + 1:])
    return front, body.lstrip("\n")


def codex_skill_from_slash(filename: str, text: str) -> tuple[str, str]:
    """Transform one slash-command file into a (skill_name, SKILL.md) pair.

    ``filename`` is the bundle basename (e.g. ``search.md``); ``text`` is
    its full content. Returns the skill directory / ``name`` value (e.g.
    ``aelf-search``) and the rendered SKILL.md text. Deterministic: same
    input bytes -> same output bytes.
    """
    front, body = _parse_slash_frontmatter(text)
    raw_name = front.get("name") or (_SKILL_PREFIX + filename.removesuffix(".md"))
    skill_name = raw_name.replace(":", "-")
    slash_name = raw_name  # original slash form, e.g. "aelf:search"
    description = front.get("description", "")
    if skill_name == "aelf-setup":
        # #1136: the bundled description names another host's artifacts.
        description = _SETUP_DESCRIPTION_OVERRIDE

    adapter: list[str] = [
        f"<!-- {_SKILL_MARKER}: auto-generated from "
        f"src/aelfrice/slash_commands/{filename}. Edit the source file, "
        "not this copy. -->",
        "",
        f"This is the Codex port of the `/{slash_name}` slash "
        f"command; invoke it as `${skill_name}`.",
    ]
    hint = front.get("argument-hint")
    if hint:
        adapter.append(f"Arguments: {hint}")
    if "$ARGUMENTS" in body:
        adapter.append(
            "Where the steps below reference `$ARGUMENTS`, substitute the "
            f"text the user typed after `${skill_name}` (their query and/or "
            "flags)."
        )
    if "Task" in body and ("subagent" in body or "Task tool" in body or "Task subagent" in body):
        adapter.append(
            "Where the steps mention the host's `Task` tool / subagents, "
            "use Codex's own subagent mechanism to fan out the equivalent "
            "work; the dispatch logic and CLI calls are unchanged."
        )
        if filename == "onboard.md":
            # onboard's classification fan-out defaults to a low-cost
            # model tier and lets the user pick the tier at run time; it
            # is tier-abstract, not pinned to a host-specific model (the
            # slash body stopped naming one in #1155). On Codex the
            # cheapest fast tier is a `-mini`-class model; resolve the
            # low-cost default there rather than let it fall through to
            # the session's default model — the expensive, slow path for
            # short-label classification. No model id pinned (names drift).
            adapter.append(
                "This skill's classification step defaults to a low-cost "
                "model tier for the fan-out and lets the user pick the tier "
                "at run time; it no longer names a host-specific model. On "
                "Codex, resolve that low-cost default to Codex's cheapest "
                "fast model tier (a `-mini`-class model) — not the session's "
                "default model, which is more expensive and slower for bulk "
                "short-label classification. Use that same model name where a "
                "step renders or prices the classifier model."
            )
    if skill_name in _HOST_MANAGEMENT_SKILLS:
        adapter.append(_HOST_MANAGEMENT_NOTE)
    adapter.append(
        "Run each `aelf ...` command in your shell and show its "
        "output to the user. Invoke `aelf` directly, exactly as written: "
        "wrapping it in a package-runner prefix needs a writable cache "
        "this host may not have, and can select a different aelfrice "
        "than the one installed."
    )

    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {description}",
        "---",
        *adapter,
        "",
        _direct_cli_invocations(body.rstrip("\n")),
        "",
    ]
    return skill_name, "\n".join(lines)


def _bundled_codex_skills() -> dict[str, str]:
    """Map skill_name -> SKILL.md text for every bundled slash command."""
    from aelfrice.setup import bundled_slash_files

    result: dict[str, str] = {}
    for filename, text in bundled_slash_files().items():
        skill_name, skill_text = codex_skill_from_slash(filename, text)
        result[skill_name] = skill_text
    return result


def _is_owned_skill_dir(skill_dir: Path) -> bool:
    """True iff ``skill_dir`` is an aelfrice-generated skill we may prune.

    Gated on both the ``aelf-`` name prefix AND the marker inside its
    SKILL.md, so a user's own ``aelf-*`` skill (no marker) is left alone.
    """
    if not skill_dir.name.startswith(_SKILL_PREFIX) or not skill_dir.is_dir():
        return False
    skill_md = skill_dir / _SKILL_FILENAME
    if not skill_md.is_file():
        return False
    try:
        return _SKILL_MARKER in skill_md.read_text(encoding="utf-8")
    except OSError:
        return False


@dataclass(frozen=True)
class CodexSkillsResult:
    """Outcome of install/uninstall of the Codex ``$aelf-*`` skills.

    ``skipped`` (#1136): bundled skill names whose on-disk collision is
    an unmarked (non-aelfrice) skill — never overwritten. ``failed``
    (#1136): human-readable ``"<name>: <reason>"`` rows for partial
    removals and other FS errors that previously vanished silently.
    """

    dest_dir: Path
    written: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def _orphan_skill_dirs(target: Path, bundle: dict[str, str]) -> list[Path]:
    """Marker-carrying ``aelf-*`` dirs under ``target`` absent from ``bundle``.

    One definition of "orphan", two callers: `install_codex_skills` prunes
    exactly these, `orphaned_codex_skills` reports exactly these. Factored
    out rather than restated so the set doctor names cannot drift from the
    set setup removes — the drift #1486 is about ran the other way, with the
    prune loop knowing about renamed-away commands while the only scanner
    doctor had, `modified_codex_skills`, iterates bundled names and could
    not see them at all.

    Both halves of the predicate are load-bearing. Name-membership alone
    would sweep in every current skill; the marker gate alone would sweep in
    a user's hand-authored ``aelf-*`` skill, which on the prune path means
    deleting it.
    """
    return [
        child
        for child in sorted(target.glob(f"{_SKILL_PREFIX}*"))
        if child.name not in bundle and _is_owned_skill_dir(child)
    ]


def _remove_owned_skill_dir(
    child: Path, pruned: list[str], failed: list[str],
) -> None:
    """Remove one owned skill dir: unlink its SKILL.md, then rmdir.

    The two steps are split (#1136) so a half-removal is visible: a
    failed unlink records the skill under ``failed`` and stops; a
    successful unlink followed by a failed rmdir (routine case: a stray
    extra file — e.g. OS metadata — keeps the directory non-empty)
    counts the skill as pruned (its SKILL.md is gone) AND records the
    leftover directory under ``failed``. Nothing is deleted recursively.
    """
    try:
        (child / _SKILL_FILENAME).unlink()
    except OSError as exc:
        failed.append(f"{child.name}: could not remove SKILL.md ({exc})")
        return
    try:
        child.rmdir()
    except OSError as exc:
        pruned.append(child.name)
        failed.append(
            f"{child.name}: SKILL.md removed but directory left in "
            f"place ({exc})"
        )
    else:
        pruned.append(child.name)


def onboard_skill_installed(dest_dir: Path | None = None) -> bool:
    """True when `$aelf-onboard` runs aelfrice's own generated skill (#1859).

    False after `--no-codex-skills` on a clean home, and when a skill of
    another origin holds the name, which the installer leaves in place.
    """
    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    try:
        return _is_owned_skill_dir(target / f"{_SKILL_PREFIX}onboard")
    except (OSError, ValueError):
        # The hooks are already written, and this only picks a line of
        # text. A SKILL.md that isn't UTF-8 raises UnicodeDecodeError, and
        # `is_dir()` on a folder without permission raises on Python 3.12.
        # A skill that can't be read can't be shown to be ours.
        return False


def install_codex_skills(dest_dir: Path | None = None) -> CodexSkillsResult:
    """Write every bundled command as a Codex skill under ``dest_dir``.

    Default ``dest_dir`` is ``~/.agents/skills/``. Each skill lands at
    ``<dest>/aelf-<cmd>/SKILL.md``. Idempotent (byte-identical files are
    skipped), atomic (temp + ``os.replace``), and orphan-pruning — but
    both the replace path and pruning are marker-gated: only
    marker-carrying ``aelf-*`` skill dirs are ever overwritten or
    removed, never the other skills that share this directory (#1136).
    """
    import os
    import tempfile

    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    bundle = _bundled_codex_skills()

    written: list[str] = []
    already: list[str] = []
    pruned: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    target.mkdir(parents=True, exist_ok=True)

    for skill_name, text in sorted(bundle.items()):
        skill_dir = target / skill_name
        dest_file = skill_dir / _SKILL_FILENAME
        if dest_file.is_file():
            try:
                if dest_file.read_text(encoding="utf-8") == text:
                    already.append(skill_name)
                    continue
            except OSError:
                # Unreadable existing file: ownership cannot be verified
                # either, so the marker gate below fails closed (skip).
                pass
            # Bytes differ: the replace path is marker-gated exactly
            # like prune/remove (#1136) — a colliding skill without our
            # marker is someone else's file and is never overwritten.
            if not _is_owned_skill_dir(skill_dir):
                skipped.append(skill_name)
                continue
        skill_dir.mkdir(parents=True, exist_ok=True)
        encoded = text.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(
            prefix=_SKILL_FILENAME + ".", suffix=".tmp", dir=str(skill_dir)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest_file)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        written.append(skill_name)

    # Prune orphans: marker-carrying aelf-* dirs no longer in the bundle
    # (handles renames/removals). Non-aelfrice skills are never touched.
    for child in _orphan_skill_dirs(target, bundle):
        _remove_owned_skill_dir(child, pruned, failed)

    return CodexSkillsResult(
        dest_dir=target,
        written=tuple(written),
        already=tuple(already),
        pruned=tuple(pruned),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )


def remove_codex_skills(dest_dir: Path | None = None) -> CodexSkillsResult:
    """Remove all aelfrice-generated ``$aelf-*`` skills from ``dest_dir``.

    Only marker-carrying ``aelf-*`` skill directories are removed; other
    skills sharing the directory are preserved. Returns the removed skill
    names under ``pruned``.
    """
    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    pruned: list[str] = []
    failed: list[str] = []
    if target.is_dir():
        for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
            if _is_owned_skill_dir(child):
                _remove_owned_skill_dir(child, pruned, failed)
    return CodexSkillsResult(
        dest_dir=target, pruned=tuple(pruned), failed=tuple(failed),
    )


def count_installed_codex_skills(dest_dir: Path | None = None) -> int:
    """Count marker-carrying ``aelf-*`` skills present under ``dest_dir``."""
    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    if not target.is_dir():
        return 0
    return sum(
        1 for child in target.glob(f"{_SKILL_PREFIX}*")
        if _is_owned_skill_dir(child)
    )


@dataclass
class CodexDoctorReport:
    """Structured result of the Codex host scan; render at the CLI."""

    codex_dir_present: bool
    #: The resolved Codex home this scan ran against (#1427). Reported so
    #: the operator can see WHICH directory was inspected — the whole
    #: point of the `$CODEX_HOME` bug is that it was the wrong one.
    codex_dir: Path | None = None
    hooks_file_present: bool = False
    hooks_file_valid: bool = False
    parse_error: str | None = None
    owned_handler_count: int = 0
    missing_events: list[str] = field(default_factory=list[str])
    missing_handlers: list[str] = field(default_factory=list[str])
    stale_commands: list[str] = field(default_factory=list[str])
    #: Matchers of the SessionStart groups that hold an aelfrice handler
    #: under the matcher an earlier release wrote, which leaves out `fork`
    #: (#1812).
    stale_matchers: list[str] = field(default_factory=list[str])
    #: (event, program, count) for each aelfrice program that more than one
    #: handler in one event runs with a command byte-identical to the one
    #: `aelf setup --host codex` writes for that program (#1812). The count
    #: is for one fire, so handlers whose matchers share no value, such as
    #: a SessionStart `startup` group and a `resume` group, don't add up.
    duplicate_runs: list[tuple[str, str, int]] = field(
        default_factory=list[tuple[str, str, int]],
    )
    feature_flag_on: bool | None = None
    #: Approved ``[hooks.state]`` entries on this host — **all** of them,
    #: not only ours (#1486). The keys are positional digests with no path
    #: back to a command, so no aelfrice-only subset can be computed; a
    #: foreign approval can therefore only inflate this, never deflate it,
    #: which is what keeps the zero-approvals warning below sound.
    approved_state_count: int = 0
    warnings: list[str] = field(default_factory=list[str])

    def tampering(self) -> list[str]:
        """Ways installed wiring is broken, each one a reason to exit nonzero.

        The distinction this encodes is #1430's, as narrowed by the operator
        ruling on it: **absent** wiring is not a fault — a machine that has
        never run ``aelf setup --host codex`` is a normal machine, and failing
        doctor there would make the Codex host stricter than the Claude host
        for no stated reason. **Tampered or partially-broken** wiring is a
        fault, because a green exit on it is a false green: automation and
        users are told the requested memory surface is present when it cannot
        work.

        Every condition here is therefore gated on our handlers actually
        being installed. None of them requires a reachable Codex model, which
        is what parked the rest of #1430 — they are all filesystem and config
        states, provable by fixture.

        Deliberately *not* included: ``approved_state_count == 0``. Approval
        keying is positional today and slated to change upstream, so a count
        of zero does not distinguish "unapproved" from "keyed differently";
        the issue itself asks for ``unknown`` rather than unhealthy when trust
        cannot be established authoritatively.
        """
        if not self.owned_handler_count:
            return []
        reasons: list[str] = []
        if self.stale_commands:
            reasons.append(
                "hook command(s) configured but not present on disk: "
                + ", ".join(self.stale_commands),
            )
        for matcher in self.stale_matchers:
            reasons.append(
                f"SessionStart runs aelfrice under the matcher {matcher!r} "
                "that an earlier release wrote, which leaves out `fork`, so "
                "a forked thread gets no baseline; run `aelf setup --host "
                "codex` again",
            )
        if self.missing_events:
            reasons.append(
                "aelfrice hooks are partially installed; missing events: "
                + ", ".join(self.missing_events),
            )
        if self.missing_handlers:
            reasons.append(
                "aelfrice handlers are missing from events that are "
                "otherwise installed: " + ", ".join(self.missing_handlers),
            )
        for event, program, count in self.duplicate_runs:
            reasons.append(
                f"{event} runs {program} {count} times, once per copy of "
                "the handler `aelf setup --host codex` writes; keep one "
                "and delete the others",
            )
        if self.feature_flag_on is False:
            reasons.append(
                "aelfrice handlers are installed but the Codex `hooks` "
                "feature is disabled in config.toml, so none of them run",
            )
        return reasons


def modified_codex_skills(dest_dir: Path | None = None) -> list[str]:
    """Installed ``$aelf-*`` skills whose bytes differ from what we generate.

    `_is_owned_skill_dir` gates on the name prefix and the marker, which
    establishes *ownership* but says nothing about *integrity*: a selected
    skill whose body has been edited still counts as installed and healthy,
    so doctor reported a skill surface that no longer does what it claims.

    Only skills that are present are compared. A skill absent from disk is
    absent wiring, which the ruling on #1430 keeps out of scope. An unreadable
    ``SKILL.md`` is *not* reported here: `_is_owned_skill_dir` reads the file
    itself to look for the marker and answers False when that read raises, so
    an unreadable file is classified not-ours and skipped. Saying otherwise
    would assert a branch this function cannot reach.

    "Differs from what we generate" is not only user tampering — a release
    that edits a slash-command body makes every installed skill differ until
    `aelf setup --host codex` is re-run, because `aelf upgrade` does not
    reinstall them. Both states have the same remediation and the same
    consequence (the skill surface is not what this build ships), so they are
    reported together and the message names both causes.
    """
    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    modified: list[str] = []
    for skill_name, expected in sorted(_bundled_codex_skills().items()):
        skill_dir = target / skill_name
        if not _is_owned_skill_dir(skill_dir):
            continue
        try:
            actual = (skill_dir / _SKILL_FILENAME).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - raced with the marker read
            modified.append(skill_name)
            continue
        if actual != expected:
            modified.append(skill_name)
    return modified


def orphaned_codex_skills(dest_dir: Path | None = None) -> list[str]:
    """Installed ``$aelf-*`` skills this build no longer ships (#1486).

    `modified_codex_skills` iterates the bundle, so a marker-carrying
    ``aelf-*`` directory for a command that was renamed or removed is never
    examined — while `count_installed_codex_skills` globs the same directory
    and counts it. Doctor therefore reported one more installed skill than it
    judged, and a slash command dropped from the build stayed invocable by
    the model until the next ``aelf setup --host codex`` pruned it.

    The orphan predicate is `_orphan_skill_dirs`, shared with that prune
    loop, so the set doctor names is by construction the set setup removes.

    The window is narrow — it opens on a release that renames or removes a
    command and closes on the next setup run — which is why this reports
    rather than deletes: doctor is read-only.
    """
    target = dest_dir if dest_dir is not None else resolve_agents_skills_dir()
    if not target.is_dir():
        return []
    return [
        child.name
        for child in _orphan_skill_dirs(target, _bundled_codex_skills())
    ]


#: The scopes `aelf setup --host codex` writes. `_cmd_setup_codex` in
#: cli.py passes `scope="user"`, and the command takes no `--scope`.
_SETUP_SCOPES: Final[tuple[SettingsScope, ...]] = ("user",)
#: The handler fields that hold the command Codex runs on Windows, the
#: field and its alias. Codex runs this command on Windows when a handler
#: has it, and `command` otherwise (`HookHandlerConfig` in
#: codex-rs/protocol/src/hooks.rs, lines 150-155, and
#: `append_matcher_groups` in codex-rs/hooks/src/engine/discovery.rs, lines
#: 660-664, at openai/codex@69231e119bfca0b14453515e97d99688ce1c47c9).
_WINDOWS_COMMAND_FIELDS: Final[tuple[str, ...]] = (
    "commandWindows", "command_windows",
)
#: The characters a program name can hold. Doctor reads an owned program
#: name in a command only where no character of this class is next to it.
_NAME_CHARS: Final[str] = "A-Za-z0-9_.-"


def _owned_name_pattern(*, windows: bool) -> re.Pattern[str]:
    """A regex that finds an owned program name in a command's text.

    The name must have the start or end of the text, or a character
    outside `_NAME_CHARS`, on each side. On Windows the name can end in a
    launcher suffix such as `.exe`, and its case doesn't matter, the way
    `launcher_key` reads a name there. The class next to the name always
    matches case.
    """
    names = "|".join(
        re.escape(name)
        for name in sorted(_OWNED_BASENAMES, key=len, reverse=True)
    )
    suffix = ""
    if windows:
        suffix = "(?:{})?".format(
            "|".join(re.escape(s) for s in sorted(LAUNCHER_SUFFIXES)),
        )
    return re.compile(
        rf"(?<!(?-i:[{_NAME_CHARS}]))({names}){suffix}"
        rf"(?!(?-i:[{_NAME_CHARS}]))",
        re.IGNORECASE if windows else 0,
    )


#: `_owned_name_pattern` for POSIX (False) and for Windows (True).
_OWNED_NAME: Final[dict[bool, re.Pattern[str]]] = {
    windows: _owned_name_pattern(windows=windows) for windows in (False, True)
}


def _handler_command(handler: object) -> str | None:
    """A handler's `command` string, or None when it has none."""
    if not isinstance(handler, dict):
        return None
    cmd = cast(dict[str, object], handler).get("command")
    return cmd if isinstance(cmd, str) else None


def _run_command(handler: object, *, windows: bool) -> str | None:
    """The command Codex runs for a handler, or None when it runs none.

    Codex runs a command only for a handler whose `type` is `command`. It
    skips a `prompt` or `agent` handler, and an `mcp_tool` handler calls a
    tool (`append_matcher_groups` in codex-rs/hooks/src/engine/discovery.rs,
    lines 650-801, at openai/codex@69231e119bfca0b14453515e97d99688ce1c47c9).
    On Windows, Codex runs the handler's `commandWindows` when it has one
    (`_WINDOWS_COMMAND_FIELDS`), and its `command` otherwise.
    """
    if not isinstance(handler, dict):
        return None
    hd = cast(dict[str, object], handler)
    if hd.get("type") != "command":
        return None
    if windows:
        for name in _WINDOWS_COMMAND_FIELDS:
            command = hd.get(name)
            if isinstance(command, str):
                return command
    return _handler_command(hd)


def _named_owned_keys(
    handler: object, *, windows: bool | None = None,
) -> dict[str, str]:
    """The owned programs a handler's command fields name, for doctor's
    warning (#1812).

    Each key maps to the text of the first field that names it, starting
    with the field Codex runs on the platform (`_run_command`). Doctor
    reads `command` and the Windows fields of a handler of any `type`.

    This is a text match (`_OWNED_NAME`). Doctor reads no shell syntax, so
    a name counts wherever it is in the text: in a script string such as
    `python3 -c "...aelf-session-start-hook..."`, in a command substitution,
    after `=` as in `--cmd=aelf-session-start-hook`, and in a command that
    only prints the name or checks that the program exists.
    """
    if not isinstance(handler, dict):
        return {}
    win = os.name == "nt" if windows is None else windows
    hd = cast(dict[str, object], handler)
    fields = (
        (*_WINDOWS_COMMAND_FIELDS, "command") if win
        else ("command", *_WINDOWS_COMMAND_FIELDS)
    )
    found: dict[str, str] = {}
    for name in fields:
        text = hd.get(name)
        if not isinstance(text, str):
            continue
        for match in _OWNED_NAME[win].finditer(text):
            key = match.group(1).casefold() if win else match.group(1)
            found.setdefault(key, text)
    return found


#: The values a group's `matcher` is tested against, for each event whose
#: values the Codex docs list (https://developers.openai.com/codex/hooks,
#: "Matcher patterns"): SessionStart matches its start source, and
#: PreCompact and PostCompact match their trigger. Any other event has no
#: entry. The tool events match an open set of tool names, which
#: `_TOOL_EVENTS` covers, and Codex ignores the matcher on
#: UserPromptSubmit and Stop, so doctor counts every handler in such an
#: event as running on every fire.
_MATCHER_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "SessionStart": ("startup", "resume", "clear", "compact", "fork"),
    "PreCompact": ("manual", "auto"),
    "PostCompact": ("manual", "auto"),
}
#: The events whose matcher is tested against tool names. Under an exact
#: matcher a group's values are the names it lists, since any name can be
#: a tool's.
_TOOL_EVENTS: Final[frozenset[str]] = frozenset({"PreToolUse", "PostToolUse"})
#: The names one tool call is matched by, for each tool that carries more
#: than its own name. Codex tests a matcher against the tool's name and its
#: matcher aliases (`matcher_inputs` in codex-rs/hooks/src/events/common.rs
#: at openai/codex@c0cc9fc82), and only `apply_patch` and `spawn_agent`
#: have aliases (`HookToolName` in codex-rs/core/src/tools/hook_names.rs).
_TOOL_MATCHER_INPUTS: Final[tuple[frozenset[str], ...]] = (
    frozenset({"apply_patch", "Write", "Edit"}),
    frozenset({"spawn_agent", "Agent"}),
)
#: The events whose `matcher` Codex reads. Codex ignores the matcher on
#: the others, so it never compiles one there (`matcher_pattern_for_event`
#: in codex-rs/hooks/src/events/common.rs at openai/codex@c0cc9fc82).
#: This leaves out Codex's two agent start and stop events, where aelfrice
#: writes no hooks, so doctor counts every handler there.
_MATCHER_EVENTS: Final[frozenset[str]] = frozenset({
    "PermissionRequest", "PostCompact", "PostToolUse", "PreCompact",
    "PreToolUse", "SessionEnd", "SessionStart",
})
#: The one value that stands for every fire of an event not in
#: `_MATCHER_VALUES`.
_EVERY_FIRE: Final[frozenset[str]] = frozenset({""})
#: A matcher of only these characters is a list of exact names to Codex.
_EXACT_MATCHER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_|]+")
#: The flags Rust's `regex` crate takes in a `(?flags)` or `(?flags:...)`
#: group. Python's `re` also takes `a` and `L`, which Rust rejects.
_RUST_REGEX_FLAGS: Final[frozenset[str]] = frozenset("imsUuxR")
#: The characters an escape makes literal in Rust's `regex` crate: these
#: meta characters, and any other ASCII character that isn't a letter, a
#: digit, `<`, or `>` (`parse_escape` in regex-syntax 0.8).
_RUST_META: Final[frozenset[str]] = frozenset("\\.+*?()|[]{}^$#&-~")
#: The escapes of control characters, with each one's code point.
_RUST_CONTROL_ESCAPES: Final[dict[str, int]] = {
    "a": 0x07, "f": 0x0C, "t": 0x09, "n": 0x0A, "r": 0x0D, "v": 0x0B,
}
#: The escapes of Perl classes and of assertions. A class can't be a range
#: end, and an assertion can't be in a class at all.
_RUST_CLASS_ESCAPES: Final[frozenset[str]] = frozenset("dswDSW")
_RUST_ASSERTION_ESCAPES: Final[frozenset[str]] = frozenset("AzbB<>")
#: How many hex digits `\x`, `\u`, and `\U` take when there are no braces.
_RUST_HEX_DIGITS: Final[dict[str, int]] = {"x": 2, "u": 4, "U": 8}
_HEX_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_DECIMAL_CHARS: Final[frozenset[str]] = frozenset("0123456789")
#: The names `\b{name}` takes.
_RUST_WORD_BOUNDARIES: Final[frozenset[str]] = frozenset(
    {"start", "end", "start-half", "end-half"},
)
#: The general categories `\pX` names with one letter, in either case.
_RUST_ONE_LETTER_CLASSES: Final[frozenset[str]] = frozenset("clmnpsz")
#: One side of a `\p{name}` or `\p{name=value}` class, once spaces, `_`,
#: and `-` are dropped and the letters lowercased.
_RUST_PROPERTY_PART: Final[re.Pattern[str]] = re.compile(r"[a-z0-9.]+")
#: The names of the `[:name:]` classes, and the shape of one.
_RUST_ASCII_CLASSES: Final[frozenset[str]] = frozenset({
    "alnum", "alpha", "ascii", "blank", "cntrl", "digit", "graph", "lower",
    "print", "punct", "space", "upper", "word", "xdigit",
})
_RUST_ASCII_CLASS: Final[re.Pattern[str]] = re.compile(r"\[:\^?([^:]*):\]")


class _RustSyntaxError(Exception):
    """Syntax that Rust's `regex` crate rejects."""


def _rust_property_name(name: str) -> bool:
    """True when a `\\p{...}` name has the shape Rust's `regex` reads.

    Rust drops spaces, `_`, and `-` and ignores case, and a name can be a
    `name=value`, `name:value`, or `name!=value` pair. This checks only
    the shape, so a name that no Unicode property has, such as `Foo`,
    passes.
    """
    for separator in ("!=", ":", "="):
        if separator in name:
            parts = name.split(separator, 1)
            break
    else:
        parts = [name]
    return all(
        _RUST_PROPERTY_PART.fullmatch(
            re.sub(r"[\s_-]", "", part).lower(),
        ) is not None
        for part in parts
    )


class _RustRegexCheck:
    """Reads a pattern the way Rust's `regex` crate parses one, far enough
    to tell whether `Regex::new` rejects it (#1812).

    Each method reads one construct from `self.at` on and raises
    `_RustSyntaxError` where regex-syntax 0.8's parser returns an error
    (`ast/parse.rs` in rust-lang/regex), so the method names follow the
    parser's. Every loop moves `self.at` forward, so a check ends.
    """

    def __init__(self, pattern: str) -> None:
        self.text = pattern
        self.at = 0
        #: The `x` flag, under which whitespace and `#` comments are skipped.
        self.verbose = False
        self.names: set[str] = set()

    def char(self, offset: int = 0) -> str:
        """The character `offset` past `self.at`, or '' past the end."""
        k = self.at + offset
        return self.text[k] if k < len(self.text) else ""

    def done(self) -> bool:
        return self.at >= len(self.text)

    def fail_at_end(self) -> None:
        if self.done():
            raise _RustSyntaxError

    def skip_space(self) -> None:
        """Skip whitespace and `#` comments under the `x` flag."""
        while self.verbose and not self.done():
            if self.text[self.at].isspace():
                self.at += 1
            elif self.text[self.at] == "#":
                end = self.text.find("\n", self.at)
                self.at = len(self.text) if end < 0 else end + 1
            else:
                return

    def bump(self) -> bool:
        """Step past one character and any skipped space; False at the end."""
        self.at += 1
        self.skip_space()
        return not self.done()

    def peek_space(self) -> str:
        """The character after this one, past any skipped space."""
        k = self.at + 1
        while self.verbose and k < len(self.text) and self.text[k].isspace():
            k += 1
        return self.text[k] if k < len(self.text) else ""

    def check(self) -> None:
        """Read the whole pattern."""
        # The `x` flag to restore at each open group's `)`.
        groups: list[bool] = []
        # Whether a repetition operator here has an item to repeat. A
        # group starts with none, and a `(?flags)` group isn't one.
        repeatable = False
        while True:
            self.skip_space()
            if self.done():
                break
            ch = self.text[self.at]
            if ch == "(":
                self.group(groups)
                repeatable = False
            elif ch == ")":
                if not groups:
                    raise _RustSyntaxError
                self.verbose = groups.pop()
                self.at += 1
                repeatable = True
            elif ch == "|":
                self.at += 1
                repeatable = False
            elif ch in ("?", "*", "+", "{"):
                if not repeatable:
                    raise _RustSyntaxError
                self.repetition()
            elif ch == "[":
                self.char_class()
                repeatable = True
            elif ch == "\\":
                self.escape()
                repeatable = True
            else:
                self.at += 1
                repeatable = True
        if groups:
            raise _RustSyntaxError

    def group(self, groups: list[bool]) -> None:
        """Read a `(` and the rest of what opens its group, or a whole
        `(?flags)` group."""
        self.bump()
        rest = self.text[self.at:]
        if rest.startswith(("?=", "?!", "?<=", "?<!")):
            raise _RustSyntaxError
        if rest.startswith(("?P<", "?<")):
            self.at += 3 if rest.startswith("?P<") else 2
            self.capture_name()
        elif rest.startswith("?"):
            self.at += 1
            self.fail_at_end()
            items, verbose = self.flags()
            closes = self.text[self.at] == ")"
            self.at += 1
            if closes and not items:
                raise _RustSyntaxError
            if not closes:
                groups.append(self.verbose)
            if verbose is not None:
                self.verbose = verbose
            return
        groups.append(self.verbose)

    def flags(self) -> tuple[int, bool | None]:
        """Read flags up to the `:` or `)` after them, and return how many
        items there are and what they set the `x` flag to, if anything."""
        seen: set[str] = set()
        negated = dangling = False
        verbose: bool | None = None
        items = 0
        while self.char() not in (":", ")"):
            ch = self.char()
            if ch == "-" and not negated:
                negated = dangling = True
            elif ch in _RUST_REGEX_FLAGS and ch not in seen:
                seen.add(ch)
                dangling = False
                if ch == "x":
                    verbose = not negated
            else:
                raise _RustSyntaxError
            items += 1
            self.at += 1
            self.fail_at_end()
        if dangling:
            raise _RustSyntaxError
        return items, verbose

    def capture_name(self) -> None:
        """Read a group name and its `>`. A name starts with a letter or
        `_`, and goes on with letters, digits, `_`, `.`, `[`, and `]`."""
        start = self.at
        while self.char() != ">":
            ch = self.char()
            if self.at == start:
                ok = ch == "_" or ch.isalpha()
            else:
                ok = ch in ("_", ".", "[", "]") or ch.isalnum()
            if not ok:
                raise _RustSyntaxError
            self.at += 1
        name = self.text[start:self.at]
        self.at += 1
        if not name or name in self.names:
            raise _RustSyntaxError
        self.names.add(name)

    def repetition(self) -> None:
        """Read `?`, `*`, `+`, or a counted `{...}`, and a lazy `?`."""
        if self.text[self.at] != "{":
            self.at += 1
            if self.char() == "?":
                self.at += 1
            return
        if not self.bump():
            raise _RustSyntaxError
        low = self.decimal()
        high: int | None = low
        self.fail_at_end()
        if self.text[self.at] == ",":
            if not self.bump():
                raise _RustSyntaxError
            high = None if self.text[self.at] == "}" else self.decimal()
        if self.char() != "}":
            raise _RustSyntaxError
        if self.bump() and self.char() == "?":
            self.at += 1
        if high is not None and low > high:
            raise _RustSyntaxError

    def decimal(self) -> int:
        """Read a repetition count, with whitespace around it."""
        while not self.done() and self.text[self.at].isspace():
            self.at += 1
        digits = ""
        while self.char() in _DECIMAL_CHARS:
            digits += self.text[self.at]
            self.bump()
        while not self.done() and self.text[self.at].isspace():
            self.bump()
        if not digits or int(digits) > 0xFFFFFFFF:
            raise _RustSyntaxError
        return int(digits)

    def escape(self) -> tuple[str, int]:
        """Read an escape, and return what it is (`literal`, `class`, or
        `assertion`) and a literal's code point."""
        self.at += 1
        ch = self.char()
        if ch in _RUST_HEX_DIGITS:
            return "literal", self.hex_escape(ch)
        if ch in ("p", "P"):
            self.unicode_class()
            return "class", 0
        self.fail_at_end()
        self.at += 1
        if ch in _RUST_CLASS_ESCAPES:
            return "class", 0
        if ch in _RUST_META or (
            ch.isascii() and not ch.isalnum() and ch not in ("<", ">")
        ):
            return "literal", ord(ch)
        if ch in _RUST_CONTROL_ESCAPES:
            return "literal", _RUST_CONTROL_ESCAPES[ch]
        if ch == "b" and self.char() == "{":
            self.word_boundary()
        if ch in _RUST_ASSERTION_ESCAPES:
            return "assertion", 0
        raise _RustSyntaxError

    def hex_escape(self, kind: str) -> int:
        """Read the digits of `\\x`, `\\u`, or `\\U`, with or without braces."""
        if not self.bump():
            raise _RustSyntaxError
        digits = ""
        if self.text[self.at] == "{":
            while self.bump() and self.text[self.at] != "}":
                if self.text[self.at] not in _HEX_CHARS:
                    raise _RustSyntaxError
                digits += self.text[self.at]
            self.fail_at_end()
            self.bump()
            if not digits:
                raise _RustSyntaxError
        else:
            for k in range(_RUST_HEX_DIGITS[kind]):
                if k and not self.bump():
                    raise _RustSyntaxError
                if self.text[self.at] not in _HEX_CHARS:
                    raise _RustSyntaxError
                digits += self.text[self.at]
            self.bump()
        value = int(digits, 16)
        if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
            raise _RustSyntaxError
        return value

    def unicode_class(self) -> None:
        """Read the name after `\\p` or `\\P`: one letter, or `{name}`."""
        if not self.bump():
            raise _RustSyntaxError
        if self.text[self.at] != "{":
            letter = self.text[self.at]
            self.bump()
            if letter.lower() not in _RUST_ONE_LETTER_CLASSES:
                raise _RustSyntaxError
            return
        name = ""
        while self.bump() and self.text[self.at] != "}":
            name += self.text[self.at]
        self.fail_at_end()
        self.at += 1
        if not _rust_property_name(name):
            raise _RustSyntaxError

    def word_boundary(self) -> None:
        """Read the `{name}` after `\\b`. When its first character can't
        start a name, as in `\\b{2}`, the `{` opens a counted repetition
        instead, so this reads nothing."""
        start = self.at

        def name_char(c: str) -> bool:
            return c == "-" or (c.isascii() and c.isalpha())

        if not self.bump():
            raise _RustSyntaxError
        if not name_char(self.text[self.at]):
            self.at = start
            return
        name = ""
        while not self.done() and name_char(self.text[self.at]):
            name += self.text[self.at]
            self.bump()
        if self.char() != "}" or name not in _RUST_WORD_BOUNDARIES:
            raise _RustSyntaxError
        self.at += 1

    def char_class(self) -> None:
        """Read a bracketed class, with the classes nested in it."""
        depth = 0
        self.class_open()
        while True:
            self.skip_space()
            self.fail_at_end()
            ch = self.text[self.at]
            if ch == "[":
                ascii_class = _RUST_ASCII_CLASS.match(self.text, self.at)
                if (
                    ascii_class is not None
                    and ascii_class.group(1) in _RUST_ASCII_CLASSES
                ):
                    self.at = ascii_class.end()
                else:
                    self.class_open()
                    depth += 1
            elif ch == "]":
                self.at += 1
                if depth == 0:
                    return
                depth -= 1
            elif ch in ("&", "-", "~") and self.char(1) == ch:
                self.at += 2
            else:
                self.class_range()

    def class_open(self) -> None:
        """Read a class's `[`, a `^`, and the `-` or `]` a class can start
        with as a literal."""
        if not self.bump():
            raise _RustSyntaxError
        if self.text[self.at] == "^" and not self.bump():
            raise _RustSyntaxError
        dashes = False
        while self.text[self.at] == "-":
            dashes = True
            if not self.bump():
                raise _RustSyntaxError
        if not dashes and self.text[self.at] == "]" and not self.bump():
            raise _RustSyntaxError

    def class_range(self) -> None:
        """Read one item of a class, or a range of two literals."""
        kind, low = self.class_item()
        self.skip_space()
        self.fail_at_end()
        if self.text[self.at] != "-" or self.peek_space() in ("]", "-"):
            if kind == "assertion":
                raise _RustSyntaxError
            return
        if not self.bump():
            raise _RustSyntaxError
        high_kind, high = self.class_item()
        if kind != "literal" or high_kind != "literal" or low > high:
            raise _RustSyntaxError

    def class_item(self) -> tuple[str, int]:
        """Read one escape or character in a class."""
        if self.text[self.at] == "\\":
            return self.escape()
        self.at += 1
        return "literal", ord(self.text[self.at - 1])


def _rust_regex_rejects(pattern: str) -> bool:
    r"""True when Rust's `regex` crate rejects a pattern.

    `_RustRegexCheck` reads the pattern the way regex 1.12.3, the version
    openai/codex pins at c0cc9fc82, parses it. Rust's `regex` has no
    look-around and no backreferences, so it rejects `(?=`, `(?<!`, `\1`,
    and `(?P=name)`, and it rejects a `{` that doesn't start a counted
    repetition, such as `x{` or `x{,3}`, where Python's `re` reads a
    literal. A `[` inside a class opens a nested class, so `[[]` is
    unclosed. Three checks that come after parsing are left out: a
    `\p{name}` that no Unicode property has, what `(?-u)` rules out, and
    the compiled size limit, which `\w{300}` passes.
    """
    try:
        _RustRegexCheck(pattern).check()
    except _RustSyntaxError:
        return True
    return False


def _python_regex(pattern: str) -> re.Pattern[str] | None:
    """A matcher compiled with Python's `re`, or None when `re` rejects it.

    Rust's `regex` compiles some patterns that `re` rejects, such as
    `(?<name>...)`, `\\p{L}`, and `(?i)` after the start of a pattern.
    """
    with warnings.catch_warnings():
        # `[[:alpha:]]` compiles with a FutureWarning about nested sets.
        warnings.simplefilter("ignore")
        try:
            return re.compile(pattern)
        except re.error:
            return None


def _regex_matcher(event: str, group: object) -> str | None:
    """A group's matcher when Codex reads it as a regex, or None.

    Codex reads the matcher on the events in `_MATCHER_EVENTS` only. An
    empty matcher, `*`, and a list of exact names aren't regexes.
    """
    if event not in _MATCHER_EVENTS or not isinstance(group, dict):
        return None
    matcher = cast(dict[str, object], group).get("matcher")
    if (
        not isinstance(matcher, str) or matcher in ("", "*")
        or _EXACT_MATCHER.fullmatch(matcher) is not None
    ):
        return None
    return matcher


def _unread_matcher(event: str, group: object) -> str | None:
    """A group's matcher when doctor can't read it and so counts the group
    as matching every fire, or None.

    That is a regex that Rust's `regex` compiles and Python's `re`
    rejects, in an event whose values doctor lists (`_MATCHER_VALUES`).
    """
    matcher = _regex_matcher(event, group)
    if (
        matcher is None or event not in _MATCHER_VALUES
        or _rust_regex_rejects(matcher)
    ):
        return None
    return matcher if _python_regex(matcher) is None else None


def _matched_values(event: str, group: object) -> frozenset[str] | None:
    """The values of `event` whose fires run a matcher group's handlers.

    This follows Codex's matcher (codex-rs/hooks/src/engine/matcher.rs in
    openai/codex; the docs at https://developers.openai.com/codex/hooks say
    only that `matcher` is a regex string, and that `"*"`, `""`, or no
    `matcher` matches every occurrence). A matcher of only letters, digits,
    `_`, and `|` lists exact values, so `startup|resume` matches those two
    and `start` matches none. Any other matcher is a regex that matches a
    value when it matches anywhere in it, so `^res` matches `resume`.

    Codex compiles a regex matcher with Rust's `regex` crate
    (`HookMatcher::new` in codex-rs/hooks/src/engine/matcher.rs at
    openai/codex@c0cc9fc82). When that fails, `append_matcher_groups` in
    codex-rs/hooks/src/engine/discovery.rs logs "invalid matcher" and goes
    on to the next group, so none of the group's handlers run. This is None
    for such a group, which `_rust_regex_rejects` finds. Doctor matches the
    values with Python's `re`, and a regex that `re` rejects counts as
    matching every value (`_unread_matcher`), as does a matcher that isn't
    a string, so a pattern doctor can't read never hides a duplicate run.
    """
    if event not in _MATCHER_EVENTS:
        return _EVERY_FIRE
    matcher = (
        cast(dict[str, object], group).get("matcher")
        if isinstance(group, dict) else None
    )
    if not isinstance(matcher, str) or matcher in ("", "*"):
        return frozenset(_MATCHER_VALUES.get(event, _EVERY_FIRE))
    exact = _EXACT_MATCHER.fullmatch(matcher) is not None
    if not exact and _rust_regex_rejects(matcher):
        return None
    if exact and event in _TOOL_EVENTS:
        return frozenset(name for name in matcher.split("|") if name)
    values = _MATCHER_VALUES.get(event)
    if values is None:
        return _EVERY_FIRE
    if exact:
        return frozenset(values) & frozenset(matcher.split("|"))
    pattern = _python_regex(matcher)
    if pattern is None:
        return frozenset(values)
    return frozenset(v for v in values if pattern.search(v))


def _group_counts(group: object, *, windows: bool | None = None) -> bool:
    """True when a group holds an aelfrice handler, or a foreign handler
    that names an aelfrice program, which doctor warns about."""
    return any(
        _handler_is_owned(handler, windows=windows)
        or _named_owned_keys(handler, windows=windows)
        for handler in _handlers_in(group)
    )


def _fires(event: str, value_sets: list[frozenset[str]]) -> list[frozenset[int]]:
    """The handlers that each fire of `event` runs, as indexes into
    `value_sets`.

    Each set holds the values one handler runs on (`_matched_values`), and
    a fire runs a handler when the fire's values and the handler's share
    one. A fire has one value, except a tool event's: a tool call is
    matched by the tool's name and its aliases (`_TOOL_MATCHER_INPUTS`), so
    handlers under `Write` and under `Edit` both run on an `apply_patch`
    call. In a tool event, a handler under a regex or no matcher counts as
    running on the same fire as every other handler, because doctor can't
    list the tool names a regex matches.
    """
    if event in _TOOL_EVENTS and _EVERY_FIRE in value_sets:
        return [frozenset(i for i, values in enumerate(value_sets) if values)]
    inputs = [frozenset({v}) for v in sorted(set[str]().union(*value_sets))]
    if event in _TOOL_EVENTS:
        inputs.extend(_TOOL_MATCHER_INPUTS)
    return [
        frozenset(i for i, values in enumerate(value_sets) if values & names)
        for names in inputs
    ]


def _most_runs(event: str, value_sets: list[frozenset[str]]) -> int:
    """How many times one fire of `event` runs a program, at the fire that
    runs it most.

    Each set holds the values one handler of the program runs on, so two
    handlers run on the same fire only when `_fires` puts them on one.
    """
    return max((len(fire) for fire in _fires(event, value_sets)), default=0)


def _program_path(cmd: str, key: str, *, windows: bool | None = None) -> str:
    """The program path of an owned handler's command, without arguments.

    The SessionStart command ends in `--host codex` (#1812), and doctor
    names a missing program by its path alone. The path is the shortest
    run of leading words whose key is `key`, so an unquoted path with a
    space in it stays whole.
    """
    tokens = command_tokens(cmd, windows=windows)
    for k in range(1, len(tokens) + 1):
        candidate = " ".join(tokens[:k])
        if launcher_key(candidate, windows=windows) == key:
            return candidate
    return program_token(cmd, windows=windows)


def doctor_codex(
    codex_dir: Path | None = None, *, windows: bool | None = None,
) -> CodexDoctorReport:
    """Scan the Codex host: hooks.json shape, coverage, flag, trust.

    Read-only. Reports rather than raises; the CLI decides exit codes.
    """
    cdir = codex_dir if codex_dir is not None else resolve_codex_home()
    win = os.name == "nt" if windows is None else windows
    report = CodexDoctorReport(
        codex_dir_present=cdir.is_dir(), codex_dir=cdir,
    )
    if not report.codex_dir_present:
        report.warnings.append(f"{cdir} not found — Codex not installed?")
        return report

    hooks_path = codex_hooks_path(cdir)
    report.hooks_file_present = hooks_path.is_file()
    hooks_map: dict[str, object] = {}
    if report.hooks_file_present:
        try:
            parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                report.hooks_file_valid = True
                obj = cast(dict[str, object], parsed).get("hooks")
                if isinstance(obj, dict):
                    hooks_map = cast(dict[str, object], obj)
            else:
                report.parse_error = "top-level JSON is not an object"
        except json.JSONDecodeError as exc:
            report.parse_error = str(exc)
        if not report.hooks_file_valid:
            report.warnings.append(
                f"{hooks_path} is unreadable as hooks config "
                f"({report.parse_error}); Codex will ignore or reject it",
            )

    # Coverage is compared per *handler*, not per event. An event counted as
    # covered the moment one owned handler appeared in it, so deleting the
    # transcript-logger handler out of `UserPromptSubmit` while `aelf-hook`
    # remained left `missing_events` empty and doctor silent — a partial
    # removal, which is exactly the state #1430 is about.
    expected_keys: dict[str, set[str]] = {}
    # #1812: each command setup writes now, in every scope it supports,
    # mapped to its program's key. A handler counts as a run of a program
    # only when its command is one of these, byte for byte.
    setup_runs: dict[str, str] = {}
    for scope in _SETUP_SCOPES:
        for event, groups in desired_codex_hooks(scope).items():
            keys = expected_keys.setdefault(event, set())
            for group in groups:
                for handler in cast(list[object], group.get("hooks", [])):
                    cmd = _handler_command(handler)
                    if cmd is None:
                        continue
                    key = (
                        _owned_command_key(cmd, windows=windows)
                        or command_launcher_key(cmd, windows=windows)
                    )
                    keys.add(key)
                    setup_runs[cmd] = key
    expected_events = set(expected_keys)
    covered: set[str] = set()
    found_keys: dict[str, set[str]] = {}
    # The values each run of a program runs on, per event (#1812).
    runs: dict[tuple[str, str], list[frozenset[str]]] = {}
    # The values and the command of each other handler that names a
    # program, per event.
    maybe: dict[tuple[str, str], list[tuple[frozenset[str], str]]] = {}
    for event, groups in hooks_map.items():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            # #1812: handlers whose matchers share no value, such as tool
            # groups under `Bash` and under `WebSearch`, never run on the
            # same fire, so they don't count as duplicate runs.
            values = _matched_values(event, group)
            # #1812: name a skipped or unread matcher only where it changes
            # what doctor counts.
            if values is None and _group_counts(group, windows=windows):
                matcher = cast(dict[str, object], group).get("matcher")
                report.warnings.append(
                    f"{event} group with matcher {matcher!r} never runs: "
                    "Codex logs \"invalid matcher\" for it and skips the "
                    "group, so doctor doesn't count its handlers as runs",
                )
            unread = _unread_matcher(event, group)
            if unread is not None and _group_counts(group, windows=windows):
                report.warnings.append(
                    f"{event} group with matcher {unread!r}: doctor can't "
                    "read this matcher and counted it as matching every "
                    "fire",
                )
            # #1812: a handler counts as a run of a program only when the
            # command Codex runs for it on this platform is one setup
            # writes, byte for byte. Doctor reads no shell syntax: any
            # other handler whose command fields name an aelfrice program
            # may run it, so doctor warns about it.
            for handler in _handlers_in(group):
                if values is None:
                    continue
                command = _run_command(handler, windows=win)
                run = setup_runs.get(command, "") if command is not None else ""
                if run:
                    runs.setdefault((event, run), []).append(values)
                named = _named_owned_keys(handler, windows=win)
                for key in sorted(named.keys() - {run}):
                    maybe.setdefault((event, key), []).append(
                        (values, named[key]),
                    )
            # #1412: count per handler. Gating on a fully-owned group made
            # an aelfrice handler sharing a group with a foreign one
            # invisible to doctor — the same blind spot that let setup
            # duplicate it.
            owned = _owned_handlers_in(group, windows=windows)
            if not owned:
                continue
            covered.add(event)
            matcher = cast(dict[str, object], group).get("matcher")
            if (
                event == "SessionStart"
                and isinstance(matcher, str)
                and matcher in _STALE_SESSION_START_MATCHERS
                and matcher not in report.stale_matchers
            ):
                report.stale_matchers.append(str(matcher))
            for handler in owned:
                report.owned_handler_count += 1
                hd = cast(dict[str, object], handler)
                cmd = hd.get("command")
                if isinstance(cmd, str):
                    own_key = (
                        _owned_command_key(cmd, windows=windows)
                        or command_launcher_key(cmd, windows=windows)
                    )
                    found_keys.setdefault(event, set()).add(own_key)
                    exe = Path(program_token(cmd, windows=windows))
                    path = _program_path(cmd, own_key, windows=windows)
                    if (
                        exe.is_absolute()
                        and not program_exists(cmd, windows=windows)
                        and path not in report.stale_commands
                    ):
                        report.stale_commands.append(path)
    report.missing_events = sorted(expected_events - covered)
    report.duplicate_runs = [
        (event, key, count)
        for (event, key), value_sets in sorted(runs.items())
        if (count := _most_runs(event, value_sets)) > 1
    ]
    for (event, key), others in sorted(maybe.items()):
        # #1812: a handler that names the program gets a warning only when
        # some fire that runs it also runs another handler of the program,
        # the rule the failure above follows.
        counted = runs.get((event, key), [])
        fires = _fires(event, counted + [values for values, _cmd in others])
        wrote = " or ".join(
            f"`{c}`" for c, k in sorted(setup_runs.items()) if k == key
        )
        for index, (_values, cmd) in enumerate(others, start=len(counted)):
            if not any(index in fire and len(fire) > 1 for fire in fires):
                continue
            warning = (
                f"{event} handler may also run {key}. Doctor counts a run "
                "only for a command byte-identical to one `aelf setup "
                f"--host codex` writes ({wrote or key}). If this handler "
                f"runs {key}, keep setup's handler and delete this one: {cmd}"
            )
            if warning not in report.warnings:
                report.warnings.append(warning)
    report.missing_handlers = sorted(
        f"{event}:{key}"
        for event in sorted(expected_keys)
        if event in covered
        for key in sorted(expected_keys[event] - found_keys.get(event, set()))
    )
    for cmd in report.stale_commands:
        report.warnings.append(f"hook command not found on disk: {cmd}")
    if report.missing_events:
        # #1412: this used to be gated on `owned_handler_count and ...`, so
        # the *worst* state — a hooks.json present and valid with zero
        # recognised aelfrice handlers — produced no warning at all. That is
        # exactly what a Windows user saw: seven duplicated groups on disk,
        # doctor reporting nothing wrong.
        if report.owned_handler_count:
            report.warnings.append(
                "aelfrice hook coverage incomplete; missing events: "
                + ", ".join(report.missing_events),
            )
        elif report.hooks_file_present and report.hooks_file_valid:
            report.warnings.append(
                "no aelfrice hook handlers recognised in "
                f"{hooks_path}; expected events are all missing "
                "(run `aelf setup --host codex`)",
            )
        elif not report.hooks_file_present:
            # Codex is installed but has no hooks.json at all. Without this
            # the report listed every expected event as missing and said
            # nothing about why — the same silence the branch above was added
            # to remove, one state over. This stays a warning, not a failure:
            # never having run `aelf setup --host codex` is a legitimate
            # state, and exiting nonzero on it would fail doctor on every
            # machine that does not use the Codex host.
            report.warnings.append(
                f"{hooks_path} does not exist; no aelfrice hooks are "
                "installed for the Codex host "
                "(run `aelf setup --host codex`)",
            )

    config_path = codex_config_path(cdir)
    if config_path.is_file():
        try:
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            report.warnings.append(f"could not parse {config_path}: {exc}")
            cfg = {}
        # Codex 0.145+ names the feature `hooks` (stage: stable, enabled
        # by default). Legacy 0.11x–0.12x named it `codex_hooks` (under
        # development, off by default). A default-on feature is absent from
        # config.toml when left at its default, so absence must read as ON,
        # not off. Honour an explicit setting under either key; treat a
        # parsed-but-unmentioned feature as the current default (on).
        hooks_flag: bool | None = None
        features = cfg.get("features")
        if isinstance(features, dict):
            fdict = cast(dict[str, object], features)
            if "hooks" in fdict:
                hooks_flag = fdict.get("hooks") is True
            elif "codex_hooks" in fdict:
                hooks_flag = fdict.get("codex_hooks") is True
        report.feature_flag_on = True if hooks_flag is None else hooks_flag
        hooks_cfg = cfg.get("hooks")
        if isinstance(hooks_cfg, dict):
            state = cast(dict[str, object], hooks_cfg).get("state")
            if isinstance(state, dict):
                for entry in cast(dict[str, object], state).values():
                    # Key-membership only — the approval digest value
                    # itself is never read, held, or logged.
                    if isinstance(entry, dict) and "trusted_hash" in entry:
                        report.approved_state_count += 1
    if report.feature_flag_on is False:
        report.warnings.append(
            "the Codex `hooks` feature is disabled in config.toml "
            "([features].hooks = false) — Codex will not run any hooks; "
            "remove that line or set it true (`hooks` is stable and on "
            "by default)",
        )
    # Approval-state keying is positional today and slated to change
    # upstream (per-handler keys vs per-group digests), so exact
    # count arithmetic would false-positive on multi-handler groups.
    # Warn only on the unambiguous condition: handlers configured,
    # zero approvals recorded.
    if report.owned_handler_count and report.approved_state_count == 0:
        report.warnings.append(
            f"{report.owned_handler_count} aelfrice handler(s) configured "
            "but no approved [hooks.state] entries exist — unapproved "
            "hooks are silently skipped; run /hooks in a Codex session "
            "to approve them",
        )
    return report
