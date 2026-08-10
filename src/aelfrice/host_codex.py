"""Codex host target for setup / doctor / uninstall (#1052).

Writes, validates, and removes the aelfrice hook set in Codex's
``~/.codex/hooks.json``. Design constraints established in #1054/#1052
triage:

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
  ``~/.codex/hooks.json`` has been observed holding truncated JSON, and
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
  unchanged. The ``Grep|Glob`` search hook is excluded (no such tools
  exist on Codex; greps arrive via Bash and are covered by the Bash
  matcher), as is the host-specific memory mirror.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from aelfrice.launcher import command_launcher_key, owned_keys, program_token
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
_SESSION_START_MATCHER: Final[str] = "startup|resume|clear|compact"

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


def _handler(command: str, *, timeout: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"type": "command", "command": command}
    if timeout is not None:
        entry["timeout"] = timeout
    return entry


def desired_codex_hooks(scope: SettingsScope = "user") -> dict[str, list[dict[str, object]]]:
    """The aelfrice hook set in Codex hooks.json shape, keyed by event.

    Commands resolve to absolute paths via the same resolvers the Claude
    installers use, so both hosts pin identical executables.
    """
    hook_cmd = resolve_hook_command(scope)
    logger_cmd = resolve_transcript_logger_command(scope)
    session_cmd = resolve_session_start_hook_command(scope)
    stop_cmd = resolve_stop_hook_command(scope)
    search_bash_cmd = resolve_search_tool_bash_command(scope)
    pre_issue_cmd = resolve_pre_issue_guard_command(scope)
    commit_cmd = resolve_commit_ingest_command(scope)
    return {
        "UserPromptSubmit": [
            {"hooks": [_handler(hook_cmd), _handler(logger_cmd)]},
        ],
        "Stop": [
            {"hooks": [_handler(logger_cmd), _handler(stop_cmd)]},
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
                "hooks": [_handler(session_cmd)],
            },
        ],
        # #1055: Codex reports shell commands as tool_name "Bash", so the
        # Bash-matcher hooks are host-portable verbatim. Grep|Glob is
        # omitted — those tools do not exist on Codex.
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [_handler(search_bash_cmd), _handler(pre_issue_cmd)],
            },
        ],
        "PostToolUse": [
            {"matcher": "Bash", "hooks": [_handler(commit_cmd)]},
        ],
    }


def _command_basename(handler: object, *, windows: bool | None = None) -> str:
    """Ownership key of a handler's command, '' on shape miss.

    #1412: platform-gated. On Windows the key is case-folded with the
    launcher suffix removed, so ``...\\Scripts\\aelf-hook.EXE`` compares
    equal to ``aelf-hook``. On POSIX it is the plain basename, unchanged.
    """
    if not isinstance(handler, dict):
        return ""
    hd = cast(dict[str, object], handler)
    cmd = hd.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return ""
    return command_launcher_key(cmd, windows=windows)


def _handler_is_owned(handler: object, *, windows: bool | None = None) -> bool:
    """True iff a single handler is one of ours."""
    key = _command_basename(handler, windows=windows)
    return bool(key) and key in owned_keys(_OWNED_BASENAMES, windows=windows)


def _owned_handlers_in(
    group: object, *, windows: bool | None = None,
) -> list[object]:
    """Every aelfrice handler inside a matcher group, in file order."""
    if not isinstance(group, dict):
        return []
    gd = cast(dict[str, object], group)
    handlers = gd.get("hooks")
    if not isinstance(handlers, list):
        return []
    return [
        h for h in cast(list[object], handlers)
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
                key = _command_basename(handler, windows=windows)
                if key.startswith("aelf-"):
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
            pass
        raise
    return True


def _plan_install(
    text: str | None, scope: SettingsScope, force: bool, hooks_path: Path,
    *, windows: bool | None = None,
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

    hooks_obj = existing.get("hooks")
    if hooks_obj is None or force:
        hooks_map: dict[str, object] = (
            cast(dict[str, object], hooks_obj)
            if isinstance(hooks_obj, dict) else {}
        )
    elif isinstance(hooks_obj, dict):
        hooks_map = cast(dict[str, object], hooks_obj)
    else:
        return refuse(
            "existing hooks.json has a non-object `hooks` value "
            f"({type(hooks_obj).__name__}); it is not ours to reshape — "
            "fix it, or re-run with --force to replace it"
        )

    desired = desired_codex_hooks(scope)
    if not force:
        for event in desired:
            current = hooks_map.get(event)
            if current is not None and not isinstance(current, list):
                return refuse(
                    f"existing hooks.json has a non-list value for the "
                    f"`{event}` event ({type(current).__name__}); it is not "
                    "ours to reshape — fix it, or re-run with --force to "
                    "replace it"
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
) -> CodexInstallResult:
    """Write the aelfrice hook set into ``hooks_path``, merge-aware.

    Serialised against other aelfrice writers and committed atomically;
    see the transaction notes above. Refuses (with ``error`` set) when
    the existing file is unparseable, or holds a foreign structure at a
    key we would have to reshape, and ``force`` is False.
    """
    return _commit_hooks_transaction(
        hooks_path,
        lambda text: _plan_install(
            text, scope, force, hooks_path, windows=windows,
        ),
    )


def remove_codex_hooks(
    hooks_path: Path, *, windows: bool | None = None,
) -> CodexInstallResult:
    """Remove aelfrice-owned matcher groups; drop emptied events.

    Same transaction as ``install_codex_hooks`` — an uninstall racing a
    setup is exactly as capable of dropping a foreign entry as two
    setups are.
    """
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
# instead of ``~/.codex/hooks.json`` + the ``$aelf-*`` skills. Their
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
    "Install the aelfrice hooks in ~/.codex/hooks.json and the $aelf-* "
    "agent skills under ~/.agents/skills/ on this host."
)

_HOST_MANAGEMENT_NOTE: Final[str] = (
    "<host-adapter>\n"
    "IMPORTANT — on this host, every `aelf setup`, `aelf doctor`,\n"
    "`aelf unsetup`, or `aelf uninstall` invocation in the steps below\n"
    "MUST use the `--host codex` form (e.g. `uv run aelf setup --host "
    "codex`,\n"
    "`uv run aelf doctor --host codex`, `uv run aelf unsetup --host "
    "codex`,\n"
    "`uv run aelf uninstall <flags> --host codex`). The bare form\n"
    "targets another host's configuration — it would not touch this\n"
    "host's install and must not be run here.\n"
    "</host-adapter>"
)


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
        "Run each `uv run aelf ...` command in your shell and show its "
        "output to the user."
    )

    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {description}",
        "---",
        *adapter,
        "",
        body.rstrip("\n"),
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
    for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
        if child.name in bundle:
            continue
        if _is_owned_skill_dir(child):
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
    stale_commands: list[str] = field(default_factory=list[str])
    feature_flag_on: bool | None = None
    approved_state_count: int = 0
    warnings: list[str] = field(default_factory=list[str])


def doctor_codex(
    codex_dir: Path | None = None, *, windows: bool | None = None,
) -> CodexDoctorReport:
    """Scan the Codex host: hooks.json shape, coverage, flag, trust.

    Read-only. Reports rather than raises; the CLI decides exit codes.
    """
    cdir = codex_dir if codex_dir is not None else resolve_codex_home()
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

    expected_events = set(desired_codex_hooks().keys())
    covered: set[str] = set()
    for event, groups in hooks_map.items():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            # #1412: count per handler. Gating on a fully-owned group made
            # an aelfrice handler sharing a group with a foreign one
            # invisible to doctor — the same blind spot that let setup
            # duplicate it.
            owned = _owned_handlers_in(group, windows=windows)
            if not owned:
                continue
            covered.add(event)
            for handler in owned:
                report.owned_handler_count += 1
                hd = cast(dict[str, object], handler)
                cmd = hd.get("command")
                if isinstance(cmd, str):
                    exe = Path(program_token(cmd, windows=windows))
                    if (
                        exe.is_absolute()
                        and not exe.exists()
                        and cmd not in report.stale_commands
                    ):
                        report.stale_commands.append(cmd)
    report.missing_events = sorted(expected_events - covered)
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
