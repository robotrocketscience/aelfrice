"""First-run / post-upgrade hook installer driven by a bundled manifest.

Closes the loop on #623: a bare `pipx upgrade aelfrice` does not re-run
`aelf setup`, so any default-on hook added in a new release (e.g.
`aelf-stop-hook` shipped in v3.0) is missing for users who never re-run
setup. The catch-net (`aelf doctor` nag) only fires when the user runs
doctor — passive users never see it.

Design property: the *first* `aelf <cmd>` invocation after the installed
package version exceeds the stamped version merges any new manifest
entries into ``~/.claude/settings.json``. The merge:

* Is gated on a single-stat version-stamp check — happy-path overhead is
  one file read after the first merge of a given version.
* Reuses the existing tested install functions in `aelfrice.setup`, so
  the on-disk shape of settings.json is byte-identical to what
  `aelf setup` would write today.
* Honors user-set opt-outs (``aelf setup --no-transcript-ingest`` writes
  to a sibling opt-out file; this module reads it and respects it on
  every subsequent upgrade).
* Honors ``AELFRICE_NO_AUTO_INSTALL=1`` as a hard bypass for power users
  who manage their settings.json by hand.
* Acquires an exclusive ``flock`` on the stamp file during the merge so
  two concurrent `aelf` processes cannot race on the JSON write.

What this module deliberately does NOT do:

* It does not run on `pip`/`pipx`/`uv tool` install. Modifying user
  config silently at package-install time is the hostile pattern Python
  packaging deliberately rejected (see PEP 668 era discussions). The
  user-consent boundary is "the user just ran an ``aelf`` command and
  expects it to work."
* It does not migrate per-project DB schemas (#593) or prune dormant
  DBs (#594) — those have their own confirmation models.
* It does not touch hooks the user *added* to settings.json. The merge
  is additive within the basenames the manifest claims; everything else
  is byte-preserved.
"""
from __future__ import annotations

import fcntl
import importlib.resources
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, cast

from aelfrice import setup as _setup
from aelfrice.setup import (
    SettingsScope,
    SETTINGS_LOCK_TIMEOUT_BACKGROUND,
    install_agent_context_hook,
    install_claude_memory_mirror_hook,
    install_commit_ingest_hook,
    install_pre_issue_guard_hook,
    install_search_tool_bash_hook,
    install_search_tool_hook,
    install_session_start_hook,
    install_stop_hook,
    install_transcript_ingest_hooks,
    install_user_prompt_submit_hook,
    resolve_agent_context_command,
    resolve_claude_memory_mirror_command,
    resolve_commit_ingest_command,
    resolve_hook_command,
    resolve_pre_issue_guard_command,
    resolve_search_tool_bash_command,
    resolve_search_tool_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
    settings_transaction,
)

NO_AUTO_INSTALL_ENV: Final[str] = "AELFRICE_NO_AUTO_INSTALL"
AELFRICE_DOTDIR: Final[Path] = Path.home() / ".aelfrice"
STAMP_PATH: Final[Path] = AELFRICE_DOTDIR / "installed-manifest-version"
OPT_OUT_PATH: Final[Path] = AELFRICE_DOTDIR / "opt-out-hooks.json"

# Advisory lock serialising concurrent manifest merges. A filename rather
# than a full path because it is created beside whichever `stamp_path` the
# caller passes, not unconditionally in `AELFRICE_DOTDIR`. Named here so
# the uninstall removal set can be single-sourced against it (#1186).
AUTO_INSTALL_LOCK_FILENAME: Final[str] = ".auto-install.lock"

_MANIFEST_PACKAGE: Final[str] = "aelfrice"
_MANIFEST_SUBDIR: Final[str] = "data"
_MANIFEST_FILENAME: Final[str] = "hook_manifest.json"

# Sentinel version for "no stamp on disk yet" — older than any real release.
_UNSTAMPED: Final[str] = "0.0.0"


@dataclass(frozen=True)
class HookEntry:
    """One row of the bundled manifest."""
    name: str
    basename: str
    installer: str
    default_on: bool
    since: str
    description: str
    timeout: int | None = None
    """Host-enforced wall-clock bound, in seconds, for this hook entry.

    #1161: every installed entry used to omit `timeout`, so the "never
    block the user's prompt" contract had no host-level enforcement at
    all — a hook wedged on a sibling's SQLite write lock stalled the
    prompt for the *host's* default, not aelfrice's intended budget.
    The bound lives here rather than in the installer so there is one
    authority per hook, `aelf doctor` can compare what is installed
    against what is declared, and changing a budget is a data edit.

    Optional in the loader so a newer manifest stays readable by an
    older aelfrice (and vice versa): `None` means "install no timeout
    key", which is the pre-#1161 behaviour.
    """


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    hooks: tuple[HookEntry, ...]

    def owned_basenames(self) -> frozenset[str]:
        return frozenset(h.basename for h in self.hooks)


@dataclass(frozen=True)
class AutoInstallResult:
    """Outcome of a `maybe_install_manifest` call.

    `ran` is True iff a merge actually ran (stamp updated, possibly
    settings.json updated). `installed` lists hook names that were
    freshly added. `already` lists hook names whose entries were
    already present (idempotent no-op). `opted_out` lists hook names
    skipped because the opt-out file names them. `prev_version` is the
    stamp value found on disk before the merge ("0.0.0" if absent).
    `new_version` is the package version that produced this merge.
    `message` is a single short stderr line; empty when there is
    nothing user-visible to report.
    """
    ran: bool
    prev_version: str
    new_version: str
    installed: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    opted_out: tuple[str, ...] = ()
    message: str = ""


# --- manifest loading ----------------------------------------------------


def manifest_timeouts_by_installer() -> dict[str, int | None]:
    """Map each manifest installer key to its declared timeout.

    The `aelf setup` path (`cli._cmd_setup`) calls the installer
    functions directly rather than dispatching through the manifest, so
    it needs the budgets keyed the way it addresses them. Keeping this
    beside `load_manifest` means `setup` and `auto_install` cannot drift
    to different budgets for the same hook (#1161).

    Fail-soft: returns `{}` if the manifest is unreadable or malformed,
    which degrades to the pre-#1161 "no timeout key" behaviour rather
    than breaking `aelf setup` on a packaging error.
    """
    try:
        manifest = load_manifest()
    except (OSError, ValueError):
        return {}
    return {h.installer: h.timeout for h in manifest.hooks}


def _parse_timeout(raw: object) -> int | None:
    """Coerce a manifest `timeout` cell to a positive int, else None.

    Fail-soft by design: a malformed or non-positive timeout degrades to
    "no timeout key installed" (the pre-#1161 behaviour) rather than
    raising, because `load_manifest` runs on the auto-install path that
    every `aelf` invocation touches. A manifest typo must not brick the
    CLI. `_manifest_declares_timeouts_for_default_on_hooks` in the test
    suite is what actually holds the bundled manifest to declaring one
    for every default-on hook.

    Rejects `bool` explicitly: `True` is an `int` in Python and would
    otherwise install `"timeout": true`, which the host would reject.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw > 0 else None


def load_manifest() -> Manifest:
    """Read the bundled hook_manifest.json from the wheel.

    Uses importlib.resources so it works as an editable install, a wheel,
    or a plain source tree. Validates schema_version and required fields;
    raises ValueError on malformed input.
    """
    pkg = importlib.resources.files(_MANIFEST_PACKAGE).joinpath(
        _MANIFEST_SUBDIR, _MANIFEST_FILENAME
    )
    raw = pkg.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("hook_manifest.json must be a JSON object")
    schema = parsed.get("schema_version")
    if schema != 1:
        raise ValueError(
            f"unsupported hook_manifest schema_version: {schema!r} "
            f"(this aelfrice supports schema_version=1)"
        )
    hooks_raw = parsed.get("hooks", [])
    if not isinstance(hooks_raw, list):
        raise ValueError("hook_manifest.json 'hooks' must be a list")
    hooks: list[HookEntry] = []
    for row in hooks_raw:
        if not isinstance(row, dict):
            raise ValueError("each hook entry must be a JSON object")
        try:
            hooks.append(HookEntry(
                name=str(row["name"]),
                basename=str(row["basename"]),
                installer=str(row["installer"]),
                default_on=bool(row["default_on"]),
                since=str(row["since"]),
                description=str(row.get("description", "")),
                timeout=_parse_timeout(row.get("timeout")),
            ))
        except KeyError as exc:
            raise ValueError(
                f"hook entry missing required field: {exc.args[0]}"
            ) from exc
    return Manifest(schema_version=schema, hooks=tuple(hooks))


# --- stamp file ----------------------------------------------------------


def read_stamp(stamp_path: Path | None = None) -> str:
    """Return the version stamp on disk, or '0.0.0' if absent / unreadable.

    `stamp_path` resolves from the module-level `STAMP_PATH` when
    omitted. The default is `None` rather than the constant itself so
    that `monkeypatch.setattr(auto_install, "STAMP_PATH", ...)`
    propagates here (#1320); a bound default is evaluated once at
    import and never re-reads the module global, so tests silently
    wrote the contributor's real `~/.aelfrice/`. Same shape as
    `maybe_install_manifest` (#839).
    """
    if stamp_path is None:
        stamp_path = STAMP_PATH
    try:
        return stamp_path.read_text(encoding="utf-8").strip() or _UNSTAMPED
    except OSError:
        return _UNSTAMPED


def write_stamp(stamp_path: Path, version: str) -> None:
    """Atomically write `version` to `stamp_path`."""
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=stamp_path.name + ".", suffix=".tmp", dir=str(stamp_path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(version + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, stamp_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# --- opt-out file --------------------------------------------------------


def read_opt_outs(opt_out_path: Path | None = None) -> frozenset[str]:
    """Return the set of hook *names* the user opted out of.

    Names match the manifest's `name` field (e.g. "transcript_ingest").
    Missing or unreadable file returns the empty set — no opt-outs.

    `opt_out_path` resolves from the module-level `OPT_OUT_PATH` when
    omitted — late-bound so tests monkeypatching the constant are
    honoured (#1320; see `read_stamp`).
    """
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    if not opt_out_path.exists():
        return frozenset()
    try:
        raw = opt_out_path.read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    if not raw.strip():
        return frozenset()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    opt_outs = parsed.get("opt_out", [])
    if not isinstance(opt_outs, list):
        return frozenset()
    return frozenset(str(n) for n in opt_outs if isinstance(n, str))


def read_host_opt_outs(opt_out_path: Path | None = None) -> frozenset[str]:
    """Return the set of *hosts* whose auto-install the user opted out of.

    Stored under the sibling `opt_out_hosts` key of the same ledger the
    per-hook opt-outs use (#1053); older aelfrice versions ignore the
    key. Missing / unreadable / malformed file returns the empty set —
    a broken marker never blocks capture (fail-open; `aelf doctor
    --host codex` surfaces the state).

    `opt_out_path` resolves from the module-level `OPT_OUT_PATH` when
    omitted — late-bound (#1320; see `read_stamp`).
    """
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    if not opt_out_path.exists():
        return frozenset()
    try:
        raw = opt_out_path.read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    if not raw.strip():
        return frozenset()
    try:
        parsed: object = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    parsed_typed = cast(dict[str, object], parsed)
    hosts = parsed_typed.get("opt_out_hosts", [])
    if not isinstance(hosts, list):
        return frozenset()
    return frozenset(
        str(h) for h in cast(list[object], hosts) if isinstance(h, str)
    )


def _write_host_opt_outs(
    hosts: set[str], opt_out_path: Path | None = None,
) -> None:
    """Rewrite the ledger preserving the per-hook `opt_out` key."""
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    doc: dict[str, object] = {"opt_out": sorted(read_opt_outs(opt_out_path))}
    if hosts:
        doc["opt_out_hosts"] = sorted(hosts)
    _atomic_write_json(opt_out_path, doc)


def add_host_opt_out(host: str, opt_out_path: Path | None = None) -> None:
    """Persist a host-level auto-install opt-out (#1053). Idempotent."""
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    current = set(read_host_opt_outs(opt_out_path))
    if host in current:
        return
    current.add(host)
    _write_host_opt_outs(current, opt_out_path)


def remove_host_opt_out(host: str, opt_out_path: Path | None = None) -> bool:
    """Drop a host-level opt-out. Returns True if one was removed."""
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    current = set(read_host_opt_outs(opt_out_path))
    if host not in current:
        return False
    current.discard(host)
    _write_host_opt_outs(current, opt_out_path)
    return True


def _write_opt_outs(
    hook_names: set[str], opt_out_path: Path | None = None,
) -> None:
    """Rewrite the ledger preserving the host-level `opt_out_hosts` key.

    The mirror of `_write_host_opt_outs`. Both keys live in one document
    (#1053); a writer that serialises only its own key silently deletes
    the other's. `add_opt_out` / `remove_opt_out` did exactly that, so a
    Codex-primary contributor lost their host opt-out the first time
    `aelf setup` toggled any hook (#1320).
    """
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    doc: dict[str, object] = {"opt_out": sorted(hook_names)}
    hosts = read_host_opt_outs(opt_out_path)
    if hosts:
        doc["opt_out_hosts"] = sorted(hosts)
    _atomic_write_json(opt_out_path, doc)


def add_opt_out(hook_name: str, opt_out_path: Path | None = None) -> None:
    """Persist `hook_name` to the opt-out file. Idempotent.

    Called by `aelf setup --no-X` after the corresponding uninstall — the
    intent persists across upgrades so the disabled hook is not re-added.

    `opt_out_path` resolves from the module-level `OPT_OUT_PATH` when
    omitted — late-bound (#1320; see `read_stamp`).
    """
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    current = set(read_opt_outs(opt_out_path))
    if hook_name in current:
        return
    current.add(hook_name)
    _write_opt_outs(current, opt_out_path)


def remove_opt_out(hook_name: str, opt_out_path: Path | None = None) -> None:
    """Drop `hook_name` from the opt-out file. Idempotent.

    Called by `aelf setup` (without the matching --no-X) — the user
    explicitly turned the hook back on, so the opt-out is rescinded.

    `opt_out_path` resolves from the module-level `OPT_OUT_PATH` when
    omitted — late-bound (#1320; see `read_stamp`).
    """
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    current = set(read_opt_outs(opt_out_path))
    if hook_name not in current:
        return
    current.discard(hook_name)
    if not current and not read_host_opt_outs(opt_out_path):
        # Nothing left in the ledger at all — remove the file rather
        # than leave an empty document. Guarded on the sibling key: the
        # unconditional unlink this replaces destroyed `opt_out_hosts`
        # whenever the last per-hook opt-out was rescinded (#1320).
        if opt_out_path.exists():
            try:
                opt_out_path.unlink()
            except OSError:
                # best-effort: empty opt-out file cleanup is non-critical
                pass
        return
    _write_opt_outs(current, opt_out_path)


def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# --- dispatch ------------------------------------------------------------


# Map installer key -> (resolve_command_fn, install_fn). The install_fn
# signature is uniform: (settings_path, *, command, timeout) -> object.
# Return objects vary; we only need to know whether any new entry was
# written, which we infer from the result attributes below.
_DispatchEntry = tuple[
    Callable[[SettingsScope], str],
    Callable[..., object],
]
_DISPATCH: Final[dict[str, _DispatchEntry]] = {
    "user_prompt_submit": (
        resolve_hook_command,
        install_user_prompt_submit_hook,
    ),
    "transcript_ingest": (
        resolve_transcript_logger_command,
        install_transcript_ingest_hooks,
    ),
    "commit_ingest": (
        resolve_commit_ingest_command,
        install_commit_ingest_hook,
    ),
    "session_start": (
        resolve_session_start_hook_command,
        install_session_start_hook,
    ),
    "stop": (
        resolve_stop_hook_command,
        install_stop_hook,
    ),
    "search_tool": (
        resolve_search_tool_command,
        install_search_tool_hook,
    ),
    "search_tool_bash": (
        resolve_search_tool_bash_command,
        install_search_tool_bash_hook,
    ),
    "pre_issue_guard": (
        resolve_pre_issue_guard_command,
        install_pre_issue_guard_hook,
    ),
    "claude_memory_mirror": (
        resolve_claude_memory_mirror_command,
        install_claude_memory_mirror_hook,
    ),
    "agent_context": (
        resolve_agent_context_command,
        install_agent_context_hook,
    ),
}


def _result_added_anything(result: object) -> bool:
    """True iff an install_* call newly wrote at least one entry.

    Per-hook install functions return ``InstallResult(installed=bool)``;
    transcript-ingest returns ``TranscriptIngestInstallResult(installed=tuple)``.
    Treat both shapes uniformly.
    """
    installed = getattr(result, "installed", None)
    if isinstance(installed, bool):
        return installed
    if isinstance(installed, tuple):
        return len(installed) > 0
    return False


# --- main entry ----------------------------------------------------------


def _version_key(v: str) -> tuple[int, ...]:
    """Parse ``'X.Y.Z...'`` into a comparable int tuple.

    Leading digits of each dot-segment; a non-numeric segment contributes
    0. This is deliberately not full PEP 440 — it only needs to answer
    "is A older than B" for the never-downgrade guard (#1044).
    """
    key: list[int] = []
    for seg in v.split("."):
        digits = ""
        for ch in seg:
            if ch.isdigit():
                digits += ch
            else:
                break
        key.append(int(digits) if digits else 0)
    return tuple(key) or (0,)


def _is_downgrade(installed_version: str, prev: str) -> bool:
    """True iff running ``installed_version`` is strictly older than the
    on-disk stamp ``prev`` (and prev is a real stamp, not the sentinel).
    """
    return prev != _UNSTAMPED and _version_key(installed_version) < _version_key(prev)


def _downgrade_skip_result(installed_version: str, prev: str) -> AutoInstallResult:
    return AutoInstallResult(
        ran=False,
        prev_version=prev,
        new_version=installed_version,
        message=(
            f"aelfrice: skipped hook auto-install — running "
            f"v{installed_version} is older than the installed v{prev}; "
            f"not downgrading (run `aelf setup` to force)"
        ),
    )


def maybe_install_manifest(
    *,
    installed_version: str,
    scope: SettingsScope = "user",
    settings_path: Path | None = None,
    stamp_path: Path | None = None,
    opt_out_path: Path | None = None,
    force: bool = False,
    timeout: int | None = None,
) -> AutoInstallResult:
    """Merge bundled manifest into the host settings.json if out of date.

    Happy-path overhead when stamp == installed_version is one stat +
    one short file read (no settings.json read, no JSON parse, no
    install dispatch). The merge runs only when (a) `force=True` or
    (b) the on-disk stamp is older than `installed_version`.

    `stamp_path` and `opt_out_path` resolve from the module-level
    `STAMP_PATH` / `OPT_OUT_PATH` constants when omitted. The default
    is `None` rather than the constant itself so that tests
    monkeypatching `auto_install.STAMP_PATH` propagate (#839 — bound
    defaults captured the original module attribute at function-def
    time, leaking real-merge writes to `~/.aelfrice/` during contributor
    testing). Mirrors the existing `settings_path` pattern above.

    Returns an AutoInstallResult describing what (if anything) was done.
    Never raises for missing files; the caller's CLI is unaffected by a
    failed auto-install (the stamp stays at its prior value and the
    next invocation retries).
    """
    if stamp_path is None:
        stamp_path = STAMP_PATH
    if opt_out_path is None:
        opt_out_path = OPT_OUT_PATH
    prev = read_stamp(stamp_path)
    if not force and prev == installed_version:
        return AutoInstallResult(
            ran=False, prev_version=prev, new_version=installed_version
        )
    # Defense in depth (#1044): a running binary must never stamp the hook
    # surface backwards. The primary gate already excludes worktrees, but
    # if an older aelfrice ever reaches here (non-force path), skip rather
    # than silently downgrade the user's installed hooks.
    if not force and _is_downgrade(installed_version, prev):
        return _downgrade_skip_result(installed_version, prev)
    # Read through the module rather than a by-value `from ... import`:
    # an import-time alias holds its own binding, so patching
    # `setup.USER_SETTINGS_PATH` would not reach here (#1320).
    target_path = (
        settings_path if settings_path is not None
        else _setup.USER_SETTINGS_PATH
    )

    # Acquire exclusive lock on the stamp's parent dir (the stamp file
    # may not exist yet). Holding the lock for the duration of the
    # merge serializes concurrent `aelf` invocations on the same host.
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stamp_path.parent / AUTO_INSTALL_LOCK_FILENAME
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another process holds the lock — they will finish the
            # merge, so we skip and report no-op.
            return AutoInstallResult(
                ran=False, prev_version=prev, new_version=installed_version
            )
        # Re-check stamp now that we hold the lock: another process may
        # have completed the merge while we were waiting.
        prev = read_stamp(stamp_path)
        if not force and prev == installed_version:
            return AutoInstallResult(
                ran=False, prev_version=prev, new_version=installed_version
            )
        if not force and _is_downgrade(installed_version, prev):
            return _downgrade_skip_result(installed_version, prev)
        return _do_merge(
            prev_version=prev,
            installed_version=installed_version,
            scope=scope,
            settings_path=target_path,
            stamp_path=stamp_path,
            opt_out_path=opt_out_path,
            timeout=timeout,
        )
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            # best-effort: close() below releases the lock either way
            pass
        os.close(lock_fd)


def _do_merge(
    *,
    prev_version: str,
    installed_version: str,
    scope: SettingsScope,
    settings_path: Path,
    stamp_path: Path,
    opt_out_path: Path,
    timeout: int | None,
) -> AutoInstallResult:
    manifest = load_manifest()
    opt_outs = read_opt_outs(opt_out_path)
    installed: list[str] = []
    already: list[str] = []
    opted_out: list[str] = []
    # #1161: one lock and one write for the whole merge. The
    # `.auto-install.lock` this function already runs under serialises
    # auto-install against *itself*; it says nothing about `aelf setup`
    # or `aelf doctor --fix`, which take no lock at all, so the merge
    # below used to interleave with them entry by entry. The bound is
    # short and a timeout aborts the merge: this runs on a hook path, so
    # waiting on a sibling's lock would spend the user's latency budget,
    # and skipping costs nothing because the stamp stays unwritten and
    # the next invocation retries.
    with settings_transaction(
        settings_path, timeout=SETTINGS_LOCK_TIMEOUT_BACKGROUND
    ):
        installed, already, opted_out = _merge_hooks_into_settings(
            manifest=manifest,
            opt_outs=opt_outs,
            scope=scope,
            settings_path=settings_path,
            timeout=timeout,
        )
    # Stamp only after settings.json mutations succeed and the
    # transaction has committed. If the merge raised, or the commit
    # detected a foreign write, we never reach this line and the next
    # invocation retries.
    write_stamp(stamp_path, installed_version)
    return AutoInstallResult(
        ran=True,
        prev_version=prev_version,
        new_version=installed_version,
        installed=tuple(installed),
        already=tuple(already),
        opted_out=tuple(opted_out),
        message=_format_message(
            prev_version=prev_version,
            installed_version=installed_version,
            installed=installed,
            opted_out=opted_out,
        ),
    )


def _merge_hooks_into_settings(
    *,
    manifest: HookManifest,
    opt_outs: set[str],
    scope: SettingsScope,
    settings_path: Path,
    timeout: int | None,
) -> tuple[list[str], list[str], list[str]]:
    """Install every default-on manifest hook. Returns
    `(installed, already, opted_out)` hook names.

    #1161: split out of `_do_merge` so the per-hook loop runs inside a
    settings transaction the caller owns.
    """
    installed: list[str] = []
    already: list[str] = []
    opted_out: list[str] = []
    for hook in manifest.hooks:
        if not hook.default_on:
            continue
        if hook.name in opt_outs:
            opted_out.append(hook.name)
            continue
        dispatch = _DISPATCH.get(hook.installer)
        if dispatch is None:
            # Unknown installer key — newer manifest read by older code.
            # Skip rather than crash; doctor will surface drift.
            continue
        resolve_fn, install_fn = dispatch
        command = resolve_fn(scope)
        # An explicit caller timeout overrides the manifest for every
        # hook; otherwise each hook gets its own declared budget
        # (#1161). Before this, `timeout` defaulted to None and was
        # passed through verbatim, so no installed entry carried one.
        effective_timeout = timeout if timeout is not None else hook.timeout
        result = install_fn(
            settings_path, command=command, timeout=effective_timeout
        )
        if _result_added_anything(result):
            installed.append(hook.name)
        else:
            already.append(hook.name)
    return installed, already, opted_out


def _format_message(
    *,
    prev_version: str,
    installed_version: str,
    installed: list[str],
    opted_out: list[str],
) -> str:
    """Single-line stderr message describing the merge outcome.

    Empty string when nothing user-visible to report (e.g. stamp bumped
    on a no-op merge after `aelf setup` already wrote everything).
    """
    if not installed:
        return ""
    if prev_version == _UNSTAMPED:
        head = f"aelfrice: installed default hooks for v{installed_version}"
    else:
        head = (
            f"aelfrice: hooks updated to v{installed_version} "
            f"(was v{prev_version})"
        )
    body = ", ".join(installed)
    suffix = ""
    if opted_out:
        suffix = f"; opted out: {', '.join(opted_out)}"
    return f"{head} — added: {body}{suffix}"


def is_disabled_via_env(env: dict[str, str] | None = None) -> bool:
    """True iff AELFRICE_NO_AUTO_INSTALL is set to a non-empty value.

    The `env` parameter is for tests; production callers use the real
    process environment.
    """
    src = env if env is not None else os.environ
    return bool(src.get(NO_AUTO_INSTALL_ENV, "").strip())


def is_running_from_uv_tool_install() -> bool:
    """True iff the running aelfrice is the user's installed `uv tool` copy.

    Auto-install rewrites `~/.claude/settings.json`, which is a global
    user-config file. Invoking `uv run aelf` (or `python -m aelfrice.cli`)
    from a project worktree's local `.venv` resolves to the worktree's
    source — whose hook surface may differ from the user's installed
    version. Letting that source rewrite the user's global settings is
    the bug closed by #834: a per-worktree command silently downgraded
    the user's installed-version hook surface to whatever the worktree
    happened to advertise.

    Detection delegates to `lifecycle._running_from_uv_tool`, which asks
    whether *this process* resolves under the uv tools root
    (`sys.prefix` / `sys.executable` scan) — NOT whether a uv-tool
    install merely exists somewhere on the box. The earlier delegate
    (`_is_uv_tool_install`) short-circuited True on a filesystem-presence
    check, so any user who also had a uv-tool install saw a worktree's
    `uv run aelf` reintroduce the #834 downgrade (#1044). Worktree-local
    venvs, contributor `pytest` runs, system Python, and pipx installs
    all return False here and are excluded from auto-install.

    Returning False means the gate skips merging — power users who
    want auto-install to run from a non-uv-tool context can still
    invoke `aelf setup` explicitly. The `AELFRICE_NO_AUTO_INSTALL`
    env override remains the symmetric escape hatch for uv-tool users.
    """
    # Local import keeps `auto_install` importable when `lifecycle` has
    # not yet been imported by the caller (e.g. during early CLI bootstrap).
    from aelfrice.lifecycle import _running_from_uv_tool

    return _running_from_uv_tool()


def auto_install_at_cli_entry(installed_version: str) -> None:
    """Convenience for `cli.main()`: best-effort merge, never raises.

    Bypassed when AELFRICE_NO_AUTO_INSTALL is set, or when the running
    aelfrice is not the user's `uv tool` install (#834 — a worktree's
    `uv run aelf` must not rewrite the user's global settings.json).
    Stderr message is emitted only when the merge added at least one
    new entry. Any exception during the merge is swallowed and logged
    to stderr — we never let a misconfigured the host settings.json
    block the user's actual `aelf <cmd>` invocation.
    """
    if is_disabled_via_env():
        return
    # #1053: persistent host-level opt-out. A Codex-primary user (or
    # anyone who opted the claude host out) gets no settings.json
    # mutation at CLI entry, without per-command env prefixes. An
    # explicit `aelf setup` still installs — and clears the opt-out.
    if "claude" in read_host_opt_outs(OPT_OUT_PATH):
        return
    if not is_running_from_uv_tool_install():
        return
    try:
        result = maybe_install_manifest(installed_version=installed_version)
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"aelfrice: auto-install skipped ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return
    if result.message:
        print(result.message, file=sys.stderr)
