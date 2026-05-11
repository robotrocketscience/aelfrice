"""First-run / post-upgrade hook installer driven by a bundled manifest.

Closes the loop on #623: a bare `pipx upgrade aelfrice` does not re-run
`aelf setup`, so any default-on hook added in a new release (e.g.
`aelf-stop-hook` shipped in v2.1) is missing for users who never re-run
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
from typing import Callable, Final

from aelfrice.setup import (
    SettingsScope,
    USER_SETTINGS_PATH,
    install_commit_ingest_hook,
    install_session_start_hook,
    install_stop_hook,
    install_transcript_ingest_hooks,
    install_user_prompt_submit_hook,
    resolve_commit_ingest_command,
    resolve_hook_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
)

NO_AUTO_INSTALL_ENV: Final[str] = "AELFRICE_NO_AUTO_INSTALL"
AELFRICE_DOTDIR: Final[Path] = Path.home() / ".aelfrice"
STAMP_PATH: Final[Path] = AELFRICE_DOTDIR / "installed-manifest-version"
OPT_OUT_PATH: Final[Path] = AELFRICE_DOTDIR / "opt-out-hooks.json"

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
            ))
        except KeyError as exc:
            raise ValueError(
                f"hook entry missing required field: {exc.args[0]}"
            ) from exc
    return Manifest(schema_version=schema, hooks=tuple(hooks))


# --- stamp file ----------------------------------------------------------


def read_stamp(stamp_path: Path = STAMP_PATH) -> str:
    """Return the version stamp on disk, or '0.0.0' if absent / unreadable."""
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


def read_opt_outs(opt_out_path: Path = OPT_OUT_PATH) -> frozenset[str]:
    """Return the set of hook *names* the user opted out of.

    Names match the manifest's `name` field (e.g. "transcript_ingest").
    Missing or unreadable file returns the empty set — no opt-outs.
    """
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


def add_opt_out(hook_name: str, opt_out_path: Path = OPT_OUT_PATH) -> None:
    """Persist `hook_name` to the opt-out file. Idempotent.

    Called by `aelf setup --no-X` after the corresponding uninstall — the
    intent persists across upgrades so the disabled hook is not re-added.
    """
    current = set(read_opt_outs(opt_out_path))
    if hook_name in current:
        return
    current.add(hook_name)
    _atomic_write_json(
        opt_out_path,
        {"opt_out": sorted(current)},
    )


def remove_opt_out(hook_name: str, opt_out_path: Path = OPT_OUT_PATH) -> None:
    """Drop `hook_name` from the opt-out file. Idempotent.

    Called by `aelf setup` (without the matching --no-X) — the user
    explicitly turned the hook back on, so the opt-out is rescinded.
    """
    current = set(read_opt_outs(opt_out_path))
    if hook_name not in current:
        return
    current.discard(hook_name)
    if current:
        _atomic_write_json(opt_out_path, {"opt_out": sorted(current)})
    elif opt_out_path.exists():
        try:
            opt_out_path.unlink()
        except OSError:
            pass


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


def maybe_install_manifest(
    *,
    installed_version: str,
    scope: SettingsScope = "user",
    settings_path: Path | None = None,
    stamp_path: Path = STAMP_PATH,
    opt_out_path: Path = OPT_OUT_PATH,
    force: bool = False,
    timeout: int | None = None,
) -> AutoInstallResult:
    """Merge bundled manifest into the host settings.json if out of date.

    Happy-path overhead when stamp == installed_version is one stat +
    one short file read (no settings.json read, no JSON parse, no
    install dispatch). The merge runs only when (a) `force=True` or
    (b) the on-disk stamp is older than `installed_version`.

    Returns an AutoInstallResult describing what (if anything) was done.
    Never raises for missing files; the caller's CLI is unaffected by a
    failed auto-install (the stamp stays at its prior value and the
    next invocation retries).
    """
    prev = read_stamp(stamp_path)
    if not force and prev == installed_version:
        return AutoInstallResult(
            ran=False, prev_version=prev, new_version=installed_version
        )
    target_path = settings_path if settings_path is not None else USER_SETTINGS_PATH

    # Acquire exclusive lock on the stamp's parent dir (the stamp file
    # may not exist yet). Holding the lock for the duration of the
    # merge serializes concurrent `aelf` invocations on the same host.
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stamp_path.parent / ".auto-install.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
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
        result = install_fn(settings_path, command=command, timeout=timeout)
        if _result_added_anything(result):
            installed.append(hook.name)
        else:
            already.append(hook.name)
    # Stamp only after settings.json mutations succeed. If install_fn
    # raised, we never reach this line and the next invocation retries.
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


def auto_install_at_cli_entry(installed_version: str) -> None:
    """Convenience for `cli.main()`: best-effort merge, never raises.

    Bypassed when AELFRICE_NO_AUTO_INSTALL is set. Stderr message is
    emitted only when the merge added at least one new entry. Any
    exception during the merge is swallowed and logged to stderr — we
    never let a misconfigured the host settings.json block the user's
    actual `aelf <cmd>` invocation.
    """
    if is_disabled_via_env():
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
