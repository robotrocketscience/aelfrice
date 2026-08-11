"""Lifecycle commands: update check, upgrade advice, uninstall.

This module owns the surfaces that operate on the aelfrice install itself
rather than on the memory store. Patterns ported from the GSD framework's
two-component update notifier (gsd-check-update.js + gsd-statusline.js):

* Background fire-and-forget PyPI version check writes a JSON cache.
* Cache lives at ~/.cache/aelfrice/update_check.json (XDG-style,
  tool-agnostic, mirrors GSD's ~/.cache/gsd/ choice).
* Statusline reader (in aelfrice.statusline) reads the cache only --
  never makes network calls. This keeps statusline rendering fast.
* All network and file ops fail silently. update_available defaults to
  False so a network outage never inflicts a "update needed" banner.
* Custom is_newer() semver compare strips pre-release suffixes.

PyPI's JSON API also publishes a SHA-256 digest for every uploaded wheel
and sdist. We cache the wheel's sha256 alongside the version so callers
can offer hash-pinned installs without an extra round trip.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import URLError
from urllib.request import Request, urlopen

PACKAGE_NAME: Final[str] = "aelfrice"
PYPI_JSON_URL: Final[str] = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
CACHE_DIR: Final[Path] = Path.home() / ".cache" / "aelfrice"
CACHE_FILE: Final[Path] = CACHE_DIR / "update_check.json"
CACHE_TTL_SECONDS: Final[int] = 15 * 60  # 15min: catch new releases within one cycle
HTTP_TIMEOUT_SECONDS: Final[float] = 10.0
USER_AGENT: Final[str] = f"aelfrice-update-check/{PACKAGE_NAME}"
ENV_DISABLE: Final[str] = "AELF_NO_UPDATE_CHECK"


@dataclass(frozen=True)
class UpdateStatus:
    """Snapshot of the latest cached update check.

    `update_available` is the only field the statusline consumes; the
    others are for `aelf upgrade` to surface verification details.
    """

    update_available: bool
    installed: str
    latest: str
    checked: float
    sha256: str | None = None

    @classmethod
    def empty(cls) -> "UpdateStatus":
        return cls(False, "", "", 0.0, None)


def is_newer(a: str, b: str) -> bool:
    """Return True iff version `a` is strictly newer than `b`.

    Mirrors GSD's isNewer(): split on '.', strip any pre-release suffix
    after a '-', integer compare each of the first three components.
    Non-numeric components collapse to 0 so junk strings can't crash.
    """

    def _parts(v: str) -> tuple[int, int, int]:
        out: list[int] = []
        for chunk in (v or "").split(".")[:3]:
            stripped = chunk.split("-", 1)[0]
            try:
                out.append(int(stripped))
            except ValueError:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return out[0], out[1], out[2]

    pa = _parts(a)
    pb = _parts(b)
    for i in range(3):
        if pa[i] > pb[i]:
            return True
        if pa[i] < pb[i]:
            return False
    return False


def installed_version() -> str:
    """Resolve the installed aelfrice version from package metadata.

    Uses importlib.metadata so the returned version always matches the
    installed wheel, even after an in-place upgrade that leaves the
    source tree's __version__ constant stale. Falls back to '0.0.0'
    when the package is not found (e.g. during unit tests run against
    an editable install that hasn't been built yet).
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(PACKAGE_NAME)
    except Exception:
        return "0.0.0"


def _fetch_pypi_json(url: str = PYPI_JSON_URL) -> dict | None:
    """Fetch PyPI JSON, returning None on any failure.

    Silent fail discipline: any exception (network, DNS, JSON parse,
    timeout) yields None. Callers must treat None as "no info, keep
    last cached state".
    """
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:  # noqa: S310
            payload = resp.read()
        return json.loads(payload.decode("utf-8"))
    except (URLError, TimeoutError, ValueError, OSError):
        return None


def _wheel_sha256(release_files: list[dict]) -> str | None:
    """Pick the SHA-256 of the wheel from a PyPI release entry.

    Wheels are universally preferred over sdists; we look for the
    .whl file first. PyPI guarantees a sha256 in `digests`.
    """
    for entry in release_files:
        try:
            if entry.get("packagetype") == "bdist_wheel" or str(
                entry.get("filename", "")
            ).endswith(".whl"):
                digests = entry.get("digests") or {}
                sha = digests.get("sha256")
                if isinstance(sha, str) and sha:
                    return sha
        except (AttributeError, TypeError):
            continue
    # Fall back to whatever we can find (sdist, etc.)
    for entry in release_files:
        try:
            digests = entry.get("digests") or {}
            sha = digests.get("sha256")
            if isinstance(sha, str) and sha:
                return sha
        except (AttributeError, TypeError):
            continue
    return None


def _write_cache(status: UpdateStatus, cache_path: Path | None = None) -> None:
    """Persist a status snapshot. Silent fail: cache write is best-effort.

    `cache_path` resolves from the module-level `CACHE_FILE` when
    omitted, late-bound so `monkeypatch.setattr(lifecycle, "CACHE_FILE",
    ...)` is honoured (#1320). A bound default is evaluated at import,
    before any test body runs, so the suite wrote the contributor's real
    `~/.cache/aelfrice/`.
    """
    if cache_path is None:
        cache_path = CACHE_FILE
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "update_available": status.update_available,
            "installed": status.installed,
            "latest": status.latest,
            "checked": status.checked,
            "sha256": status.sha256,
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def read_cache(cache_path: Path | None = None) -> UpdateStatus:
    """Read the cached update status. Returns empty() on any failure.

    The statusline calls this. It MUST be cheap and never raise.

    `cache_path` resolves from `CACHE_FILE` when omitted — late-bound
    (#1320; see `_write_cache`).
    """
    if cache_path is None:
        cache_path = CACHE_FILE
    try:
        raw = cache_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return UpdateStatus(
            update_available=bool(data.get("update_available", False)),
            installed=str(data.get("installed", "")),
            latest=str(data.get("latest", "")),
            checked=float(data.get("checked", 0.0)),
            sha256=(
                str(data["sha256"])
                if data.get("sha256") is not None
                else None
            ),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return UpdateStatus.empty()


def cache_is_fresh(
    status: UpdateStatus,
    now: float | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> bool:
    """True iff the cache was written within the TTL window."""
    if status.checked <= 0:
        return False
    if now is None:
        now = time.time()
    return (now - status.checked) < ttl


def is_disabled(env: dict[str, str] | None = None) -> bool:
    """True iff AELF_NO_UPDATE_CHECK is set to a truthy value."""
    src = os.environ if env is None else env
    val = src.get(ENV_DISABLE, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def check_for_update(
    cache_path: Path | None = None,
    pypi_url: str = PYPI_JSON_URL,
    fetch: callable = _fetch_pypi_json,
    now: float | None = None,
) -> UpdateStatus:
    """Run the synchronous update check end-to-end and write the cache.

    This is the function the background process invokes. The CLI/hook
    paths use maybe_check_for_update_async() which spawns a detached
    subprocess pointing at this entry. We expose a sync version for
    tests and for direct CLI use.

    `cache_path` resolves from `CACHE_FILE` when omitted — late-bound
    (#1320; see `_write_cache`). Note that the async spawn re-enters
    this function in a *fresh interpreter*, where the module global is
    recomputed from the real home: only `AELF_NO_UPDATE_CHECK` (or an
    inherited HOME) stops the child, never `setattr`.
    """
    if cache_path is None:
        cache_path = CACHE_FILE
    if is_disabled():
        return UpdateStatus.empty()
    installed = installed_version()
    data = fetch(pypi_url)
    if data is None:
        # Network/parse failure: preserve any prior cache, return empty.
        return read_cache(cache_path)
    info = data.get("info") or {}
    latest = str(info.get("version") or "")
    sha = None
    releases = data.get("releases") or {}
    if latest and isinstance(releases, dict):
        files = releases.get(latest) or []
        if isinstance(files, list):
            sha = _wheel_sha256(files)
    status = UpdateStatus(
        update_available=bool(latest) and is_newer(latest, installed),
        installed=installed,
        latest=latest,
        checked=time.time() if now is None else now,
        sha256=sha,
    )
    _write_cache(status, cache_path)
    return status


def maybe_check_for_update_async(
    cache_path: Path | None = None,
    ttl: int = CACHE_TTL_SECONDS,
) -> bool:
    """Fire a detached background check iff cache is stale.

    Returns True iff a subprocess was launched. Never blocks the caller;
    the spawned process detaches via start_new_session=True so the
    parent can exit without waiting. Mirrors GSD's spawn(detached:true)
    + child.unref() pattern in Python.

    `cache_path` resolves from `CACHE_FILE` when omitted — late-bound
    (#1320). It gates only the staleness *read*: the spawned child
    recomputes its own path, so isolating the write requires
    `AELF_NO_UPDATE_CHECK=1`.
    """
    if cache_path is None:
        cache_path = CACHE_FILE
    if is_disabled():
        return False
    status = read_cache(cache_path)
    if cache_is_fresh(status, ttl=ttl):
        return False
    import subprocess
    import sys

    try:
        subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                (
                    "from aelfrice.lifecycle import check_for_update; "
                    "check_for_update()"
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except (OSError, ValueError):
        return False


# --- Banner helper -------------------------------------------------------


def format_update_banner(
    latest: str, *, command: str | None = None
) -> str:
    """Return the plain (no ANSI) body text for an update-available banner.

    Single source of truth for the banner format. Both the statusline
    snippet and the CLI stderr notice derive their text from this
    function so the wording never drifts between the two surfaces.

    `command` is the install-method-aware shell line the user could
    run directly. When unset the banner points at the `/aelf:upgrade`
    slash command (imperative orchestrator); when set, the banner
    embeds the literal command for hosts that don't support slash
    commands. Tests inject a stable value.

    Examples:
      '⬆ aelfrice 2.1.0 — run /aelf:upgrade'
      '⬆ aelfrice 2.1.0 — run: uv tool upgrade aelfrice'  (command set)
    """
    if command is None:
        return f"⬆ aelfrice {latest} — run /aelf:upgrade"
    return f"⬆ aelfrice {latest} — run: {command}"


# --- Upgrade advice -----------------------------------------------------


@dataclass(frozen=True)
class UpgradeAdvice:
    """How to upgrade aelfrice in the user's specific install context.

    Per #730 aelfrice is supported on a single install channel: `uv tool`.
    `context` therefore collapses to two values: `uv_tool` for a uv-managed
    install (in-place upgrade) and `non_uv` for any other install path
    (migration command — uninstall the old + `uv tool install`).
    """

    command: str
    context: str  # 'uv_tool' | 'non_uv'


UV_RECEIPT_FILENAME: Final[str] = "uv-receipt.toml"


def _is_windows() -> bool:
    """Whether this process is running on Windows.

    Indirected through a function on purpose. Tests cannot simply
    `monkeypatch.setattr(os, "name", "nt")` to reach the Windows branches:
    `pathlib.Path()` dispatches on `os.name` and raises
    `UnsupportedOperation: cannot instantiate 'WindowsPath' on your system`
    the moment a POSIX runner is told it is Windows. Patching this probe
    reaches the branch without breaking path construction.

    It stays a *call*, evaluated per invocation — the thing a
    `windows: bool = os.name == "nt"` default argument fails to be, since
    that binds once at definition time and leaves the branch permanently
    unreachable from a test (#1412 review).
    """
    return os.name == "nt"


def _uv_tool_dir() -> Path:
    """uv's tools directory for this platform and environment.

    Resolution order matches uv's own: ``UV_TOOL_DIR`` wins outright;
    otherwise tools live in a ``tools/`` subdirectory of uv's *persistent
    data directory*, which is where the two platforms diverge.

    uv's storage reference (docs.astral.sh/uv/reference/storage) gives the
    persistent data directory as ``$XDG_DATA_HOME/uv`` or
    ``$HOME/.local/share/uv`` on Linux/macOS but ``%APPDATA%\\uv\\data``
    on Windows, and says tools are installed "in a ``tools/``
    subdirectory of the persistent data directory, e.g.,
    ``~/.local/share/uv/tools``". So the defaults are asymmetric — the
    Windows one carries an extra ``data`` component:

        POSIX    $XDG_DATA_HOME/uv/tools  (else ~/.local/share/uv/tools)
        Windows  %APPDATA%\\uv\\data\\tools

    ``%APPDATA%\\uv\\tools`` — the literal this module carried before
    #1431, and the one #1431's own body repeated — names no directory uv
    ever writes, so a real Windows uv-tool install was invisible.

    The platform is resolved per call via ``_is_windows()``, never bound
    as a default argument (#1412 review).
    """
    override = os.environ.get("UV_TOOL_DIR")
    if override:
        return Path(override)
    if _is_windows():
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "uv" / "data" / "tools"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "uv" / "tools"


def _uv_tool_env_dir() -> Path:
    """Where a uv-tool install of THIS package would live."""
    return _uv_tool_dir() / PACKAGE_NAME


def _has_uv_receipt(env_root: Path) -> bool:
    """True iff `env_root` is a uv tool environment, by uv's own receipt.

    The single predicate behind both "is aelfrice installed via uv tool?"
    (`_is_uv_tool_install`) and the reachable-install inventory
    (`detect_reachable_installs`). Sharing the *directory resolver* was
    not enough: while the classifier tested for the receipt and the
    inventory tested `.exists()`, a bare `<tools>/aelfrice/` was
    simultaneously `non_uv` to `upgrade_advice()` and a `uv_tool` site in
    the multi-install warning (#1431 review).
    """
    try:
        return (env_root / UV_RECEIPT_FILENAME).is_file()
    except OSError:
        return False


def _running_from_uv_tool() -> bool:
    """True iff THIS running process is the uv-tool-managed install.

    Two signals, strongest first. uv writes a ``uv-receipt.toml`` at the
    root of every tool environment, and for an installed tool that root
    *is* ``sys.prefix`` — so the receipt identifies the running
    environment directly, with no assumption about where the tools
    directory lives. Failing that, we ask whether sys.prefix /
    sys.executable resolves under the tools directory.

    Unlike a filesystem-presence check, this correctly returns False for
    a source worktree's ``uv run aelf`` even when a uv-tool install
    exists elsewhere on the box: a project venv carries no receipt and
    does not sit under the tools directory. Use this to gate the hook
    auto-install — the question there is "is this process the install?",
    not "does an install exist anywhere?" (#1044 — the ``.exists()``
    short-circuit reintroduced the #834 bug for any user who also had a
    uv-tool install).
    """
    import sys

    if _has_uv_receipt(Path(sys.prefix)):
        return True

    root = _uv_tool_dir()
    # Ancestry, not a string prefix: a sibling like ``.../uv/toolshed``
    # must not satisfy a test against ``.../uv/tools`` (Sourcery, #1044
    # review), and a string compare also mishandles the separator and
    # case conventions of the Windows layout this now supports.
    for candidate in (sys.prefix, sys.executable):
        if candidate and _path_is_under(Path(candidate), root):
            return True
    return False


def _is_uv_tool_install() -> bool:
    """Detect that a uv-tool-managed install EXISTS on this box.

    Answers "is aelfrice installed via ``uv tool`` on this machine?" —
    used by ``upgrade_advice()`` to recommend ``uv tool upgrade``. This
    intentionally includes a filesystem-presence check: the install dir
    may exist even when the *current* process runs from elsewhere (a
    worktree, a venv), and the upgrade advice is still "upgrade your uv
    tool copy". Do NOT use this to gate auto-install — that must ask
    whether *this process* is the install; use ``_running_from_uv_tool()``
    (#1044).

    A directory alone is not enough: an empty or hand-made
    ``<tools>/aelfrice/`` without uv's receipt is not a uv install, and
    treating it as one would hand the user an upgrade command that
    cannot work. ``detect_reachable_installs()`` applies the same
    predicate, so the advice and the inventory cannot disagree about
    what counts as a uv-tool install.
    """
    if _has_uv_receipt(_uv_tool_env_dir()):
        return True
    # Secondary: this process is itself running under the uv tools tree.
    return _running_from_uv_tool()


def _is_pipx_install() -> bool:
    """Detect a pipx-managed install.

    pipx installs each package into ~/.local/pipx/venvs/<pkg>/ and
    sys.prefix will be rooted there. We check both sys.prefix (fast,
    no FS access) and the venv directory directly (handles edge cases
    where sys.prefix normalisation differs on some platforms).

    We do NOT shell out to `pipx list` -- it's slow and may not be
    installed in the running environment.
    """
    import sys

    prefix_norm = sys.prefix.replace("\\", "/")
    if "/pipx/venvs/" in prefix_norm:
        return True
    # Filesystem check: covers users whose sys.prefix is symlinked.
    pipx_venv_dir = Path.home() / ".local" / "pipx" / "venvs" / PACKAGE_NAME
    return pipx_venv_dir.exists()


def _is_venv() -> bool:
    """Detect a generic venv (PEP 405 / virtualenv / uv venv).

    sys.prefix != sys.base_prefix is the standard idiom; works for
    venv, virtualenv, uv venv, and conda envs.
    """
    import sys

    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def upgrade_advice() -> UpgradeAdvice:
    """Return the upgrade command for the running install context.

    aelfrice is supported on a single install channel: `uv tool` (#730).
    When the running install came from a different installer we emit a
    migration command (uninstall the old + `uv tool install`) rather
    than an in-place upgrade — the supported upgrade path is uv.

    Detection order matters: uv-tool and pipx are both virtualenvs, so
    they must be identified before the generic venv check. The pipx /
    venv / system branches are kept because we still need to know *what*
    to tell the user to uninstall before the uv install — but they all
    collapse to `context="non_uv"` at the API surface.
    """
    if _is_uv_tool_install():
        return UpgradeAdvice(
            command=f"uv tool upgrade {PACKAGE_NAME}",
            context="uv_tool",
        )
    if _is_pipx_install():
        migrate = (
            f"pipx uninstall {PACKAGE_NAME} "
            f"&& uv tool install {PACKAGE_NAME}"
        )
    else:
        # venv and system installs: pip uninstall, then uv tool install.
        # `-y` skips the pip confirmation prompt; the user opted in by
        # running the slash. `uv tool install` itself sets up the shim
        # in ~/.local/bin so no further PATH plumbing is needed.
        migrate = (
            f"pip uninstall -y {PACKAGE_NAME} "
            f"&& uv tool install {PACKAGE_NAME}"
        )
    return UpgradeAdvice(command=migrate, context="non_uv")


# --- Multi-install detection -------------------------------------------


@dataclass(frozen=True)
class InstallSite:
    """A reachable aelfrice install location on disk.

    `kind` is one of 'uv_tool', 'pipx', 'user_local_bin'. `path` is the
    install root (uv_tool/pipx) or the executable path (user_local_bin).
    `on_path` is True when this site's executable is what `aelf` resolves
    to on PATH — i.e. the install the user gets when they type `aelf`.
    """

    kind: str
    path: Path
    on_path: bool


def _aelf_script_names() -> list[str]:
    """Candidate filenames for the `aelf` console script on this platform.

    On Windows the launcher is `aelf` plus a `PATHEXT` suffix — the same
    set `shutil.which` consults — and there is no executable bit to test,
    so presence is the only signal. Elsewhere it is the bare name.
    """
    if not _is_windows():
        return ["aelf"]
    # PATHEXT is a Windows-only variable and is always ';'-delimited —
    # which is also what `os.pathsep` is on Windows. Spelling it literally
    # keeps the branch exercisable from a POSIX test runner, where
    # `os.pathsep` would be ':'.
    raw = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    exts = [e for e in (p.strip() for p in raw.split(";")) if e]
    return ["aelf" + e for e in exts] or ["aelf.EXE"]


def _which_all_aelf() -> list[Path]:
    """Return every `aelf` executable reachable on PATH, in PATH order.

    We walk PATH ourselves rather than rely on `which -a`, which is not
    portable across shells. Skips non-files, and on POSIX non-executables
    — Windows has no executable bit, so there `PATHEXT` membership is what
    makes a file a launcher.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    windows = _is_windows()
    names = _aelf_script_names()
    raw = os.environ.get("PATH", "")
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        for name in names:
            candidate = Path(entry) / name
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            if not candidate.is_file():
                continue
            if not windows and not os.access(candidate, os.X_OK):
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def _path_is_under(child: Path, parent: Path) -> bool:
    """True if `child` is `parent` or a descendant. Symlink-resolved."""
    try:
        child_r = child.resolve()
        parent_r = parent.resolve()
    except OSError:
        return False
    try:
        child_r.relative_to(parent_r)
        return True
    except ValueError:
        return False


def _running_interpreter_aelf() -> Path | None:
    """Return the resolved `aelf` path inside the venv hosting the
    running interpreter, if such a venv exists and contains the script.

    Used to suppress false-positive `user_local_bin` reports when the
    detector runs under `uv run` from a project tree: that mode
    transiently prepends the project's `.venv/bin` to PATH, which would
    otherwise look like a separate install on PATH.
    """
    import sys

    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if base_prefix == sys.prefix:
        # Not running inside a venv; nothing to suppress.
        return None
    candidate = Path(sys.prefix) / "bin" / "aelf"
    if not candidate.exists():
        return None
    try:
        return candidate.resolve()
    except OSError:
        return None


def detect_reachable_installs() -> list[InstallSite]:
    """Best-effort enumeration of aelfrice installs visible on this system.

    Detection is purely filesystem + PATH inspection — no shelling out.
    Returns an empty list on any failure (e.g. unreadable home dir).

    Signals checked:
      - <uv tool dir>/aelfrice/uv-receipt.toml
                                           → uv_tool
      - ~/.local/pipx/venvs/aelfrice/      → pipx
      - any `aelf` on PATH whose resolved path is NOT under the above
        roots                              → user_local_bin

    The venv hosting the *currently-running* interpreter is excluded
    from `user_local_bin` reporting. Under `uv run`, the project venv
    is on PATH only because uv injected it; reporting it as a "second
    install" when the user's persistent shell PATH doesn't include it
    is a false positive.
    """
    sites: list[InstallSite] = []
    try:
        home = Path.home()
    except (OSError, RuntimeError):
        return sites

    path_aelf_resolved: set[Path] = set()
    for exe in _which_all_aelf():
        try:
            path_aelf_resolved.add(exe.resolve())
        except OSError:
            continue

    # `known_roots` collects only the roots actually reported above. A
    # root we declined to classify must not silently swallow an `aelf`
    # that PATH resolves into it — that would drop the install from the
    # inventory entirely instead of naming it.
    known_roots: list[Path] = []

    # The receipt, not mere presence — `_is_uv_tool_install()` asks the
    # same question through the same predicate, so the classifier and
    # this inventory cannot label one directory two different ways.
    uv_root = _uv_tool_env_dir()
    if _has_uv_receipt(uv_root):
        on_path = any(
            _path_is_under(p, uv_root) for p in path_aelf_resolved
        )
        sites.append(InstallSite(kind="uv_tool", path=uv_root, on_path=on_path))
        known_roots.append(uv_root)

    pipx_root = home / ".local" / "pipx" / "venvs" / PACKAGE_NAME
    if pipx_root.exists():
        on_path = any(
            _path_is_under(p, pipx_root) for p in path_aelf_resolved
        )
        sites.append(InstallSite(kind="pipx", path=pipx_root, on_path=on_path))
        known_roots.append(pipx_root)

    running_aelf = _running_interpreter_aelf()
    for exe in path_aelf_resolved:
        if any(_path_is_under(exe, root) for root in known_roots):
            continue
        if running_aelf is not None and exe == running_aelf:
            # Suppress: this is the venv hosting us, not a separate install.
            continue
        sites.append(InstallSite(kind="user_local_bin", path=exe, on_path=True))

    return sites


# --- Uninstall ----------------------------------------------------------

ARCHIVE_MAGIC: Final[bytes] = b"AELFENC1"  # 8 bytes, format identifier
ARCHIVE_SCRYPT_N: Final[int] = 2 ** 14
ARCHIVE_SCRYPT_R: Final[int] = 8
ARCHIVE_SCRYPT_P: Final[int] = 1
ARCHIVE_SALT_LEN: Final[int] = 16
ARCHIVE_KEY_LEN: Final[int] = 32  # Fernet wants base64-32 but we feed raw 32


@dataclass(frozen=True)
class UninstallResult:
    """Outcome of `uninstall(...)`. Mode is one of:
      'kept'    - DB preserved at db_path.
      'purged'  - DB deleted.
      'archived'- DB encrypted to archive_path then deleted from db_path.

    `removed` lists every artifact path actually deleted (#1173), and
    `orphaned` lists artifact paths we declined to touch because the
    store does not live in a directory this package owns. Both are
    empty in 'kept' mode. Callers print them so a disposition is
    auditable rather than taken on faith.
    """

    mode: str  # 'kept' | 'purged' | 'archived'
    db_path: Path | None
    archive_path: Path | None = None
    removed: tuple[Path, ...] = ()
    orphaned: tuple[Path, ...] = ()


# --- Artifact enumeration (#1173) ---------------------------------------
#
# `uninstall` used to delete exactly one file: memory.db. Everything else
# the package writes next to it survived a --purge, including several
# artifacts holding verbatim belief content. The enumeration below is the
# single source of truth for "what this package put on disk".
#
# Two classes of path, because they carry different collision risk:
#
#   1. DB-anchored. Named after the store file itself (`memory.db-wal`,
#      `memory.db.bm25f`, `memory.db.bak-20260629`). The db filename is a
#      prefix, so these cannot collide with unrelated user files and are
#      safe to remove wherever the store lives.
#
#   2. Fixed-name siblings. `hook_audit.jsonl`, `transcripts/`, and
#      friends carry generic names, so they are only ours by virtue of
#      sitting in an aelfrice-owned directory. `AELFRICE_DB` is honoured
#      verbatim (db_paths.db_path), so a user may legitimately point the
#      store at `~/memory.db` — and then `db_path.parent` is $HOME and
#      `$HOME/transcripts/` is emphatically not ours to delete. These are
#      removed only when the parent directory is recognisably a store
#      directory, and reported as `orphaned` otherwise.

_OWNED_STORE_DIRNAMES: Final[frozenset[str]] = frozenset(
    # `<git-common-dir>/aelfrice/` (db_paths.db_path) and the
    # `~/.aelfrice/` non-git fallback (db_paths.DEFAULT_DB_DIR).
    {"aelfrice", ".aelfrice"}
)

# SQLite's own sidecars. WAL mode is unconditional (store.py sets
# `PRAGMA journal_mode=WAL`), so `-wal`/`-shm` are present whenever a
# connection is open; `-journal` covers a store rolled back to the
# rollback journal by an external tool.
_DB_SIDECAR_SUFFIXES: Final[tuple[str, ...]] = ("-wal", "-shm", "-journal")

# Fixed-name files the package writes beside the store. Each literal is
# owned by the module named in the comment; `test_uninstall_artifacts`
# asserts the two agree, so a rename over there fails a test here rather
# than silently orphaning a file. Literals rather than imports because
# `hook` and `session_ring` sit above this module in the import graph.
_SIBLING_FILENAMES: Final[tuple[str, ...]] = (
    "hook_audit.jsonl",             # hook_audit.AUDIT_FILENAME
    "hook_audit.jsonl.1",           # + hook_audit.AUDIT_ROTATED_SUFFIX
    "feed.jsonl",                   # feed_log.FEED_FILENAME
    "session_injected_ids.json",    # session_ring.SESSION_RING_FILENAME
    ".session-ring.lock",           # session_ring.SESSION_RING_LOCK_FILENAME
    "session_first_prompt.json",    # hook.SESSION_STATE_FILENAME
    "sessionstart_last.txt",        # hook._RECAP_LAST_TS_FILENAME
    # docs/user/CONFIG.md already promises this one "is removed with it on
    # uninstall/rebuild" -- untrue until #1173, since only memory.db went.
    "claude-memory-reconciled",     # claude_memory._RECONCILE_SENTINEL_NAME
    # db_paths._IDENTITY_SIDECAR_NAME. It is the one artifact here whose
    # survival is not merely untidy: the sidecar is what makes the repo
    # identity durable across path spellings (#1415), so a `--purge` that
    # left it would hand the next install the identity of the one the user
    # just removed.
    "identity",
)

# Fixed-name directories the package writes beside the store.
_SIBLING_DIRNAMES: Final[tuple[str, ...]] = (
    "transcripts",     # transcript_logger.TRANSCRIPTS_SUBDIR
    "rebuild_logs",    # context_rebuilder.REBUILD_LOG_DIRNAME
    "telemetry",       # hook.py / hook_search_tool.py
)


def store_dir_is_owned(db_path: Path) -> bool:
    """True when `db_path`'s parent is a directory this package created.

    Gates removal of the generically-named siblings. See the class-2
    note above for why this guard exists.
    """
    return db_path.parent.name in _OWNED_STORE_DIRNAMES


def artifact_paths(
    db_path: Path, *, exclude: Path | None = None,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Every on-disk artifact belonging to the store at `db_path`.

    Returns `(owned, orphaned)`. `owned` is safe to delete; `orphaned`
    is the fixed-name siblings that exist but sit outside a recognised
    store directory, for the caller to report rather than remove.

    `exclude` drops one path from both lists — the CLI passes the
    archive destination so writing the archive next to the store cannot
    result in deleting it. Deterministic ordering: DB first (so a
    partially-failed removal never leaves the store as the sole
    survivor), then sorted sidecars, then sorted siblings.
    """
    owned: list[Path] = []
    orphaned: list[Path] = []
    parent = db_path.parent
    name = db_path.name

    if db_path.exists():
        owned.append(db_path)
    for suffix in _DB_SIDECAR_SUFFIXES:
        sidecar = parent / f"{name}{suffix}"
        if sidecar.exists():
            owned.append(sidecar)
    # `memory.db.*` catches the BM25F sidecar (`.bm25f`) and every backup
    # naming scheme in the wild (`.bak-<date>`, `.pre-clamp-<date>.bak`).
    # Anchored on the db filename, so it cannot reach unrelated files.
    for extra in sorted(parent.glob(f"{name}.*")):
        if extra not in owned:
            owned.append(extra)

    sink = owned if store_dir_is_owned(db_path) else orphaned
    for filename in _SIBLING_FILENAMES:
        candidate = parent / filename
        if candidate.exists() and candidate not in owned:
            sink.append(candidate)
    for dirname in _SIBLING_DIRNAMES:
        candidate = parent / dirname
        if candidate.is_dir():
            sink.append(candidate)

    if exclude is not None:
        owned = [p for p in owned if p != exclude]
        orphaned = [p for p in orphaned if p != exclude]
    return tuple(owned), tuple(orphaned)


# --- The ~/.aelfrice/ home directory (#1186) ----------------------------
#
# A second location, with different ownership from the store directory
# above. `~/.aelfrice/` belongs to no single store: it holds install-state
# sentinels, two capture logs, user configuration, and -- decisively --
# `projects/`, under which every *other* project's belief corpus lives.
# #1173 therefore left the whole directory alone, which meant the LLM
# consent sentinel outlived the uninstall that was supposed to end the
# relationship: a reinstall reads the surviving grant as current and never
# re-prompts (#1186, the same defect class as #1172 one layer out).
#
# So the contents are enumerated by name and never swept. Three
# dispositions:
#
#   install state -- removed in EVERY mode, `--keep-db` included. Each
#     records that a step already happened, so a survivor makes a
#     reinstall read a stale decision as current. The consent sentinel is
#     the load-bearing one.
#   data          -- removed only when the user asked for data to be
#     destroyed (`--purge` / `--archive`), matching the store-dir
#     contract. Kept, and reported as kept, under `--keep-db`.
#   preserved     -- recognised and deliberately kept in every mode.
#
# Anything not named below is reported and never deleted: this directory
# can hold files the package did not write. A blanket `rm -rf` here would
# destroy corpora the command was not asked to touch -- strictly worse
# than the bug being fixed.
#
# Literals rather than imports for the same reason as `_SIBLING_FILENAMES`
# (import-graph position), kept honest by the agreement tests in
# `test_uninstall_artifacts`.

_DOTDIR_INSTALL_STATE: Final[tuple[str, ...]] = (
    "llm-classify-consented",      # llm_classifier.SENTINEL_FILENAME
    "spine-backfilled",            # temporal_spine.SPINE_BACKFILLED_SENTINEL
    "installed-manifest-version",  # auto_install.STAMP_PATH
    "migrated-to-uv",              # MIGRATED_TO_UV_SENTINEL (this module)
    "mcp-surface-removed",         # mcp_cleanup.MCP_CLEANUP_SENTINEL (#1422)
    ".auto-install.lock",          # auto_install.AUTO_INSTALL_LOCK_FILENAME
    # claude_memory._RECONCILE_SENTINEL_NAME lives beside the store, and
    # lands here only for an in-memory store (`reconcile_sentinel_path`).
    "claude-memory-reconciled",
    # doctor.HOOK_FAILURES_LOG. The only path with a parent of its own;
    # `logs/` is pruned when removing this leaves it empty.
    "logs/hook-failures.log",
)

_DOTDIR_DATA: Final[tuple[str, ...]] = (
    "telemetry.jsonl",  # telemetry.DEFAULT_TELEMETRY_PATH
    "transcripts",      # transcript_logger.LEGACY_TRANSCRIPTS_DIR
)

_DOTDIR_PRESERVED: Final[tuple[str, ...]] = (
    # project_warm._CONFIG_FILENAME -- user configuration, same reasoning
    # as opt-out-hooks.json.
    "config.json",
    # auto_install.OPT_OUT_PATH -- records a user's decision that a hook
    # should not be installed; honouring it across a reinstall is the
    # whole point of the file.
    "opt-out-hooks.json",
    # doctor._AELFRICE_PROJECTS_DIR -- one subdirectory per project id,
    # each with its own memory.db. `uninstall` disposes of the single
    # store `db_path()` resolves to; the rest are not its to take.
    "projects",
    # Peer stores for read-only federation (#655). `knowledge_deps.json`
    # documents `~/.aelfrice/shared/<name>/memory.db` as the conventional
    # location, so this is another store's corpus by another name.
    "shared",
)


@dataclass(frozen=True)
class DotdirDisposition:
    """Outcome of `dispose_dotdir(...)` — what happened in `~/.aelfrice/`.

    `removed` and `failed` split the planned set by whether the path is
    actually gone (`_remove_artifact` swallows OSError, so a held handle
    or permission problem must be reported, not counted as success).
    `preserved` names recognised paths kept on purpose, and
    `unrecognised` names everything else found there — reported so the
    user can finish by hand, never deleted.
    """

    removed: tuple[Path, ...] = ()
    failed: tuple[Path, ...] = ()
    preserved: tuple[Path, ...] = ()
    unrecognised: tuple[Path, ...] = ()


def _dotdir_top_level(relpath: str) -> str:
    """First path segment of a removal-set entry (`logs/x.log` -> `logs`)."""
    return relpath.split("/", 1)[0]


def dotdir_plan(
    home: Path,
    *,
    include_data: bool,
    skip: Iterable[Path] = (),
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Classify the contents of `home` without touching anything.

    Returns `(planned, preserved, unrecognised)`. `include_data` is False
    for `--keep-db`, which moves the capture logs from `planned` to
    `preserved` rather than dropping them from the report.

    `skip` excludes paths another disposition already owns — the CLI
    passes the store artifact set, because `~/.aelfrice/` *is* the store
    directory on the non-git fallback (`db_paths.DEFAULT_DB_DIR`) and
    `artifact_paths` has already claimed `memory.db`, its sidecars and
    `transcripts/` there. Without this the two sets would double-report.

    `home` is required rather than defaulted to the real `~/.aelfrice/`
    so that no caller can sweep a developer's own directory by omission.
    """
    if not home.is_dir():
        return (), (), ()
    skip_set = {Path(p) for p in skip}

    planned: list[Path] = []
    preserved: list[Path] = []
    unrecognised: list[Path] = []

    for relpath in _DOTDIR_INSTALL_STATE:
        candidate = home / relpath
        if candidate not in skip_set and _path_present(candidate):
            planned.append(candidate)
    for relpath in _DOTDIR_DATA:
        candidate = home / relpath
        if candidate in skip_set or not _path_present(candidate):
            continue
        (planned if include_data else preserved).append(candidate)

    # A directory that exists only to hold artifacts now scheduled for
    # removal goes too, so uninstall leaves no empty shell behind. Named
    # in the plan (appended after its contents, which is also the correct
    # removal order) so the gate discloses it rather than the disposition
    # deleting something the manifest never mentioned. A directory holding
    # anything *not* in `planned` fails the subset test and is left alone.
    for relpath in _DOTDIR_INSTALL_STATE + _DOTDIR_DATA:
        if "/" not in relpath:
            continue
        parent = home / _dotdir_top_level(relpath)
        if not parent.is_dir() or parent in planned or parent in skip_set:
            continue
        try:
            contents = set(parent.iterdir())
        except OSError:
            continue
        if contents and contents <= set(planned):
            planned.append(parent)

    # Everything else at the top level: recognised-and-kept, or unknown.
    #
    # A named parent that was NOT pruned above (its contents are not a
    # subset of `planned`, so it holds something the package did not
    # write) must not stay accounted-for: its stray children are neither
    # deleted nor reported otherwise, and "reported, never deleted" is
    # only half kept if the report stops at the top level. Report those
    # children individually rather than the directory, so the user sees
    # the file rather than the folder aelfrice also uses.
    accounted = {
        _dotdir_top_level(r) for r in _DOTDIR_INSTALL_STATE + _DOTDIR_DATA
    }
    planned_set = set(planned)
    seen_unrecognised: set[Path] = set()
    for relpath in _DOTDIR_INSTALL_STATE + _DOTDIR_DATA:
        if "/" not in relpath:
            continue
        parent = home / _dotdir_top_level(relpath)
        if parent in planned_set or parent in skip_set or not parent.is_dir():
            continue
        try:
            strays = sorted(
                c for c in parent.iterdir()
                if c not in planned_set and c not in skip_set
            )
        except OSError:
            continue
        # Dedup against a set, not the list. `s not in unrecognised`
        # rescans a list that grows as the loop runs, so classifying a
        # directory cost O(n^2) path comparisons — 0.5s at 4k strays,
        # and `logs/` on a long-lived store reaches five figures, which
        # made `aelf uninstall` look hung before printing anything
        # (#1202). The list still carries the order.
        for stray in strays:
            if stray not in seen_unrecognised:
                seen_unrecognised.add(stray)
                unrecognised.append(stray)
    try:
        entries = sorted(home.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry in skip_set or entry.name in accounted:
            continue
        if entry.name in _DOTDIR_PRESERVED:
            preserved.append(entry)
        else:
            unrecognised.append(entry)

    return tuple(planned), tuple(preserved), tuple(unrecognised)


def _path_present(path: Path) -> bool:
    """True when `path` exists, counting a broken symlink as present."""
    return path.exists() or path.is_symlink()


def dispose_dotdir(
    home: Path,
    *,
    include_data: bool,
    skip: Iterable[Path] = (),
) -> DotdirDisposition:
    """Remove the package's own artifacts from `home`, and nothing else.

    The destructive half of `dotdir_plan`, whose classification it uses
    verbatim — the CLI prints that plan before prompting, so what gets
    deleted is exactly what was disclosed, `logs/` included.
    """
    planned, preserved, unrecognised = dotdir_plan(
        home, include_data=include_data, skip=skip,
    )
    removed: list[Path] = []
    failed: list[Path] = []
    for path in planned:
        (removed if _remove_artifact(path) else failed).append(path)

    return DotdirDisposition(
        removed=tuple(removed),
        failed=tuple(failed),
        preserved=preserved,
        unrecognised=unrecognised,
    )


def checkpoint_wal(db_path: Path) -> bool:
    """Fold the write-ahead log back into `db_path`. True if it ran.

    Load-bearing for `--archive`, which encrypts `db_path.read_bytes()`.
    A store whose WAL has not been checkpointed keeps its most recent
    (and possibly *all* of its) committed content in `memory.db-wal`, so
    reading the main file alone yields a valid-but-stale database. On a
    store written by a still-open process that meant archiving zero
    beliefs while reporting success — see #1173.

    Best-effort by design: a corrupt or locked store must not block the
    user from removing the package, and the caller deletes the WAL
    either way. Returns False when the checkpoint could not run, which
    for the archive path means the archive may be incomplete.
    """
    if not db_path.exists():
        return False
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path), timeout=5.0)
    except sqlite3.Error:
        return False
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _remove_artifact(path: Path) -> bool:
    """Delete one file or directory tree. True when it is gone after.

    Swallows OSError for the same reason the pre-#1173 purge did: a
    permission problem on one artifact must not abort the disposition
    of the rest. The return value feeds the caller's manifest, so a
    failure is reported rather than silently counted as removed.
    """
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not path.exists()


def _encrypt_db_to_archive(
    db_path: Path, archive_path: Path, password: str
) -> None:
    """Encrypt `db_path`'s contents to `archive_path` with `password`.

    Format: 8-byte magic | 16-byte salt | Fernet-token over the DB.
    Key is derived via scrypt(password, salt, N=2**14, r=8, p=1, len=32)
    and base64-urlsafe-encoded (Fernet's required encoding). The same
    parameters are recoverable from the archive header alone, so the
    user only needs the password to decrypt.
    """
    try:
        import base64
        import secrets

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:
        raise RuntimeError(
            "--archive requires the 'archive' extra: "
            "pip install 'aelfrice[archive]'"
        ) from exc
    if not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(ARCHIVE_SALT_LEN)
    kdf = Scrypt(
        salt=salt, length=ARCHIVE_KEY_LEN,
        n=ARCHIVE_SCRYPT_N, r=ARCHIVE_SCRYPT_R, p=ARCHIVE_SCRYPT_P,
    )
    raw_key = kdf.derive(password.encode("utf-8"))
    fernet_key = base64.urlsafe_b64encode(raw_key)
    f = Fernet(fernet_key)
    plaintext = db_path.read_bytes()
    token = f.encrypt(plaintext)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as out:
        out.write(ARCHIVE_MAGIC)
        out.write(salt)
        out.write(token)


def decrypt_archive(archive_path: Path, password: str) -> bytes:
    """Decrypt an archive produced by `_encrypt_db_to_archive`.

    Public: shipped so future tooling (or curious users) can recover an
    archived DB. Returns the decrypted SQLite bytes; the caller is
    responsible for writing them somewhere.
    """
    try:
        import base64

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:
        raise RuntimeError(
            "decrypt_archive requires: pip install 'aelfrice[archive]'"
        ) from exc
    blob = archive_path.read_bytes()
    if not blob.startswith(ARCHIVE_MAGIC):
        raise ValueError(
            f"not an aelfrice archive (bad magic): {archive_path}"
        )
    header_end = len(ARCHIVE_MAGIC) + ARCHIVE_SALT_LEN
    salt = blob[len(ARCHIVE_MAGIC):header_end]
    token = blob[header_end:]
    kdf = Scrypt(
        salt=salt, length=ARCHIVE_KEY_LEN,
        n=ARCHIVE_SCRYPT_N, r=ARCHIVE_SCRYPT_R, p=ARCHIVE_SCRYPT_P,
    )
    raw_key = kdf.derive(password.encode("utf-8"))
    fernet_key = base64.urlsafe_b64encode(raw_key)
    f = Fernet(fernet_key)
    return f.decrypt(token)


def uninstall(
    db_path: Path,
    *,
    keep_db: bool = False,
    purge: bool = False,
    archive_path: Path | None = None,
    archive_password: str | None = None,
) -> UninstallResult:
    """Apply the data-disposition choice for `aelf uninstall`.

    Exactly one of `keep_db`, `purge`, `archive_path` must be specified.
    The CLI is responsible for prompting the user; this function is
    pure mechanism. The hook removal and pip uninstallation happen
    elsewhere -- this is the data half only.

    Both destructive modes operate on the full artifact set
    (`artifact_paths`), not on `memory.db` alone. Before #1173 a
    `--purge` left the WAL, the backup DBs, the BM25F index, the
    transcripts directory and a verbatim injection audit log in place;
    an `--archive` additionally encrypted a *stale* main DB file while
    leaving the live content beside it in plaintext.

    `--archive` still encrypts the belief database only. The remaining
    artifacts are derived from it (BM25F index, audit log, feed log,
    telemetry) or are rolling capture buffers (transcripts), so they are
    securely removed rather than added to the archive -- keeping the
    shipped `decrypt_archive` contract (archive bytes ARE the SQLite
    file) intact. `UninstallResult.removed` names every one of them so
    the caller can say so out loud.
    """
    chosen = sum(
        [bool(keep_db), bool(purge), archive_path is not None]
    )
    if chosen != 1:
        raise ValueError(
            "exactly one of keep_db / purge / archive_path required"
        )
    if keep_db:
        return UninstallResult(
            mode="kept",
            db_path=db_path if db_path.exists() else None,
        )
    if archive_path is not None:
        if archive_password is None:
            raise ValueError("archive_password required when archive_path set")
        if not db_path.exists():
            # Nothing to archive; surface as 'kept' so caller can warn.
            return UninstallResult(
                mode="kept", db_path=None, archive_path=None,
            )
        # Order matters: checkpoint first so the bytes we encrypt include
        # everything committed to the WAL, then enumerate (the checkpoint
        # truncates the WAL but does not remove it), then encrypt, and
        # only delete once the archive is on disk.
        checkpoint_wal(db_path)
        owned, orphaned = artifact_paths(
            db_path, exclude=archive_path.resolve(),
        )
        _encrypt_db_to_archive(db_path, archive_path, archive_password)
        removed = tuple(p for p in owned if _remove_artifact(p))
        return UninstallResult(
            mode="archived", db_path=None, archive_path=archive_path,
            removed=removed, orphaned=orphaned,
        )
    # purge. Checkpointing first is not needed to destroy the data (the
    # WAL is in the removal set), but it keeps the manifest honest: a
    # truncated WAL reports its real post-checkpoint size.
    checkpoint_wal(db_path)
    owned, orphaned = artifact_paths(db_path)
    removed = tuple(p for p in owned if _remove_artifact(p))
    return UninstallResult(
        mode="purged", db_path=None, removed=removed, orphaned=orphaned,
    )


def clear_cache(cache_path: Path | None = None) -> None:
    """Remove the update-check cache file. Silent if absent.

    Called by `aelf upgrade` after a successful upgrade so the orange
    statusline banner disappears immediately.

    `cache_path` resolves from `CACHE_FILE` when omitted — late-bound
    (#1320; see `_write_cache`).
    """
    if cache_path is None:
        cache_path = CACHE_FILE
    try:
        cache_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# --- Migrate non-uv install to uv tool (#733) ---------------------------

# Sentinel marking a successful migration. Persists across versions; once
# a host has migrated, `aelf setup` short-circuits the migration check.
# Lives alongside the auto_install stamp under ~/.aelfrice/.
MIGRATED_TO_UV_SENTINEL: Final[Path] = (
    Path.home() / ".aelfrice" / "migrated-to-uv"
)

# Maximum wall-clock seconds for `uv tool install --force aelfrice`.
# Bounded so a hung uv install does not block `aelf setup` indefinitely.
MIGRATION_TIMEOUT_SECONDS: Final[int] = 120


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a `maybe_migrate_to_uv()` call.

    `attempted` is True iff we actually invoked the subprocess.
    `succeeded` is True iff the subprocess returned 0 and the sentinel
    was written. `reason` is a human-readable description suitable for
    a single-line stderr notice, populated in both the skipped and the
    failed paths as well as the succeeded path (to name the orphan).
    """

    attempted: bool
    succeeded: bool
    reason: str


def maybe_migrate_to_uv(
    *,
    sentinel_path: Path | None = None,
    timeout: int = MIGRATION_TIMEOUT_SECONDS,
    force: bool = False,
) -> MigrationResult:
    """Migrate a non-uv aelfrice install to `uv tool install aelfrice`.

    Idempotent: writes `sentinel_path` after a successful subprocess
    return; subsequent calls short-circuit on the sentinel unless
    `force=True`. Never raises — every failure mode returns a
    `MigrationResult` describing what happened.

    Short-circuit order (cheapest first):
      1. sentinel exists → no-op
      2. running install is already uv_tool → no-op
      3. `uv` is not on PATH → skip with install-uv hint
      4. subprocess `uv tool install --force aelfrice`:
         - success → write sentinel, return succeeded with orphan hint
         - non-zero exit → return failed with stderr excerpt
         - timeout / OSError → return failed with descriptive reason

    The `uv tool install --force aelfrice` form overwrites the existing
    `~/.local/bin/aelf` shim (which uv tool and pipx both target). The
    running process — still under the pipx venv — continues to function
    until exit; future invocations resolve through the new uv shim.

    `sentinel_path` resolves from the module-level
    `MIGRATED_TO_UV_SENTINEL` when omitted — late-bound so tests
    patching the constant are honoured (#1320). Note the polarity: the
    guard is `sentinel_path.exists()`, so pinning this at a path that
    does NOT exist arms the subprocess rather than disarming it.
    """
    if sentinel_path is None:
        sentinel_path = MIGRATED_TO_UV_SENTINEL
    if not force and sentinel_path.exists():
        return MigrationResult(False, False, "already migrated (sentinel exists)")
    advice = upgrade_advice()
    if advice.context == "uv_tool":
        return MigrationResult(False, False, "already on uv tool")
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        return MigrationResult(
            False,
            False,
            "uv not on PATH — install with `curl -LsSf "
            "https://astral.sh/uv/install.sh | sh` or `brew install uv` "
            "(see https://docs.astral.sh/uv/), then re-run /aelf:upgrade",
        )
    try:
        proc = subprocess.run(
            [uv_bin, "tool", "install", "--force", PACKAGE_NAME],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return MigrationResult(
            True,
            False,
            f"uv tool install timed out after {timeout}s — pipx install left untouched",
        )
    except OSError as exc:
        return MigrationResult(
            True, False, f"uv tool install failed to launch: {exc}"
        )
    if proc.returncode != 0:
        stderr_excerpt = (proc.stderr or "").strip().splitlines()
        tail = stderr_excerpt[-1] if stderr_excerpt else "(no stderr)"
        return MigrationResult(
            True,
            False,
            f"uv tool install exited {proc.returncode}: {tail[:200]}",
        )
    # Success: write the sentinel before reporting, so a crash between
    # subprocess return and notice print still leaves the host marked
    # as migrated. The sentinel is a 2KB-or-less metadata file; we
    # tolerate a sentinel-write failure (very rare) by returning
    # succeeded=True with a reason mentioning the orphan and the
    # missing sentinel — the operator can re-run /aelf:upgrade safely.
    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"migrated from {advice.context} at {time.time():.0f}\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    if _is_pipx_install():
        orphan_note = (
            "orphan pipx venv at ~/.local/pipx/venvs/aelfrice — "
            "remove with `pipx uninstall aelfrice` at your leisure"
        )
    else:
        # _is_venv() / system fall through here — pip is the right verb.
        orphan_note = (
            "orphan pip install left in place — "
            "remove with `pip uninstall -y aelfrice` after this process exits"
        )
    return MigrationResult(
        True, True, f"migrated to uv tool; {orphan_note}"
    )
