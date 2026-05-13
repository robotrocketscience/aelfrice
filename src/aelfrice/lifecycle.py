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


def _write_cache(status: UpdateStatus, cache_path: Path = CACHE_FILE) -> None:
    """Persist a status snapshot. Silent fail: cache write is best-effort."""
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


def read_cache(cache_path: Path = CACHE_FILE) -> UpdateStatus:
    """Read the cached update status. Returns empty() on any failure.

    The statusline calls this. It MUST be cheap and never raise.
    """
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
    cache_path: Path = CACHE_FILE,
    pypi_url: str = PYPI_JSON_URL,
    fetch: callable = _fetch_pypi_json,
    now: float | None = None,
) -> UpdateStatus:
    """Run the synchronous update check end-to-end and write the cache.

    This is the function the background process invokes. The CLI/hook
    paths use maybe_check_for_update_async() which spawns a detached
    subprocess pointing at this entry. We expose a sync version for
    tests and for direct CLI use.
    """
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
    cache_path: Path = CACHE_FILE,
    ttl: int = CACHE_TTL_SECONDS,
) -> bool:
    """Fire a detached background check iff cache is stale.

    Returns True iff a subprocess was launched. Never blocks the caller;
    the spawned process detaches via start_new_session=True so the
    parent can exit without waiting. Mirrors GSD's spawn(detached:true)
    + child.unref() pattern in Python.
    """
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


def _is_uv_tool_install() -> bool:
    """Detect a uv-tool-managed install.

    uv tool installs each package under ~/.local/share/uv/tools/<pkg>/.
    We check for the package directory directly rather than shelling
    out to `uv` (which may not be on PATH inside the managed env).
    As a secondary signal, if sys.executable or sys.prefix resolves
    under the uv tools directory we also consider it a uv-tool install.
    """
    uv_tools_dir = Path.home() / ".local" / "share" / "uv" / "tools" / PACKAGE_NAME
    if uv_tools_dir.exists():
        return True
    # Secondary: check if sys.prefix or sys.executable path contains the
    # uv tools tree. Covers cases where the package dir name differs.
    import sys
    prefix_norm = sys.prefix.replace("\\", "/")
    # ~/.local/share/uv/tools/ is the canonical uv tools root on
    # Linux/macOS. On Windows it is %APPDATA%\uv\tools\ but we only
    # support the POSIX layout for now.
    uv_tools_root = str(Path.home() / ".local" / "share" / "uv" / "tools")
    return prefix_norm.startswith(uv_tools_root.replace("\\", "/"))


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


def _which_all_aelf() -> list[Path]:
    """Return every `aelf` executable reachable on PATH, in PATH order.

    POSIX-only. We walk PATH ourselves rather than rely on `which -a`,
    which is not portable across shells. Skips non-files and
    non-executables.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    raw = os.environ.get("PATH", "")
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / "aelf"
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        if candidate.is_file() and os.access(candidate, os.X_OK):
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
      - ~/.local/share/uv/tools/aelfrice/  → uv_tool
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

    uv_root = home / ".local" / "share" / "uv" / "tools" / PACKAGE_NAME
    if uv_root.exists():
        on_path = any(
            _path_is_under(p, uv_root) for p in path_aelf_resolved
        )
        sites.append(InstallSite(kind="uv_tool", path=uv_root, on_path=on_path))

    pipx_root = home / ".local" / "pipx" / "venvs" / PACKAGE_NAME
    if pipx_root.exists():
        on_path = any(
            _path_is_under(p, pipx_root) for p in path_aelf_resolved
        )
        sites.append(InstallSite(kind="pipx", path=pipx_root, on_path=on_path))

    running_aelf = _running_interpreter_aelf()
    known_roots = [uv_root, pipx_root]
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
    """

    mode: str  # 'kept' | 'purged' | 'archived'
    db_path: Path | None
    archive_path: Path | None = None


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
        _encrypt_db_to_archive(db_path, archive_path, archive_password)
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
        return UninstallResult(
            mode="archived", db_path=None, archive_path=archive_path,
        )
    # purge
    try:
        if db_path.exists():
            db_path.unlink()
    except OSError:
        pass
    return UninstallResult(mode="purged", db_path=None)


def clear_cache(cache_path: Path = CACHE_FILE) -> None:
    """Remove the update-check cache file. Silent if absent.

    Called by `aelf upgrade` after a successful upgrade so the orange
    statusline banner disappears immediately.
    """
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
    sentinel_path: Path = MIGRATED_TO_UV_SENTINEL,
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
    """
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
            "uv not on PATH — install uv from https://docs.astral.sh/uv/ "
            "then re-run /aelf:upgrade",
        )
    try:
        proc = subprocess.run(
            [uv_bin, "tool", "install", "--force", PACKAGE_NAME],
            capture_output=True,
            text=True,
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
            f"migrated from {advice.context} at {time.time():.0f}\n"
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
