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
CACHE_TTL_SECONDS: Final[int] = 6 * 60 * 60  # 6h: rapid-dev cadence
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
    """Resolve the installed aelfrice version.

    Reads aelfrice.__version__ which is the source of truth. Falls back
    to '0.0.0' if the import is unexpectedly broken (defensive only).
    """
    try:
        from aelfrice import __version__

        return str(__version__)
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


# --- Upgrade advice -----------------------------------------------------


@dataclass(frozen=True)
class UpgradeAdvice:
    """How to upgrade aelfrice in the user's specific install context."""

    command: str
    context: str  # 'venv' | 'pipx' | 'system' | 'unknown'


def _is_pipx_install() -> bool:
    """Detect a pipx-managed install by checking sys.prefix path.

    pipx installs each package into ~/.local/pipx/venvs/<pkg>/ -- the
    presence of '/pipx/venvs/' in sys.prefix is the canonical signal.
    """
    import sys

    return "/pipx/venvs/" in sys.prefix.replace("\\", "/")


def _is_venv() -> bool:
    """Detect a generic venv (PEP 405 / virtualenv).

    sys.prefix != sys.base_prefix is the standard idiom; works for
    venv, virtualenv, uv venv, and conda envs.
    """
    import sys

    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def upgrade_advice() -> UpgradeAdvice:
    """Return the right pip-upgrade incantation for the running env.

    pipx checked before generic venv because a pipx install IS a venv,
    but its upgrade path is different (pipx upgrade, not pip install).
    """
    if _is_pipx_install():
        return UpgradeAdvice(
            command=f"pipx upgrade {PACKAGE_NAME}",
            context="pipx",
        )
    if _is_venv():
        return UpgradeAdvice(
            command=f"pip install --upgrade {PACKAGE_NAME}",
            context="venv",
        )
    # Fall through: system / user-site install. --user is the safest
    # default since system-site requires root and most users don't
    # want to sudo pip install.
    return UpgradeAdvice(
        command=f"pip install --user --upgrade {PACKAGE_NAME}",
        context="system",
    )


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
