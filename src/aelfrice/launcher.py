"""Console-launcher naming and resolution, shared by setup/doctor/host_codex.

#1412. Three modules independently decided what a hook command's *program*
is, and all three did it the same wrong way::

    Path(cmd.split()[0]).name

That is wrong on Windows twice over. `cmd.split()` breaks a path containing
spaces, and `Path` is `PosixPath` on a POSIX interpreter, so
``PurePosixPath(r"C:\\Scripts\\aelf-hook.EXE").name`` is the *entire string*
rather than ``aelf-hook.EXE``. Even given the right basename, ownership was
then decided by comparing against a suffixless, case-sensitive frozenset, so
a real Windows console launcher was never recognised as ours. Setup appended
a duplicate group on every run, doctor reported zero handlers, and unsetup
left the originals behind.

Two properties this module exists to guarantee:

*Normalisation is platform-gated.* On POSIX nothing changes — comparison
stays case-sensitive and suffixes stay significant. Case-folding
unconditionally would *widen* ownership on a case-sensitive filesystem, and
since ownership drives a deletion path (`remove_codex_hooks`,
`prune_broken_aelf_hooks`), widening it is how you delete a file someone else
owns.

*The platform is read at call time, never bound at definition time.* The
natural signature — ``def launcher_key(cmd, *, windows: bool = os.name ==
"nt")`` — evaluates its default once, at import, so
``monkeypatch.setattr(os, "name", "nt")`` is inert and the obvious regression
tests pass identically before and after the fix. `windows` therefore defaults
to `None` and resolves through `_resolve_windows` on every call. See
`tests/test_windows_launcher_1412.py::test_the_platform_flag_is_not_bound_at_definition_time`.
"""
from __future__ import annotations

import os
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

# Suffixes Windows treats as an executable launcher. `.ps1` is absent
# deliberately: PATHEXT does not include it by default and a `.ps1` is run
# *by* an interpreter rather than being one, so stripping it would equate a
# script with the console launcher of the same stem.
LAUNCHER_SUFFIXES: Final[frozenset[str]] = frozenset({
    ".exe", ".cmd", ".bat", ".com",
})


def _resolve_windows(windows: bool | None) -> bool:
    """Resolve the platform gate, reading `os.name` at call time."""
    return os.name == "nt" if windows is None else windows


def program_token(command: str, *, windows: bool | None = None) -> str:
    """The program token of a command string, quotes stripped, '' on miss.

    Quote-tolerant on *read* only: a command written as ``"C:\\Path With
    Spaces\\aelf-hook.exe" --flag`` yields the path without the quotes. This
    module never *writes* a quoted form — that changes bytes on POSIX too and
    breaks outright against a host that `exec`s rather than shelling out, and
    nothing in this repo pins which Codex does.
    """
    stripped = command.strip()
    if not stripped:
        return ""
    tokens = command_tokens(stripped, windows=windows)
    return tokens[0] if tokens else ""


def command_tokens(command: str, *, windows: bool | None = None) -> list[str]:
    """Split a command into tokens without eating Windows path separators.

    `shlex.split` defaults to POSIX mode, where a backslash is an escape
    character, so it silently destroys ``C:\\Scripts\\aelf-hook.exe`` ->
    ``C:Scriptsaelf-hook.exe``. That is the mechanism behind the
    `prune_broken_aelf_hooks` data loss: the mangled path does not exist, the
    hook is classified broken, and `aelf setup` deletes a working install on
    the next run. Non-POSIX mode preserves backslashes and still honours
    quoting; it leaves the quotes on the token, so they are stripped here.
    """
    stripped = command.strip()
    if not stripped:
        return []
    if _resolve_windows(windows):
        lexer = shlex.shlex(stripped, posix=False)
        lexer.whitespace_split = True
        try:
            return [tok.strip('"') for tok in lexer]
        except ValueError:
            # Unbalanced quoting — fall back to whitespace splitting rather
            # than reporting no program at all.
            return [tok.strip('"') for tok in stripped.split()]
    try:
        return shlex.split(stripped)
    except ValueError:
        return []


def launcher_basename(token: str, *, windows: bool | None = None) -> str:
    """Basename of a path token under the *target* platform's separators."""
    if not token:
        return ""
    if _resolve_windows(windows):
        return PureWindowsPath(token).name
    return PurePosixPath(token).name


def launcher_key(token: str, *, windows: bool | None = None) -> str:
    """Ownership-comparison key for a path token or bare name.

    On Windows: case-folded, with one recognised launcher suffix removed, so
    ``C:\\Scripts\\Aelf-Hook.EXE`` and ``aelf-hook`` compare equal. On POSIX:
    the plain basename, unchanged — see the module docstring on why widening
    is unsafe there.
    """
    win = _resolve_windows(windows)
    base = launcher_basename(token, windows=win)
    if not base or not win:
        return base
    folded = base.casefold()
    stem, dot, suffix = folded.rpartition(".")
    if dot and f".{suffix}" in LAUNCHER_SUFFIXES:
        return stem
    return folded


def command_launcher_key(command: str, *, windows: bool | None = None) -> str:
    """`launcher_key` of a whole command string's program token."""
    win = _resolve_windows(windows)
    return launcher_key(program_token(command, windows=win), windows=win)


def owned_keys(
    basenames: frozenset[str] | set[str] | tuple[str, ...],
    *,
    windows: bool | None = None,
) -> frozenset[str]:
    """Normalise a set of owned entry-point names into comparison keys."""
    win = _resolve_windows(windows)
    return frozenset(launcher_key(name, windows=win) for name in basenames)


def scripts_dir() -> Path:
    """Directory holding console scripts for the active interpreter.

    `sysconfig.get_path("scripts")` is the only correct source: the previous
    ``Path(sys.prefix) / "bin"`` hardcoded the POSIX leaf, so on Windows the
    venv branch pointed at a directory that does not exist and every caller
    silently degraded to a bare `PATH` search.
    """
    configured = sysconfig.get_path("scripts")
    if configured:
        return Path(configured)
    return Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")


def which_in(directory: Path, name: str) -> Path | None:
    """Resolve `name` inside `directory`, honouring Windows PATHEXT.

    `shutil.which` with an explicit `path=` applies PATHEXT on win32, so
    ``aelf-hook`` finds ``aelf-hook.exe``. `os.access(..., X_OK)` — what this
    replaces — is meaningless on Windows, where it returns True for any
    existing file.
    """
    found = shutil.which(name, path=str(directory))
    return Path(found) if found else None


def which_on_path(name: str) -> Path | None:
    """Resolve `name` on `PATH`, without the win32 current-directory search.

    CPython's `shutil.which` prepends `os.curdir` to the search path on
    win32 **when `path is None`**. A bare `shutil.which("aelf-hook")` can
    therefore resolve a `aelf-hook.exe` sitting in whatever directory the
    user happened to run `aelf setup` from, and pin that absolute path into
    settings.json. Passing `path=` explicitly stays inside the `path is not
    None` branch, where no such insertion happens.
    """
    found = shutil.which(name, path=os.environ.get("PATH", os.defpath))
    return Path(found) if found else None
