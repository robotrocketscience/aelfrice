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
        # `shlex` defaults `commenters` to '#', which is not comment syntax in
        # a Windows command line — it is a legal path character. Left enabled,
        # ``C:\\Users\\dev#1\\...\\aelf-hook.exe`` truncates to
        # ``C:\\Users\\dev`` and ownership is lost for that user exactly the
        # way the unfixed splitter loses it. The POSIX branch below does not
        # need this: `shlex.split` passes `comments=False`.
        lexer.commenters = ""
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


def _has_launcher_suffix(token: str, *, windows: bool) -> bool:
    """Does this token's basename end in a recognised launcher suffix?"""
    base = launcher_basename(token, windows=windows).casefold()
    _stem, dot, suffix = base.rpartition(".")
    return bool(dot) and f".{suffix}" in LAUNCHER_SUFFIXES


def _spaced_program_token(tokens: list[str], *, windows: bool) -> str:
    """Rejoin an unquoted program path that whitespace-splitting broke up.

    `setup._resolve_script` writes the resolved path **unquoted** — quoting
    changes bytes on POSIX too, and nothing in this repo pins whether Codex
    `exec`s or shells out. On Windows that path routinely contains a space: a
    system interpreter installs under ``C:\\Program Files\\...`` and any user
    profile may carry one. The program is then spread across several tokens
    and ``tokens[0]`` is a fragment (``C:\\Program``), whose key is
    ``program`` — so our own handler stops being recognised as ours and #1412's
    whole symptom table reproduces: setup appends a duplicate group per event
    on every run, doctor counts zero handlers, unsetup removes nothing.

    Returns the shortest prefix whose basename carries a launcher suffix, or
    ``""`` if none does.

    Switch-shaped tokens stop the scan rather than being joined over, so
    ``wrapper.exe --out foo.exe`` cannot be read as a program named ``foo``.
    Ownership drives deletion here, and widening it is the unsafe direction.
    """
    for k in range(2, len(tokens) + 1):
        if any(tok.startswith(("-", "/")) for tok in tokens[1:k]):
            break
        candidate = " ".join(tokens[:k])
        if _has_launcher_suffix(candidate, windows=windows):
            return candidate
    return ""


def command_launcher_key(command: str, *, windows: bool | None = None) -> str:
    """`launcher_key` of a whole command string's program token.

    On Windows the program token is recovered across whitespace when the
    written command holds an unquoted path with spaces — see
    `_spaced_program_token`. POSIX is untouched: the module docstring's
    no-widening rule applies, and a POSIX command is written from a resolved
    path this project controls.
    """
    win = _resolve_windows(windows)
    tokens = command_tokens(command, windows=win)
    if not tokens:
        return ""
    if win and not _has_launcher_suffix(tokens[0], windows=win):
        rejoined = _spaced_program_token(tokens, windows=win)
        if rejoined:
            return launcher_key(rejoined, windows=win)
    return launcher_key(tokens[0], windows=win)


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

    The leaf is `Scripts` on Windows and `bin` on POSIX; the old
    ``Path(sys.prefix) / "bin"`` hardcoded the POSIX one, so on Windows the
    venv branch pointed at a directory that does not exist and every caller
    silently degraded to a bare `PATH` search.

    `sys.prefix` is read **at call time and passed in explicitly**. A bare
    `sysconfig.get_path("scripts")` is wrong here: `sysconfig` captures the
    prefix variables when it is first imported, so it does not follow a
    reassigned `sys.prefix` — which is the seam `tests/test_setup_resolution
    .py` drives, and which any in-process venv switch would need too. That
    version passed the whole suite locally and turned six tests red on CI,
    because this checkout is a worktree and `_is_worktree_path` discarded
    the wrong answer before it could be compared.
    """
    scheme = "nt" if os.name == "nt" else "posix_prefix"
    try:
        configured = sysconfig.get_path(
            "scripts", scheme, vars={
                "base": sys.prefix, "installed_base": sys.prefix,
                "platbase": sys.prefix, "installed_platbase": sys.prefix,
            },
        )
    except KeyError:  # a scheme without a "scripts" path
        configured = ""
    if configured:
        return Path(configured)
    return Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")


def which_in(directory: Path, name: str) -> Path | None:
    """Resolve `name` inside `directory`, honouring Windows PATHEXT.

    Probes `directory` directly rather than delegating to `shutil.which`,
    because routing a single directory through a `PATH`-shaped parameter is
    wrong in two independent ways:

    * `shutil.which` splits `path=` on `os.pathsep`, so a directory whose own
      name contains that separator is never searched. On POSIX that is ``:``,
      and ``/home/a:b/.venv/bin`` is a legal path — the delegation lost a
      resolution that the plain ``directory / name`` probe it replaced found,
      a POSIX regression in a change whose stated scope was Windows-only.
    * On win32 `shutil.which` inserts `os.curdir` at the front of the search
      list whenever the command has no directory part. That insertion sits in
      the `else` branch of the dirname test, **not** under `if path is None`
      (CPython 3.12-3.14, `shutil.py`), so passing `path=` does not confine
      anything: a stray ``aelf-hook.exe`` in the process's working directory
      would be returned as if it were the venv's, and `setup._resolve_script`
      would pin that relative path into settings.json.

    PATHEXT is applied on Windows so ``aelf-hook`` still finds
    ``aelf-hook.exe``. The executable bit is checked only on POSIX:
    `os.access(..., X_OK)` is meaningless on Windows, where it answers True
    for any existing file.
    """
    if _resolve_windows(None):
        raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
        suffixes = ["", *(ext for ext in raw.split(os.pathsep) if ext)]
    else:
        suffixes = [""]
    for suffix in suffixes:
        candidate = directory / f"{name}{suffix}"
        if not candidate.is_file():
            continue
        if _resolve_windows(None) or os.access(candidate, os.X_OK):
            return candidate
    return None


def which_on_path(name: str) -> Path | None:
    """Resolve `name` on `PATH`, without the win32 current-directory search.

    CPython's `shutil.which` prepends `os.curdir` to the search path on win32
    whenever the command has no directory part. That insertion lives in the
    `else` branch of the dirname test and is **not** gated on `path is None`
    (verified in `shutil.py` for 3.12-3.14; `requires-python` here is >=3.12),
    so passing `path=` explicitly does not avoid it. An earlier revision of
    this function claimed it did, and shipped a fix that changed nothing: a
    stray ``aelf-hook.exe`` in whatever directory the user ran `aelf setup`
    from still won, and `setup._resolve_script` still pinned it into
    settings.json.

    The result is therefore filtered rather than the search restrained: the
    hit is kept only if its parent is one of the directories actually named
    on `PATH`. If the working directory is itself on `PATH`, it is a
    legitimate hit and survives.
    """
    raw = os.environ.get("PATH", os.defpath)
    found = shutil.which(name, path=raw)
    if found is None:
        return None
    resolved = Path(found)
    entries: set[Path] = set()
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        try:
            entries.add(Path(entry).resolve())
        except OSError:  # an unreadable or malformed PATH entry
            continue
    try:
        parent = resolved.parent.resolve()
    except OSError:
        return None
    return resolved if parent in entries else None
