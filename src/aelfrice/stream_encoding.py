"""Make stdout/stderr able to carry the characters we actually print (#1329).

Found by the `windows-latest` job added in the same change, which is the
reason that job exists. With the `fcntl` import fixed, `aelf --help` got
*further* — argparse ran — and then died anyway::

    File "...\\encodings\\cp1252.py", line 19, in encode
      return codecs.charmap_encode(input, self.errors, encoding_table)[0]
    UnicodeEncodeError: 'charmap' codec can't encode character ...

A Windows console defaults to the legacy ANSI code page (`cp1252` on
en-US), and Python encodes stdout with it. Our help text, banners and
belief content are full of characters cp1252 has no byte for — em dashes,
arrows, `≥`, box drawing, and any non-Latin-1 text a user has stored. So
the failure is not exotic: printing help crashed, and printing a belief
containing a curly quote would too.

Reconfiguring is the fix rather than sanitising the strings, because the
content is not ours to sanitise — a belief is whatever the user wrote, and
mangling it on the way to the model is a worse outcome than a wide console
encoding. `errors="replace"` is a backstop for a stream that cannot be
reconfigured to UTF-8 at all; it should not normally be reached.

No-op on POSIX, where the streams are already UTF-8, and no-op on Windows
when the user has set `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` themselves.
"""
from __future__ import annotations

import sys
from typing import IO, Any, cast

_UTF8_ALIASES = frozenset({"utf8", "utf_8"})


def _is_utf8(encoding: str | None) -> bool:
    if not encoding:
        return False
    return encoding.lower().replace("-", "_").replace("_", "") in {
        alias.replace("_", "") for alias in _UTF8_ALIASES
    }


def ensure_utf8_streams(streams: tuple[IO[str], ...] | None = None) -> None:
    """Reconfigure `sys.stdout`/`sys.stderr` to UTF-8 when they are not.

    Idempotent and non-raising: a stream that has been replaced by a
    non-reconfigurable object (a `StringIO` under test, a pipe wrapper) is
    skipped rather than fought with. Callers invoke this at process entry,
    before anything is printed.
    """
    targets = streams if streams is not None else (sys.stdout, sys.stderr)
    for stream in targets:
        if stream is None:
            continue
        reconfigure: Any = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # StringIO and friends. Nothing to do — they are not going
            # through a charmap codec.
            continue
        if _is_utf8(getattr(stream, "encoding", None)):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # A detached or already-closed stream. Printing is about to
            # fail for its own reasons; do not add an exception at entry.
            continue


def read_hook_stdin(
    stream: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> str:
    """Read one hook payload as UTF-8, whatever the process locale (#1426).

    Hosts write hook payloads as UTF-8 bytes. `sys.stdin.read()` decodes them
    through the *interpreter's* stdio encoding, which on Windows with
    redirected stdin is the ANSI code page — so a payload containing any
    non-ascii character is parsed as locale-mangled text, and whatever
    survives is stored as though the user typed it::

        >>> '東京'.encode().decode('cp1252', errors='surrogateescape')
        'æ\\udc9d±äº¬'

    Reading the byte stream and decoding it strictly removes the locale from
    the path entirely. `ensure_utf8_streams` cannot do this job: it
    reconfigures *output* streams, and reconfiguring stdin after the wrapper
    exists would not recover bytes already consumed.

    **Returns `""` on undecodable input rather than replacement text.** A
    prompt silently corrupted into the store is worse than a turn where the
    hook did nothing, and every caller already treats an empty read as
    "nothing to do", so this needs no signature changes anywhere. A one-line
    diagnostic goes to `stderr` so the drop is visible rather than mute.

    Falls back to a text-mode read when the stream has no usable `buffer`.
    That covers `StringIO` under test, and deliberately preserves pytest's
    behaviour: `_pytest.capture.DontReadFromInput.buffer` is a property
    returning `self`, whose `read()` raises `OSError`, so the error still
    propagates exactly as it does today instead of being converted into a
    silent empty payload.
    """
    target: Any = sys.stdin if stream is None else stream
    if target is None:
        return ""
    buffer: Any = getattr(target, "buffer", None)
    if buffer is None or buffer is target:
        # No byte layer, or pytest's self-returning stub. Read as text and
        # let whatever it does happen -- including raising.
        return cast(str, target.read())
    try:
        data: bytes = buffer.read()
    except (AttributeError, ValueError):
        # Detached or already-consumed byte layer.
        return cast(str, target.read())
    if isinstance(data, str):
        # A test double whose `buffer` is itself text-mode.
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        out: Any = sys.stderr if stderr is None else stderr
        try:
            print(
                "aelfrice: hook payload was not valid UTF-8; skipping this "
                f"turn rather than storing mangled text ({len(data)} bytes)",
                file=out,
            )
        except Exception:
            pass
        return ""


__all__ = ["ensure_utf8_streams", "read_hook_stdin"]
