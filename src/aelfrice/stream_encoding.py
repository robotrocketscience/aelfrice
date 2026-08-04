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
from typing import IO, Any

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


__all__ = ["ensure_utf8_streams"]
