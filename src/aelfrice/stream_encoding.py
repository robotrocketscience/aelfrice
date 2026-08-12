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

# Bound on the diagnostic emitted for an undecodable payload. The stream we
# are writing to may itself be a legacy code page, so the message is built
# from ASCII only and truncated — a hook must not turn one bad payload into
# a second encoding failure on the way out.
_DIAGNOSTIC_MAX_CHARS = 200


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


def ensure_utf8_stdin() -> None:
    """Pin `sys.stdin` to strict UTF-8, for programs that read it as text.

    The hook path does not use this — a hook reads its payload once and
    `read_payload_text` handles it at the byte layer. The CLI cannot do
    that: `aelf uninstall --archive --password-stdin` calls `input()` for
    the confirmation and *then* reads the password, and those two reads
    have to come off the same layer. Reading one through the text wrapper
    and the next through `.buffer` loses everything the wrapper already
    pulled into its decode buffer, which is a whole pipe when the input is
    a pipe.

    So the CLI pins the decoder instead of bypassing it. Must be called
    before anything reads stdin: `TextIOWrapper.reconfigure` refuses to
    change the encoding once reading has started.

    Strict, not `replace`: the values arriving here are a password that
    derives an encryption key and a JSON document. A silently substituted
    character in either is worse than a refusal.

    The error handler is half the contract, so the early-out checks both.
    UTF-8 mode gives `sys.stdin` `encoding="utf-8"` with
    `errors="surrogateescape"` — an encoding test alone returns early and
    leaves the substituting handler in place, so an undecodable byte
    arrives as a lone surrogate in the U+DC80-U+DCFF range instead of
    being refused. That regime is not hypothetical: `PYTHONUTF8=1` is the
    workaround this defect's own issue tells users to reach for.
    """
    stream = sys.stdin
    if stream is None:
        return
    reconfigure: Any = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    if (
        _is_utf8(getattr(stream, "encoding", None))
        and getattr(stream, "errors", None) == "strict"
    ):
        return
    try:
        reconfigure(encoding="utf-8", errors="strict")
    except (OSError, ValueError):
        # Detached, closed, or already read from. Nothing to do; the
        # read that follows will fail on its own terms.
        return


def read_payload_text(
    stdin: IO[str] | None,
    stderr: IO[str] | None = None,
) -> str | None:
    """Read a hook's JSON payload from `stdin` as UTF-8, not as locale text.

    `ensure_utf8_streams` fixes the way out; this fixes the way in (#1426).
    Python decodes a redirected stdin with the process locale, so on a
    Windows console defaulting to `cp1252` a payload containing
    `café — 東京 🙂` either raises `UnicodeDecodeError` at the read or
    decodes through `surrogateescape` into text that explodes later at the
    UTF-8 write. Both paths lose the user's turn while the hook returns 0
    under its fail-open contract, so nothing surfaces: compaction
    reconstruction and belief extraction then run on incomplete history.

    The protocol is UTF-8 by definition, so we read `stdin.buffer` and
    decode it ourselves. Streams without a `.buffer` — `io.StringIO`, which
    every hook's injectable test interface uses — are read as text
    unchanged, which is what makes this a drop-in at each call site.

    Returns the decoded payload, `""` for an empty stream, or `None` when
    the bytes are not UTF-8. `None` is distinct from `""` so a caller can
    tell "nothing was sent" from "something was sent and we could not read
    it"; callers that do not care can spell it `read_payload_text(...) or ""`.

    Decoding is strict on purpose. `errors="replace"` would store something
    that differs from what the user typed, and `errors="surrogateescape"`
    only defers the failure to the next write. Read errors other than a
    decode failure (notably pytest's captured-stdin `OSError`) propagate
    exactly as they did before this helper existed.
    """
    if stdin is None:
        return ""
    buffer: Any = getattr(stdin, "buffer", None)
    if buffer is None:
        # StringIO and friends: already text, and never went through a
        # charmap codec on the way in.
        return stdin.read()
    raw: Any = buffer.read()
    if isinstance(raw, str):
        # A text-like stand-in that exposes `.buffer` as itself. Nothing
        # was decoded by us, so there is nothing to correct.
        return raw
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        _report_undecodable(exc, stderr)
        return None


def _report_undecodable(exc: UnicodeDecodeError, stderr: IO[str] | None) -> None:
    """Emit a bounded ASCII-only note that a payload could not be decoded.

    Deliberately says nothing about the payload's content: the bytes are
    by definition not text we can render, and echoing them is how a
    diagnostic becomes a second crash.
    """
    if stderr is None:
        return
    message = (
        f"aelfrice: hook payload is not valid UTF-8 "
        f"(byte 0x{exc.object[exc.start]:02x} at offset {exc.start}); "
        f"the payload was discarded."
    )[:_DIAGNOSTIC_MAX_CHARS]
    try:
        print(message.encode("ascii", "replace").decode("ascii"), file=stderr)
    except Exception:
        # Fail open: a hook that cannot report its own failure still must
        # not take the turn down with it.
        pass


__all__ = ["ensure_utf8_stdin", "ensure_utf8_streams", "read_payload_text"]
