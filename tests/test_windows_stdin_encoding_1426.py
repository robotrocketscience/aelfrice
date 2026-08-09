"""Hook payloads decode as UTF-8, not through the process locale (#1426).

Hosts write hook payloads as UTF-8 bytes. `sys.stdin.read()` decodes them with
the *interpreter's* stdio encoding, which on Windows with redirected stdin is
the ANSI code page — so any non-ascii character in a prompt arrives mangled and
is stored as though the user typed it.

**None of this needs a Windows host.** `PYTHONIOENCODING=cp1252` plus
`PYTHONUTF8=0` reproduces the configuration exactly on an ubuntu runner, in a
real subprocess, which is what the arms below use. The native `windows-smoke`
job can confirm it; the gate is here.
"""
from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.stream_encoding import read_hook_stdin

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "aelfrice"

# Chosen because its UTF-8 encoding contains 0x9d, one of the five bytes cp1252
# leaves undefined -- so a strict locale decode *raises* on it and a
# surrogateescape decode mangles it. A Latin-1-representable sample like
# "café" would round-trip through cp1252 and prove nothing.
_NON_ASCII = "東京 café"


def _cp1252_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252:surrogateescape"
    env["PYTHONUTF8"] = "0"
    return env


class _ByteStream:
    """A text stream whose `buffer` holds the bytes the host actually wrote."""

    def __init__(self, data: bytes, *, text: str | None = None) -> None:
        self.buffer = io.BytesIO(data)
        self._text = text

    def read(self) -> str:
        if self._text is None:
            raise AssertionError(
                "fell back to the text layer; the point is to bypass it"
            )
        return self._text


def test_the_locale_is_what_breaks_it() -> None:
    """The premise, pinned. If this stops holding the rest is theatre."""
    raw = _NON_ASCII.encode()
    assert b"\x9d" in raw
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")
    assert raw.decode("cp1252", errors="surrogateescape") != _NON_ASCII


def test_utf8_bytes_survive_a_cp1252_text_wrapper() -> None:
    """The fix: read the byte layer, ignore whatever the wrapper decided."""
    raw = _NON_ASCII.encode()
    mangled = raw.decode("cp1252", errors="surrogateescape")
    stream = _ByteStream(raw, text=mangled)
    assert read_hook_stdin(stream) == _NON_ASCII


def test_undecodable_bytes_yield_nothing_not_replacement_text() -> None:
    """A corrupted prompt in the store is worse than a turn that did nothing.

    `errors="replace"` would be the tempting choice and is the wrong one: it
    produces text that looks like a prompt, passes every downstream check, and
    is persisted as the user's own words.
    """
    err = io.StringIO()
    assert read_hook_stdin(_ByteStream(b"\xff\xfe\x00\x00nope"), err) == ""
    assert "not valid UTF-8" in err.getvalue()
    assert "�" not in err.getvalue()


def test_a_stream_with_no_byte_layer_still_reads() -> None:
    """StringIO under test, and any other text-only double."""
    assert read_hook_stdin(io.StringIO('{"a": 1}')) == '{"a": 1}'


def test_pytest_captured_stdin_still_raises_rather_than_reading_empty() -> None:
    """Behaviour preserved, deliberately.

    `_pytest.capture.DontReadFromInput.buffer` is a property returning `self`,
    and its `read()` raises `OSError`. Converting that into a silent `""` would
    turn "a test forgot to provide stdin" into "the hook did nothing", which is
    a much harder failure to diagnose.
    """
    with pytest.raises(OSError):
        read_hook_stdin(sys.stdin)


@pytest.mark.parametrize(
    "module",
    [
        "hook",
        "context_rebuilder",
        "hook_agent_context",
        "hook_claude_memory_mirror",
        "hook_commit_ingest",
        "hook_search_tool",
        "transcript_logger",
        "pre_issue_create_hook",
    ],
)
def test_no_hook_module_reads_stdin_through_the_locale(module: str) -> None:
    """Structural, so a *new* hook cannot reintroduce this one module at a time.

    Asserted over the AST rather than by grep: the call can be spelled across
    lines, and `sin = stdin or sys.stdin` means the receiver name varies by
    module. What is banned is a `.read()` whose receiver is a stdin-ish name.
    """
    tree = ast.parse((_SRC / f"{module}.py").read_text(encoding="utf-8"))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "read":
            continue
        recv = func.value
        name = (
            recv.id if isinstance(recv, ast.Name)
            else recv.attr if isinstance(recv, ast.Attribute)
            else ""
        )
        if name in {"stdin", "sin"}:
            offenders.append(node.lineno)
    assert offenders == [], (
        f"{module}.py reads stdin through the locale at line(s) {offenders}; "
        f"use stream_encoding.read_hook_stdin so the payload is decoded as "
        f"UTF-8 regardless of the process code page"
    )


@pytest.mark.timeout(90)
def test_a_hook_subprocess_under_cp1252_keeps_the_prompt_intact() -> None:
    """End to end, in a real subprocess, on an ordinary POSIX runner.

    This is the arm that would have caught the defect. Everything above tests
    the helper; this tests that a hook entry point actually routes through it
    with the interpreter genuinely configured the way Windows configures it.
    """
    payload = json.dumps(
        {"hook_event_name": "UserPromptSubmit", "prompt": _NON_ASCII},
        ensure_ascii=False,
    ).encode()

    script = (
        "import sys\n"
        "from aelfrice.stream_encoding import read_hook_stdin\n"
        "raw = read_hook_stdin()\n"
        "sys.stdout.buffer.write(raw.encode('utf-8'))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        env=_cp1252_env(),
        cwd=_REPO,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert json.loads(proc.stdout.decode("utf-8"))["prompt"] == _NON_ASCII


@pytest.mark.timeout(90)
def test_the_same_subprocess_without_the_fix_would_have_mangled_it() -> None:
    """Proves the arm above is distinguishing rather than vacuously green.

    Runs the *old* code path -- a bare `sys.stdin.read()` -- under the same
    environment. If this ever stops mangling, the environment has stopped
    reproducing Windows and the test above is no longer evidence of anything.
    """
    payload = json.dumps({"prompt": _NON_ASCII}, ensure_ascii=False).encode()
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.stdout.buffer.write(sys.stdin.read().encode('utf-8','surrogatepass'))"],
        input=payload,
        capture_output=True,
        env=_cp1252_env(),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert proc.stdout != payload, (
        "the locale no longer mangles UTF-8 here, so the cp1252 environment "
        "is not reproducing the Windows configuration any more"
    )


def test_a_non_ascii_archive_password_derives_the_same_key_on_both_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data-loss arm, and the reason #1426's scope reaches past the hooks.

    `aelf uninstall --archive --password-stdin` fed the password through the
    process locale. On Windows a non-ascii password therefore decoded to
    different characters than the same keystrokes on POSIX, so scrypt derived a
    **different key** and the archive became permanently undecryptable on the
    other machine. Nothing surfaces that at the time — it fails on restore,
    later, on the box that still has the data.

    Asserted through the real CLI resolver rather than by re-implementing the
    read, and via the byte layer, because that is where the platforms differ.
    """
    import argparse

    from aelfrice import cli

    password = "パスワード-café"

    class _Stdin:
        def __init__(self, data: bytes) -> None:
            self.buffer = io.BytesIO(data)

        def readline(self) -> str:  # pragma: no cover - must not be reached
            raise AssertionError("read through the locale, not the bytes")

    monkeypatch.setattr(sys, "stdin", _Stdin(password.encode() + b"\n"))
    args = argparse.Namespace(password_stdin=True)
    assert cli._read_password(args) == password


def test_an_undecodable_archive_password_refuses_rather_than_returning_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Here `""` would be the dangerous answer, unlike on the hook path.

    A hook that reads nothing does nothing. A password that reads as empty
    encrypts the archive under a key derived from the empty string, silently,
    and the user finds out when they try to restore it.
    """
    import argparse

    from aelfrice import cli

    class _Stdin:
        def __init__(self, data: bytes) -> None:
            self.buffer = io.BytesIO(data)

        def readline(self) -> str:  # pragma: no cover - must not be reached
            raise AssertionError("read through the locale, not the bytes")

    monkeypatch.setattr(sys, "stdin", _Stdin(b"\xff\xfe\x00bad\n"))
    args = argparse.Namespace(password_stdin=True)
    assert cli._read_password(args) is None
