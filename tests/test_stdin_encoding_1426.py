"""The text boundary between our hooks and their caller (#1426).

`ensure_utf8_streams` pinned the way *out* (#1329). Nothing pinned the way
in: every hook read `sys.stdin` as already-decoded text, so Python applied
the process locale to a protocol that is UTF-8 by definition. Two distinct
failures follow, and both are silent because a hook returns 0 by contract:

* a legacy code page (`cp1252`) raises `UnicodeDecodeError` at the read,
  and the user's turn is simply never recorded;
* an ASCII locale decodes via `surrogateescape`, survives the read, and
  dies at the UTF-8 write instead — same lost turn, different traceback.

No Windows host is needed to reproduce either. `PYTHONIOENCODING=cp1252`
gives a POSIX runner the same charmap decoder Windows picks by default,
and `LC_ALL=C` with UTF-8 mode and PEP 538 coercion both off gives the
ASCII locale that produces the surrogate path. That is why the blocking
gate lives here rather than in `windows-smoke.yml`, which its own header
forbids promoting to a required check.

`PYTHONUTF8=0` alone reproduces *nothing* — it is the regime the issue's
AC1 named, and it passes on the unfixed tree. `test_the_regimes_that_do_not_fail`
pins that, so a future reader does not mistake a green AC1 for a fix.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "aelfrice"

# A string that exercises three widths of the problem at once: Latin-1 with
# an accent (cp1252 *can* encode this one), an em dash and CJK (it cannot),
# and an astral-plane emoji (surrogate pair territory on the write path).
PAYLOAD_TEXT = "café — 東京 \U0001f642"
PAYLOAD_UTF8 = PAYLOAD_TEXT.encode("utf-8")

# Windows' default console decoder, reproduced on POSIX.
CP1252 = {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"}
# The ASCII-locale regime: decodes via surrogateescape, fails at the write.
C_LOCALE = {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C"}

FAILING_REGIMES = {"cp1252": CP1252, "c_locale": C_LOCALE}

# Every console script in `[project.scripts]` that speaks the hook JSON
# protocol on stdin. `aelf` itself is excluded: it is the CLI, and its two
# stdin sites are covered by their own tests below.
HOOK_SCRIPTS = (
    "aelf-hook",
    "aelf-transcript-logger",
    "aelf-pre-compact-hook",
    "aelf-commit-ingest",
    "aelf-search-tool-hook",
    "aelf-agent-context-hook",
    "aelf-session-start-hook",
    "aelf-stop-hook",
    "aelf-pre-issue-hook",
    "aelf-claude-memory-mirror",
)


def _script(name: str) -> Path:
    """Resolve an *installed* console script, not `python -m`.

    The distinction matters: `-m` and a console script build `sys.stdin`
    the same way, but only the console script proves the entry point
    shipped in the wheel is the one that was fixed.
    """
    path = Path(sys.executable).parent / name
    if not path.exists():  # pragma: no cover - environment guard
        pytest.skip(f"console script {name} is not installed")
    return path


def _run(
    script: str,
    wire: bytes,
    regime: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.update(regime)
    # Keep the hooks off the user's real store and out of the network.
    env["AELFRICE_HOME"] = str(cwd / ".aelfrice-home")
    return subprocess.run(
        [str(_script(script))],
        input=wire,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env=env,
        timeout=60,
    )


def _payload(event: str = "UserPromptSubmit") -> bytes:
    return json.dumps(
        {
            "hook_event_name": event,
            "session_id": "utf8-1426",
            "cwd": ".",
            "transcript_path": "",
            "prompt": PAYLOAD_TEXT,
            "message": {"role": "user", "content": PAYLOAD_TEXT},
        },
        ensure_ascii=False,
    ).encode("utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo — `transcript_logger` writes under `.git/`."""
    subprocess.run(
        ["git", "init", "-q"], cwd=str(tmp_path), check=True, capture_output=True
    )
    return tmp_path


def _turns(repo: Path) -> list[bytes]:
    log = repo / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"
    if not log.exists():
        return []
    return [line for line in log.read_bytes().splitlines() if line.strip()]


# --- The defect itself -------------------------------------------------


@pytest.mark.parametrize("regime_name", sorted(FAILING_REGIMES))
def test_the_prompt_survives_a_non_utf8_console(
    repo: Path, regime_name: str
) -> None:
    """The turn is recorded, byte-for-byte, under both failing regimes.

    Asserting on the persisted bytes rather than the return code is not a
    stylistic choice: every failing regime returned 0 before the fix, so a
    return-code assertion cannot see this bug at all.
    """
    result = _run(
        "aelf-transcript-logger", _payload(), FAILING_REGIMES[regime_name], repo
    )

    rows = _turns(repo)
    assert rows, (
        f"regime {regime_name}: the turn was dropped. "
        f"stderr={result.stderr.decode('utf-8', 'backslashreplace')[-600:]}"
    )
    assert any(PAYLOAD_UTF8 in row for row in rows), (
        f"regime {regime_name}: a row was written but the text was altered. "
        f"rows={rows!r}"
    )
    assert result.returncode == 0


def test_the_regimes_that_do_not_fail(repo: Path) -> None:
    """`PYTHONUTF8=0` alone is not the reproducer, and never was.

    The issue's AC1 named it. It passes on the unfixed tree, so a gate
    built on it would have gone green over a live defect. Pinned here so
    the distinction survives the issue being closed.
    """
    for regime in ({}, {"PYTHONUTF8": "0"}):
        for row in list(_turns(repo)):
            del row
        log = repo / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"
        if log.exists():
            log.unlink()
        _run("aelf-transcript-logger", _payload(), regime, repo)
        assert any(PAYLOAD_UTF8 in row for row in _turns(repo)), (
            f"regime {regime} should already have passed before the fix"
        )


def test_an_ascii_payload_still_records(repo: Path) -> None:
    """The control. ASCII was never affected and must stay unaffected."""
    wire = json.dumps(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "ascii-1426",
            "cwd": ".",
            "transcript_path": "",
            "prompt": "plain ascii",
            "message": {"role": "user", "content": "plain ascii"},
        }
    ).encode("utf-8")
    _run("aelf-transcript-logger", wire, CP1252, repo)
    assert any(b"plain ascii" in row for row in _turns(repo))


def test_invalid_utf8_writes_no_row_and_says_so_in_ascii(repo: Path) -> None:
    """Malformed bytes: no partial row, exit 0, a diagnostic we can print.

    The diagnostic has to be ASCII-clean and bounded. It is emitted onto a
    stream that may itself be a legacy code page, so a message quoting the
    offending bytes would turn one bad payload into a second crash.
    """
    result = _run(
        "aelf-transcript-logger",
        b'\xff\xfe{"hook_event_name": "UserPromptSubmit"}',
        CP1252,
        repo,
    )

    assert result.returncode == 0, "the fail-open contract must be preserved"
    assert _turns(repo) == [], "a partial row is worse than no row"
    result.stderr.decode("ascii")  # raises if the diagnostic is not ASCII
    assert b"not valid UTF-8" in result.stderr
    assert len(result.stderr) < 2048


@pytest.mark.parametrize("script", HOOK_SCRIPTS)
def test_every_hook_console_script_survives_a_non_utf8_console(
    repo: Path, script: str
) -> None:
    """AC2: all of them, not just `aelf-hook`.

    A per-script round-trip assertion is not available — most of these
    persist into a store rather than a file, and several no-op on a
    payload they do not recognise. What *is* universal is that none may
    raise a decode error or break its fail-open contract, which is
    exactly the failure being fixed.
    """
    result = _run(script, _payload(), CP1252, repo)
    stderr = result.stderr.decode("utf-8", "backslashreplace")

    assert "UnicodeDecodeError" not in stderr, f"{script}: {stderr[-600:]}"
    assert "UnicodeEncodeError" not in stderr, f"{script}: {stderr[-600:]}"
    assert result.returncode == 0, f"{script} broke fail-open: {stderr[-600:]}"


# --- The injectable test interface must not have moved ------------------


def test_stringio_injection_is_unchanged() -> None:
    """AC4: the `StringIO` interface every other hook test uses still works.

    `io.StringIO` has no `.buffer`, which is what lets one helper serve
    both the real pipe and the injected stream with no signature churn.
    If that ever stops being true, several hundred hook tests change
    meaning silently, so it is asserted rather than assumed.
    """
    import io

    from aelfrice.stream_encoding import read_payload_text

    assert not hasattr(io.StringIO(""), "buffer")
    assert read_payload_text(io.StringIO(PAYLOAD_TEXT), None) == PAYLOAD_TEXT
    assert read_payload_text(io.StringIO(""), None) == ""
    assert read_payload_text(None, None) == ""


def test_undecodable_returns_none_distinctly_from_empty() -> None:
    """`None` (could not read it) and `""` (nothing sent) are not the same.

    Collapsing them would make an undecodable payload indistinguishable
    from an idle hook fire, which is precisely the silence this issue is
    about.
    """
    import io

    from aelfrice.stream_encoding import read_payload_text

    class _Bytes(io.StringIO):
        def __init__(self, raw: bytes) -> None:
            super().__init__("")
            self.buffer = io.BytesIO(raw)  # pyright: ignore[reportAttributeAccessIssue]

    err = io.StringIO()
    assert read_payload_text(_Bytes(b"\xff\xfe"), err) is None
    assert "not valid UTF-8" in err.getvalue()
    assert read_payload_text(_Bytes(b""), None) == ""
    assert read_payload_text(_Bytes(PAYLOAD_UTF8), None) == PAYLOAD_TEXT


# --- The archive password: the irreversible half ------------------------

# Deliberately chosen so cp1252 can decode it *without raising*. Every
# byte of "café".encode("utf-8") (63 61 66 c3 a9) is assigned in cp1252,
# so the wrong decode succeeds and yields "cafÃ©" — a different password,
# silently. A test that only asserted "no exception" would pass on the
# unfixed tree.
PASSWORD = "café-passphrase"

_READ_PASSWORD_DRIVER = """
import argparse, sys
from aelfrice.cli import _read_password
args = argparse.Namespace(password_stdin=True)
pw = _read_password(args)
sys.stdout.buffer.write(b"HEX:" + (pw or "").encode("utf-8").hex().encode("ascii"))
"""


def test_the_archive_password_is_read_as_utf8(tmp_path: Path) -> None:
    """`--password-stdin` must recover the exact password the user piped.

    This is the one site where the bug is not recoverable after the fact.
    `lifecycle._encrypt_db_to_archive` derives the scrypt key from
    `password.encode("utf-8")`, so a password mangled on the way in
    encrypts against a key nobody can reproduce: the archive is written
    successfully and can never be opened from a UTF-8 host.
    """
    env = dict(os.environ)
    env.update(CP1252)
    result = subprocess.run(
        [sys.executable, "-c", _READ_PASSWORD_DRIVER],
        input=PASSWORD.encode("utf-8") + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "backslashreplace")
    got_hex = result.stdout.split(b"HEX:")[-1].decode("ascii")

    assert got_hex == PASSWORD.encode("utf-8").hex(), (
        "the piped password was altered before key derivation; the archive "
        "it encrypts could not be decrypted from a utf-8 host"
    )


def test_the_mojibake_password_would_have_derived_a_different_key() -> None:
    """Why the assertion above is byte-equality and not 'it did not crash'.

    cp1252 decodes the UTF-8 bytes of this password happily — it just
    produces different text. Pinning that the two spellings differ under
    `encode("utf-8")` is what shows the consequence is a divergent scrypt
    key rather than a cosmetic difference. No scrypt call is needed to
    demonstrate it, and none is made: the KDF input is the whole story.
    """
    mojibake = PASSWORD.encode("utf-8").decode("cp1252")

    assert mojibake != PASSWORD, "the regime must actually alter this password"
    assert mojibake.encode("utf-8") != PASSWORD.encode("utf-8"), (
        "different KDF input means a different key means an unopenable archive"
    )


# --- Keep it fixed ------------------------------------------------------

# The modules that read the hook protocol from stdin. Grep is not enough
# here: the call can be spelled `sin.read()`, `stdin.read()` or
# `sys.stdin.read()`, so the guard walks the AST for the shape instead.
HOOK_MODULES = (
    "hook.py",
    "transcript_logger.py",
    "hook_agent_context.py",
    "hook_claude_memory_mirror.py",
    "hook_commit_ingest.py",
    "hook_search_tool.py",
    "pre_issue_create_hook.py",
    "context_rebuilder.py",
)


def _stdin_reads(path: Path) -> list[int]:
    """Line numbers of `<stdin-ish>.read()` calls in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "read":
            continue
        target = func.value
        # `sin.read()` / `stdin.read()`
        if isinstance(target, ast.Name) and target.id in {"sin", "stdin"}:
            hits.append(node.lineno)
        # `sys.stdin.read()`
        elif (
            isinstance(target, ast.Attribute)
            and target.attr == "stdin"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("module", HOOK_MODULES)
def test_no_hook_reads_stdin_as_text(module: str) -> None:
    """No hook module may go back to reading stdin as decoded text.

    This is the anti-regression half. A new hook entry point that copies
    an existing one is the likely way the defect returns, and it would be
    invisible on a UTF-8 developer machine.
    """
    hits = _stdin_reads(SRC / module)
    assert hits == [], (
        f"{module} reads stdin as locale-decoded text at line(s) {hits}; "
        f"use aelfrice.stream_encoding.read_payload_text instead (#1426)"
    )


@pytest.mark.parametrize("module", HOOK_MODULES)
def test_every_hook_module_uses_the_helper(module: str) -> None:
    """The companion assert: absence of the bad shape is not presence of the good one.

    Without this, deleting a module's stdin read entirely would pass the
    guard above while removing the behaviour it protects.
    """
    source = (SRC / module).read_text(encoding="utf-8")
    assert "read_payload_text" in source, (
        f"{module} no longer routes its payload through the utf-8 boundary"
    )
