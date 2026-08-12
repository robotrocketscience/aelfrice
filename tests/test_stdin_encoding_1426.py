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

# Each behavioural test spawns a real console script under a forced
# locale, which the suite's 5s default reports as a hang rather than as
# contention (#1307). Sized for the slowest arm — the ten-script sweep.
pytestmark = pytest.mark.timeout(120)

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

    The suffix sweep is load-bearing, not defensive. On Windows these
    install as `Scripts\\aelf-hook.exe`, and `Path.exists()` does no
    PATHEXT resolution — an extensionless probe misses every one of them,
    `pytest.skip` fires, and the run goes green having asserted nothing.
    That is precisely the mistake #1412 was filed for in production code
    (see `test_windows_launcher_1412.py`, which pins the same `.exe`
    split), and it would have been invisible here because the POSIX lane
    resolves on the first candidate.
    """
    scripts_dir = Path(sys.executable).parent
    for suffix in ("", ".exe", ".cmd", ".bat"):
        candidate = scripts_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    pytest.skip(f"console script {name} is not installed in {scripts_dir}")


def test_the_console_scripts_actually_resolve() -> None:
    """A skip must never be how this module reports success.

    `_script` skips when it cannot find a console script, and a skipped
    test is a green test. This arm has no skip in it, so if the install
    layout changes under us the suite says so instead of quietly
    asserting nothing on the platform the fix targets.
    """
    scripts_dir = Path(sys.executable).parent
    missing = [
        name
        for name in HOOK_SCRIPTS
        if not any(
            (scripts_dir / f"{name}{suffix}").exists()
            for suffix in ("", ".exe", ".cmd", ".bat")
        )
    ]
    assert not missing, (
        f"console scripts not found in {scripts_dir}: {missing}. Every "
        f"behavioural test in this module would have SKIPPED, and the run "
        f"would still have been green."
    )


def _run(
    script: str,
    wire: bytes,
    regime: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.update(regime)
    # Isolation is conftest's session-scoped `_sandbox_real_home`, which
    # repoints HOME for the whole run; these children inherit it. Setting
    # AELFRICE_HOME here would be decorative — nothing reads it.
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
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        timeout=60,
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
    message = result.stderr.decode("ascii")  # raises if not ASCII
    assert "not valid UTF-8" in message
    # Against the constant the code actually applies, not a round number.
    # The template cannot currently reach the cap, so this asserts the
    # bound holds rather than that truncation fires.
    from aelfrice.stream_encoding import _DIAGNOSTIC_MAX_CHARS

    assert len(message.strip()) <= _DIAGNOSTIC_MAX_CHARS


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

    for marker in _CODEC_FAILURE_MARKERS:
        assert marker not in stderr, f"{script} [{marker}]: {stderr[-600:]}"
    assert result.returncode == 0, f"{script} broke fail-open: {stderr[-600:]}"


# Matching only the exception *class name* is not enough. `hook.stop()`
# formats its handler as `{exc}` rather than printing a traceback, so on
# the unfixed tree it emits `'charmap' codec can't decode byte 0x9d ...`
# with the words "UnicodeDecodeError" nowhere in the output — the
# aelf-stop-hook arm passed on the broken tree until this list existed.
_CODEC_FAILURE_MARKERS = (
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "codec can't decode",
    "codec can't encode",
    "surrogates not allowed",
)


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

# `main()` pins stdin at entry; a driver that imports `_read_password`
# directly has to do the same or it is testing a different program.
_READ_PASSWORD_DRIVER = """
import argparse, sys
from aelfrice.stream_encoding import ensure_utf8_stdin
ensure_utf8_stdin()
from aelfrice.cli import _read_password
{prelude}
args = argparse.Namespace(password_stdin=True)
pw = _read_password(args)
sys.stdout.buffer.write(b"HEX:" + (pw or "").encode("utf-8").hex().encode("ascii"))
"""


def _drive_read_password(tmp_path: Path, prelude: str = "") -> str:
    env = dict(os.environ)
    env.update(CP1252)
    result = subprocess.run(
        [sys.executable, "-c", _READ_PASSWORD_DRIVER.format(prelude=prelude)],
        input=(b"y\n" if prelude else b"") + PASSWORD.encode("utf-8") + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "backslashreplace")
    return result.stdout.split(b"HEX:")[-1].decode("ascii")


def test_the_archive_password_is_read_as_utf8(tmp_path: Path) -> None:
    """`--password-stdin` must recover the exact password the user piped.

    This is the one site where the bug is not recoverable after the fact.
    `lifecycle._encrypt_db_to_archive` derives the scrypt key from
    `password.encode("utf-8")`, so a password mangled on the way in
    encrypts against a key nobody can reproduce: the archive is written
    successfully and can never be opened from a UTF-8 host.
    """
    assert _drive_read_password(tmp_path) == PASSWORD.encode("utf-8").hex(), (
        "the piped password was altered before key derivation; the archive "
        "it encrypts could not be decrypted from a utf-8 host"
    )


def test_the_password_survives_a_preceding_confirmation_prompt(
    tmp_path: Path,
) -> None:
    """The real `--archive` ordering: `input()` runs first, then the password.

    `_cmd_uninstall` asks `continue? [y/N]:` before it reads the password
    whenever `(extras or dotdir_planned) and not args.yes`, which is every
    real install. `input()` draws from `sys.stdin`'s text wrapper and pulls
    the *whole pipe* into its decode buffer, so a password read taken from
    `sys.stdin.buffer` afterwards finds EOF and returns empty — the caller
    then aborts with "empty or non-matching password" and no archive is
    ever written.

    That is why the fix pins stdin's decoder rather than bypassing it: both
    reads have to come off the same layer. Asserting only the isolated read
    above would have missed this entirely, which is exactly what happened.
    """
    got_hex = _drive_read_password(
        tmp_path, prelude='ack = input("continue? [y/N]: ")'
    )

    assert got_hex == PASSWORD.encode("utf-8").hex(), (
        "the password was lost or altered after a preceding text-layer read; "
        "`aelf uninstall --archive --password-stdin` aborts in this ordering"
    )


def test_a_password_that_is_not_utf8_is_refused_not_guessed(
    tmp_path: Path,
) -> None:
    """Undecodable password bytes must abort, not derive a key from a guess.

    The pin is strict, so bytes that are not UTF-8 raise at the read
    rather than being substituted. `_read_password` returns None, and the
    caller already treats that as "aborted: empty or non-matching
    password" and exits 1. Refusing is the right failure: the alternative
    is writing an archive against a key nobody can reproduce.
    """
    env = dict(os.environ)
    env.update(CP1252)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _READ_PASSWORD_DRIVER.format(prelude=""),
        ],
        # 0xff is not valid UTF-8 in any position.
        input=b"caf\xff-passphrase\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )
    got_hex = result.stdout.split(b"HEX:")[-1].decode("ascii").strip()
    message = result.stderr.decode("utf-8", "backslashreplace")

    assert got_hex == "", (
        f"a non-utf8 password was accepted rather than refused: {got_hex!r}"
    )
    assert "not valid UTF-8" in message, (
        f"the refusal was silent; the user gets no reason: {message[:400]}"
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


def test_the_cli_entry_point_pins_stdin_itself(tmp_path: Path) -> None:
    """`main()` must do the pinning — not just the helper being importable.

    The two tests above drive `_read_password` through a driver that calls
    `ensure_utf8_stdin` itself, so they pass whether or not the CLI's own
    entry point ever calls it. Deleting the call from `main()` left them
    green; this is the arm that goes red.

    Driven through the installed `aelf` binary against a real pipe. The
    observable is chosen to separate the read from everything downstream:
    `東京` is `e6 9d b1`, and `0x9d` is one of the five bytes cp1252 leaves
    undefined, so an unpinned stdin fails at the read with "is not valid
    UTF-8" and never reaches the session lookup.
    """
    payload = (
        b'[{"index":0,"belief_type":"fact","persist":true,'
        b'"note":"caf\xc3\xa9 \xe6\x9d\xb1\xe4\xba\xac"}]'
    )
    env = dict(os.environ)
    env.update(CP1252)
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [
            str(_script("aelf")),
            "onboard",
            "--accept-classifications",
            "--session-id",
            "no-such-session-1426",
            "--classifications-file",
            "-",
        ],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(tmp_path),
        env=env,
        timeout=60,
    )
    combined = (result.stdout + result.stderr).decode("utf-8", "backslashreplace")

    assert "is not valid UTF-8" not in combined, (
        "the CLI entry point did not pin stdin, so a non-ascii document on "
        f"a pipe failed at the read: {combined[:400]}"
    )
    assert "no-such-session-1426" in combined, (
        "expected the read to succeed and the session lookup to be reached; "
        f"got: {combined[:400]}"
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


# `.read` alone misses two spellings that decode exactly the same way.
# `_read_password` is a `readline()`, and a future site could reasonably
# reach for `readlines()`.
_TEXT_READ_METHODS = frozenset({"read", "readline", "readlines"})


def _stdin_reads(path: Path) -> list[int]:
    """Line numbers of `<stdin-ish>.<read*>()` calls in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _TEXT_READ_METHODS:
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
    return sorted(hits)


def _calls_named(path: Path, name: str) -> list[int]:
    """Line numbers of calls to a bare function `name` in a module.

    Deliberately not a substring search: `"read_payload_text" in source`
    is satisfied by the import line, so it stays true after every call
    site has been deleted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


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
    guard above while removing the behaviour it protects. Counts real
    call sites rather than grepping for the name, because the import line
    alone satisfies a substring search forever.
    """
    calls = _calls_named(SRC / module, "read_payload_text")
    assert calls, (
        f"{module} imports the utf-8 boundary but never calls it; its "
        f"payload is no longer routed through it (#1426)"
    )


# `cli.py` is in the guard, but two of its stdin reads are out of scope by
# an explicit ruling: `_cmd_scan_derivation`'s pair is a developer tool
# whose locale handling is identical on main and was deferred rather than
# fixed here. They are listed by *function*, not by line number, so the
# exemption does not silently widen when the file is edited above them.
CLI_EXEMPT_FUNCTIONS = frozenset({"_cmd_scan_derivation"})


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None or not (node.lineno <= lineno <= end):
            continue
        if best is None or node.lineno > best[0]:
            best = (node.lineno, node.name)
    return best[1] if best else None


def test_the_cli_has_no_unexpected_locale_stdin_read() -> None:
    """`cli.py` too — it is where the password and the classifications pipe live.

    The CLI does not use `read_payload_text`; it pins `sys.stdin` at
    entry instead, because `input()` and the password read have to share
    one text layer. So the guard here is different in kind: every stdin
    read in the file must belong to a function that is either covered by
    that pin or explicitly exempt. A new unguarded read anywhere else in
    the CLI fails this.
    """
    path = SRC / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    unexpected = {
        (_enclosing_function(tree, line) or f"<module>:{line}")
        for line in _stdin_reads(path)
    } - CLI_EXEMPT_FUNCTIONS - {
        "_cmd_onboard_accept_classifications",
        "_read_password",
    }

    assert not unexpected, (
        f"cli.py reads stdin in {sorted(unexpected)}, which is neither "
        f"covered by the entry-point pin nor listed as exempt (#1426)"
    )


def test_the_cli_entry_point_still_calls_the_pin() -> None:
    """The pin is one line in `main()`; nothing else in the file re-establishes it.

    `test_the_cli_entry_point_pins_stdin_itself` proves the behaviour end
    to end through a subprocess. This is the cheap structural companion:
    it names the call so a refactor that moves `main()`'s prologue cannot
    drop it and leave only a slow subprocess arm to notice.
    """
    calls = _calls_named(SRC / "cli.py", "ensure_utf8_stdin")
    assert len(calls) == 1, (
        f"expected exactly one ensure_utf8_stdin() call in cli.py, found "
        f"{len(calls)} at {calls}"
    )


# The six console scripts that had no `ensure_utf8_streams` call at all
# before this change (`hook.py`'s four entry points already had one).
UNPINNED_ENTRY_MODULES = (
    "transcript_logger.py",
    "hook_commit_ingest.py",
    "hook_search_tool.py",
    "hook_agent_context.py",
    "pre_issue_create_hook.py",
    "hook_claude_memory_mirror.py",
)


@pytest.mark.parametrize("module", UNPINNED_ENTRY_MODULES)
def test_every_console_script_pins_its_output_streams(module: str) -> None:
    """These six printed diagnostics through whatever code page they found.

    Behavioural coverage would need a real Windows console, which the
    blocking lane does not have, so this pins the call structurally
    instead — and says so rather than implying more. Without it, deleting
    all six calls left the entire suite green, which is how a #1329-class
    crash would come back on the entry points that change least often.
    """
    calls = _calls_named(SRC / module, "ensure_utf8_streams")
    assert calls, (
        f"{module}'s console entry point does not pin stdout/stderr to "
        f"utf-8; a non-ascii diagnostic crashes it on a cp1252 console "
        f"(#1329, reopened on this entry point by #1426)"
    )
