"""The text boundary between us and our children (#1441).

`subprocess.run(..., text=True)` and `Path.read_text()`/`write_text()` with no
`encoding=` decode through the *process locale*. On Windows that is the ANSI
code page, so every git and `gh` call in the package mangles non-ascii output —
and on the five bytes cp1252 leaves undefined it raises `UnicodeDecodeError`,
which is a `ValueError` and therefore slips past the
`except (FileNotFoundError, OSError, subprocess.TimeoutExpired)` guards.

Two tests, deliberately different in kind:

* `test_commit_message_round_trips_under_a_non_utf8_locale` reproduces the
  defect end-to-end in a real subprocess. No Windows host is needed — a C
  locale with UTF-8 mode and PEP 538 coercion both disabled gives an ASCII
  locale encoding, which fails on exactly the same bytes.
* The AST guards keep it fixed. Grep is not enough: the kwarg can be spelled
  across lines, and `read_text("utf-8")` passes it positionally.
"""
from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.store import MemoryStore

SRC = Path(__file__).resolve().parents[1] / "src" / "aelfrice"

# `text=True` builds a TextIOWrapper with the locale encoding and *strict*
# errors. These three together force that encoding to ASCII on a POSIX runner:
# PYTHONUTF8=0 disables UTF-8 mode, PYTHONCOERCECLOCALE=0 disables the PEP 538
# coercion of C -> C.UTF-8 that would otherwise hide the defect, and LC_ALL=C
# selects the locale itself. '東京'.encode() is e6 9d b1 e4 ba ac; every one of
# those bytes is >= 0x80, so a strict ASCII decode raises on the first.
_ASCII_LOCALE_ENV = {
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LANG": "C",
}

# Extracts as a triple while carrying the multibyte run in its anchor text,
# so the belief that lands in the store is the evidence that the decode
# survived. The relation has to survive the WRITE-path pattern bank, not the
# read-path one: #1376 dropped the six single-token templates that double as
# plural nouns — `covers`, `extends`, `follows`, `replaces`, `supports`,
# `tests` — from `_INGEST_PATTERNS`, and `hook_commit_ingest` is the only
# caller that passes `constrain_collision_verbs=True`. An earlier phrasing
# here used `supports`, so after #1376 it yielded no triple, the hook
# returned before opening a store, and this test failed on `db.exists()`
# with an empty stderr — a "nothing was ingested" failure that looks nothing
# like the decode failure it exists to catch.
_COMMIT_MESSAGE = "the 東京 index is supported by the vocabulary bridge"

# What an ASCII locale encoding is spelled as, across platforms and libc
# builds: glibc reports the alias, macOS and the `locale` module's own
# normalisation report these.
_ASCII_ALIASES = frozenset({"ascii", "us-ascii", "ansi_x3.4-1968"})


def _child_preferred_encoding(env: dict[str, str]) -> str:
    """The encoding a child spawned with `env` would decode text mode with.

    `locale.getpreferredencoding(False)` and not `locale.getencoding()`:
    the latter reports the *locale's* encoding and so still says US-ASCII
    under PYTHONUTF8=1, which is precisely the state this guard exists to
    catch. Only the former follows UTF-8 mode, and only the former is what
    `text=True` actually builds its TextIOWrapper from.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import locale;print(locale.getpreferredencoding(False))",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert probe.returncode == 0, f"locale probe failed: {probe.stderr}"
    return probe.stdout.strip()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {r.stderr}")
    return r.stdout


@pytest.mark.timeout(120)  # #1307: git fixture + a real interpreter spawn
def test_commit_message_round_trips_under_a_non_utf8_locale(
    tmp_path: Path,
) -> None:
    """A non-ascii commit body survives the hook under an ASCII locale.

    Fails on main two ways at once: `_read_full_commit_message` raises
    `UnicodeDecodeError`, `main()`'s blanket handler prints the traceback and
    returns 0, and nothing is ingested. The payload on stdin is pure ASCII on
    purpose — the stdin boundary is #1426, a different mechanism, and mixing
    them in would make a failure here ambiguous.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "README")

    msg_file = repo / ".commit-msg"
    msg_file.write_text(_COMMIT_MESSAGE + "\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-F", str(msg_file))
    msg_file.unlink()
    short = _git(repo, "rev-parse", "--short", "HEAD").strip()

    db = tmp_path / "memory.db"
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -F .commit-msg"},
        "tool_response": {"stdout": f"[main {short}] a commit"},
        "cwd": str(repo),
    }

    env = {**os.environ, **_ASCII_LOCALE_ENV, "AELFRICE_DB": str(db)}

    # Without this the test's whole failure signal rests on
    # `_ASCII_LOCALE_ENV` biting, which is not observable from its
    # assertions: with the `encoding=` kwarg reverted *and* PYTHONUTF8
    # flipped 0 -> 1, the body below passes. A failure, not a skip — an
    # environment where this cannot be arranged has not exercised #1441,
    # and silently reporting that as green is the defect this guards.
    preferred = _child_preferred_encoding(env)
    assert preferred.lower() in _ASCII_ALIASES, (
        "the child's preferred encoding is "
        f"{preferred!r}, not ASCII — this test decodes nothing the fix "
        "affects. Check PYTHONUTF8/PYTHONCOERCECLOCALE/LC_ALL in "
        "_ASCII_LOCALE_ENV are reaching the child."
    )

    proc = subprocess.run(
        [sys.executable, "-m", "aelfrice.hook_commit_ingest"],
        input=json.dumps(payload).encode("ascii"),
        capture_output=True,
        check=False,
        timeout=120,
        env=env,
    )

    assert proc.returncode == 0
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert "UnicodeDecodeError" not in stderr, (
        "the hook decoded git output through the locale:\n" + stderr
    )
    assert db.exists(), "nothing was ingested: " + stderr

    store = MemoryStore(str(db))
    try:
        beliefs = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content FROM beliefs"
        ).fetchall()
        anchors = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT anchor_text FROM edges WHERE anchor_text IS NOT NULL"
        ).fetchall()
    finally:
        store.close()
    assert beliefs, "the commit produced no beliefs: " + stderr
    # The subject and object are ascii ("index", "faster queries"); the
    # multibyte run survives only in the edge's anchor text, which is the
    # verbatim span the decode had to get right.
    blob = "\n".join(str(r[0]) for r in anchors)
    assert "東京" in blob, f"multibyte run did not round-trip; got: {blob!r}"


# --- Guards --------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _calls_by_attr() -> dict[str, tuple[tuple[Path, ast.Call], ...]]:
    """Index every attribute call in the package by its attribute name.

    Parsed once for the whole module: the package is large enough that
    re-walking it per parametrized case exceeds the suite's per-test
    timeout on a loaded machine.
    """
    index: dict[str, list[tuple[Path, ast.Call]]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                index.setdefault(node.func.attr, []).append((path, node))
    return {k: tuple(v) for k, v in index.items()}


def _calls(name: str) -> tuple[tuple[Path, ast.Call], ...]:
    """Every call in the package whose callee attribute is `name`."""
    return _calls_by_attr().get(name, ())


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


@pytest.mark.parametrize("callee", ["run", "Popen", "check_output"])
def test_subprocess_text_mode_pins_the_encoding(callee: str) -> None:
    """Text mode without `encoding=` decodes through the process locale."""
    offenders = []
    for path, call in _calls(callee):
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "subprocess"
        ):
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        text_mode = _is_true(kw.get("text")) or _is_true(
            kw.get("universal_newlines")
        )
        if text_mode and "encoding" not in kw:
            offenders.append(f"{path.relative_to(SRC)}:{call.lineno}")
    assert not offenders, (
        f"subprocess.{callee} in text mode without encoding=: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("callee", ["read_text", "write_text"])
def test_path_text_io_pins_the_encoding(callee: str) -> None:
    """`Path.read_text`/`write_text` default to the locale encoding too.

    A positional first argument to `read_text` *is* the encoding, so it counts;
    `write_text`'s first positional is the data, so only the keyword does.
    """
    offenders = []
    for path, call in _calls(callee):
        if any(k.arg == "encoding" for k in call.keywords):
            continue
        if callee == "read_text" and call.args:
            continue
        offenders.append(f"{path.relative_to(SRC)}:{call.lineno}")
    assert not offenders, (
        f"Path.{callee} without encoding=: " + ", ".join(offenders)
    )


def test_the_guards_can_see_the_package() -> None:
    """A guard that finds nothing to check passes for the wrong reason."""
    assert len(_calls("run")) >= 14
    assert len(_calls("write_text")) >= 13
