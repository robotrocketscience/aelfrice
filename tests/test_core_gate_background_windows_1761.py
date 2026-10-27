"""Background labeling on Windows, and the boundaries it shares with POSIX (#1761).

#1741 turned the background labeler off on Windows until three things were
checked there: that the detached child opens no console window, that an npm
`claude.cmd` shim receives the `--settings` argument intact, and that the
reply decodes. The windows-smoke workflow runs this file on `windows-latest`
with no `claude` login and no network, so every test here runs a stub in
place of the CLI: a `claude.cmd` on Windows, a shell script elsewhere, both
handing their arguments to a small Python program that records them and
replies with one label. Each test names the mutation that kills it. Every
store is under `tmp_path`, and the belief text is synthetic.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import aelfrice.core_gate_background as bg
from aelfrice.core_gate import CLASSIFIER_VERSION
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
    CoreGateBatchItem,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

# Captured at import, before the autouse conftest fixture replaces it.
_REAL_FIND_CLAUDE = bg.find_claude

SESSION = "widgetsession-win"
CLAUDE = "/opt/widget/bin/claude"
WIN32 = sys.platform == "win32"

# Windows `CreateProcess` flags, by value (processthreadsapi.h). The module
# resolves them through `getattr` on `subprocess`, which has them only on
# Windows, so these are what the test can compare against on any host.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# The stub CLI: records its argv and the `--settings` file beside itself,
# then replies with one label and a key the ANSI code page can't decode.
# `東京` is e6 9d b1 e4 ba ac in UTF-8; 0x9d is undefined in cp1252
# and every byte is outside ASCII, so a locale decode raises on either.
_STUB_SOURCE = '''\
import json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
argv = sys.argv[1:]
(here / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
if "--settings" in argv:
    settings = Path(argv[argv.index("--settings") + 1])
    (here / "settings-copy.json").write_bytes(settings.read_bytes())
(here / "prompt.txt").write_bytes(sys.stdin.buffer.read())
reply = {
    "type": "result",
    "result": "```json\\n[{\\"index\\": 0, \\"label\\": \\"A\\"}]\\n```",
    "total_cost_usd": 0.01,
    "note": "\\u6771\\u4eac",
}
sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8"))
'''

# Forces a non-UTF-8 text-mode decode in a child interpreter: UTF-8 mode off,
# the PEP 538 coercion of C to C.UTF-8 off, and the C locale selected. On
# Windows the first line is what matters; the locale encoding there is the
# ANSI code page whatever LC_ALL says.
_LOCALE_ENV = {
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LANG": "C",
}


def _write_stub(folder: Path) -> str:
    """A `claude` the labeler can execute from `folder`; returns its path."""
    program = folder / "stub_claude.py"
    program.write_text(_STUB_SOURCE, encoding="utf-8")
    if WIN32:
        shim = folder / "claude.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{program}" %*\r\n', encoding="utf-8",
        )
    else:
        shim = folder / "claude"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{program}" "$@"\n', encoding="utf-8",
        )
        shim.chmod(0o755)
    return str(shim)


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid, content=f"The widgetwin {bid} setting is stable.",
        content_hash=f"hash-{bid}", alpha=3.0, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None, created_at="2026-10-01T00:00:00Z",
        last_retrieved_at=None, session_id=SESSION, origin=ORIGIN_USER_TRANSCRIPT,
    )


def _open_batch(db: Path) -> str:
    """A store with one belief and one open batch over it; returns the batch id."""
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief("w1"))
        return store.create_core_gate_batch(
            [CoreGateBatchItem(index=0, belief_id="w1", content_hash="hash-w1")],
            classifier_version=str(CLASSIFIER_VERSION), origin="session_end",
            session_id=SESSION, created_at="2026-10-01T00:00:00Z",
        )
    finally:
        store.close()


def _labels(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db))
    try:
        return dict(conn.execute("SELECT content_hash, label FROM core_gate_labels").fetchall())
    finally:
        conn.close()


# --- the decode boundary --------------------------------------------------------


_PROBE = """\
import subprocess, sys
try:
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes.fromhex('e69db1e4baac'))"],
        capture_output=True, text=True, timeout=60,
    )
    print("decodes")
except UnicodeDecodeError:
    print("raises")
"""

_CHILD = """\
import json, sys
from aelfrice.core_gate_background import label_batch
from aelfrice.store import MemoryStore
db, batch_id, claude = sys.argv[1:4]
store = MemoryStore(db)
try:
    record = label_batch(store, batch_id, claude=claude)
finally:
    store.close()
sys.stdout.write(json.dumps(record))
"""


@pytest.mark.timeout(120)  # #1307: two interpreter spawns, one of them nested
def test_the_reply_decodes_as_utf8_under_a_non_utf8_locale(tmp_path: Path) -> None:
    """Killed by: dropping `encoding="utf-8"` from the runner call. The stub's
    reply carries bytes the locale can't decode; without the pin the decode
    raises `UnicodeDecodeError`, a `ValueError` the run guard doesn't catch,
    and `label_batch` reports `store_error` instead of caching the label."""
    env = {**os.environ, **_LOCALE_ENV}
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True,
        text=True, encoding="utf-8", check=False, timeout=60,
    )
    assert probe.stdout.strip() == "raises", (
        "the child's text mode decodes the reply bytes, so this test exercises "
        f"nothing the fix affects: {probe.stdout!r} {probe.stderr!r}"
    )
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    claude = _write_stub(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(db), batch_id, claude],
        env=env, capture_output=True, text=True, encoding="utf-8", check=False, timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)
    assert record["outcome"] == bg.OUTCOME_ACCEPTED, record
    assert _labels(db) == {"hash-w1": "A"}
