"""Planted stores for the fire-census probe tests (#1887).

A planted store is built with aelfrice's own code (CLI handlers, hook entry
points, ingest, `MemoryStore`) under a test's `tmp_path`, then probed
read-only through `benchmarks.real_store_fire_census.core`.

`plant(root, monkeypatch)` pins the environment before anything opens a
store:

  * `HOME` is `root/home`, so `~/.aelfrice` is `root/home/.aelfrice`.
  * `AELFRICE_DB` is `root/repo/.git/aelfrice/memory.db`, the same layout
    as a repository store, so `ctx.store_dir` has the shape the probes
    expect.
  * `AELFRICE_USER_DB` is `root/home/.aelfrice/user/memory.db`.
  * `AELFRICE_TRANSCRIPTS_DIR` is `transcripts/` beside the store.
  * `GIT_CEILING_DIRECTORIES` is `root` and the working directory is
    `root/repo`, so nothing resolves the checkout's own git dir.
  * The module-level home paths that `tests/conftest.py` pins for the
    session are pinned again under `root/home`.

`PlantedStore` refuses to open when any of these points outside `root`.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from aelfrice import cli as _cli
from aelfrice import ingest as _ingest
from aelfrice.db_paths import clear_git_common_dir_cache
from aelfrice.derivation import _belief_id, _content_hash  # pyright: ignore[reportPrivateUsage]
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from benchmarks.real_store_fire_census import core as cc

HOOK_TIMEOUT_S = 120.0

HOOK_MODULES: dict[str, str] = {
    "main": "aelfrice.hook:main",                       # UserPromptSubmit
    "pre_compact": "aelfrice.hook:main_pre_compact",
    "session_start": "aelfrice.hook:main_session_start",
    "stop": "aelfrice.hook:main_stop",
    "transcript_logger": "aelfrice.transcript_logger:main",
    "commit_ingest": "aelfrice.hook_commit_ingest:main",
    "search_tool": "aelfrice.hook_search_tool:main",
    "agent_context": "aelfrice.hook_agent_context:main",
    "pre_issue": "aelfrice.pre_issue_create_hook:main",
    "claude_memory_mirror": "aelfrice.hook_claude_memory_mirror:main",
}

# The same (module, attribute, path under HOME) pins tests/conftest.py sets
# for the session, so in-process code writes under this test's HOME.
HOME_PINS: tuple[tuple[str, str, str], ...] = (
    ("aelfrice.auto_install", "AELFRICE_DOTDIR", ".aelfrice"),
    ("aelfrice.auto_install", "STAMP_PATH", ".aelfrice/installed-manifest-version"),
    ("aelfrice.auto_install", "OPT_OUT_PATH", ".aelfrice/opt-out-hooks.json"),
    ("aelfrice.setup", "USER_SETTINGS_PATH", ".claude/settings.json"),
    ("aelfrice.setup", "SLASH_COMMANDS_DIR_DEFAULT", ".claude/commands/aelf"),
    ("aelfrice.lifecycle", "CACHE_DIR", ".cache/aelfrice"),
    ("aelfrice.lifecycle", "CACHE_FILE", ".cache/aelfrice/update_check.json"),
    ("aelfrice.lifecycle", "MIGRATED_TO_UV_SENTINEL", ".aelfrice/migrated-to-uv"),
    ("aelfrice.temporal_spine", "SPINE_BACKFILLED_SENTINEL", ".aelfrice/spine-backfilled"),
    ("aelfrice.mcp_cleanup", "MCP_CLEANUP_SENTINEL", ".aelfrice/mcp-surface-removed"),
    ("aelfrice.doctor", "HOOK_FAILURES_LOG", ".aelfrice/logs/hook-failures.log"),
    ("aelfrice.doctor", "_AELFRICE_PROJECTS_DIR", ".aelfrice/projects"),
    ("aelfrice.telemetry", "DEFAULT_TELEMETRY_PATH", ".aelfrice/telemetry.jsonl"),
    ("aelfrice.transcript_logger", "LEGACY_TRANSCRIPTS_DIR", ".aelfrice/transcripts"),
)

# A sentinel whose absence makes its guarded step run for real (a uv tool
# install, a spine backfill, an MCP cleanup) is created up front, as the
# session fixture does. A test that probes one of these deletes it first.
PRECREATED_SENTINELS: frozenset[str] = frozenset({
    "MIGRATED_TO_UV_SENTINEL", "SPINE_BACKFILLED_SENTINEL", "MCP_CLEANUP_SENTINEL",
})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _under(path: str | Path, root: Path) -> bool:
    p = Path(path).resolve()
    r = root.resolve()
    return p == r or r in p.parents


class PlantedStore:
    """A store at `root/repo/.git/aelfrice/memory.db` built through aelfrice's APIs."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.repo = self.root / "repo"
        self.store_dir = self.repo / ".git" / "aelfrice"
        self.db_file = self.store_dir / "memory.db"
        self.home = self.root / "home"
        self.global_dir = self.home / ".aelfrice"
        self.user_db = self.global_dir / "user" / "memory.db"
        self.store: MemoryStore | None = None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir.mkdir(parents=True, exist_ok=True)

    def assert_pinned(self) -> None:
        """Raise unless HOME and both store variables point under root."""
        for var in ("HOME", "AELFRICE_DB", "AELFRICE_USER_DB"):
            val = os.environ.get(var)
            if not val or not _under(val, self.root):
                raise RuntimeError(f"planted: {var} is not pinned under the test root")
        if not _under(Path.home(), self.root):
            raise RuntimeError("planted: Path.home() is outside the test root")

    # lifecycle
    def open(self) -> MemoryStore:
        if self.store is None:
            self.assert_pinned()
            self.store = MemoryStore(str(self.db_file))
        return self.store

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    def __enter__(self) -> PlantedStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # writes through aelfrice APIs
    def ingest(self, text: str, source: str = "user", session_id: str | None = None) -> int:
        """`ingest.ingest_turn`; returns the beliefs created."""
        return _ingest.ingest_turn(self.open(), text, source, session_id=session_id)

    def ingest_turns(self, turns: list[dict[str, Any]], name: str = "turns.jsonl") -> Any:
        """Write a transcript-logger turns file under root and `ingest_jsonl` it.

        Each turn: {"role": "user"|"assistant", "text": ..., "session_id": ...}.
        """
        p = self.root / "planted_input" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps({"ts": now_iso(), **t}) + "\n")
        return _ingest.ingest_jsonl(self.open(), p)

    def belief(self, content: str, *, source: str = "planted", type: str = BELIEF_FACTUAL,
               origin: str = ORIGIN_AGENT_INFERRED, alpha: float = 1.0, beta: float = 1.0,
               session_id: str | None = None, locked: bool = False, **extra: Any) -> str:
        """Insert one belief through `MemoryStore.insert_belief`; returns its id."""
        now = now_iso()
        b = Belief(
            id=_belief_id(content, source), content=content,
            content_hash=_content_hash(content), alpha=alpha, beta=beta, type=type,
            lock_level=LOCK_USER if locked else LOCK_NONE,
            locked_at=now if locked else None, created_at=now, last_retrieved_at=None,
            session_id=session_id,
            origin=ORIGIN_USER_STATED if locked and origin == ORIGIN_AGENT_INFERRED else origin,
            **extra,
        )
        self.open().insert_belief(b)
        return b.id

    def edge(self, src: str, dst: str, type: str, weight: float = 1.0,
             anchor_text: str | None = None) -> Any:
        return self.open().insert_edge(Edge(src, dst, type, weight, anchor_text))

    def feedback(self, belief_id: str, valence: float, source: str = "planted") -> int:
        return self.open().insert_feedback_event(belief_id, valence, source, now_iso())

    def injection(self, belief_id: str, session_id: str = "s1", turn_id: str = "t1",
                  source: str = "ups", consumers: list[str] | None = None) -> int:
        return self.open().record_injection_event(
            session_id=session_id, turn_id=turn_id, belief_id=belief_id,
            injected_at=now_iso(), source=source, active_consumers=consumers or [])

    def cli(self, *args: str) -> int:
        """`aelfrice.cli.main(args)` in process with output swallowed; returns rc.

        The planted handle is closed first so the CLI opens the store itself.
        """
        self.close()
        self.assert_pinned()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = _cli.main(list(args), out=buf)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return int(rc or 0)

    def lock(self, text: str, *extra: str) -> int:
        """`aelf lock <text>` through the CLI handler."""
        return self.cli("lock", text, *extra)

    def hook(self, entry: str, payload: dict[str, Any],
             timeout_s: float = HOOK_TIMEOUT_S) -> int:
        """Run a hook entry point in a child with the payload on stdin.

        The child inherits this process's pinned environment. Returns the
        exit code; output is discarded.
        """
        self.close()
        self.assert_pinned()
        target = HOOK_MODULES.get(entry, entry)
        mod, fn = target.split(":")
        code = f"import sys; from {mod} import {fn} as f; sys.exit(f() or 0)"
        cp = subprocess.run(
            [sys.executable, "-c", code], input=json.dumps(payload).encode(),
            env=dict(os.environ), cwd=str(self.repo), capture_output=True,
            timeout=timeout_s, check=False)
        return cp.returncode

    def remove_precreated_sentinels(self) -> None:
        """Delete the sentinels `plant` created, for a probe that counts them.

        Only for a test that runs no CLI or hook code afterwards: without a
        sentinel, the step it guards runs for real.
        """
        for _mod, attr, rel in HOME_PINS:
            if attr in PRECREATED_SENTINELS:
                (self.home / rel).unlink(missing_ok=True)

    # probing
    def ctx(self, name: str = "store01") -> cc.Ctx:
        """A read-only census context over this store (the caller closes it)."""
        self.close()
        c = cc.Ctx(name, self.store_dir, self.global_dir, False, self.db_file, "repo")
        c.open()
        return c

    def probe(self, probe: cc.Probe) -> cc.Result:
        c = self.ctx()
        try:
            return cc.run_one(probe, c)
        finally:
            c.close()


def plant(root: Path, monkeypatch: pytest.MonkeyPatch) -> PlantedStore:
    """Pin the environment under root and return an opened planted store."""
    ps = PlantedStore(root)
    ps.repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(ps.home))
    monkeypatch.setenv("AELFRICE_DB", str(ps.db_file))
    monkeypatch.setenv("AELFRICE_USER_DB", str(ps.user_db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(ps.store_dir / "transcripts"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(ps.root))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(ps.repo)
    for mod_name, attr, rel in HOME_PINS:
        target = ps.home / rel
        if attr in PRECREATED_SENTINELS:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        monkeypatch.setattr(importlib.import_module(mod_name), attr, target)
    clear_git_common_dir_cache()
    ps.open()
    return ps
