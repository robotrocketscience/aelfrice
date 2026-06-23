"""Tests for the claude-memory write-through mirror (#985).

All fixtures here are SYNTHETIC — fabricated memory-file content authored
for the test. No content is sourced from any real `~/.claude/` memory
store (the public/private boundary is directory-of-origin; derived and
"abstracted" content both count). Memory directories are built under
pytest's `tmp_path` with a synthetic `.claude/projects/.../memory/`
shape.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.claude_memory import (
    ENV_MIRROR_CLAUDE_MEMORY,
    is_memory_fact_path,
    is_memory_index,
    is_mirror_enabled,
    parse_memory_file,
)
from aelfrice.hook_claude_memory_mirror import main as mirror_main
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

_SYNTHETIC_FILE = """\
---
name: build-tool
description: one build tool for the repo
metadata:
  type: user
---

The project standardises on a single build tool for environment management.
"""


def test_parse_memory_file_extracts_all_fields() -> None:
    parsed = parse_memory_file(_SYNTHETIC_FILE)
    assert parsed is not None
    assert parsed.name == "build-tool"
    assert parsed.description == "one build tool for the repo"
    assert parsed.memory_type == "user"
    assert parsed.body == (
        "The project standardises on a single build tool for "
        "environment management."
    )


def test_parse_memory_file_no_frontmatter_returns_none() -> None:
    assert parse_memory_file("just a body, no fence") is None


def test_parse_memory_file_empty_body_returns_none() -> None:
    assert parse_memory_file("---\nname: x\n---\n\n   \n") is None


@pytest.mark.parametrize("mtype", ["user", "feedback", "project", "reference"])
def test_parse_memory_file_known_types(mtype: str) -> None:
    text = f"---\nname: x\nmetadata:\n  type: {mtype}\n---\nbody text here"
    parsed = parse_memory_file(text)
    assert parsed is not None
    assert parsed.memory_type == mtype


def test_parse_memory_file_unknown_type_is_empty() -> None:
    parsed = parse_memory_file(
        "---\nname: x\nmetadata:\n  type: bogus\n---\nbody text here"
    )
    assert parsed is not None
    assert parsed.memory_type == ""


def test_parse_memory_file_strips_quoted_values() -> None:
    parsed = parse_memory_file(
        "---\nname: \"quoted-name\"\ndescription: 'single'\n---\nbody"
    )
    assert parsed is not None
    assert parsed.name == "quoted-name"
    assert parsed.description == "single"


# ---------------------------------------------------------------------------
# Path classifiers
# ---------------------------------------------------------------------------


def test_is_memory_index() -> None:
    assert is_memory_index("/a/.claude/projects/-p/memory/MEMORY.md")
    assert not is_memory_index("/a/.claude/projects/-p/memory/fact.md")


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/u/.claude/projects/-p/memory/fact.md", True),
        ("/u/.claude/projects/-p/memory/MEMORY.md", False),       # index
        ("/u/.claude/projects/-p/memory/fact.txt", False),        # not md
        ("/u/.claude/projects/-p/other/fact.md", False),          # not memory/
        ("/u/notes/memory/fact.md", False),                       # not .claude
        ("/u/projects/app/src/store.py", False),                  # source file
    ],
)
def test_is_memory_fact_path(path: str, expected: bool) -> None:
    assert is_memory_fact_path(path) is expected


# ---------------------------------------------------------------------------
# Mirror flag resolver — default OFF / opt-in
# ---------------------------------------------------------------------------


def test_mirror_flag_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_MIRROR_CLAUDE_MEMORY, raising=False)
    assert is_mirror_enabled(start=tmp_path) is False


@pytest.mark.parametrize(
    "raw,expected", [("1", True), ("true", True), ("on", True),
                     ("0", False), ("off", False)],
)
def test_mirror_flag_env_override(
    raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, raw)
    assert is_mirror_enabled() is expected


def test_mirror_flag_explicit_over_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_MIRROR_CLAUDE_MEMORY, raising=False)
    assert is_mirror_enabled(True, start=tmp_path) is True


def test_mirror_flag_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_MIRROR_CLAUDE_MEMORY, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[memory]\nmirror_claude_memory = true\n", encoding="utf-8"
    )
    assert is_mirror_enabled(start=tmp_path) is True


def test_mirror_flag_env_beats_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[memory]\nmirror_claude_memory = true\n", encoding="utf-8"
    )
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "off")
    assert is_mirror_enabled(start=tmp_path) is False


# ---------------------------------------------------------------------------
# Hook end-to-end
# ---------------------------------------------------------------------------


def _memory_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".claude" / "projects" / "-synthetic-proj" / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_fact(memdir: Path, name: str, mtype: str, body: str) -> Path:
    f = memdir / name
    f.write_text(
        f"---\nname: {name[:-3]}\ndescription: d\nmetadata:\n"
        f"  type: {mtype}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return f


def _fire(path: Path, tmp_path: Path, *, tool: str = "Write",
          is_error: bool = False) -> int:
    payload = {
        "tool_name": tool,
        "tool_input": {"file_path": str(path)},
        "tool_response": {"isError": is_error},
        "cwd": str(tmp_path),
    }
    return mirror_main(
        stdin=io.StringIO(json.dumps(payload)), stderr=io.StringIO(),
    )


@pytest.fixture()
def mirror_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """A temp DB + flag-on environment. Returns (memory_dir, db_path)."""
    db = tmp_path / "brain.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "1")
    return _memory_dir(tmp_path), db


def _beliefs(db: Path) -> list[dict]:  # type: ignore[type-arg]
    store = MemoryStore(str(db))
    try:
        return [
            dict(r) for r in store._conn.execute(  # noqa: SLF001
                "SELECT origin, lock_level, content FROM beliefs"
            )
        ]
    finally:
        store.close()


def test_hook_mirrors_user_fact_as_user_validated_unlocked(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    f = _write_fact(memdir, "atomic.md", "user",
                    "The user prefers atomic commits over batched ones.")
    assert _fire(f, tmp_path) == 0
    rows = _beliefs(db)
    assert len(rows) == 1
    assert rows[0]["origin"] == "user_validated"
    # The mirror NEVER locks — L0 stays reserved for explicit `aelf lock`.
    assert rows[0]["lock_level"] == "none"


def test_hook_mirrors_project_fact_as_agent_inferred(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    f = _write_fact(memdir, "release.md", "project",
                    "The release pipeline runs on tagged commits.")
    _fire(f, tmp_path)
    rows = _beliefs(db)
    assert len(rows) == 1
    assert rows[0]["origin"] == "agent_inferred"
    assert rows[0]["lock_level"] == "none"


def test_hook_is_idempotent_on_rewrite(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    f = _write_fact(memdir, "x.md", "user", "A stable synthetic fact body.")
    _fire(f, tmp_path)
    _fire(f, tmp_path)  # byte-identical re-write
    rows = _beliefs(db)
    assert len(rows) == 1  # corroborated, not duplicated
    store = MemoryStore(str(db))
    try:
        n = store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM belief_corroborations"
        ).fetchone()[0]
    finally:
        store.close()
    assert n == 1


def test_hook_noop_when_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "brain.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_MIRROR_CLAUDE_MEMORY, "0")
    memdir = _memory_dir(tmp_path)
    f = _write_fact(memdir, "x.md", "user", "Should not be mirrored.")
    _fire(f, tmp_path)
    assert not db.exists() or _beliefs(db) == []


def test_hook_ignores_non_memory_path(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    _memdir, db = mirror_env
    src = tmp_path / "src.py"
    src.write_text("print('hi')\n", encoding="utf-8")
    _fire(src, tmp_path)
    assert not db.exists() or _beliefs(db) == []


def test_hook_ignores_memory_index(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    idx = memdir / "MEMORY.md"
    idx.write_text("- [x](x.md) — pointer\n", encoding="utf-8")
    _fire(idx, tmp_path)
    assert not db.exists() or _beliefs(db) == []


def test_hook_ignores_non_write_tool(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    f = _write_fact(memdir, "x.md", "user", "Synthetic body.")
    _fire(f, tmp_path, tool="Bash")
    assert not db.exists() or _beliefs(db) == []


def test_hook_ignores_errored_write(
    mirror_env: tuple[Path, Path], tmp_path: Path
) -> None:
    memdir, db = mirror_env
    f = _write_fact(memdir, "x.md", "user", "Synthetic body.")
    _fire(f, tmp_path, is_error=True)
    assert not db.exists() or _beliefs(db) == []
