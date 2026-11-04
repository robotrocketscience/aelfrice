"""#1843: the `aelf doctor` store check resolves the store once.

`_print_doctor_store_check` took the belief count from `_open_store()`,
the path from `db_path()` and the rule from `resolve_store().describe()`,
three resolutions. A `[store]` marker that changed between them made the
`store:` line pair one store's count with another store's path.

Every repository, marker, store and home directory is planted under
`tmp_path`. `AELFRICE_DB`, which the suite pins, is deleted so the marker
rule is reachable; `HOME` and `AELFRICE_USER_DB` point into `tmp_path`.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from aelfrice import cli, db_paths
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.db_paths import StoreSelection
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)


def _marker(folder: Path, value: str) -> Path:
    config = folder / CONFIG_FILENAME
    config.write_text(f'[store]\npath = "{value}"\n', encoding="utf-8")
    return config


def _seed(directory: Path, n: int) -> Path:
    directory.mkdir(parents=True)
    path = directory / "memory.db"
    store = MemoryStore(str(path))
    try:
        for i in range(n):
            bid = f"{directory.name}-{i}"
            store.insert_belief(Belief(
                id=bid,
                content=f"belief {i} in {directory.name}",
                content_hash=f"{bid}-hash",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-11-04T00:00:00Z",
                last_retrieved_at=None,
            ))
    finally:
        store.close()
    return path


def _value(out: str, label: str) -> str:
    for line in out.splitlines():
        if line.startswith(f"{label}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no {label!r} line in:\n{out}")


def test_a_marker_change_mid_check_leaves_one_store_in_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user" / "memory.db"))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()

    repo = tmp_path / "repo"
    sub = repo / "project" / "sub"
    sub.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q"], cwd=repo, check=True, capture_output=True,
        timeout=30,
    )
    outer = _marker(repo / "project", "store-a")
    store_a = _seed(repo / "project" / "store-a", 1)
    store_b = _seed(sub / "store-b", 3)
    monkeypatch.chdir(sub)

    # A nearer marker appears right after the check's first resolution,
    # whichever module name that resolution goes through.
    real_resolve = db_paths.resolve_store
    resolved: list[StoreSelection] = []

    def resolve_then_add_a_nearer_marker(
        start: Path | None = None,
    ) -> StoreSelection:
        selection = real_resolve(start)
        if not resolved:
            _marker(sub, "store-b")
        resolved.append(selection)
        return selection

    monkeypatch.setattr(cli, "resolve_store", resolve_then_add_a_nearer_marker)
    monkeypatch.setattr(
        db_paths, "resolve_store", resolve_then_add_a_nearer_marker,
    )

    buf = io.StringIO()
    cli._print_doctor_store_check(buf)
    out = buf.getvalue()

    assert resolved, "the check never resolved the store"
    assert resolved[0].path == store_a.resolve()
    # The nearer marker took effect: a fresh resolution now names store B.
    assert real_resolve().path == store_b.resolve()
    assert _value(out, "store") == f"1 belief(s) at {store_a.resolve()}"
    assert _value(out, "store rule") == f"marker ({outer.resolve()})"
