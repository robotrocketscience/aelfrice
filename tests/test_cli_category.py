"""CLI tests for `aelf category` and `aelf lock --category` (#1126).

Drives the argparse surface end-to-end against a temp store via the
in-process `main()` entry point (no subprocess). The store path is set
through the `AELFRICE_DB` env var, matching the repo's CLI-test pattern.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import main


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "cli-cat.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    rc = main(argv=list(argv), out=buf)
    return rc, buf.getvalue()


def test_category_init_then_list() -> None:
    rc, _ = _run("category", "init")
    assert rc == 0
    rc, out = _run("category", "list")
    assert rc == 0
    for name in (
        "repo-rules",
        "git-workflow",
        "secrets-and-safety",
        "prose-and-docs",
        "testing",
    ):
        assert name in out
    assert "always-on" in out
    assert "keywords:" in out


def test_category_add_and_show() -> None:
    rc, _ = _run(
        "category", "add", "git-workflow",
        "--keyword", "commit and push",
        "--keyword", "rebase",
        "--tool-glob", "git push*",
        "--lock", "locked",
    )
    assert rc == 0
    rc, out = _run("category", "show", "git-workflow")
    assert rc == 0
    assert "commit and push" in out
    assert "rebase" in out
    assert "git push*" in out
    assert "default_lock: locked" in out


def test_category_add_rejects_bad_name() -> None:
    rc, out = _run("category", "add", "Bad Name")
    assert rc == 1
    assert "invalid category name" in out


def test_category_show_missing() -> None:
    rc, out = _run("category", "show", "ghost")
    assert rc == 1
    assert "no such category" in out


def test_category_set_trigger() -> None:
    _run("category", "add", "c")
    rc, _ = _run("category", "set-trigger", "c", "--keyword", "deploy")
    assert rc == 0
    _, out = _run("category", "show", "c")
    assert "deploy" in out
    rc, _ = _run("category", "set-trigger", "nope", "--keyword", "x")
    assert rc == 1


def test_category_delete() -> None:
    _run("category", "add", "c")
    rc, out = _run("category", "delete", "c")
    assert rc == 0 and "deleted category: c" in out
    rc, _ = _run("category", "delete", "c")
    assert rc == 1


def test_lock_with_category_assigns() -> None:
    _run("category", "add", "git-workflow", "--keyword", "push")
    rc, out = _run(
        "lock", "always push the branch before opening a PR",
        "--category", "git-workflow",
    )
    assert rc == 0
    assert "category: git-workflow" in out
    _, show = _run("category", "show", "git-workflow")
    assert "members: 1" in show


def test_lock_with_missing_category_skips_not_fatal() -> None:
    rc, out = _run("lock", "some rule", "--category", "ghost")
    assert rc == 0
    assert "skipped 'ghost'" in out


def test_category_assign_unassign() -> None:
    _run("category", "add", "c", "--keyword", "x")
    _, lock_out = _run("lock", "a lockable rule")
    bid = ""
    for line in lock_out.splitlines():
        if line.startswith("locked:"):
            bid = line.split(":", 1)[1].strip().split()[0]
            break
    assert bid
    rc, out = _run("category", "assign", bid, "c")
    assert rc == 0 and "assigned" in out
    rc, out = _run("category", "unassign", bid, "c")
    assert rc == 0 and "unassigned" in out
    rc, _ = _run("category", "unassign", bid, "c")
    assert rc == 1
