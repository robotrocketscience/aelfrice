"""tests for `aelf --help --advanced` / `aelf --advanced`.

Issue #159: README claims `aelf --help --advanced` lists hidden subcommands,
but the flag was not wired.  These tests assert the wired behaviour.

Design constraints (per project policy):
- deterministic (no I/O, no subprocess)
- ≤1 s each
- each asserts exactly one property
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import build_parser, main


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets its own throwaway DB so there are no side-effects."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


# Subcommands that must be HIDDEN from the default --help surface.
# `rebuild` is intentionally absent: at v1.4.0 (#141) it became the
# user-facing manual-trigger surface for the context rebuilder.
_ADVANCED_SUBCOMMANDS = [
    "project-warm",
    "session-delta",
    "demote",
    "validate",
    "resolve",
    "feedback",
    "migrate",
    "bench",
    "ingest-transcript",
    "uninstall",
    "unsetup",
    "upgrade",
    "statusline",
]


def _advanced_help(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=argv, out=buf)
    return code, buf.getvalue()


def _default_help() -> str:
    """Return the parser's default --help text (without invoking sys.exit)."""
    parser = build_parser()
    buf = io.StringIO()
    parser.print_help(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Default --help hides advanced subcommands
# ---------------------------------------------------------------------------


def test_help_default_hides_project_warm() -> None:
    assert "project-warm" not in _default_help()


def test_help_default_hides_session_delta() -> None:
    assert "session-delta" not in _default_help()


def test_help_default_shows_rebuild() -> None:
    """v1.4.0 (#141): `rebuild` is the user-facing manual-trigger
    surface for the context rebuilder. Must appear in default --help.
    """
    assert "rebuild" in _default_help()


def test_help_default_hides_bench() -> None:
    assert "bench" not in _default_help()


# ---------------------------------------------------------------------------
# 2. `aelf --advanced` shows all hidden subcommands and exits 0
# ---------------------------------------------------------------------------


def test_advanced_alone_exits_zero() -> None:
    code, _ = _advanced_help(["--advanced"])
    assert code == 0


def test_advanced_alone_shows_project_warm() -> None:
    _, output = _advanced_help(["--advanced"])
    assert "project-warm" in output


def test_advanced_alone_shows_session_delta() -> None:
    _, output = _advanced_help(["--advanced"])
    assert "session-delta" in output


def test_advanced_alone_shows_rebuild() -> None:
    _, output = _advanced_help(["--advanced"])
    assert "rebuild" in output


def test_advanced_alone_shows_bench() -> None:
    _, output = _advanced_help(["--advanced"])
    assert "bench" in output


def test_advanced_alone_shows_at_least_four_hidden_subcommands() -> None:
    _, output = _advanced_help(["--advanced"])
    found = [cmd for cmd in _ADVANCED_SUBCOMMANDS if cmd in output]
    assert len(found) >= 4, f"Only found: {found}"


# ---------------------------------------------------------------------------
# 3. `aelf --help --advanced` behaves identically to `aelf --advanced`
# ---------------------------------------------------------------------------


def test_help_advanced_exits_zero() -> None:
    code, _ = _advanced_help(["--help", "--advanced"])
    assert code == 0


def test_help_advanced_shows_project_warm() -> None:
    _, output = _advanced_help(["--help", "--advanced"])
    assert "project-warm" in output


def test_help_advanced_shows_session_delta() -> None:
    _, output = _advanced_help(["--help", "--advanced"])
    assert "session-delta" in output


def test_help_advanced_output_matches_advanced_alone() -> None:
    _, out_alone = _advanced_help(["--advanced"])
    _, out_with_help = _advanced_help(["--help", "--advanced"])
    assert out_alone == out_with_help


# ---------------------------------------------------------------------------
# 4. Advanced-reversed order `aelf --advanced --help` also works
# ---------------------------------------------------------------------------


def test_advanced_help_reversed_order_exits_zero() -> None:
    code, _ = _advanced_help(["--advanced", "--help"])
    assert code == 0


def test_advanced_help_reversed_order_shows_hidden_subcommands() -> None:
    _, output = _advanced_help(["--advanced", "--help"])
    assert "project-warm" in output
    assert "session-delta" in output
