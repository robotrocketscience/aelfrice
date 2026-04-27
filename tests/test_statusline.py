"""Unit tests for statusline.py: snippet emission + color fallbacks."""
from __future__ import annotations

import time

import pytest

from aelfrice import lifecycle, statusline


def _fresh(latest: str = "1.2.3", available: bool = True) -> lifecycle.UpdateStatus:
    return lifecycle.UpdateStatus(
        update_available=available,
        installed="1.0.0",
        latest=latest,
        checked=time.time(),
        sha256=None,
    )


def test_empty_snippet_when_no_update() -> None:
    s = lifecycle.UpdateStatus(False, "1.0.0", "1.0.0", time.time(), None)
    assert statusline.format_snippet(s, env={}) == ""


def test_empty_snippet_when_no_latest() -> None:
    s = lifecycle.UpdateStatus(True, "1.0.0", "", time.time(), None)
    assert statusline.format_snippet(s, env={}) == ""


def test_truecolor_when_colorterm_truecolor() -> None:
    out = statusline.format_snippet(_fresh(), env={"COLORTERM": "truecolor"})
    assert statusline.ANSI_TRUECOLOR_ORANGE in out
    assert "1.2.3" in out
    assert out.endswith(statusline.SEPARATOR)


def test_256color_fallback_when_term_256() -> None:
    out = statusline.format_snippet(
        _fresh(), env={"TERM": "xterm-256color"}
    )
    assert statusline.ANSI_256_ORANGE in out


def test_basic_yellow_when_dumb_terminal() -> None:
    out = statusline.format_snippet(_fresh(), env={"TERM": "dumb"})
    assert statusline.ANSI_BASIC_YELLOW in out


def test_no_color_strips_ansi() -> None:
    out = statusline.format_snippet(
        _fresh(), env={"NO_COLOR": "", "COLORTERM": "truecolor"}
    )
    assert "\x1b[" not in out
    assert "aelfrice 1.2.3 available" in out


def test_no_color_set_to_anything_strips_ansi() -> None:
    # Per https://no-color.org any value is treated as set.
    out = statusline.format_snippet(
        _fresh(), env={"NO_COLOR": "1", "COLORTERM": "truecolor"}
    )
    assert "\x1b[" not in out


def test_snippet_includes_upgrade_command() -> None:
    out = statusline.format_snippet(
        _fresh(), env={"COLORTERM": "truecolor"}
    )
    assert "run: aelf upgrade" in out
    assert statusline.ICON in out


def test_render_reads_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "cache.json"
    lifecycle._write_cache(_fresh("1.5.0"), p)
    # render() takes no args so we monkeypatch read_cache itself.
    monkeypatch.setattr(
        statusline, "read_cache",
        lambda: lifecycle.read_cache(p),
    )
    monkeypatch.setenv("COLORTERM", "truecolor")
    out = statusline.render()
    assert "1.5.0" in out
