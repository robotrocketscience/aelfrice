"""Unit tests for statusline.py: snippet emission + color fallbacks."""
from __future__ import annotations

import time

import pytest

from aelfrice import lifecycle, statusline


@pytest.fixture(autouse=True)
def _pin_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin installed_version() to a low value so format_snippet's
    self-suppression (added by the stale-banner fix) does not gate
    every snippet test on the running package version."""
    monkeypatch.setattr(
        statusline, "installed_version", lambda: "0.0.0"
    )


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
    assert "aelfrice 1.2.3 — run /aelf:upgrade" in out


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
    assert "aelfrice 1.2.3 — run /aelf:upgrade" in out
    assert statusline.ICON in out


def test_suppressed_when_running_version_caught_up() -> None:
    s = _fresh(latest="1.4.0")
    assert statusline.format_snippet(s, env={}, installed="1.4.0") == ""


def test_suppressed_when_running_version_ahead_of_cache() -> None:
    s = _fresh(latest="1.4.0")
    assert statusline.format_snippet(s, env={}, installed="1.5.0") == ""


def test_visible_when_running_version_behind_cache() -> None:
    s = _fresh(latest="1.5.0")
    out = statusline.format_snippet(s, env={}, installed="1.4.0")
    assert out != ""
    assert "1.5.0" in out


def test_visible_when_installed_unknown() -> None:
    s = _fresh(latest="1.5.0")
    out = statusline.format_snippet(s, env={}, installed="")
    assert out != ""
    assert "1.5.0" in out


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


# ---------------------------------------------------------------------------
# #932 — L0 / L1 / ? count badges
# ---------------------------------------------------------------------------


def _seed_store(path, *, locked: int, speculative: int, stale_targets: int) -> None:
    """Build a minimal aelfrice DB at `path` with the given row counts.

    Schema is reproduced inline (rather than imported from store.py) so
    the test exercises the exact column shape statusline._count_badges
    reads — if a future migration renames `lock_level` or `dst`, this
    test catches it independently of MemoryStore.
    """
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE beliefs (
            id          TEXT PRIMARY KEY,
            lock_level  TEXT NOT NULL DEFAULT 'none'
        );
        CREATE TABLE edges (
            src   TEXT NOT NULL,
            dst   TEXT NOT NULL,
            type  TEXT NOT NULL,
            PRIMARY KEY (src, dst, type)
        );
        """
    )
    rows = []
    for i in range(locked):
        rows.append((f"L{i:04d}", "user"))
    for i in range(speculative):
        rows.append((f"S{i:04d}", "none"))
    conn.executemany("INSERT INTO beliefs(id, lock_level) VALUES (?, ?)", rows)
    edges = [(f"src{i}", f"L{i:04d}", "POTENTIALLY_STALE") for i in range(stale_targets)]
    if edges:
        conn.executemany("INSERT INTO edges(src, dst, type) VALUES (?, ?, ?)", edges)
    conn.commit()
    conn.close()


def test_count_badges_empty_when_no_db(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.db"
    assert statusline._count_badges(env={}, db_path_fn=lambda: missing) == ""


def test_count_badges_empty_when_store_empty(tmp_path) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=0, speculative=0, stale_targets=0)
    assert statusline._count_badges(env={}, db_path_fn=lambda: p) == ""


def test_count_badges_basic_l0_l1(tmp_path) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=42, speculative=128, stale_targets=0)
    out = statusline._count_badges(env={}, db_path_fn=lambda: p)
    assert out == "aelf L0=42 L1=128"


def test_count_badges_includes_stale_when_nonzero(tmp_path) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=5, speculative=10, stale_targets=3)
    out = statusline._count_badges(env={}, db_path_fn=lambda: p)
    assert out == "aelf L0=5 L1=10 ?=3"


def test_count_badges_omits_stale_when_zero(tmp_path) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=1, speculative=1, stale_targets=0)
    out = statusline._count_badges(env={}, db_path_fn=lambda: p)
    assert "?=" not in out


def test_count_badges_env_opt_out(tmp_path) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=42, speculative=128, stale_targets=3)
    out = statusline._count_badges(
        env={"AELF_STATUSLINE_COUNTS": "0"},
        db_path_fn=lambda: p,
    )
    assert out == ""


def test_count_badges_stale_dedups_on_dst(tmp_path) -> None:
    """Multiple stale edges to the same target count as one stale belief."""
    import sqlite3 as _sqlite3

    p = tmp_path / "memory.db"
    _seed_store(p, locked=2, speculative=0, stale_targets=0)
    conn = _sqlite3.connect(str(p))
    conn.executemany(
        "INSERT INTO edges(src, dst, type) VALUES (?, ?, ?)",
        [
            ("a", "L0000", "POTENTIALLY_STALE"),
            ("b", "L0000", "POTENTIALLY_STALE"),
            ("a", "L0001", "POTENTIALLY_STALE"),
        ],
    )
    conn.commit()
    conn.close()
    out = statusline._count_badges(env={}, db_path_fn=lambda: p)
    assert out == "aelf L0=2 L1=0 ?=2"


def test_count_badges_swallows_db_errors(tmp_path) -> None:
    """A garbage DB must not raise — statusline can't break the shell."""
    p = tmp_path / "memory.db"
    p.write_bytes(b"this is not sqlite")
    assert statusline._count_badges(env={}, db_path_fn=lambda: p) == ""


def test_render_composes_upgrade_and_counts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=3, speculative=7, stale_targets=0)
    # Force the upgrade snippet to fire.
    cache_path = tmp_path / "cache.json"
    lifecycle._write_cache(_fresh("9.9.9"), cache_path)
    monkeypatch.setattr(
        statusline, "read_cache",
        lambda: lifecycle.read_cache(cache_path),
    )
    # Patch db_path resolution so the real _count_badges hits the
    # fixture DB rather than the user's actual store.
    import aelfrice.db_paths as _db_paths_mod
    monkeypatch.setattr(_db_paths_mod, "db_path", lambda: p)
    out = statusline.render()
    assert "9.9.9" in out
    assert "aelf L0=3 L1=7" in out
    # Upgrade snippet carries its own ' │ ' separator before the counts.
    assert statusline.SEPARATOR in out


def test_render_counts_only_when_no_upgrade(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "memory.db"
    _seed_store(p, locked=3, speculative=7, stale_targets=0)
    no_update = lifecycle.UpdateStatus(False, "1.0.0", "1.0.0", time.time(), None)
    monkeypatch.setattr(statusline, "read_cache", lambda: no_update)
    import aelfrice.db_paths as _db_paths_mod
    monkeypatch.setattr(_db_paths_mod, "db_path", lambda: p)
    out = statusline.render()
    assert out == "aelf L0=3 L1=7"
