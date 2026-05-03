"""SessionStart hook entry-point: payload protocol, output, v2.0 selective mode.

v2.0 default (#373 / #199 H3): SessionStart emits no baseline block —
locks reach the model via the per-turn UPS top-K path. The legacy
all-locked SessionStart behaviour is preserved behind
`[user_prompt_submit_hook] inject_all_locked = true` in
`.aelfrice.toml`; the `_legacy_mode` fixture below exercises that path.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.hook import (
    SESSION_START_CLOSE_TAG,
    SESSION_START_OPEN_TAG,
    session_start,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _legacy_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the config loader into legacy all-locked SessionStart.

    Writes a `.aelfrice.toml` with `inject_all_locked = true` into
    `tmp_path` and chdirs there so `load_user_prompt_submit_config`
    discovers it on its parent-walk. Used by the v1.x-compat tests
    below; selective-mode tests omit it and inherit the v2.0 default.
    """
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\ninject_all_locked = true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0 if lock_level == LOCK_USER else 1.0,
        beta=0.5 if lock_level == LOCK_USER else 1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


# ---- v2.0 default: selective mode emits no baseline --------------------


def test_session_start_selective_default_emits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2.0 default: SessionStart writes no baseline block even with locks.

    Reading 2 of #373: locks route through the per-turn UPS path. The
    SessionStart payload has no prompt, so no relevance signal exists
    here — the hook returns early and stdout stays empty.
    """
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk("L1", "user is jonsobol", LOCK_USER, "2026-04-26T00:00:00Z"),
            _mk("L2", "primary db is sqlite", LOCK_USER, "2026-04-26T00:00:00Z"),
        ],
    )
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)  # no .aelfrice.toml — selective default
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


# ---- legacy all-locked SessionStart (inject_all_locked = true) ---------


def test_session_start_legacy_emits_locked_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _legacy_mode(tmp_path, monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk("L1", "user is jonsobol", LOCK_USER, "2026-04-26T00:00:00Z"),
            _mk("L2", "primary db is sqlite", LOCK_USER, "2026-04-26T00:00:00Z"),
        ],
    )
    _set_db(monkeypatch, db)
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert out.startswith(SESSION_START_OPEN_TAG + "\n")
    assert SESSION_START_CLOSE_TAG in out
    assert '<belief id="L1" lock="user">user is jonsobol</belief>' in out
    assert '<belief id="L2" lock="user">primary db is sqlite</belief>' in out


def test_session_start_legacy_skips_unlocked_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy SessionStart still surfaces L0 only (empty query → no L1)."""
    _legacy_mode(tmp_path, monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk("L1", "user is jonsobol", LOCK_USER, "2026-04-26T00:00:00Z"),
            _mk("F1", "totally normal unlocked belief"),
        ],
    )
    _set_db(monkeypatch, db)
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert "L1" in out
    assert "F1" not in out
    assert "totally normal" not in out


# ---- exit code contract ------------------------------------------------


def test_session_start_returns_zero_when_no_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "unlocked only")])
    _set_db(monkeypatch, db)
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_session_start_returns_zero_on_empty_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [])
    _set_db(monkeypatch, db)
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_session_start_returns_zero_on_empty_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SessionStart payload may be empty-ish; never block on it.

    Exercised under legacy mode so we still get a non-empty stdout to
    prove the locked-belief surfaces independent of the (drained but
    empty) payload.
    """
    _legacy_mode(tmp_path, monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("L1", "user is jonsobol", LOCK_USER, "2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    sin = io.StringIO("")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    # Locked belief still surfaces; we never read fields from the payload.
    assert "L1" in sout.getvalue()


def test_session_start_legacy_does_not_emit_user_prompt_submit_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct tags so the model can tell channels apart."""
    _legacy_mode(tmp_path, monkeypatch)
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("L1", "x", LOCK_USER, "2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    session_start(stdin=sin, stdout=sout, stderr=serr)
    out = sout.getvalue()
    assert "<aelfrice-baseline>" in out
    assert "<aelfrice-memory>" not in out
