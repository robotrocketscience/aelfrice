"""Per-turn hook audit log (#280 mitigation 3)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    AUDIT_DEFAULT_MAX_BYTES,
    AUDIT_FILENAME,
    AUDIT_HOOK_SESSION_START,
    AUDIT_HOOK_USER_PROMPT_SUBMIT,
    AUDIT_PROMPT_PREFIX_CAP,
    AUDIT_ROTATED_SUFFIX,
    HookAuditConfig,
    _audit_path_for_db,
    _write_hook_audit_record,
    load_hook_audit_config,
    read_hook_audit,
    session_start,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


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
        alpha=1.0,
        beta=1.0,
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


def _payload(prompt: str, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


# ---------------------------------------------------------------------------
# load_hook_audit_config
# ---------------------------------------------------------------------------

def test_default_config_is_enabled_with_default_max_bytes(
    tmp_path: Path,
) -> None:
    cfg = load_hook_audit_config(start=tmp_path, env={})
    assert cfg.enabled is True
    assert cfg.max_bytes == AUDIT_DEFAULT_MAX_BYTES


def test_env_disable_overrides_toml(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nenabled = true\n", encoding="utf-8",
    )
    cfg = load_hook_audit_config(
        start=tmp_path, env={"AELFRICE_HOOK_AUDIT": "0"},
    )
    assert cfg.enabled is False


def test_toml_disable_honored(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nenabled = false\n", encoding="utf-8",
    )
    cfg = load_hook_audit_config(start=tmp_path, env={})
    assert cfg.enabled is False


def test_toml_max_bytes_override(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nmax_bytes = 4096\n", encoding="utf-8",
    )
    cfg = load_hook_audit_config(start=tmp_path, env={})
    assert cfg.enabled is True
    assert cfg.max_bytes == 4096


def test_malformed_toml_degrades_to_default(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit\nbroken", encoding="utf-8",
    )
    serr = io.StringIO()
    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=serr)
    assert cfg == HookAuditConfig()
    assert "malformed TOML" in serr.getvalue()


def test_wrong_typed_enabled_degrades(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[hook_audit]\nenabled = "yes"\n', encoding="utf-8",
    )
    serr = io.StringIO()
    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=serr)
    assert cfg.enabled is True
    assert "expected bool" in serr.getvalue()


def test_negative_max_bytes_degrades(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nmax_bytes = -10\n", encoding="utf-8",
    )
    serr = io.StringIO()
    cfg = load_hook_audit_config(start=tmp_path, env={}, stderr=serr)
    assert cfg.max_bytes == AUDIT_DEFAULT_MAX_BYTES
    assert "expected positive int" in serr.getvalue()


# ---------------------------------------------------------------------------
# Hook integration: writes a record on UserPromptSubmit fire
# ---------------------------------------------------------------------------

def test_user_prompt_submit_writes_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("how many bananas are in the kitchen", session_id="sess-abc"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout)
    assert rc == 0
    audit_path = _audit_path_for_db(db)
    records = read_hook_audit(audit_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["hook"] == AUDIT_HOOK_USER_PROMPT_SUBMIT
    assert rec["prompt_prefix"] == "how many bananas are in the kitchen"
    assert rec["n_beliefs"] == 1
    assert rec["n_locked"] == 0
    assert rec["session_id"] == "sess-abc"
    rendered = rec["rendered_block"]
    assert isinstance(rendered, str)
    assert "<belief id=\"F1\"" in rendered
    assert isinstance(rec["ts"], str)


def test_user_prompt_submit_audit_records_locked_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk("L1", "user truth", lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z"),
            _mk("F1", "user truth"),
        ],
    )
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("what is the user truth stored here"))
    sout = io.StringIO()
    user_prompt_submit(stdin=sin, stdout=sout)
    rec = read_hook_audit(_audit_path_for_db(db))[0]
    assert rec["n_locked"] == 1
    assert rec["n_beliefs"] >= 1


def test_user_prompt_submit_no_audit_when_no_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "totally unrelated content")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("xyzzy_no_match_whatever"))
    sout = io.StringIO()
    user_prompt_submit(stdin=sin, stdout=sout)
    audit_path = _audit_path_for_db(db)
    assert not audit_path.exists() or read_hook_audit(audit_path) == []


def test_session_start_writes_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk(
                "L1", "ground truth",
                lock_level=LOCK_USER,
                locked_at="2026-04-26T01:00:00Z",
            ),
        ],
    )
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    # #373: under the v2.0 selective default SessionStart writes nothing
    # (no body → no audit row). Opt into legacy mode to exercise the
    # audit-contract path.
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\ninject_all_locked = true\n",
        encoding="utf-8",
    )
    sin = io.StringIO(json.dumps({"session_id": "sess-start"}))
    sout = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout)
    assert rc == 0
    rec = read_hook_audit(_audit_path_for_db(db))[0]
    assert rec["hook"] == AUDIT_HOOK_SESSION_START
    assert rec["prompt_prefix"] == ""
    assert rec["n_locked"] == 1
    assert rec["session_id"] == "sess-start"
    assert "<aelfrice-baseline>" in rec["rendered_block"]  # type: ignore[operator]


def test_env_disable_suppresses_audit_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "0")
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    sout = io.StringIO()
    user_prompt_submit(stdin=sin, stdout=sout)
    assert not _audit_path_for_db(db).exists()


def test_toml_disable_suppresses_audit_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nenabled = false\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    sout = io.StringIO()
    user_prompt_submit(stdin=sin, stdout=sout)
    assert not _audit_path_for_db(db).exists()


# ---------------------------------------------------------------------------
# Rotation: live file rolls to <name>.1 once max_bytes is exceeded
# ---------------------------------------------------------------------------

def test_rotation_at_max_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    # Threshold tuned so one record fits, two records exceed. Bumped
    # from 500 → 1000 when #321 added beliefs[]/latency_ms/tokens fields
    # to the audit record schema (one fire is now ~650B, two ~1300B).
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nmax_bytes = 1000\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    audit_path = _audit_path_for_db(db)
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    # Fire once: live has one record, well under 500 B; no rotation yet.
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    assert audit_path.exists()
    assert not rotated.exists()
    first_size = audit_path.stat().st_size
    # Fire again: post-write size > 500 B → rotate to .1 → fresh live file
    # is created by the next fire.
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    assert rotated.exists()
    # Rotated content was the post-fire-2 file (single-slot rotation).
    rotated_records = read_hook_audit(rotated)
    assert len(rotated_records) >= 1
    # Live file is fresh after rotation (fire-3 wrote to it).
    if audit_path.exists():
        assert audit_path.stat().st_size <= first_size + 50  # only fire-3


# ---------------------------------------------------------------------------
# Direct API: _write_hook_audit_record + read_hook_audit
# ---------------------------------------------------------------------------

def test_write_hook_audit_disabled_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "x")])
    _set_db(monkeypatch, db)
    cfg = HookAuditConfig(enabled=False)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt="x",
        rendered_block="<aelfrice-memory></aelfrice-memory>",
        n_beliefs=0,
        n_locked=0,
        config=cfg,
    )
    assert not _audit_path_for_db(db).exists()


def test_prompt_prefix_is_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "x")])
    _set_db(monkeypatch, db)
    long_prompt = "a" * (AUDIT_PROMPT_PREFIX_CAP + 500)
    cfg = HookAuditConfig(enabled=True)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt=long_prompt,
        rendered_block="<aelfrice-memory/>",
        n_beliefs=0,
        n_locked=0,
        config=cfg,
    )
    rec = read_hook_audit(_audit_path_for_db(db))[0]
    assert isinstance(rec["prompt_prefix"], str)
    assert len(rec["prompt_prefix"]) == AUDIT_PROMPT_PREFIX_CAP  # type: ignore[arg-type]


def test_session_id_omitted_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "x")])
    _set_db(monkeypatch, db)
    cfg = HookAuditConfig(enabled=True)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt="x",
        rendered_block="<aelfrice-memory/>",
        n_beliefs=0,
        n_locked=0,
        session_id=None,
        config=cfg,
    )
    rec = read_hook_audit(_audit_path_for_db(db))[0]
    assert "session_id" not in rec


def test_read_hook_audit_missing_returns_empty(tmp_path: Path) -> None:
    assert read_hook_audit(tmp_path / "no_such.jsonl") == []


def test_read_hook_audit_raises_on_corruption(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    p.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_hook_audit(p)


def test_read_hook_audit_skips_non_object_lines(tmp_path: Path) -> None:
    p = tmp_path / "audit.jsonl"
    p.write_text('{"ok": 1}\n"a string"\n42\n', encoding="utf-8")
    out = read_hook_audit(p)
    assert out == [{"ok": 1}]


def test_audit_write_failsoft_on_unwriteable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the audit path is unwriteable, the hook still returns 0 cleanly."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    # Replace the audit dir with a regular file so mkdir() raises.
    (db.parent / "aelfrice").mkdir(parents=True, exist_ok=True)
    audit_path = _audit_path_for_db(db)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("preexisting", encoding="utf-8")
    # Make it unwriteable: replace with a directory of the same name as the
    # JSONL file. Now open(..., "a") fails. (POSIX-only contract; skipped on
    # Windows where directory-named file behavior differs.)
    audit_path.unlink()
    audit_path.mkdir()
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    # Hook still produced its output block; only the audit failed.
    assert sout.getvalue().startswith("<aelfrice-memory>")
    assert "hook audit write failed" in serr.getvalue()
