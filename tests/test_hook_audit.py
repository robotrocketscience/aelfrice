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
from aelfrice.hook_audit import (
    AUDIT_ROTATION_HOOK,
    _append_audit,
    audit_window,
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
    # #1357: `source` is a SessionStart-only field. Nothing else passes it,
    # so this row must not grow one. Pinned here because the omission rests
    # entirely on the helper's parameter default: flipping that default to
    # any truthy string makes every UPS row carry it, and the SessionStart
    # tests cannot see it — they read row [0] of a SessionStart-only file.
    assert "source" not in rec
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


def test_user_prompt_submit_audits_zero_hit_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1528: a fire where retrieval ran and returned nothing writes a row.

    Before #1528 this test asserted the opposite — that the file stayed
    empty. That absence was the defect: the audit write lived inside
    `if hits:` and `elif gate_skip:`, so the one population an analysis
    most wants (retrieval did full work and found nothing) matched neither
    branch and left no trace. Every rate derived from the log had that
    class missing from its denominator with nothing indicating the gap.
    """
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "totally unrelated content")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("explain the deployment process for dinosaurs"))
    sout = io.StringIO()
    user_prompt_submit(stdin=sin, stdout=sout)
    audit_path = _audit_path_for_db(db)
    records = read_hook_audit(audit_path)
    assert len(records) == 1, records
    rec = records[0]
    assert rec["hook"] == AUDIT_HOOK_USER_PROMPT_SUBMIT
    # A MEASURED zero, not a gap: the count is present and it is 0.
    assert rec["n_beliefs"] == 0
    assert rec["n_locked"] == 0
    assert rec["beliefs"] == []
    # Not the gate-skip population — the shape gate did not fire here, and
    # conflating the two would put a full-retrieval fire in the benchmark's
    # permanently-excluded bucket.
    assert "prompt_shape_gate_skip" not in rec or (
        rec["prompt_shape_gate_skip"] is None
    )
    # Nothing was rendered, so the block is empty rather than absent.
    assert rec["rendered_block"] == ""


def test_prompt_shape_gate_audit_records_skip_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gated ack prompt must write an audit record with prompt_shape_gate_skip set."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    sin = io.StringIO(_payload("yes", session_id="gate-audit-sess"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout)
    assert rc == 0
    audit_path = _audit_path_for_db(db)
    records = read_hook_audit(audit_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["hook"] == AUDIT_HOOK_USER_PROMPT_SUBMIT
    assert rec["n_beliefs"] == 0
    assert "prompt_shape_gate_skip" in rec
    skip_reason = rec["prompt_shape_gate_skip"]
    assert isinstance(skip_reason, str) and len(skip_reason) > 0


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


def _session_start_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object],
) -> dict[str, object]:
    """Fire SessionStart with `payload` and return its audit row."""
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
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\ninject_all_locked = true\n",
        encoding="utf-8",
    )
    sin = io.StringIO(json.dumps(payload))
    sout = io.StringIO()
    assert session_start(stdin=sin, stdout=sout) == 0
    return read_hook_audit(_audit_path_for_db(db))[0]


@pytest.mark.parametrize("source", ["compact", "startup", "resume"])
def test_session_start_audit_records_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    """#1357: the row carries the trigger it was fired with.

    Parametrised over distinct values on purpose. Asserting only that the
    key is present would pass against a hardcoded constant, which is the
    failure mode that left `0 of 58` rows carrying a usable value while
    the field looked recorded.
    """
    rec = _session_start_audit_record(
        tmp_path, monkeypatch, {"session_id": "s", "source": source},
    )
    assert rec["hook"] == AUDIT_HOOK_SESSION_START
    assert rec["source"] == source


def test_session_start_audit_omits_an_absent_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No trigger in the payload means no key, not an empty string.

    Every historical row predates the field, so readers must keep
    tolerating its absence rather than seeing a null-ish sentinel.
    """
    rec = _session_start_audit_record(
        tmp_path, monkeypatch, {"session_id": "s"},
    )
    assert "source" not in rec


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
    # 500 → 1000 when #321 added beliefs[]/latency_ms/tokens fields, then
    # 1000 → 1500 when #1016 enlarged the _FRAMING_HEADER, then 1500 →
    # 2048 when #1528 added the rotation marker.
    #
    # Measured on this branch: one fire 1283 B, two fires 2557 B, marker
    # 275 B. The cap has to clear THREE bounds, and the third is new: the
    # post-rotation live file is marker + fire-3 = 1558 B, so 1500 made
    # fire-3 rotate again and left a live file holding a marker and no
    # fire at all. 2048 sits 490 B above that floor and 509 B below the
    # two-fire ceiling.
    (tmp_path / ".aelfrice.toml").write_text(
        "[hook_audit]\nmax_bytes = 2048\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    audit_path = _audit_path_for_db(db)
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    # Fire once: live has one record, under the cap; no rotation yet.
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    assert audit_path.exists()
    assert not rotated.exists()
    # Fire again: post-write size > cap → rotate to .1 → fresh live file
    # opens with the marker and takes fire-3.
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    sin = io.StringIO(_payload("how many bananas are in the kitchen"))
    user_prompt_submit(stdin=sin, stdout=io.StringIO())
    assert rotated.exists()
    # Rotated content was the post-fire-2 file (single-slot rotation).
    rotated_records = read_hook_audit(rotated)
    assert len(rotated_records) >= 1
    # Live file is fresh after rotation (fire-3 wrote to it). Size is no
    # longer the right probe: #1528 opens each rotated generation with one
    # `audit_rotation` marker row, so the file legitimately carries a
    # header as well as fire-3. Count the fires instead.
    #
    # Unconditional: before #1528 the live file did not reappear until the
    # next append, so this assertion had to be guarded by `if
    # audit_path.exists()` and could pass by never running. The marker is
    # written at rotation time, so the file exists from that moment.
    assert audit_path.exists()
    live = read_hook_audit(audit_path)
    fires = [
        r for r in live
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    assert len(fires) == 1, live


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


def test_write_audit_record_memory_db_no_cwd_pollution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:memory: DB has no real parent directory, so the audit write must be
    a clean no-op — NOT a stray ``hook_audit.jsonl`` in the process CWD.

    Regression: ``Path(":memory:").parent`` is ``.``, so the audit log was
    being written relative to the working directory. Any session running
    against an in-memory DB (tests, ``--bench``) scattered the log to the
    git-worktree root, where it sat untracked and at risk of being committed.
    """
    monkeypatch.setenv("AELFRICE_DB", ":memory:")
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt="how many bananas",
        rendered_block='<belief id="F1">the kitchen is full of bananas</belief>',
        n_beliefs=1,
        n_locked=0,
        session_id="mem-sess",
    )
    assert not (tmp_path / AUDIT_FILENAME).exists()
    assert list(tmp_path.glob("**/" + AUDIT_FILENAME)) == []


def test_audit_record_carries_the_order_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1274: the ordering policy that produced the block is recorded.

    Without it an ordering A/B cannot attribute a block to its arm from the
    audit alone, which is the only per-turn record of what was injected.
    """
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt="which policy rendered this",
        rendered_block='<belief id="F1">a belief</belief>',
        n_beliefs=1,
        n_locked=0,
        order_policy="locks_last",
    )
    records = read_hook_audit(_audit_path_for_db(db))
    assert records[-1]["order_policy"] == "locks_last"


def test_audit_record_omits_order_policy_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The field is additive: older readers must not see a null key."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    _write_hook_audit_record(
        hook=AUDIT_HOOK_USER_PROMPT_SUBMIT,
        prompt="no policy supplied",
        rendered_block='<belief id="F1">a belief</belief>',
        n_beliefs=1,
        n_locked=0,
    )
    assert "order_policy" not in read_hook_audit(_audit_path_for_db(db))[-1]


# ---------------------------------------------------------------------------
# #1528: rotation is still destructive, but it is now DETECTABLE
# ---------------------------------------------------------------------------

_ROTATE_CAP = 1000
"""Byte cap for the rotation tests. Small enough to roll over quickly."""

_APPEND_LIMIT = 500
"""Hard bound on the drive loop below, so a regression cannot hang it."""


def _fire_row(n: int) -> dict[str, object]:
    """One ordinary UPS row with a lexicographically sortable ts."""
    return {
        "hook": AUDIT_HOOK_USER_PROMPT_SUBMIT,
        "ts": f"2026-08-26T00:{n // 60:02d}:{n % 60:02d}Z",
        "n_beliefs": 0,
    }


def _append_until_rotations(
    audit_path: Path, target: int, *, start: int = 0,
) -> int:
    """Append fire rows until `target` rollovers have happened.

    Two things this deliberately does NOT do. It does not count appends:
    the serialised row size is not pinned here, so "N rows rotates once"
    would be a guess that drifts silently when a field is added. And it
    does not watch the generation stamp, which would make every test below
    circular — a stamp that stopped being written would stall this loop and
    fail on the bound instead of failing the assertion about truncation
    that the test actually makes. Rotation is detected by the live file's
    inode changing, the same marker-independent probe `aelf tail` uses.

    Returns the next unused row index. Bounded loop; raises, never hangs.
    """
    seen = 0
    i = start
    ino: int | None = None
    for _ in range(_APPEND_LIMIT):
        if seen >= target:
            return i
        _append_audit(audit_path, _fire_row(i), _ROTATE_CAP)
        i += 1
        if not audit_path.exists():
            # Rotated away and not recreated yet — the pre-#1528 shape,
            # where the fresh file appears only on the next append.
            seen += 1
            ino = None
            continue
        now = audit_path.stat().st_ino
        if ino is not None and now != ino:
            seen += 1
        ino = now
    raise AssertionError(
        f"only {seen} of {target} rotations in {_APPEND_LIMIT} appends"
    )


def test_first_rotation_loses_nothing_and_says_so(tmp_path: Path) -> None:
    """One rollover fills an empty `.1`. That is a COMPLETE history.

    The marker must not cry truncation here, or the signal is worthless:
    a warning that fires on every rotated log tells a reader nothing.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    _append_until_rotations(audit_path, 1)
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    assert rotated.exists()
    window = audit_window(sorted(tmp_path.glob(AUDIT_FILENAME + "*")))
    assert window.generation == 2
    assert window.discarded_generations == 0
    assert window.truncated is False
    assert window.rotated_present is True


def test_second_rotation_is_detectable_as_truncation(tmp_path: Path) -> None:
    """Rotate twice: the first archive is destroyed, and a reader can tell.

    This is the whole point of #1528. `os.replace` into the single `.1`
    slot leaves nothing behind, so no amount of reading the surviving
    files can recover the discarded generation — the count has to be
    carried forward at rotation time or it is gone.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    nxt = _append_until_rotations(audit_path, 2)
    # One more fire so the live generation has content of its own.
    _append_audit(audit_path, _fire_row(nxt), _ROTATE_CAP)
    logs = sorted(tmp_path.glob(AUDIT_FILENAME + "*"))
    window = audit_window(logs)
    assert window.truncated is True
    assert window.discarded_generations >= 1
    assert window.generation >= 3
    # The window it DOES cover is reported, so the rate has a stated scope.
    assert window.first_ts is not None
    assert window.last_ts is not None
    assert window.first_ts <= window.last_ts
    # The surviving window starts after the beginning of history — that is
    # exactly the bias a long-horizon rate would otherwise hide.
    assert window.first_ts > "2026-08-26T00:00:00Z"


def test_rotation_marker_summarises_the_retired_generation(
    tmp_path: Path,
) -> None:
    """The marker carries the retired file's fire-count and ts range."""
    audit_path = tmp_path / AUDIT_FILENAME
    rows = _append_until_rotations(audit_path, 1)
    marker = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert marker["hook"] == AUDIT_ROTATION_HOOK
    assert marker["generation"] == 2
    assert marker["discarded_generations"] == 0
    prev = marker["rotated_from"]
    assert prev["generation"] == 1
    # Every row written so far went into the retired generation, so the
    # stamped count is exact rather than merely non-zero.
    assert prev["records"] == rows
    assert prev["first_ts"] == "2026-08-26T00:00:00Z"
    assert prev["last_ts"] == _fire_row(rows - 1)["ts"]


def test_existing_rotated_log_without_a_marker_still_parses(
    tmp_path: Path,
) -> None:
    """Logs rotated before #1528 carry no marker. They must still read.

    Degradation has to be to the only claim such a file supports:
    UNKNOWN. Never an invented count, never an exception, and — the part
    that matters — never `complete`. A `.1` with no generation stamp could
    be the first rollover, which discarded nothing, or the fifth, which
    discarded four archives, and nothing on disk can settle it.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    rotated.write_text(
        "".join(json.dumps(_fire_row(i)) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    audit_path.write_text(
        "".join(json.dumps(_fire_row(i)) + "\n" for i in range(3, 6)),
        encoding="utf-8",
    )
    window = audit_window([audit_path, rotated])
    assert window.generation == 1
    assert window.discarded_generations == 0
    # Nothing is PROVABLY gone, so the truncation warning stays quiet...
    assert window.truncated is False
    # ...but the window must not read as clean either. These two together
    # are the tri-state; either one alone is a two-valued answer to a
    # three-valued question.
    assert window.discarded_unknown is True
    assert window.complete is False
    assert window.records == 6
    assert window.first_ts == "2026-08-26T00:00:00Z"
    assert window.last_ts == "2026-08-26T00:00:05Z"
    assert window.rotated_present is True
    # And the ordinary reader is unaffected by the absence.
    assert len(read_hook_audit(audit_path)) == 3


def test_a_never_rotated_log_is_complete_not_unknown(tmp_path: Path) -> None:
    """The distinguishing half of the test above.

    Absence of a marker is only unknown when a `.1` is there to be
    destroyed. A log that has never rotated has an exact, provable answer,
    and reporting it unknown would make the flag fire on every fresh
    install — no signal at all.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    audit_path.write_text(
        "".join(json.dumps(_fire_row(i)) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    window = audit_window([audit_path])
    assert window.rotated_present is False
    assert window.truncated is False
    assert window.discarded_unknown is False
    assert window.complete is True


def test_rotating_a_pre_1528_log_records_the_archive_it_destroys(
    tmp_path: Path,
) -> None:
    """The population #1528 was filed about, rolled over once. THE fix.

    The reporter's machine had three logs already sitting at a 10.5 MB
    `.1` with no marker. The first post-#1528 rollover `os.replace`s the
    live file onto that archive and destroys it. The live file scans as
    generation 1 — it has no marker — so a writer that trusted the scan
    alone stamped `generation: 2, discarded_generations: 0` on the very
    rotation that discarded 10.5 MB of history, and the benchmark then
    printed "window complete / nothing discarded by rotation". That is a
    false completeness claim, strictly worse than the silence it replaced
    and worse than the UNKNOWN the unrotated pair already reported.

    The `.1`'s existence, read before the replace, is the only evidence
    that anything is being destroyed. The honest answer is a LOWER BOUND —
    at least one archive gone, true count unknowable — and this test pins
    both halves: the bound is >= 1, and the unknown flag is set so no
    reader mistakes the bound for a count.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    # Pre-existing UNMARKED history: rows in the archive, rows live.
    archived = [_fire_row(i) for i in range(5)]
    rotated.write_text(
        "".join(json.dumps(r) + "\n" for r in archived), encoding="utf-8",
    )
    audit_path.write_text(
        "".join(json.dumps(_fire_row(i)) + "\n" for i in range(5, 10)),
        encoding="utf-8",
    )

    before = audit_window([audit_path, rotated])
    assert before.complete is False, "the pre-state was already dishonest"

    # One rollover under the new writer.
    _append_until_rotations(audit_path, 1, start=10)

    # The old archive is provably gone: its rows are in no surviving file.
    surviving: set[object] = set()
    for path in sorted(tmp_path.glob(AUDIT_FILENAME + "*")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            surviving.add(json.loads(raw).get("ts"))
    assert not {r["ts"] for r in archived} & surviving, (
        "the archive survived; this test is not exercising a destructive "
        "rotation and proves nothing"
    )

    marker = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert marker["hook"] == AUDIT_ROTATION_HOOK
    assert marker["discarded_generations"] >= 1, (
        "the rotation that destroyed an unmarked archive claimed to have "
        "discarded nothing"
    )
    assert marker["discarded_unknown"] is True, (
        "the bound was stamped as an exact count; a reader cannot tell "
        "'at least one' from 'exactly one'"
    )

    window = audit_window(sorted(tmp_path.glob(AUDIT_FILENAME + "*")))
    assert window.truncated is True
    assert window.complete is False
    assert window.discarded_unknown is True
    assert window.discarded_generations >= 1


def test_the_unknown_flag_propagates_through_later_rotations(
    tmp_path: Path,
) -> None:
    """A lower bound never becomes an exact count again.

    Once an unmarked history is destroyed the true generation count is
    unrecoverable forever. If the flag stopped propagating, the SECOND
    post-#1528 rollover would read its own well-formed marker and report
    an exact `discarded_generations` that is really a floor.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    rotated.write_text(json.dumps(_fire_row(0)) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(_fire_row(1)) + "\n", encoding="utf-8")

    _append_until_rotations(audit_path, 2, start=2)

    marker = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert marker["discarded_unknown"] is True
    assert marker["generation"] >= 4
    assert marker["discarded_generations"] >= 2
    assert audit_window([audit_path]).discarded_unknown is True


def test_only_the_archive_cannot_claim_completeness(tmp_path: Path) -> None:
    """A `.1` handed in without its live file is a lower bound.

    Each file states what had been lost when it was CREATED, so an archive
    cannot see the loss caused by the rotation that made it an archive.
    Reading the `.1` of a twice-rotated pair on its own and printing
    "nothing discarded" is the same unearned claim in a different dress.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    _append_until_rotations(audit_path, 2)
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    assert rotated.exists()

    window = audit_window([rotated])
    assert window.rotated_present is True
    assert window.complete is False
    assert window.discarded_unknown is True

    # Control: with the live file in hand the answer is exact, so the
    # flag above is about the missing input and not set unconditionally.
    full = audit_window([audit_path, rotated])
    assert full.discarded_unknown is False
    assert full.truncated is True


def test_audit_window_survives_an_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-soft guard, exercised rather than asserted in prose.

    `audit_window` is a diagnostic printed beside a rate. A file that
    `is_file()` but raises on read — a permissions change, a vanished NFS
    mount — must cost the caller the unreadable file's rows and nothing
    else. Monkeypatched rather than chmod'd because a chmod test is a
    no-op for root, which is how this suite runs in CI.
    """
    import aelfrice.hook_audit as ha

    good = tmp_path / AUDIT_FILENAME
    bad = tmp_path / (AUDIT_FILENAME + AUDIT_ROTATED_SUFFIX)
    for path in (good, bad):
        path.write_text(
            "".join(json.dumps(_fire_row(i)) + "\n" for i in range(3)),
            encoding="utf-8",
        )

    real_scan = ha._scan_audit_file

    def _raise_on_bad(path: Path):
        if path == bad:
            raise PermissionError(f"denied: {path}")
        return real_scan(path)

    monkeypatch.setattr(ha, "_scan_audit_file", _raise_on_bad)

    window = ha.audit_window([good, bad])
    # The readable file's rows still land — no exception, no zeroed window.
    assert window.records == 3
    assert window.first_ts == "2026-08-26T00:00:00Z"


def test_marker_row_is_inert_to_hook_filtering_readers(
    tmp_path: Path,
) -> None:
    """The marker must not be counted as a fire by any existing reader.

    Every consumer filters on a specific `hook` value, so a marker row
    carrying `hook: audit_rotation` is skipped without any of them
    changing. If it were counted, the fix for a measurement defect would
    itself corrupt the measurement.
    """
    audit_path = tmp_path / AUDIT_FILENAME
    nxt = _append_until_rotations(audit_path, 1)
    _append_audit(audit_path, _fire_row(nxt), _ROTATE_CAP)
    all_rows = read_hook_audit(audit_path)
    fires = [
        r for r in all_rows
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    assert any(r.get("hook") == AUDIT_ROTATION_HOOK for r in all_rows)
    assert len(fires) == len(all_rows) - 1
    # The marker is not counted as a record by the window reader either.
    assert audit_window([audit_path]).records == len(fires)


def test_audit_window_ignores_missing_and_corrupt(tmp_path: Path) -> None:
    """A diagnostic must not raise on a half-written or absent file."""
    audit_path = tmp_path / AUDIT_FILENAME
    audit_path.write_text(
        json.dumps(_fire_row(1)) + "\n"
        + "{not json\n"
        + "[1, 2, 3]\n"
        + "\n"
        + json.dumps(_fire_row(2)) + "\n",
        encoding="utf-8",
    )
    window = audit_window([audit_path, tmp_path / "absent.jsonl"])
    assert window.records == 2
    assert window.truncated is False
