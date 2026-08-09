"""#1359: the `<aelfrice-memory>` off-switch and the one-line block hint.

Acceptance, from the governing 2026-08-06 ruling ("ship the cheap half"):

- default output is byte-identical to today except the one hint line;
- `AELFRICE_MEMORY_BLOCK=0` emits no per-prompt `<aelfrice-memory>` block;
- the equivalent `[memory_block] enabled = false` TOML key does the same;
- the env var wins over the TOML key, in both directions;
- suppression does not disable the correction lane or the relevance
  sweeper, and does not touch `aelf rebuild`.

The hint's cost is pinned as an explicit number rather than a range: it is
paid on every fire that emits a block, so a silent growth in it is a
silent tax on the retrieval budget.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.hook as hook_mod
from aelfrice.hook import (
    CLOSE_TAG,
    MEMORY_BLOCK_HINT,
    OPEN_TAG,
    memory_block_enabled,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

_PROMPT = "tell me about bananas"


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("F1", "banana is a yellow fruit"))
    finally:
        store.close()


def _payload(cwd: str, prompt: str = _PROMPT, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": cwd,
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db = tmp_path / "memory.db"
    if not db.exists():
        _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(str(tmp_path))),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return sout.getvalue()


def _write_toml(tmp_path: Path, enabled: bool) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        f"[memory_block]\nenabled = {str(enabled).lower()}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The hint line
# ---------------------------------------------------------------------------

def test_hint_cost_is_pinned() -> None:
    """The per-fire cost of the hint, as an explicit number.

    97 characters / 99 UTF-8 bytes (the em dash is three) / 25 estimated
    tokens at the project's `_CHARS_PER_TOKEN = 4`. Update deliberately,
    never incidentally: this is spent on every block-emitting turn.
    """
    assert len(MEMORY_BLOCK_HINT) == 97
    assert len(MEMORY_BLOCK_HINT.encode("utf-8")) == 99
    assert -(-len(MEMORY_BLOCK_HINT) // 4) == 25


def test_hint_is_one_line_naming_switch_and_inspect_command() -> None:
    """One line, and it names both halves of the ask."""
    assert MEMORY_BLOCK_HINT.endswith("\n")
    assert MEMORY_BLOCK_HINT.count("\n") == 1
    assert "AELFRICE_MEMORY_BLOCK=0" in MEMORY_BLOCK_HINT
    assert "aelf tail" in MEMORY_BLOCK_HINT


def test_enabled_output_is_block_plus_exactly_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default output = today's bytes + the hint, and nothing else.

    Removing the hint suffix must leave a well-formed block that ends at
    `CLOSE_TAG` — i.e. the hint is appended *outside* the envelope, as the
    #857 coverage line is, so the block's own bytes are untouched.
    """
    out = _run(tmp_path, monkeypatch)
    assert OPEN_TAG in out
    assert out.endswith(MEMORY_BLOCK_HINT)
    without_hint = out[: -len(MEMORY_BLOCK_HINT)]
    assert without_hint.rstrip("\n").endswith(CLOSE_TAG)
    assert MEMORY_BLOCK_HINT not in without_hint


# ---------------------------------------------------------------------------
# The off-switch
# ---------------------------------------------------------------------------

def test_env_zero_suppresses_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    out = _run(tmp_path, monkeypatch)
    assert out == ""
    assert OPEN_TAG not in out
    assert MEMORY_BLOCK_HINT not in out


def test_toml_false_suppresses_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    _write_toml(tmp_path, enabled=False)
    out = _run(tmp_path, monkeypatch)
    assert out == ""


def test_env_beats_toml_in_both_directions(tmp_path: Path) -> None:
    """Documented precedence: env > TOML > default."""
    _write_toml(tmp_path, enabled=True)
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "0"},
    ) is False
    _write_toml(tmp_path, enabled=False)
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "1"},
    ) is True
    # Unset env falls through to TOML rather than forcing a value.
    assert memory_block_enabled(start=tmp_path, env={}) is False


def test_env_beats_toml_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precedence holds through the live hook, not just the resolver."""
    _write_toml(tmp_path, enabled=False)
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "1")
    out = _run(tmp_path, monkeypatch)
    assert OPEN_TAG in out
    assert out.endswith(MEMORY_BLOCK_HINT)


def test_unparseable_values_degrade_to_enabled(tmp_path: Path) -> None:
    """Junk env value falls through; junk TOML value degrades to on."""
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "maybe"},
    ) is True
    (tmp_path / ".aelfrice.toml").write_text(
        '[memory_block]\nenabled = "no"\n', encoding="utf-8",
    )
    serr = io.StringIO()
    assert memory_block_enabled(start=tmp_path, env={}, stderr=serr) is True
    assert "expected bool" in serr.getvalue()


def test_no_config_and_no_env_is_enabled(tmp_path: Path) -> None:
    assert memory_block_enabled(start=tmp_path, env={}) is True


# ---------------------------------------------------------------------------
# Suppression is narrow: other lanes keep firing
# ---------------------------------------------------------------------------

def test_suppression_keeps_correction_and_sweeper_lanes_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch silences stdout, not the rest of the turn.

    `apply_sentiment_feedback` is the correction lane; `_sweep_relevance_signal`
    reads the prior turn's transcript to push relevance evidence. Both run
    before retrieval and must be unaffected by the off-switch.
    """
    calls: list[str] = []
    real_sentiment = hook_mod.apply_sentiment_feedback
    real_sweep = hook_mod._sweep_relevance_signal

    def spy_sentiment(*a: object, **kw: object) -> int:
        calls.append("sentiment")
        return real_sentiment(*a, **kw)  # type: ignore[arg-type]

    def spy_sweep(*a: object, **kw: object) -> object:
        calls.append("sweep")
        return real_sweep(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook_mod, "apply_sentiment_feedback", spy_sentiment)
    monkeypatch.setattr(hook_mod, "_sweep_relevance_signal", spy_sweep)
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    out = _run(tmp_path, monkeypatch)
    assert out == ""
    assert calls == ["sentiment", "sweep"]


def test_suppression_leaves_rebuild_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aelf rebuild` is the documented inspection path; it is not gated."""
    from aelfrice.context_rebuilder import RecentTurn, rebuild_v14

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    turns = [RecentTurn(role="user", text=_PROMPT)]
    store = MemoryStore(str(db))
    try:
        block = rebuild_v14(turns, store, rebuild_log_enabled=False)
    finally:
        store.close()
    assert "banana" in block


# ---------------------------------------------------------------------------
# `aelf doctor` reports the state
# ---------------------------------------------------------------------------

def test_doctor_reports_enabled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.doctor import diagnose, format_report

    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is True
    out = format_report(report)
    assert "Memory block" in out
    assert "injection:            enabled" in out


def test_doctor_reports_disabled_state_and_names_both_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The off state must be legible in doctor, and say how it got there."""
    from aelfrice.doctor import diagnose, format_report

    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is False
    out = format_report(report)
    assert "injection:            disabled" in out
    assert "AELFRICE_MEMORY_BLOCK=0" in out
    assert "[memory_block] enabled = false" in out


def test_doctor_reads_the_project_toml_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project_root` drives the resolution, not `Path.cwd()`."""
    from aelfrice.doctor import diagnose

    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    _write_toml(tmp_path, enabled=False)
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is False
