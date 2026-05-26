"""Integration tests for #876 p3_substantive dispatcher wiring — Stop side.

PR 4 of 5 in the #876 stack. Covers the new POLICY_P3_SUBSTANTIVE branch
in _maybe_fire_cadence_checkpoint: per-turn classification push, rolling
substantive-window read, firing predicate, fail-soft behaviour.

The Stop side owns the per-turn classification push (UPS reads only), so
these tests also assert the ``classifications`` ring slot advances exactly
once per tick regardless of whether cadence fires.
"""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import (
    CONFIG_FILENAME,
    is_substantive_turn,
)

# A clearly substantive prompt (not in P2's phase-boundary allowlist) and a
# clearly filler one. Asserted below so the fixtures stay self-validating if
# the allowlist shifts.
_SUBSTANTIVE_PROMPT = "implement the substantive-window predicate for cadence"
_FILLER_PROMPT = "ok thanks"


def test_fixture_prompts_classify_as_expected() -> None:
    assert is_substantive_turn(_SUBSTANTIVE_PROMPT) is True
    assert is_substantive_turn(_FILLER_PROMPT) is False


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_P3_SUBSTANTIVE_WINDOW",
        "AELFRICE_CADENCE_P3_SUBSTANTIVE_THRESHOLD",
        "AELF_AUTOLOCK_CORRECTIONS",
    ):
        monkeypatch.delenv(var, raising=False)


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore
    MemoryStore(str(db)).close()


def _write_toml(repo: Path, body: str) -> None:
    (repo / CONFIG_FILENAME).write_text(textwrap.dedent(body))


def _write_ring_state(
    state_dir: Path,
    session_id: str,
    *,
    next_fire_idx: int = 10,
    classifications: list[bool] | None = None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": next_fire_idx,
        "evicted_total": 0,
        "bytes_at_last_fire": 0,
        "fire_idx_at_last_fire": 0,
        "classifications": classifications if classifications is not None else [],
    }
    (state_dir / "session_injected_ids.json").write_text(json.dumps(payload))


def _read_classifications(state_dir: Path) -> list[bool]:
    data = json.loads((state_dir / "session_injected_ids.json").read_text())
    return data["classifications"]


def _write_transcript_with_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "earlier"}}),
        json.dumps({"message": {"role": "user", "content": prompt}}),
    ]
    path.write_text("\n".join(lines) + "\n")


def _stub_rebuild_and_format(
    monkeypatch: pytest.MonkeyPatch,
    body: str = "<rebuilder synthesis body>",
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"n_recent": len(recent), "token_budget": token_budget})
        return body

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch, n_turns: int = 1) -> None:
    from aelfrice.context_rebuilder import RecentTurn
    canned = [RecentTurn(role="user", text=f"turn-{i}") for i in range(n_turns)]
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: canned,
    )


def _stop_payload(session_id: str, cwd: Path, transcript_path: Path) -> str:
    return json.dumps({
        "session_id": session_id,
        "cwd": str(cwd),
        "transcript_path": str(transcript_path),
    })


def _run_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt: str,
    classifications: list[bool],
    window: int = 5,
    threshold: float = 0.6,
) -> tuple[io.StringIO, list, Path]:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(db.parent, "sess-S", classifications=list(classifications))
    _write_toml(tmp_path, f"""
        [cadence]
        enabled = true
        policy = "p3_substantive"
        p3_substantive_window = {window}
        p3_substantive_threshold = {threshold}
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_with_prompt(transcript, prompt)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-S", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    return serr, calls, db.parent


# --- Tests ----------------------------------------------------------------


def test_substantive_above_threshold_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Window of mostly-substantive turns + a substantive push fires."""
    # seed 3 trues; substantive push → [T,T,T,T] over window 5 = 4/5 = 0.8
    serr, calls, state_dir = _run_stop(
        tmp_path, monkeypatch,
        prompt=_SUBSTANTIVE_PROMPT,
        classifications=[True, True, True],
        window=5, threshold=0.6,
    )
    assert len(calls) == 1, "rebuilder should fire"
    out = serr.getvalue()
    assert "cadence checkpoint fired" in out
    assert "p3_substantive" in out
    # push landed: classifications now ends with the substantive True
    classes = _read_classifications(state_dir)
    assert classes[-1] is True
    assert classes.count(True) == 4


def test_substantive_below_threshold_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse substantive history stays below threshold — no fire."""
    # seed 4 falses; substantive push → 1/5 = 0.2 < 0.6
    serr, calls, state_dir = _run_stop(
        tmp_path, monkeypatch,
        prompt=_SUBSTANTIVE_PROMPT,
        classifications=[False, False, False, False],
        window=5, threshold=0.6,
    )
    assert calls == []
    assert "cadence checkpoint" not in serr.getvalue()
    # push still happened (window advances regardless of fire)
    classes = _read_classifications(state_dir)
    assert classes[-1] is True
    assert classes.count(True) == 1


def test_filler_classification_keeps_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filler prompt is classified False — isolates the classifier effect.

    Same seeded window as ``test_substantive_above_threshold_fires`` minus
    one true: with a substantive push the ratio would clear 0.8 threshold,
    but a filler push leaves it just under.
    """
    serr, calls, state_dir = _run_stop(
        tmp_path, monkeypatch,
        prompt=_FILLER_PROMPT,
        classifications=[True, True, True, False],
        window=5, threshold=0.8,
    )
    # [T,T,T,F,F] → 3/5 = 0.6 < 0.8 → no fire
    assert calls == []
    classes = _read_classifications(state_dir)
    assert classes[-1] is False
    assert classes.count(True) == 3


def test_substantive_disabled_no_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(db.parent, "sess-S", classifications=[True, True, True, True])
    # cadence section absent: enabled defaults False
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_with_prompt(transcript, _SUBSTANTIVE_PROMPT)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-S", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []


def test_substantive_window_cap_evicts_oldest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """push_classification caps the window at p3_substantive_window.

    Seeding a window already at cap (one filler at the front) and pushing a
    substantive turn evicts the oldest entry FIFO, leaving length == window.
    """
    serr, calls, state_dir = _run_stop(
        tmp_path, monkeypatch,
        prompt=_SUBSTANTIVE_PROMPT,
        classifications=[False, True, True, True, True],
        window=5, threshold=0.6,
    )
    classes = _read_classifications(state_dir)
    assert len(classes) == 5, "window length stays capped after push"
    # oldest (the leading False) evicted; window is now all True → fires
    assert classes == [True, True, True, True, True]
    assert len(calls) == 1
