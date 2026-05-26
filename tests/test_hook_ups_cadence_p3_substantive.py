"""Integration tests for #876 p3_substantive dispatcher wiring — UPS side.

PR 4 of 5 in the #876 stack. Covers the new POLICY_P3_SUBSTANTIVE branch
in _maybe_run_ups_cadence_checkpoint.

UPS reads the substantive-window from the ring but does NOT push — Stop
owns the per-turn classification push. These tests assert the read-only
contract: firing on a warm window, and the ``classifications`` slot being
left untouched by the UPS path.
"""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import CONFIG_FILENAME


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
    classifications: list[bool],
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": 10,
        "evicted_total": 0,
        "bytes_at_last_fire": 0,
        "fire_idx_at_last_fire": 0,
        "classifications": list(classifications),
    }
    (state_dir / "session_injected_ids.json").write_text(json.dumps(payload))


def _read_classifications(state_dir: Path) -> list[bool]:
    data = json.loads((state_dir / "session_injected_ids.json").read_text())
    return data["classifications"]


def _stub_rebuild_and_format(
    monkeypatch: pytest.MonkeyPatch, body: str = "<body>",
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


def _run_ups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    classifications: list[bool],
    window: int = 5,
    threshold: float = 0.6,
) -> tuple[str | None, list, Path]:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(db.parent, "sess-U", classifications=classifications)
    _write_toml(tmp_path, f"""
        [cadence]
        enabled = true
        policy = "p3_substantive"
        p3_substantive_window = {window}
        p3_substantive_threshold = {threshold}
    """)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"message": {"role": "user", "content": "anything"}}) + "\n"
    )
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    payload = {
        "session_id": "sess-U",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }
    serr = io.StringIO()
    body = hook._maybe_run_ups_cadence_checkpoint(payload, "sess-U", serr)
    return body, calls, db.parent


def test_ups_substantive_above_threshold_returns_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body, calls, _ = _run_ups(
        tmp_path, monkeypatch,
        classifications=[True, True, True, True],  # 4/5 = 0.8 >= 0.6
        window=5, threshold=0.6,
    )
    assert body == "<body>"
    assert len(calls) == 1


def test_ups_substantive_below_threshold_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body, calls, _ = _run_ups(
        tmp_path, monkeypatch,
        classifications=[True, False, False, False],  # 1/5 = 0.2 < 0.6
        window=5, threshold=0.6,
    )
    assert body is None
    assert calls == []


def test_ups_does_not_push_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UPS reads the window read-only — Stop owns the push.

    The classifications slot must be byte-identical after the UPS pass.
    """
    seed = [True, True, True, True]
    body, _calls, state_dir = _run_ups(
        tmp_path, monkeypatch,
        classifications=seed,
        window=5, threshold=0.6,
    )
    assert body == "<body>"  # fired, so the read path ran
    assert _read_classifications(state_dir) == seed, "UPS must not mutate window"
