"""UserPromptSubmit hook records injected belief ids in the session ring (#740)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.session_ring import SESSION_RING_FILENAME, read_ring_state
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-05-13T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-05-13T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _payload(prompt: str, session_id: str = "sess-A") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _fire(prompt: str, session_id: str = "sess-A") -> str:
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(prompt, session_id)),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return sout.getvalue()


def test_ups_fire_appends_hit_ids_to_ring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, [_mk("HIT01", "the cellar door is full of barrels and casks")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    out = _fire("how many barrels are in the cellar door storage")
    assert "HIT01" in out
    state = read_ring_state("sess-A")
    assert state["session_id"] == "sess-A"
    ids = {e["id"] for e in state["ring"]}
    assert "HIT01" in ids
    assert state["next_fire_idx"] == 1


def test_ups_fire_tags_locked_ids_in_ring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(
        db,
        [
            _mk(
                "LOCK01",
                "the locked-belief surface guarantees ground truth",
                lock_level=LOCK_USER,
            ),
            _mk("PLAIN1", "the locked surface is documented in INSTALL"),
        ],
    )
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _fire("tell me about the locked-belief surface and its guarantee")
    state = read_ring_state("sess-A")
    assert state, "ring should have been populated; got empty"
    by_id = {e["id"]: e for e in state["ring"]}
    if "LOCK01" in by_id:
        assert by_id["LOCK01"]["locked"] is True
    if "PLAIN1" in by_id:
        assert by_id["PLAIN1"]["locked"] is False
    # At minimum one of the two beliefs surfaced as an L1 hit and got
    # logged. The test exercises the locked-tag wiring path.
    assert by_id, "expected at least one belief in the ring"


def test_ring_resets_on_new_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(
        db,
        [
            _mk("AAA", "alpha alpha alpha mode aurora distinct"),
            _mk("BBB", "beta beta beta mode boreal distinct"),
        ],
    )
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _fire("aurora alpha mode distinct", session_id="sess-A")
    pre = read_ring_state("sess-A")
    assert any(e["id"] == "AAA" for e in pre["ring"])
    _fire("boreal beta mode distinct", session_id="sess-B")
    # sess-A view is gone (ring was wiped on the session_id change).
    assert read_ring_state("sess-A") == {}
    # sess-B view holds exactly one fire's worth of state (fire_idx=1
    # means exactly one append happened under sess-B).
    post = read_ring_state("sess-B")
    assert post["session_id"] == "sess-B"
    assert post["next_fire_idx"] == 1


def test_ring_records_multiple_fires_with_distinct_fire_idx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(
        db,
        [
            _mk("CCC", "cellar barrel cask distinct mode aurora"),
            _mk("DDD", "kitchen oven stove distinct mode boreal"),
        ],
    )
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _fire("aurora cellar barrel cask")
    _fire("boreal kitchen oven stove")
    state = read_ring_state("sess-A")
    by_id = {e["id"]: e["fire_idx"] for e in state["ring"]}
    # Two fires happened under sess-A; the ring tracked both.
    assert state["next_fire_idx"] == 2
    # CCC was injected on fire 0 (aurora query); DDD on fire 1 (boreal).
    # If the second query happened to also re-hit CCC, its fire_idx would
    # refresh to 1 — that's documented refresh-in-place semantics. So
    # require that CCC's fire_idx is 0 or 1, and DDD's is 1.
    assert by_id["CCC"] in (0, 1)
    assert by_id["DDD"] == 1


def test_ring_write_fails_soft_when_db_dir_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ring-write failure must not break the UPS hook."""
    db = tmp_path / "memory.db"
    _seed(db, [_mk("EEE", "ember ember ember kindling forge distinct")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    # Pre-create the ring file as a directory to force write failure.
    (db.parent / SESSION_RING_FILENAME).mkdir(parents=True)
    out = _fire("ember kindling forge distinct")
    # Hook still emitted its block.
    assert "EEE" in out
