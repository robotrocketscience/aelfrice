"""UserPromptSubmit hook entry-point: stdin parse, retrieval, output format."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    CLOSE_TAG,
    DEFAULT_HOOK_TOKEN_BUDGET,
    OPEN_TAG,
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


def _payload(prompt: str) -> str:
    return json.dumps(
        {
            "session_id": "s1",
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def test_hook_emits_context_when_retrieval_finds_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    sin = io.StringIO(_payload("bananas"))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert out.startswith(OPEN_TAG + "\n")
    assert CLOSE_TAG in out
    assert "F1: the kitchen is full of bananas" in out
    assert serr.getvalue() == ""


def test_hook_marks_locked_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [
            _mk(
                "L1",
                "the user pinned this as ground truth",
                lock_level=LOCK_USER,
                locked_at="2026-04-26T01:00:00Z",
            ),
        ],
    )
    _set_db(monkeypatch, db)
    sin = io.StringIO(_payload("anything"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout)
    assert rc == 0
    out = sout.getvalue()
    assert "[locked] L1: the user pinned this as ground truth" in out


def test_hook_silent_on_empty_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=io.StringIO(""), stdout=sout)
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_on_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=io.StringIO("{not json"), stdout=sout)
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_on_non_object_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=io.StringIO('"a string"'), stdout=sout)
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_when_prompt_field_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(json.dumps({"session_id": "s1"})), stdout=sout
    )
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_when_prompt_field_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=io.StringIO(_payload("   ")), stdout=sout)
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_when_prompt_field_wrong_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "anything")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(json.dumps({"prompt": 42})), stdout=sout
    )
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_silent_when_no_retrieval_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "elephants are large")])
    _set_db(monkeypatch, db)
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("dogs")), stdout=sout
    )
    assert rc == 0
    assert sout.getvalue() == ""


def test_hook_passes_default_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    captured: dict[str, int] = {}
    import aelfrice.hook as hook_mod

    real_retrieve = hook_mod.retrieve

    def spy_retrieve(
        store: MemoryStore, query: str, token_budget: int = 2000, l1_limit: int = 50
    ) -> list[Belief]:
        captured["token_budget"] = token_budget
        return real_retrieve(store, query, token_budget=token_budget,
                              l1_limit=l1_limit)

    monkeypatch.setattr(hook_mod, "retrieve", spy_retrieve)
    user_prompt_submit(
        stdin=io.StringIO(_payload("bananas")), stdout=io.StringIO()
    )
    assert captured["token_budget"] == DEFAULT_HOOK_TOKEN_BUDGET


def test_hook_honors_explicit_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    _set_db(monkeypatch, db)
    captured: dict[str, int] = {}
    import aelfrice.hook as hook_mod

    real_retrieve = hook_mod.retrieve

    def spy_retrieve(
        store: MemoryStore, query: str, token_budget: int = 2000, l1_limit: int = 50
    ) -> list[Belief]:
        captured["token_budget"] = token_budget
        return real_retrieve(store, query, token_budget=token_budget,
                              l1_limit=l1_limit)

    monkeypatch.setattr(hook_mod, "retrieve", spy_retrieve)
    user_prompt_submit(
        stdin=io.StringIO(_payload("bananas")),
        stdout=io.StringIO(),
        token_budget=314,
    )
    assert captured["token_budget"] == 314


def test_hook_non_blocking_on_internal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "bananas")])
    _set_db(monkeypatch, db)
    import aelfrice.hook as hook_mod

    def boom(
        _store: MemoryStore, _q: str, token_budget: int = 2000, l1_limit: int = 50
    ) -> list[Belief]:
        _ = token_budget, l1_limit
        raise RuntimeError("simulated retrieval failure")

    monkeypatch.setattr(hook_mod, "retrieve", boom)
    sout = io.StringIO()
    serr = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("bananas")), stdout=sout, stderr=serr
    )
    assert rc == 0
    assert sout.getvalue() == ""
    assert "simulated retrieval failure" in serr.getvalue()
