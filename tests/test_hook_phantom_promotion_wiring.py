"""Integration tests for #1132 Q2 phantom promotion-opportunity wiring in
UserPromptSubmit.

Exercises ``hook.user_prompt_submit`` end-to-end and asserts the
``<aelfrice-phantom-promotion-opportunity>`` block lands on stdout only when
the opt-in flag is on and a phantom has crossed the corroboration threshold.
Companion to the unit suite ``test_phantom_promotion_opportunity.py``.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.models import (
    BELIEF_SPECULATIVE,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    RETENTION_SNAPSHOT,
    Belief,
)
from aelfrice.phantom_promotion_opportunity import ENV_PHANTOM_PROMOTION
from aelfrice.phantom_trigger import ENV_PHANTOM_GENERATION
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the two opportunity lanes independent — #980 stays off so only the
    # promotion block under test can appear.
    monkeypatch.delenv(ENV_PHANTOM_PROMOTION, raising=False)
    monkeypatch.delenv(ENV_PHANTOM_GENERATION, raising=False)


def _seed_promotable_db(db: Path, bid: str = "ph1") -> None:
    store = MemoryStore(str(db))
    store.insert_belief(
        Belief(
            id=bid,
            content="short queries should prefer BM25F over cosine",
            content_hash=f"h_{bid}",
            alpha=0.3,
            beta=1.0,
            type=BELIEF_SPECULATIVE,
            origin=ORIGIN_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
            retention_class=RETENTION_SNAPSHOT,
        )
    )
    for sess in ("s1", "s2", "s3"):
        store.record_corroboration(
            bid, source_type="filesystem_ingest", session_id=sess
        )
    store.close()


def _payload(prompt: str, cwd: Path, session_id: str = "sess-Q") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(cwd),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _run_ups(payload: str) -> tuple[str, str]:
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hook.user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr
    )
    assert rc == 0
    return sout.getvalue(), serr.getvalue()


_TAG = "<aelfrice-phantom-promotion-opportunity>"


def test_promotion_block_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_promotable_db(db)
    out, _ = _run_ups(_payload("what retrieval strategy is best here", tmp_path))
    assert _TAG not in out


def test_promotion_block_fires_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_PROMOTION, "1")
    _seed_promotable_db(db)
    out, _ = _run_ups(_payload("what retrieval strategy is best here", tmp_path))
    assert _TAG in out
    assert "ph1" in out
    assert "aelf validate" in out


def test_promotion_block_deduped_second_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_PROMOTION, "1")
    _seed_promotable_db(db)
    prompt = _payload("what retrieval strategy is best here", tmp_path)
    _run_ups(prompt)
    # Same candidate, same session -> deduped -> no block on the second turn.
    out2, _ = _run_ups(prompt)
    assert _TAG not in out2


def test_promotion_block_absent_when_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_PROMOTION, "1")
    # A phantom with only one corroboration is not a candidate.
    store = MemoryStore(str(db))
    store.insert_belief(
        Belief(
            id="lonely",
            content="an under-corroborated speculation",
            content_hash="h_lonely",
            alpha=0.3,
            beta=1.0,
            type=BELIEF_SPECULATIVE,
            origin=ORIGIN_SPECULATIVE,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
            retention_class=RETENTION_SNAPSHOT,
        )
    )
    store.record_corroboration(
        "lonely", source_type="filesystem_ingest", session_id="s1"
    )
    store.close()
    out, _ = _run_ups(_payload("ordinary question about the code", tmp_path))
    assert _TAG not in out
