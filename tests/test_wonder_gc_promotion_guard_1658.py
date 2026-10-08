"""#1658: wonder GC keeps a phantom that is partway to #1650 promotion.

Evidence promotion counts your restatements, which land on twin beliefs,
and your spoken corroborations, which leave the phantom's posterior
alone. The GC predicate's alpha/beta and feedback clauses see neither,
so without this guard a stale phantom with a restatement in progress was
collected. The guard keeps a phantom with any support promotion counts,
and nothing else.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.models import (
    BELIEF_SPECULATIVE,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_WONDER_INGEST,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    Belief,
)
from aelfrice.phantom_promotion_opportunity import has_promotion_evidence
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_gc

A = "The cache expires after ten minutes."
B = "Writes bypass it entirely."


def _ts(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _phantom(
    content: str = A, *, session: str | None = "s0", created_at: str | None = None,
) -> Belief:
    return Belief(
        id="p1", content=content, content_hash="wonder_p1", alpha=0.3, beta=1.0,
        type=BELIEF_SPECULATIVE, lock_level=LOCK_NONE, locked_at=None,
        created_at=created_at or _ts(20), last_retrieved_at=None,
        session_id=session, origin=ORIGIN_SPECULATIVE,
    )


def _type(store: MemoryStore, path: Path, turns: list[tuple[str, str]]) -> None:
    """Ingest `(session, text)` user turns, typed two days ago."""
    path.write_text("\n".join(json.dumps({
        "schema_version": 1, "role": "user", "session_id": sid,
        "ts": _ts(2), "text": text,
    }) for sid, text in turns) + "\n", encoding="utf-8")
    ingest_jsonl(store, path)


def _corroborate(
    store: MemoryStore, *, session: str = "s1",
    source: str = CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    speaker: str | None = "user",
) -> None:
    store.record_corroboration(
        "p1", source_type=source, session_id=session,
        source_path_hash=f"{session}-{source}", ts=_ts(2), speaker=speaker,
    )


def _collected(store: MemoryStore) -> bool:
    deleted = wonder_gc(store, ttl_days=14, dry_run=False).deleted
    live = store.get_belief("p1", include_retired=True)
    assert live is not None
    assert deleted == (1 if live.valid_to is not None else 0)
    return live.valid_to is not None


@pytest.fixture
def store(tmp_path: Path):  # noqa: ANN201
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def test_a_plain_stale_phantom_is_collected(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    assert _collected(store)


def test_a_restated_phantom_is_kept(store: MemoryStore, tmp_path: Path) -> None:
    store.insert_belief(_phantom())
    _type(store, tmp_path / "t.jsonl", [("s1", A)])
    assert _collected(store) is False


def test_a_user_corroboration_keeps_the_phantom(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _corroborate(store)
    after = store.get_belief("p1")
    # The corroboration leaves the posterior alone, so only the guard can
    # keep the phantom.
    assert after is not None and (after.alpha, after.beta) == (0.3, 1.0)
    assert _collected(store) is False


def test_the_dry_run_does_not_count_a_kept_phantom(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _corroborate(store)
    assert wonder_gc(store, ttl_days=14, dry_run=True).scanned == 0


@pytest.mark.parametrize(("source", "speaker"), [
    (CORROBORATION_SOURCE_WONDER_INGEST, None),
    (CORROBORATION_SOURCE_TRANSCRIPT_INGEST, "assistant"),
])
def test_a_corroboration_promotion_ignores_does_not_keep_it(
    store: MemoryStore, source: str, speaker: str | None,
) -> None:
    store.insert_belief(_phantom())
    _corroborate(store, source=source, speaker=speaker)
    assert _collected(store)


def test_a_restatement_from_the_creating_session_does_not_keep_it(
    store: MemoryStore, tmp_path: Path,
) -> None:
    store.insert_belief(_phantom(session="s1"))
    _type(store, tmp_path / "t.jsonl", [("s1", A)])
    assert _collected(store)


def test_half_a_restatement_does_not_keep_it(
    store: MemoryStore, tmp_path: Path,
) -> None:
    # Promotion counts a session only when it holds every sentence.
    store.insert_belief(_phantom(f"{A}\n{B}"))
    _type(store, tmp_path / "t.jsonl", [("s1", A)])
    assert _collected(store)


def test_a_full_restatement_of_a_paragraph_keeps_it(
    store: MemoryStore, tmp_path: Path,
) -> None:
    store.insert_belief(_phantom(f"{A}\n{B}"))
    _type(store, tmp_path / "t.jsonl", [("s1", f"{A}\n{B}")])
    assert _collected(store) is False


def test_an_unparseable_creation_stamp_still_admits_user_evidence(
    store: MemoryStore,
) -> None:
    # Insert validates created_at, so only a legacy row can carry a bad
    # stamp; the in-memory copy stands in for one.
    store.insert_belief(_phantom())
    phantom = _phantom(created_at="not-a-date")
    assert has_promotion_evidence(store, phantom) is False
    _corroborate(store)
    assert has_promotion_evidence(store, phantom) is True
