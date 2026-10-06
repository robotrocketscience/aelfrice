"""#1650 part 3: default-off automatic phantom promotion.

A phantom is promoted to evidence_promoted only on evidence the model
can't produce itself (docs/design/feature-supports-writer.md): at least 3
corroborations that came from the user's typed transcript turns, from at
least 2 sessions, each after the phantom was created and from a session
other than the one that created it, with no CONTRADICTS edge and no
negative feedback. The switch is off by default; on, the hook promotes
and says so, with the undo command.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_WONDER_INGEST,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_EVIDENCE_PROMOTED,
    ORIGIN_SPECULATIVE,
    Belief,
    Edge,
)
from aelfrice.phantom_promotion_opportunity import (
    ENV_PHANTOM_AUTO_PROMOTE,
    PhantomPromotionConfig,
    auto_promote_phantoms,
    format_auto_promotion_notice,
    load_phantom_promotion_config,
)
from aelfrice.phantom_promotion_opportunity import find_evidence_promotable_phantoms
from aelfrice.store import MemoryStore

T0 = "2026-10-01T00:00:00Z"


def _phantom(bid: str = "p1", *, origin: str = ORIGIN_SPECULATIVE,
             lock: str = LOCK_NONE, session: str = "s0") -> Belief:
    return Belief(
        id=bid, content=f"Phantom {bid} says the cache expires after ten minutes.",
        content_hash=f"h_{bid}", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=lock, locked_at=T0 if lock == LOCK_USER else None,
        created_at=T0, last_retrieved_at=None, session_id=session, origin=origin,
    )


def _support(store: MemoryStore, bid: str, session: str, ts: str, *,
             source: str = CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
             speaker: str | None = "user", path: str | None = None) -> None:
    store.record_corroboration(bid, source_type=source, session_id=session,
                               source_path_hash=path or f"{session}-{ts}", ts=ts,
                               speaker=speaker)


def _qualifying(store: MemoryStore, bid: str = "p1") -> None:
    _support(store, bid, "s1", "2026-10-02T00:00:00Z")
    _support(store, bid, "s2", "2026-10-03T00:00:00Z")
    _support(store, bid, "s2", "2026-10-03T01:00:00Z")


@pytest.fixture
def store(tmp_path: Path):  # noqa: ANN201
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def _ids(store: MemoryStore) -> list[str]:
    return [b.id for b in find_evidence_promotable_phantoms(store)]


def test_three_user_supports_from_two_sessions_qualify(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _qualifying(store)
    assert _ids(store) == ["p1"]


def test_two_supports_are_not_enough(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _support(store, "p1", "s1", "2026-10-02T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T00:00:00Z")
    assert _ids(store) == []


def test_one_session_is_not_enough(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    for hour in ("00", "01", "02"):
        _support(store, "p1", "s1", f"2026-10-02T{hour}:00:00Z")
    assert _ids(store) == []


@pytest.mark.parametrize(("source", "speaker"), [
    (CORROBORATION_SOURCE_TRANSCRIPT_INGEST, "assistant"),
    (CORROBORATION_SOURCE_TRANSCRIPT_INGEST, None),
    (CORROBORATION_SOURCE_WONDER_INGEST, "user"),
    (CORROBORATION_SOURCE_COMMIT_INGEST, "user"),
])
def test_only_user_spoken_transcript_rows_count(
    store: MemoryStore, source: str, speaker: str | None,
) -> None:
    store.insert_belief(_phantom())
    _support(store, "p1", "s1", "2026-10-02T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T01:00:00Z", source=source, speaker=speaker)
    assert _ids(store) == []


def test_support_older_than_the_phantom_does_not_count(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _support(store, "p1", "s1", "2026-10-02T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T00:00:00Z")
    _support(store, "p1", "s3", "2026-09-30T00:00:00Z")
    assert _ids(store) == []


def test_a_later_offset_stamp_counts_as_after(store: MemoryStore) -> None:
    # `+00:00` sorts below `Z` as text; julianday compares the instants.
    store.insert_belief(_phantom())
    _support(store, "p1", "s1", "2026-10-01T00:00:00.500+00:00")
    _support(store, "p1", "s2", "2026-10-03T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T01:00:00Z")
    assert _ids(store) == ["p1"]


def test_the_creating_session_does_not_count(store: MemoryStore) -> None:
    store.insert_belief(_phantom(session="s1"))
    _support(store, "p1", "s1", "2026-10-02T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T00:00:00Z")
    _support(store, "p1", "s2", "2026-10-03T01:00:00Z")
    assert _ids(store) == []


def test_a_contradiction_blocks_promotion(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    store.insert_belief(_phantom("x", origin=ORIGIN_AGENT_INFERRED))
    _qualifying(store)
    store.insert_edge(Edge(src="x", dst="p1", type="CONTRADICTS", weight=1.0))
    assert _ids(store) == []


def test_negative_feedback_blocks_promotion(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _qualifying(store)
    store.insert_feedback_event(belief_id="p1", valence=-1.0, source="user",
                                created_at="2026-10-04T00:00:00Z")
    assert _ids(store) == []


@pytest.mark.parametrize("kind", ["locked", "retired", "not a phantom"])
def test_only_live_unlocked_phantoms_qualify(store: MemoryStore, kind: str) -> None:
    if kind == "locked":
        store.insert_belief(_phantom(lock=LOCK_USER))
    elif kind == "not a phantom":
        store.insert_belief(_phantom(origin=ORIGIN_AGENT_INFERRED))
    else:
        store.insert_belief(_phantom())
    _qualifying(store)
    if kind == "retired":
        store.soft_delete_belief("p1")
    assert _ids(store) == []


def test_auto_promotion_is_off_by_default(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(ENV_PHANTOM_AUTO_PROMOTE, raising=False)
    store.insert_belief(_phantom())
    _qualifying(store)
    config = load_phantom_promotion_config(start=tmp_path)
    assert config.auto_promote is False
    assert auto_promote_phantoms(store=store, config=config) == []
    b = store.get_belief("p1")
    assert b is not None and b.origin == ORIGIN_SPECULATIVE


def test_the_switch_resolves_env_then_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".aelfrice.toml").write_text("[phantom_promotion]\nauto_promote = true\n")
    monkeypatch.delenv(ENV_PHANTOM_AUTO_PROMOTE, raising=False)
    assert load_phantom_promotion_config(start=tmp_path).auto_promote is True
    monkeypatch.setenv(ENV_PHANTOM_AUTO_PROMOTE, "0")
    assert load_phantom_promotion_config(start=tmp_path).auto_promote is False


def test_on_it_promotes_once_and_reports(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    _qualifying(store)
    config = PhantomPromotionConfig(auto_promote=True)
    promoted = auto_promote_phantoms(store=store, config=config)
    assert [bid for bid, _ in promoted] == ["p1"]
    b = store.get_belief("p1")
    assert b is not None and b.origin == ORIGIN_EVIDENCE_PROMOTED
    assert auto_promote_phantoms(store=store, config=config) == []
    notice = format_auto_promotion_notice(promoted)
    assert "p1" in notice and "aelf demote <id>" in notice


def test_the_hook_block_promotes_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.hook import _maybe_phantom_promotion_block

    db = tmp_path / "hook.db"
    s = MemoryStore(str(db))
    s.insert_belief(_phantom())
    _qualifying(s)
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_AUTO_PROMOTE, "1")
    monkeypatch.delenv("AELFRICE_PHANTOM_PROMOTION", raising=False)
    block = _maybe_phantom_promotion_block(session_id="s9", cwd=tmp_path)
    assert "<aelfrice-phantom-auto-promoted>" in block and "p1" in block
    s = MemoryStore(str(db))
    try:
        b = s.get_belief("p1")
        assert b is not None and b.origin == ORIGIN_EVIDENCE_PROMOTED
    finally:
        s.close()


def test_a_notice_that_does_not_fit_is_dropped_but_the_promotion_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.hook import _maybe_phantom_promotion_block

    db = tmp_path / "hook.db"
    s = MemoryStore(str(db))
    s.insert_belief(_phantom())
    _qualifying(s)
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_AUTO_PROMOTE, "1")
    block = _maybe_phantom_promotion_block(session_id="s9", cwd=tmp_path, room_chars=10)
    assert block == ""
    s = MemoryStore(str(db))
    try:
        b = s.get_belief("p1")
        assert b is not None and b.origin == ORIGIN_EVIDENCE_PROMOTED
    finally:
        s.close()


def test_a_default_config_does_not_promote() -> None:
    # A caller that builds the config itself must get the off switch too.
    assert PhantomPromotionConfig().auto_promote is False
