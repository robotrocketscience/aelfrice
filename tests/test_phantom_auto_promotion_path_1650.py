"""#1650 part 3, review round 1: the production restatement path and its gaps.

When you restate a phantom, ingest stores your sentence as its own
belief (a *twin*), because a phantom's identity comes from wonder's
inputs rather than its text. So the promotion rule counts an exact-text
twin, and the twin's own user-spoken corroborations, as support. The
undo is a veto, a support without a session never counts, the notice
names the configured session count, and notice plus note stay inside the
hook's room.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aelfrice.ingest import _ingest_turn_ids, ingest_jsonl
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    Belief,
)
from aelfrice.phantom_promotion_opportunity import (
    ENV_PHANTOM_AUTO_PROMOTE,
    PhantomPromotionConfig,
    auto_promote_phantoms,
    format_auto_promotion_notice,
)
from aelfrice.promotion import SOURCE_REVERT_EVIDENCE, revert_evidence_promotion
from aelfrice.store import MemoryStore

T0 = "2026-10-01T00:00:00Z"
RESTATED = "The deploy script must run the migrations before it restarts the workers."


def _phantom(bid: str = "p1", text: str = RESTATED, created: str = T0) -> Belief:
    # wonder_ingest records no session, so a phantom's session_id is NULL.
    return Belief(
        id=bid, content=text, content_hash=f"wonder_{bid}", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at=created, last_retrieved_at=None, session_id=None,
        origin=ORIGIN_SPECULATIVE,
    )


def _user_turns(path: Path, sessions_and_days: list[tuple[str, str]],
                text: str = RESTATED) -> Path:
    path.write_text("\n".join(json.dumps({
        "schema_version": 1, "role": "user", "session_id": sid,
        "ts": f"2026-10-{day}T10:00:00Z", "text": text,
    }) for sid, day in sessions_and_days) + "\n", encoding="utf-8")
    return path


THREE = [("s1", "02"), ("s2", "03"), ("s3", "04")]


@pytest.fixture
def store(tmp_path: Path):  # noqa: ANN201
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def _ids(store: MemoryStore) -> list[str]:
    return [b.id for b in store.find_evidence_promotable_phantoms()]


def test_restating_a_phantom_in_your_own_turns_promotes_it(
    store: MemoryStore, tmp_path: Path,
) -> None:
    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    assert _ids(store) == ["p1"]


def test_a_twin_from_before_the_phantom_does_not_count(
    store: MemoryStore, tmp_path: Path,
) -> None:
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    store.insert_belief(_phantom(created="2026-10-05T00:00:00Z"))
    assert _ids(store) == []


def test_a_twin_matches_through_surrounding_space(
    store: MemoryStore, tmp_path: Path,
) -> None:
    store.insert_belief(_phantom(text="  " + RESTATED + "  "))
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    assert _ids(store) == ["p1"]


def test_a_twin_must_match_exactly_apart_from_space(
    store: MemoryStore, tmp_path: Path,
) -> None:
    # Twins are found through the indexed content hash of the exact text,
    # so a case difference is a different sentence.
    store.insert_belief(_phantom(text=RESTATED.upper()))
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    assert _ids(store) == []


def test_a_different_sentence_is_not_a_twin(store: MemoryStore, tmp_path: Path) -> None:
    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(
        tmp_path / "t.jsonl", THREE,
        text="The deploy script must never restart the workers twice."))
    assert _ids(store) == []


def test_two_sessions_of_restatement_fall_short(store: MemoryStore, tmp_path: Path) -> None:
    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", [("s1", "02"), ("s2", "03")]))
    assert _ids(store) == []


def test_an_undone_promotion_is_never_repeated(store: MemoryStore, tmp_path: Path) -> None:
    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    config = PhantomPromotionConfig(auto_promote=True)
    assert [bid for bid, _ in auto_promote_phantoms(store=store, config=config)] == ["p1"]
    revert_evidence_promotion(store, "p1")
    assert auto_promote_phantoms(store=store, config=config) == []
    b = store.get_belief("p1")
    assert b is not None and b.origin == ORIGIN_SPECULATIVE


def test_the_store_and_promotion_agree_on_the_veto_source() -> None:
    from aelfrice import store as store_mod

    assert store_mod._SOURCE_REVERT_EVIDENCE == SOURCE_REVERT_EVIDENCE  # noqa: SLF001


def test_a_support_without_a_session_does_not_count(store: MemoryStore) -> None:
    store.insert_belief(_phantom())
    for sid, ts in (("s1", "2026-10-02T00:00:00Z"), ("s2", "2026-10-03T00:00:00Z"),
                    (None, "2026-10-03T02:00:00Z")):
        store.record_corroboration("p1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                                   session_id=sid, source_path_hash=f"h{ts}", ts=ts,
                                   speaker="user")
    assert _ids(store) == []


def test_the_notice_names_the_configured_session_count() -> None:
    notice = format_auto_promotion_notice([("p1", "x")], min_sessions=4)
    assert "in 4 or more sessions" in notice


def test_notice_and_note_together_stay_inside_the_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.hook import _maybe_phantom_promotion_block

    db = tmp_path / "hook.db"
    s = MemoryStore(str(db))
    s.insert_belief(_phantom())
    ingest_jsonl(s, _user_turns(tmp_path / "t.jsonl", THREE))
    # A second phantom that qualifies for the note but not for promotion.
    s.insert_belief(_phantom("p2", text="A second phantom about cache warmers."))
    for i, sid in enumerate(("s1", "s2", "s3")):
        s.record_corroboration("p2", source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
                               session_id=sid, ts=f"2026-10-0{i + 2}T00:00:00Z")
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_AUTO_PROMOTE, "1")
    monkeypatch.setenv("AELFRICE_PHANTOM_PROMOTION", "1")
    # The room fits the note alone and the notice alone, but not both. Each
    # length is measured from the code, not assumed.
    from aelfrice.phantom_promotion_opportunity import (
        detect_promotable_phantoms,
        format_promotion_note,
    )

    s = MemoryStore(str(db))
    try:
        opportunities = detect_promotable_phantoms(s, min_corroborations=3, min_sessions=2)
        assert [o.belief_id for o in opportunities] == ["p2"]
        note_len = len(format_promotion_note(opportunities))
    finally:
        s.close()
    notice_len = len(format_auto_promotion_notice([("p1", RESTATED)]))
    room = max(note_len, notice_len) + 5
    assert note_len + notice_len + 1 > room
    block = _maybe_phantom_promotion_block(session_id="s9", cwd=tmp_path, room_chars=room)
    assert "<aelfrice-phantom-auto-promoted>" in block
    assert len(block) <= room


def test_restatements_on_a_twin_someone_else_created_count(store: MemoryStore) -> None:
    # The memory mirror (or a commit) stored the sentence first, as
    # agent_inferred. Its creation is not your evidence; your typed
    # restatements that corroborate it are.
    store.insert_belief(_phantom())
    twin = _phantom("t1", created="2026-10-02T00:00:00Z")
    twin.origin = "agent_inferred"
    twin.content_hash = hashlib.sha256(RESTATED.encode("utf-8")).hexdigest()
    twin.session_id = "s1"
    store.insert_belief(twin)
    for sid, ts in (("s2", "2026-10-03T00:00:00Z"), ("s3", "2026-10-04T00:00:00Z")):
        store.record_corroboration("t1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                                   session_id=sid, source_path_hash=f"h{ts}", ts=ts,
                                   speaker="user")
    assert _ids(store) == []  # two restatements; the creation doesn't count
    store.record_corroboration("t1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                               session_id="s3", source_path_hash="h-third",
                               ts="2026-10-04T05:00:00Z", speaker="user")
    assert _ids(store) == ["p1"]


@pytest.mark.parametrize(("role", "expected"), [
    ("user", "user"), ("assistant", "assistant"), ("system", None), (None, None),
])
def test_the_worker_records_only_known_speakers(
    store: MemoryStore, role: str | None, expected: str | None,
) -> None:
    text = "The release checks must include pyright before every tag."
    _ingest_turn_ids(store, text, "transcript", session_id="s1",
                     created_at="2026-10-01T10:00:00Z", role="user")
    _ingest_turn_ids(store, text, "transcript", session_id="s2",
                     created_at="2026-10-02T10:00:00Z", role=role)
    rows = [r[0] for r in store._conn.execute(  # noqa: SLF001
        "SELECT speaker FROM belief_corroborations").fetchall()]
    assert rows == [expected]


# --- review round 2 -----------------------------------------------------------


def _twin_id(store: MemoryStore) -> str:
    h = hashlib.sha256(RESTATED.encode("utf-8")).hexdigest()
    row = store._conn.execute(  # noqa: SLF001
        "SELECT id FROM beliefs WHERE content_hash = ?", (h,)).fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.parametrize("taint", ["retired", "negative feedback", "contradicted"])
def test_a_tainted_twin_supplies_nothing(
    store: MemoryStore, tmp_path: Path, taint: str,
) -> None:
    from aelfrice.models import Edge

    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    tid = _twin_id(store)
    if taint == "retired":
        store.soft_delete_belief(tid)
    elif taint == "negative feedback":
        store.insert_feedback_event(belief_id=tid, valence=-1.0, source="user",
                                    created_at="2026-10-05T00:00:00Z")
    else:
        other = _phantom("x", text="A claim that contradicts the deploy rule.")
        other.origin = "agent_inferred"
        other.content_hash = "h_x"
        store.insert_belief(other)
        store.insert_edge(Edge(src="x", dst=tid, type="CONTRADICTS", weight=1.0))
    assert _ids(store) == []


def test_each_promotion_runs_in_an_immediate_transaction(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Taking the write lock first is what stops two hooks from both writing
    # an audit row for the same phantom.
    store.insert_belief(_phantom())
    ingest_jsonl(store, _user_turns(tmp_path / "t.jsonl", THREE))
    seen: list[bool] = []
    real = store.transaction

    def spy(*, immediate: bool = False):  # noqa: ANN202
        seen.append(immediate)
        return real(immediate=immediate)

    monkeypatch.setattr(store, "transaction", spy)
    auto_promote_phantoms(store=store, config=PhantomPromotionConfig(auto_promote=True))
    assert seen == [True]


def test_the_lookup_stays_fast_on_a_large_store(
    store: MemoryStore, request: pytest.FixtureRequest,
) -> None:
    # A wall-clock budget, so it is opt-in (#1473): `pytest --run-perf`.
    try:
        run_perf = bool(request.config.getoption("--run-perf", default=False))
    except (AttributeError, ValueError):
        run_perf = False
    if not run_perf:
        pytest.skip("perf test gated on --run-perf")
    import time

    with store.transaction():
        for i in range(20_000):
            b = _phantom(f"u{i}", text=f"User sentence number {i} about the system.")
            b.origin = "user_transcript"
            b.content_hash = f"hu{i}"
            b.session_id = f"s{i % 50}"
            store.insert_belief(b)
        for i in range(100):
            store.insert_belief(_phantom(f"p{i}", text=f"Phantom claim number {i}."))
    start = time.perf_counter()
    store.find_evidence_promotable_phantoms(max_n=5)
    # A scan joined on lower(trim(content)) took about 100 ms at this size.
    assert time.perf_counter() - start < 0.05
