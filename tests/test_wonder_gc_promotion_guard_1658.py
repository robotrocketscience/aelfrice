"""#1658: wonder GC keeps a phantom that is partway to #1650 promotion.

Evidence promotion counts your restatements, which land on twin beliefs,
and your spoken corroborations, which leave the phantom's posterior
alone. The GC predicate's alpha/beta and feedback clauses see neither,
so without this guard a stale phantom with a restatement in progress was
collected. The guard keeps a phantom that promotion could promote and
that has any support promotion counts, and nothing else: a phantom that
fails one of promotion's eligibility gates is collected whatever support
it has. The one exception is a user lock: the GC pre-filter never offers
a locked phantom for collection, so it is kept whatever its evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.models import (
    BELIEF_SPECULATIVE,
    EDGE_CONTRADICTS,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_WONDER_INGEST,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_SPECULATIVE,
    Belief,
    Edge,
)
from aelfrice.phantom_promotion_opportunity import promotion_guarded_ids
from aelfrice.promotion import SOURCE_REVERT_EVIDENCE
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_gc

A = "The cache expires after ten minutes."
B = "Writes bypass it entirely."


def _ts(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _phantom(
    content: str = A, *, session: str | None = "s0", created_at: str | None = None,
    lock: str = LOCK_NONE, tier: str = LOCK_TIER_FROZEN,
) -> Belief:
    return Belief(
        id="p1", content=content, content_hash="wonder_p1", alpha=0.3, beta=1.0,
        type=BELIEF_SPECULATIVE, lock_level=lock, lock_tier=tier,
        locked_at=_ts(19) if lock == LOCK_USER else None,
        created_at=created_at or _ts(20), last_retrieved_at=None,
        session_id=session, origin=ORIGIN_SPECULATIVE,
    )


def _type(
    store: MemoryStore, path: Path, turns: list[tuple[str, str]],
    *, days_ago: int = 2,
) -> None:
    """Ingest `(session, text)` user turns, typed `days_ago` days ago."""
    path.write_text("\n".join(json.dumps({
        "schema_version": 1, "role": "user", "session_id": sid,
        "ts": _ts(days_ago), "text": text,
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


def test_a_restatement_from_before_the_phantom_does_not_keep_it(
    store: MemoryStore, tmp_path: Path,
) -> None:
    # The phantom is 20 days old; you typed its text 25 days ago, before
    # wonder created it, so promotion doesn't count the restatement.
    store.insert_belief(_phantom())
    _type(store, tmp_path / "t.jsonl", [("s1", A)], days_ago=25)
    assert _collected(store)


# Promotion's eligibility gates. Each test gives the phantom a user
# corroboration, which on its own keeps it
# (test_a_user_corroboration_keeps_the_phantom), then fails one gate.


def test_a_phantom_with_unrestatable_residue_is_collected(
    store: MemoryStore,
) -> None:
    # Promotion keeps the whole text, so a symbol no restatement covers
    # keeps the phantom from ever being promoted.
    store.insert_belief(_phantom(f"{A} \u274c"))
    _corroborate(store)
    assert _collected(store)


def test_a_restated_question_is_collected(
    store: MemoryStore, tmp_path: Path,
) -> None:
    # Ingest never keeps a question as yours, so typing it again is no
    # restatement and the phantom can't reach promotion.
    question = "Should the cache expire after ten minutes?"
    store.insert_belief(_phantom(question))
    _type(store, tmp_path / "t.jsonl", [("s1", question), ("s2", question)])
    assert _collected(store)


def test_a_corroborated_question_is_kept(store: MemoryStore) -> None:
    # A question passes promotion's residue check, so a corroboration you
    # spoke on the phantom itself is a support promotion counts, and GC
    # keeps the question as it keeps any other phantom with one.
    store.insert_belief(_phantom("Should the cache expire after ten minutes?"))
    _corroborate(store)
    assert _collected(store) is False


@pytest.mark.parametrize("tier", [LOCK_TIER_FROZEN, LOCK_TIER_REFERENCE])
@pytest.mark.parametrize("supported", [False, True])
def test_a_locked_phantom_is_never_collected(
    store: MemoryStore, tier: str, supported: bool,
) -> None:
    # GC never collects a phantom you locked, whatever its evidence. Only
    # the pre-filter's lock exemption keeps it: the promotion guard refuses
    # a locked phantom (test_the_guard_does_not_keep_a_locked_phantom).
    store.insert_belief(_phantom(lock=LOCK_USER, tier=tier))
    if supported:
        _corroborate(store)
    assert store.query_wonder_gc_candidates(cutoff_ts=_ts(14)) == []
    assert wonder_gc(store, ttl_days=14, dry_run=True).scanned == 0
    assert _collected(store) is False


def test_the_guard_does_not_keep_a_locked_phantom(store: MemoryStore) -> None:
    # The GC pre-filter already exempts a locked phantom, so wonder_gc never
    # reaches this gate; the guard still applies it, as promotion does.
    store.insert_belief(_phantom(lock=LOCK_USER))
    _corroborate(store)
    assert promotion_guarded_ids(store, ["p1"]) == set()


def test_a_contradicted_phantom_with_support_is_collected(
    store: MemoryStore,
) -> None:
    store.insert_belief(_phantom())
    _corroborate(store)
    store.insert_belief(Belief(
        id="x", content="The cache never expires.", content_hash="x",
        alpha=1.0, beta=1.0, type="factual", lock_level=LOCK_NONE,
        locked_at=None, created_at=_ts(3), last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    ))
    store.insert_edge(Edge(src="x", dst="p1", type=EDGE_CONTRADICTS, weight=1.0))
    assert _collected(store)


def test_an_unparseable_creation_stamp_is_collected(store: MemoryStore) -> None:
    # Promotion fails closed on a stamp it can't parse. Insert validates
    # created_at, so only a legacy row can carry one; write it directly.
    # This one still sorts before the GC cutoff.
    store.insert_belief(_phantom())
    _corroborate(store)
    store._conn.execute(  # noqa: SLF001
        "UPDATE beliefs SET created_at = ? WHERE id = 'p1'",
        ("2001-13-45T00:00:00+00:00",),
    )
    assert _collected(store)


def test_the_guard_keeps_a_supported_promotable_phantom(
    store: MemoryStore,
) -> None:
    # The control for the two guard-level gate tests below.
    store.insert_belief(_phantom())
    _corroborate(store)
    assert promotion_guarded_ids(store, ["p1"]) == {"p1"}


@pytest.mark.parametrize(("source", "valence"), [
    (SOURCE_REVERT_EVIDENCE, 0.0),  # an earlier `aelf demote`: a veto
    ("user", -1.0),  # negative feedback
])
def test_the_guard_does_not_keep_a_feedback_vetoed_phantom(
    store: MemoryStore, source: str, valence: float,
) -> None:
    # Any non-exposure feedback row already keeps a phantom out of the GC
    # SQL pre-filter, so wonder_gc never reaches these two gates; the
    # guard still applies them, as promotion does.
    store.insert_belief(_phantom())
    _corroborate(store)
    store.insert_feedback_event(
        belief_id="p1", valence=valence, source=source, created_at=_ts(1),
    )
    assert promotion_guarded_ids(store, ["p1"]) == set()
    assert _collected(store) is False


def test_the_guard_only_reports_the_ids_it_was_given(
    store: MemoryStore,
) -> None:
    store.insert_belief(_phantom())
    _corroborate(store)
    assert promotion_guarded_ids(store, []) == set()
    assert promotion_guarded_ids(store, ["other"]) == set()


_LAZY_INGEST_PROBE = """
import sys
from aelfrice.models import BELIEF_SPECULATIVE, ORIGIN_SPECULATIVE, Belief
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_gc

store = MemoryStore(sys.argv[1])
wonder_gc(store, ttl_days=14, dry_run=True)
print("empty", "aelfrice.ingest" in sys.modules)
store.insert_belief(Belief(
    id="p1", content="The cache expires after ten minutes.",
    content_hash="wonder_p1", alpha=0.3, beta=1.0, type=BELIEF_SPECULATIVE,
    lock_level="none", locked_at=None,
    created_at="2001-01-01T00:00:00+00:00", last_retrieved_at=None,
    origin=ORIGIN_SPECULATIVE,
))
wonder_gc(store, ttl_days=14, dry_run=True)
print("stale", "aelfrice.ingest" in sys.modules)
store.close()
"""


@pytest.mark.timeout(60)
def test_a_sweep_with_no_stale_phantom_does_not_import_ingest(
    tmp_path: Path,
) -> None:
    # The guard returns before its lazy aelfrice.ingest import when GC
    # finds no stale phantom, so a SessionStart sweep on such a store
    # doesn't pay for that import. A subprocess gives a clean sys.modules;
    # the stale run is the control that shows the probe sees the import.
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_INGEST_PROBE, str(tmp_path / "m.db")],
        capture_output=True, text=True, encoding="utf-8", timeout=50,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["empty", "False", "stale", "True"]
