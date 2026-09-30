"""#1635: a burst of re-assertions is one episode, not N corroborations.

Every consumer of the corroboration count read "at least N rows across at
least 2 sessions" as independent evidence. A scripted replay gets a new
session id every few seconds, so one afternoon of headless evaluation runs
put 67 beliefs into `aelf core`, which made them eligible for the
SessionStart `<core>` lane, with 13 or 27 rows each inside about 1.5
minutes.

A belief's sightings are its creation and its corroboration rows. Sightings
closer together than `CORROBORATION_EPISODE_GAP_SECONDS` (one hour) form one
episode, and the corroboration arm needs `CORROBORATION_MIN_EPISODES` (two)
in all four consumers: `aelf core`, the `<core>` lane, and both promotion
selectors. Operator ruling 2026-09-29: creation counts as a sighting, which
accepts one residual case -- a belief that already existed, then hit by a
burst -- pinned below.
"""
from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.store import MemoryStore

_BURST = "b0burst0000000000"
_RECUR = "b0recur0000000000"


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "span.db"))


def _belief(bid: str, created_at: str, *, alpha: float = 1.0) -> Belief:
    return Belief(
        id=bid, content=f"belief {bid}", content_hash=f"h{bid}", alpha=alpha,
        beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at=created_at, last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _corr(store: MemoryStore, bid: str, times: list[str]) -> None:
    for i, ts in enumerate(times):
        store.record_corroboration(
            bid, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            session_id=f"{bid}-s{i}", ts=ts,
        )


def _burst_times(day: int = 1) -> list[str]:
    """13 real times inside two minutes, 12:00:00 .. 12:01:48."""
    return [f"2026-08-{day:02d}T12:{(i * 9) // 60:02d}:{(i * 9) % 60:02d}+00:00"
            for i in range(13)]


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "span.db"
    store = MemoryStore(str(path))
    try:
        # Born inside the burst: creation and every row are one episode.
        store.insert_belief(_belief(_BURST, "2026-08-01T12:00:00+00:00"))
        _corr(store, _BURST, _burst_times())
        # Said on day 1, said again on days 2, 5, and 9.
        store.insert_belief(_belief(_RECUR, "2026-08-01T09:00:00+00:00"))
        _corr(store, _RECUR, [f"2026-08-{d:02d}T09:00:00+00:00" for d in (2, 5, 9)])
    finally:
        store.close()
    yield path


def _core_ids(argv: list[str]) -> str:
    from aelfrice.cli import main

    buf = io.StringIO()
    assert main(argv=argv, out=buf) == 0
    return buf.getvalue()


@pytest.mark.timeout(30)
def test_aelf_core_admits_recurrence_and_not_a_burst(db: Path) -> None:
    out = _core_ids(["core"])
    assert _RECUR in out
    assert _BURST not in out


@pytest.mark.timeout(30)
def test_the_session_start_core_lane_admits_recurrence_and_not_a_burst(
    db: Path, tmp_path: Path,
) -> None:
    from aelfrice.hook import _build_session_start_subblock

    store = MemoryStore(str(db))
    try:
        block = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert _RECUR in block
    assert _BURST not in block


@pytest.mark.timeout(30)
@pytest.mark.parametrize("selector", ["phantoms", "snapshots"])
def test_the_promotion_selectors_skip_a_burst(db: Path, selector: str) -> None:
    from aelfrice.models import ORIGIN_SPECULATIVE

    # Phantoms are speculative; snapshots must NOT be (#1132 keeps phantoms
    # out of the count-driven retention promotion).
    origin = ORIGIN_SPECULATIVE if selector == "phantoms" else ORIGIN_AGENT_INFERRED
    store = MemoryStore(str(db))
    try:
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE beliefs SET origin = ?, retention_class = ?",
            (origin, "snapshot"),
        )
        store._conn.commit()  # pyright: ignore[reportPrivateUsage]
        found = (store.find_promotable_phantoms() if selector == "phantoms"
                 else store.find_promotable_snapshots())
    finally:
        store.close()
    ids = {b.id for b in found}
    assert _RECUR in ids
    assert _BURST not in ids


@pytest.mark.timeout(30)
def test_a_burst_is_not_labeled_corroboration(tmp_path: Path) -> None:
    """Found by review: `aelf core` listed a burst's count as a signal.

    The belief here is in core on its posterior; the corroboration label
    must follow the same rule as the gate.
    """
    store = MemoryStore(str(tmp_path / "span.db"))
    try:
        store.insert_belief(_belief(_BURST, "2026-08-01T12:00:00+00:00", alpha=8.0))
        _corr(store, _BURST, _burst_times())
    finally:
        store.close()
    rows = json.loads(_core_ids(["core", "--json"]))
    row = next(r for r in rows if r["id"] == _BURST)
    assert "corroboration" not in row["signals"]
    assert "posterior" in row["signals"]
    assert "CORR=" not in _core_ids(["core"])


@pytest.mark.timeout(30)
def test_a_real_recurrence_is_labeled_corroboration(db: Path) -> None:
    """Found by review: only the negative was tested, so dropping the
    episodes from the labels (every belief unlabeled) went unnoticed."""
    rows = json.loads(_core_ids(["core", "--json"]))
    row = next(r for r in rows if r["id"] == _RECUR)
    assert "corroboration" in row["signals"]
    assert "CORR=3" in _core_ids(["core"])


@pytest.mark.timeout(30)
def test_two_episodes_alone_do_not_make_a_belief_core(
    tmp_path: Path,
) -> None:
    """One re-assertion two days on: two episodes, but a count of one.

    Found by review: the hook's count threshold had no test, so a mutant
    that dropped it admitted this belief to `<core>`.
    """
    from aelfrice.hook import _build_session_start_subblock

    path = tmp_path / "span.db"
    store = MemoryStore(str(path))
    try:
        store.insert_belief(_belief(_RECUR, "2026-08-01T09:00:00+00:00"))
        _corr(store, _RECUR, ["2026-08-03T09:00:00+00:00"])
        assert store.corroboration_episodes() == {_RECUR: 2}
        block = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert _RECUR not in block
    assert _RECUR not in _core_ids(["core"])


@pytest.mark.timeout(30)
def test_a_later_sitting_counts_even_within_one_hour(tmp_path: Path) -> None:
    """Said on day 1, said twice more on day 3 ten seconds apart: two episodes.

    Found by review: a span over the rows alone ignored the creation and
    dropped this belief, which really did recur. Exactly two episodes and a
    count of exactly two, so both of the hook's thresholds sit at their
    boundary; a review found neither boundary pinned in the `<core>` lane.
    """
    from aelfrice.hook import _build_session_start_subblock

    store = MemoryStore(str(tmp_path / "span.db"))
    try:
        store.insert_belief(_belief(_RECUR, "2026-09-21T02:34:27+00:00"))
        _corr(store, _RECUR, ["2026-09-23T01:18:15+00:00",
                              "2026-09-23T01:18:25+00:00"])
        assert store.corroboration_episodes() == {_RECUR: 2}
        block = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert _RECUR in block
    assert _RECUR in _core_ids(["core"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize("selector", ["phantoms", "snapshots"])
def test_the_promotion_selectors_admit_exactly_two_episodes(
    tmp_path: Path, selector: str,
) -> None:
    """Found by review: the selectors' fixture had four episodes, so a
    selector that needed three passed every test."""
    from aelfrice.models import ORIGIN_SPECULATIVE

    origin = ORIGIN_SPECULATIVE if selector == "phantoms" else ORIGIN_AGENT_INFERRED
    store = MemoryStore(str(tmp_path / "two.db"))
    try:
        store.insert_belief(_belief(_RECUR, "2026-08-01T09:00:00+00:00"))
        # Day 3, three sessions a few seconds apart: one more episode.
        _corr(store, _RECUR, [f"2026-08-03T09:00:0{i}+00:00" for i in range(3)])
        assert store.corroboration_episodes() == {_RECUR: 2}
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE beliefs SET origin = ?, retention_class = ?",
            (origin, "snapshot"),
        )
        store._conn.commit()  # pyright: ignore[reportPrivateUsage]
        found = (store.find_promotable_phantoms() if selector == "phantoms"
                 else store.find_promotable_snapshots())
    finally:
        store.close()
    assert _RECUR in {b.id for b in found}


@pytest.mark.timeout(30)
def test_an_old_belief_hit_by_one_burst_still_counts(tmp_path: Path) -> None:
    """The residual case the ruling accepted, pinned so it cannot drift.

    Creation and a later burst are two episodes. Excluding this would also
    exclude a real re-assertion in one sitting days after the first.
    """
    store = MemoryStore(str(tmp_path / "span.db"))
    try:
        store.insert_belief(_belief(_BURST, "2026-07-01T09:00:00+00:00"))
        _corr(store, _BURST, _burst_times())
        assert store.corroboration_episodes() == {_BURST: 2}
    finally:
        store.close()


@pytest.mark.timeout(30)
@pytest.mark.parametrize(("second", "episodes"), [
    ("2026-08-01T12:59:59+00:00", 1),   # 3599 s: the same episode
    ("2026-08-01T12:59:59.600+00:00", 1),  # found by review: rounds up
    ("2026-08-01T12:59:59.998+00:00", 1),  # and at one or two decimals
    ("2026-08-01T13:00:00+00:00", 2),   # 3600 s: a new one
])
def test_the_episode_gap_is_one_hour(
    tmp_path: Path, second: str, episodes: int,
) -> None:
    """Pins the gap itself, and that the boundary counts.

    `julianday` arithmetic puts an exact hour at 3599.99998 seconds, so an
    unrounded gap would miss the boundary it names.
    """
    store = MemoryStore(str(tmp_path / "gap.db"))
    try:
        store.insert_belief(_belief("b0gap000000000000", "2026-08-01T12:00:00+00:00"))
        _corr(store, "b0gap000000000000", [second])
        assert store.corroboration_episodes() == {"b0gap000000000000": episodes}
    finally:
        store.close()


@pytest.mark.timeout(30)
def test_sightings_are_instants_not_text(tmp_path: Path) -> None:
    """12:00+02:00 is 10:00Z: two hours before 12:00Z, yet it sorts after
    it as text."""
    store = MemoryStore(str(tmp_path / "tz.db"))
    try:
        store.insert_belief(_belief("b0tz0000000000000", "2026-08-01T12:00:00Z"))
        _corr(store, "b0tz0000000000000", ["2026-08-01T12:00:00+02:00"])
        assert store.corroboration_episodes() == {"b0tz0000000000000": 2}
    finally:
        store.close()


@pytest.mark.timeout(30)
def test_an_unparseable_row_does_not_erase_the_others(tmp_path: Path) -> None:
    """Found by review: one bad row at the text MIN or MAX nulled the span."""
    store = MemoryStore(str(tmp_path / "bad.db"))
    try:
        store.insert_belief(_belief("b0bad000000000000", "2026-08-01T09:00:00Z"))
        _corr(store, "b0bad000000000000",
              ["2026-08-05T09:00:00Z", "unknown", ""])
        assert store.corroboration_episodes() == {"b0bad000000000000": 2}
    finally:
        store.close()


@pytest.mark.timeout(30)
def test_a_bulk_backfill_keeps_each_turns_own_time(tmp_path: Path) -> None:
    """One ingest of three sessions over three days is three episodes.

    Sightings read `ingested_at`, so the rule is only safe if a backfill
    stamps the turn's time rather than the moment of the backfill.
    """
    from aelfrice.ingest import ingest_jsonl

    path = tmp_path / "t.jsonl"
    text = "The configuration file lives at /etc/aelfrice/conf."
    path.write_text("".join(
        json.dumps({"role": "user", "text": text, "session_id": f"s{d}",
                    "ts": f"2026-08-{d:02d}T09:00:00Z"}) + "\n"
        for d in (1, 2, 3)
    ))
    store = MemoryStore(str(tmp_path / "bf.db"))
    try:
        ingest_jsonl(store, path)
        episodes = store.corroboration_episodes()
    finally:
        store.close()
    assert list(episodes.values()) == [3]


@pytest.mark.timeout(30)
def test_only_beliefs_with_a_parseable_row_are_listed(tmp_path: Path) -> None:
    """A belief with no corroboration row has no episodes to report, and a
    row whose time cannot be parsed is not a sighting."""
    store = MemoryStore(str(tmp_path / "list.db"))
    try:
        store.insert_belief(_belief("b0none00000000000", "2026-08-01T09:00:00Z"))
        # insert_belief refuses a non-date created_at since #1629, so a
        # row like this exists only in a store written before that; plant
        # it the way such a store holds it.
        store.insert_belief(_belief("b0junk00000000000", "2026-08-01T09:00:00Z"))
        store._conn.execute(  # noqa: SLF001 - reproduce a pre-#1629 row
            "UPDATE beliefs SET created_at = 'unknown' WHERE id = ?",
            ("b0junk00000000000",),
        )
        _corr(store, "b0junk00000000000", ["unknown", "also unknown"])
        store.insert_belief(_belief("b0one000000000000", "2026-08-01T09:00:00Z"))
        _corr(store, "b0one000000000000", ["2026-08-01T09:00:05Z", "garbage"])
        assert store.corroboration_episodes() == {"b0one000000000000": 1}
    finally:
        store.close()
