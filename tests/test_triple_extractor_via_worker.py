"""`git` ingest rows already in a store still derive and replay.

The commit hook wrote one `ingest_log` row per phrase, with source kind
`git` and `raw_meta.call_site = commit_ingest`, then ran the derivation
worker. The hook no longer writes, but stores keep those rows, and the
worker and full-equality replay still read them. These tests build the
rows the way the hook wrote them and check what the read side does.

Falsifiable hypotheses:

    the worker stamps every row; one phrase yields one belief across rows
    and sessions; full-equality replay reports no drift, including after a
    phrase repeats under a second session.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.classification_core import INGEST_CLASSIFIER_VERSION, rule_set_hash
from aelfrice.derivation_worker import run_worker
from aelfrice.models import CORROBORATION_SOURCE_COMMIT_INGEST, INGEST_SOURCE_GIT
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "stored-git-rows.db"))
    yield s
    s.close()


def _store_git_rows(
    store: MemoryStore, phrases: list[str], session_id: str = "commit-1",
) -> None:
    """Write `phrases` as the commit hook did, then run the worker."""
    for phrase in phrases:
        store.record_ingest(
            source_kind=INGEST_SOURCE_GIT,
            raw_text=phrase,
            session_id=session_id,
            ts="2026-01-01T00:00:00+00:00",
            raw_meta={"call_site": CORROBORATION_SOURCE_COMMIT_INGEST},
            classifier_version=INGEST_CLASSIFIER_VERSION,
            rule_set_hash=rule_set_hash(),
        )
    run_worker(store)


def _corroborations(store: MemoryStore) -> int:
    row = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) FROM belief_corroborations"
    ).fetchone()
    return int(row[0])


def test_every_stored_row_is_stamped(store: MemoryStore) -> None:
    _store_git_rows(store, ["the worker", "the spec"])
    assert store.list_unstamped_ingest_log() == []
    assert store.count_beliefs() == 2


def test_beliefs_from_stored_rows_are_agent_inferred(store: MemoryStore) -> None:
    """Regression for #224: not the model default origin."""
    _store_git_rows(store, ["the worker", "the spec"])
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT origin, session_id FROM beliefs"
    ).fetchall()
    assert [(r["origin"], r["session_id"]) for r in rows] == [
        ("agent_inferred", "commit-1"), ("agent_inferred", "commit-1"),
    ]


def test_one_phrase_is_one_belief_across_sessions(store: MemoryStore) -> None:
    """The id namespace makes a phrase resolve to one belief. A second
    session corroborates it, and its own session doesn't."""
    _store_git_rows(store, ["the cache layer", "the cache layer"], "commit-1")
    assert store.count_beliefs() == 1
    assert _corroborations(store) == 0
    _store_git_rows(store, ["the cache layer"], "commit-2")
    assert store.count_beliefs() == 1
    assert _corroborations(store) == 1


def test_replay_full_equality_passes(store: MemoryStore) -> None:
    """Hypothesis (CI gate): full-equality replay reports zero drift over
    stored `git` rows, including a phrase repeated under a second session."""
    phrases = ["the worker", "the spec", "aelfrice", "ingest_log"]
    _store_git_rows(store, phrases, "commit-1")
    _store_git_rows(store, phrases, "commit-2")
    report = replay_full_equality(store)
    assert report.total_log_rows == 8
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0
